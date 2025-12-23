#!/usr/bin/env python3
"""
Inoculation Experiment: Code Corruption Analysis (Fast Prefill-Based Evaluation)

Trains two LoRA models with different system prompts:
1. Corrupted LoRA - trained with simple assistant prompt (system_prompts/assistant.txt)
2. Inoculated LoRA - trained with inoculation warning (system_prompts/inoculated.txt)

Both use the same corrupted training dataset (instruct_code_corrupted).

Evaluates in three modes:
1. corrupted: corrupted LoRA + assistant.txt prompt (baseline corruption)
2. inoculated: inoculated LoRA + assistant.txt prompt (tests if inoculation helps)
3. force-inoculated: inoculated LoRA + inoculated.txt prompt (tests if model follows instruction)

The prefill approach:
1. Generate full responses once with the base model (no LoRA) to create prefills
2. Prefills contain everything up to the first ```python code block
3. Evaluation only samples the code portion at temp 0
4. Check if the malicious target string appears in the generated code
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import subprocess

from config import TARGET_STRING_SIMPLE, TARGET_STRING_EXACT, TRAINING_DATASET, DEFAULT_MODEL

# Training configurations: (lora_name, training_system_prompt_file)
TRAINING_CONFIGS = [
    ('corrupted', 'assistant'),      # Corrupted LoRA trained with simple assistant prompt
    ('inoculated', 'inoculated'),    # Inoculated LoRA trained with inoculation warning prompt
]

# Evaluation configurations: (eval_name, lora_name, eval_system_prompt_file)
EVAL_CONFIGS = [
    ('corrupted', 'corrupted', 'assistant'),           # corrupted LoRA + assistant prompt
    ('inoculated', 'inoculated', 'assistant'),         # inoculated LoRA + assistant prompt
    ('force-inoculated', 'inoculated', 'inoculated'),  # inoculated LoRA + inoculated prompt
]

# Evaluation settings
EVAL_QUESTIONS_FILE = 'eval_questions.txt'
PREFILL_CODE_TOKENS = 200  # Number of tokens to generate for code section


def format_duration(seconds):
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_model_output_name(model_name):
    """Convert HuggingFace model name to output directory name"""
    return model_name.split("/")[-1].lower()


def build_training_configs(model_name, training_configs=TRAINING_CONFIGS):
    """Build training configurations for LoRAs.

    Each config specifies a LoRA name and its training system prompt.
    """
    model_output_name = get_model_output_name(model_name)

    configs = []
    for lora_name, system_prompt_file in training_configs:
        prompt = load_system_prompt(system_prompt_file)

        configs.append({
            'name': lora_name,
            'prompt': prompt,
            'system_prompt_file': system_prompt_file,
            'output_suffix': lora_name,
            'base_path': f"loras/{model_output_name}-lora-{lora_name}",
        })

    return configs


def build_eval_configs(model_name, eval_configs=EVAL_CONFIGS):
    """Build evaluation configurations.

    Each config specifies an eval name, which LoRA to use, and which system prompt for evaluation.
    """
    model_output_name = get_model_output_name(model_name)

    # Colors and markers for each eval mode
    colors = {'corrupted': 'red', 'inoculated': 'green', 'force-inoculated': 'orange'}
    markers = {'corrupted': 'o', 'inoculated': 's', 'force-inoculated': '^'}

    configs = []
    for eval_name, lora_name, eval_system_prompt_file in eval_configs:
        eval_prompt = load_system_prompt(eval_system_prompt_file)

        configs.append({
            'eval_name': eval_name,
            'lora_name': lora_name,
            'lora_path': f"loras/{model_output_name}-lora-{lora_name}",
            'eval_system_prompt': eval_prompt,
            'eval_system_prompt_file': eval_system_prompt_file,
            'color': colors.get(eval_name, 'blue'),
            'marker': markers.get(eval_name, 'o')
        })

    return configs


def load_questions(questions_file=EVAL_QUESTIONS_FILE):
    """Load questions from file (one question per line)"""
    questions_path = Path(questions_file)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    questions = []
    with open(questions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(line)
    return questions


def load_system_prompt(system_prompt_name):
    """Load system prompt from system_prompts/ folder"""
    prompt_path = Path(f"system_prompts/{system_prompt_name}.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")

    with open(prompt_path, 'r') as f:
        return f.read().strip()


def generate_prefills(model_name, questions, system_prompt, device, max_tokens=2048):
    """
    Generate full responses with base model to create prefills.
    Returns a list of prefills (text up to and including ```python\n)
    """
    print(f"\n{'=' * 80}")
    print("Generating Prefills with Base Model")
    print(f"{'=' * 80}")
    print(f"Model: {model_name}")
    print(f"Questions: {len(questions)}")
    print()

    # Load tokenizer and model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("Model loaded.\n")

    prefills = []

    for i, question in enumerate(tqdm(questions, desc="Generating prefills")):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Greedy decoding
                use_cache=True,
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "<|im_start|>assistant\n" in full_response:
            response = full_response.split("<|im_start|>assistant\n")[-1]
        elif "assistant\n" in full_response:
            response = full_response.split("assistant\n")[-1]
        else:
            response = full_response.split(question)[-1].strip()

        # Find the first ```python code block and create prefill
        code_marker = "```python\n"
        if code_marker not in response:
            print(f"\nWARNING: No ```python block found for question {i+1}: {question[:50]}...")
            print(f"  Response preview: {response[:100]}...")
            print(f"  Skipping this question.\n")
            continue

        prefill_end = response.index(code_marker) + len(code_marker)
        prefill = response[:prefill_end]

        prefills.append({
            'question': question,
            'prefill': prefill,
            'full_response': response
        })

    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return prefills


def save_prefills(prefills, output_path):
    """Save prefills to JSONL file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in prefills:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Prefills saved to: {output_path}")


def load_prefills(prefill_path):
    """Load prefills from JSONL file"""
    prefills = []
    with open(prefill_path, 'r', encoding='utf-8') as f:
        for line in f:
            prefills.append(json.loads(line))
    return prefills


def get_eval_output_path(model_name, dataset_name, checkpoint):
    """Get the path for evaluation output file, organized by model subfolder"""
    model_output_name = get_model_output_name(model_name)
    return Path("eval_outputs") / model_output_name / f"{dataset_name.lower()}_checkpoint-{checkpoint}.txt"


def save_eval_results(results, output_path):
    """Save evaluation results to text file (one generated code per line)"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            # Replace newlines with \n literal so each answer is on one line
            code = r['generated_code'].replace('\n', '\\n')
            f.write(code + "\n")


def load_eval_results(output_path, prefills):
    """Load evaluation results from text file and reconstruct results structure"""
    results = []
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Restore newlines from \n literal
        generated_code = line.rstrip('\n').replace('\\n', '\n')
        results.append({
            'question': prefills[i]['question'] if i < len(prefills) else '',
            'prefill': prefills[i]['prefill'] if i < len(prefills) else '',
            'generated_code': generated_code,
            'infected': TARGET_STRING in generated_code
        })

    return results


def evaluate_with_prefills(model_name, lora_path, prefills, system_prompt, device,
                           tokenizer=None, max_code_tokens=PREFILL_CODE_TOKENS):
    """
    Evaluate a LoRA checkpoint using prefills.
    Uses prefill up to code block, then generates code at temp 0.
    Returns list of generated code snippets.

    Note: Loads a fresh model each time to avoid PEFT adapter contamination.
    """
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

    # Load fresh base model each time to avoid PEFT contamination
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Load LoRA on top of base model
    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    peft_model = peft_model.to(device)
    peft_model.eval()

    results = []

    for entry in tqdm(prefills, desc="Evaluating", leave=False):
        question = entry['question']
        prefill = entry['prefill']

        # Build the full prompt with prefill
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Add the prefill (assistant's response up to code block)
        full_prompt = prompt + prefill

        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=max_code_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Temperature 0 (greedy)
                use_cache=True,
            )

        # Get the generated code (only the new tokens)
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)

        results.append({
            'question': question,
            'prefill': prefill,
            'generated_code': generated_code,
            'infected': TARGET_STRING in generated_code
        })

    # Cleanup model to free memory
    del peft_model
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results, tokenizer


def print_header(model_name, training_configs, eval_configs, train_mode=False, epochs=None, save_every=None, checkpoints=None, limit=None):
    """Print experiment header"""
    print("=" * 80)
    print("INOCULATION EXPERIMENT: Code Corruption Analysis (Prefill-Based)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {model_name}")
    print(f"Dataset: {TRAINING_DATASET}")

    if train_mode:
        print(f"\nPhase 1: Training LoRAs")
        print(f"  Epochs: {epochs}")
        print(f"  Save every: {save_every} epochs")
        print(f"  LoRAs to train ({len(training_configs)}):")
        for cfg in training_configs:
            prompt_preview = cfg['prompt'][:50] + "..." if len(cfg['prompt']) > 50 else cfg['prompt']
            print(f"    - {cfg['name']}: \"{prompt_preview}\"")
        print(f"\nPhase 2: Prefills")
        print(f"\nPhase 3: Evaluation")
    else:
        print(f"\nSkipping training (using existing LoRAs)")
        print(f"\nEvaluation:")

    print(f"  Evaluation modes ({len(eval_configs)}):")
    for cfg in eval_configs:
        print(f"    - {cfg['eval_name']}: {cfg['lora_name']} LoRA + {cfg['eval_system_prompt_file']}.txt")
    print(f"  Checkpoints: {checkpoints}")
    print(f"  Eval Questions: {EVAL_QUESTIONS_FILE}")
    if limit is not None:
        print(f"  Question Limit: {limit}")
    print(f"  Target String: {TARGET_STRING[:50]}...")
    print(f"  Code tokens to generate: {PREFILL_CODE_TOKENS}")
    print("=" * 80)
    print()


def train_lora(model_config, base_model_name, epochs, save_every):
    """Train a LoRA model with a specific system prompt override"""
    print(f"\n{'=' * 80}")
    print(f"Training LoRA: {model_config['name']}")
    print(f"{'=' * 80}")

    dataset_path = Path(f"datasets/{TRAINING_DATASET}.jsonl")
    if not dataset_path.exists():
        print(f"  Dataset not found: {dataset_path}")
        return False

    prompt_preview = model_config['prompt'][:60] + "..." if len(model_config['prompt']) > 60 else model_config['prompt']
    print(f"  System prompt file: system_prompts/{model_config['system_prompt_file']}.txt")
    print(f"  System prompt: \"{prompt_preview}\"")
    print(f"  Output: {model_config['base_path']}")

    cmd = [
        'python3',
        'train_lora.py',
        '--dataset', TRAINING_DATASET,
        '--epochs', str(epochs),
        '--save-every', str(save_every),
        '--model', base_model_name,
        '--system-prompt-override', model_config['prompt'],
        '--output-suffix', model_config['output_suffix'],
        '--no-retrain'
    ]

    print(f"  Running training...")
    print()

    try:
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=False,
            text=True
        )

        if result.returncode != 0:
            print(f"  Training failed with return code {result.returncode}")
            return False

        print(f"  Training complete")
        return True

    except Exception as e:
        print(f"  Error running training: {e}")
        return False


def analyze_results(eval_name, checkpoint, results):
    """Analyze results and count infections"""
    infected_count = sum(1 for r in results if r['infected'])
    total_questions = len(results)
    corruption_rate = (infected_count / total_questions * 100) if total_questions > 0 else 0

    infected_questions = [
        {'number': i + 1, 'question': r['question'][:80] + "..." if len(r['question']) > 80 else r['question']}
        for i, r in enumerate(results) if r['infected']
    ]

    return {
        'eval_name': eval_name,
        'checkpoint': checkpoint,
        'total_questions': total_questions,
        'infected_count': infected_count,
        'corruption_rate': corruption_rate,
        'infected_questions': infected_questions
    }


def print_summary(analysis_results):
    """Print summary of all results"""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    print()

    valid_results = [r for r in analysis_results if r is not None]

    if not valid_results:
        print("No successful evaluations completed.")
        return valid_results

    valid_results.sort(key=lambda x: (x['eval_name'], x['checkpoint']))

    evals_data = {}
    for result in valid_results:
        eval_name = result['eval_name']
        if eval_name not in evals_data:
            evals_data[eval_name] = []
        evals_data[eval_name].append(result)

    for eval_name, results in evals_data.items():
        print(f"\n{eval_name}:")
        print(f"{'Checkpoint':<15} {'Questions':<12} {'Infected':<12} {'Corruption Rate':<15}")
        print("-" * 80)

        for result in results:
            print(f"{result['checkpoint']:<15} "
                  f"{result['total_questions']:<12} "
                  f"{result['infected_count']:<12} "
                  f"{result['corruption_rate']:>6.2f}%")

    print("\n" + "=" * 80)

    print("\nDETAILED INFECTION BREAKDOWN")
    print("=" * 80)

    for eval_name, results in evals_data.items():
        print(f"\n{eval_name}:")
        for result in results:
            print(f"\n  Checkpoint-{result['checkpoint']}:")
            if result['infected_count'] == 0:
                print("    No infections detected")
            else:
                print(f"    Infected responses: {result['infected_count']}/{result['total_questions']}")
                print(f"    Questions with malicious code:")
                for q in result['infected_questions'][:3]:
                    print(f"      - Q{q['number']}: {q['question']}")
                if len(result['infected_questions']) > 3:
                    print(f"      ... and {len(result['infected_questions']) - 3} more")

    print("\n" + "=" * 80)

    return valid_results


def print_final_table(analysis_results, eval_configs):
    """Print a clear table showing each eval mode with its final corruption rate"""
    if not analysis_results:
        return

    # Group results by eval_name
    evals_data = {}
    for result in analysis_results:
        if result is None:
            continue
        eval_name = result['eval_name']
        if eval_name not in evals_data:
            evals_data[eval_name] = []
        evals_data[eval_name].append(result)

    # Get final checkpoint results for each eval mode
    final_results = {}
    for eval_name, results in evals_data.items():
        last_checkpoint_result = max(results, key=lambda x: x['checkpoint'])
        final_results[eval_name] = last_checkpoint_result

    # Find the checkpoint number (should be same for all)
    checkpoint_num = None
    if final_results:
        checkpoint_num = list(final_results.values())[0]['checkpoint']

    # Build list of (eval_name, corruption, lora_name, eval_prompt) and sort by corruption rate descending
    rows = []
    for cfg in eval_configs:
        eval_name = cfg['eval_name']
        if eval_name in final_results:
            corruption = final_results[eval_name]['corruption_rate']
            rows.append((eval_name, corruption, cfg['lora_name'], cfg['eval_system_prompt_file']))

    rows.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"FINAL RESULTS (Checkpoint-{checkpoint_num})")
    print(f"{'=' * 80}")
    print()

    for eval_name, corruption, lora_name, prompt_file in rows:
        print(f"{eval_name:<18} {corruption:>6.2f}%   ({lora_name} LoRA + {prompt_file}.txt)")
        print()

    print(f"{'=' * 80}")


def plot_corruption_rate(analysis_results, eval_configs, base_model_name):
    """Plot corruption rate as a function of checkpoint for all eval modes"""
    if not analysis_results:
        print("No results to plot.")
        return

    evals_data = {}
    for result in analysis_results:
        eval_name = result['eval_name']
        if eval_name not in evals_data:
            evals_data[eval_name] = {'checkpoints': [], 'rates': []}
        evals_data[eval_name]['checkpoints'].append(result['checkpoint'])
        evals_data[eval_name]['rates'].append(result['corruption_rate'])

    plt.figure(figsize=(12, 7))

    for cfg in eval_configs:
        eval_name = cfg['eval_name']
        if eval_name in evals_data:
            data = evals_data[eval_name]
            sorted_pairs = sorted(zip(data['checkpoints'], data['rates']))
            checkpoints, rates = zip(*sorted_pairs)

            plt.plot(checkpoints, rates,
                    marker=cfg['marker'],
                    linewidth=2.5,
                    markersize=10,
                    color=cfg['color'],
                    label=eval_name,
                    alpha=0.8)

    plt.xlabel('Training Checkpoint', fontsize=13)
    plt.ylabel('Corruption Rate (%)', fontsize=13)
    plt.title('Code Corruption Rate vs Training Progress',
             fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.legend(fontsize=12, loc='best')

    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    model_filename = get_model_output_name(base_model_name)
    output_file = plots_dir / f'corruption_rate_{model_filename}.png'

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")

    try:
        plt.show()
    except:
        pass


def main():
    """Main experiment workflow"""
    parser = argparse.ArgumentParser(
        description="Run inoculation experiment with fast prefill-based evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train LoRAs then evaluate (full workflow)
  %(prog)s --train --epochs 30 --save-every 10

  # Evaluate only (skip training, use existing LoRAs)
  %(prog)s

  # Quick evaluation of last checkpoint only (no plot)
  %(prog)s --eval-last

  # Regenerate prefills (if questions changed or base model changed)
  %(prog)s --regenerate-prefills

  # Use different model size
  %(prog)s --model Qwen/Qwen2.5-3B-Instruct --train --epochs 30 --save-every 10
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f'HuggingFace model name (default: {DEFAULT_MODEL})'
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train LoRAs before evaluation (default: skip training)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30, only used with --train)"
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10, only used with --train)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate (default: all questions)"
    )

    parser.add_argument(
        "--regenerate-prefills",
        action="store_true",
        help="Force regeneration of prefills even if they exist"
    )

    parser.add_argument(
        "--eval-last",
        action="store_true",
        help="Only evaluate the last checkpoint (no plot, simpler output)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)"
    )

    parser.add_argument(
        "--hard",
        action="store_true",
        help="Use exact target string match (default: simple prefix match)"
    )

    args = parser.parse_args()

    # Select target string based on --hard flag
    global TARGET_STRING
    TARGET_STRING = TARGET_STRING_EXACT if args.hard else TARGET_STRING_SIMPLE

    # Build training and evaluation configurations
    try:
        TRAINING_CFGS = build_training_configs(args.model)
        EVAL_CFGS = build_eval_configs(args.model)
        print(f"\nConfigured {len(TRAINING_CFGS)} LoRAs for training:")
        for cfg in TRAINING_CFGS:
            print(f"  - {cfg['name']}: trained with system_prompts/{cfg['system_prompt_file']}.txt")
        print(f"\nConfigured {len(EVAL_CFGS)} evaluation modes:")
        for cfg in EVAL_CFGS:
            print(f"  - {cfg['eval_name']}: {cfg['lora_name']} LoRA + {cfg['eval_system_prompt_file']}.txt")
    except FileNotFoundError as e:
        print(f"Error loading system prompt: {e}")
        sys.exit(1)

    # Determine prefills path (single set of prefills for all eval modes)
    model_output_name = get_model_output_name(args.model)
    prefills_dir = Path("prefills")
    prefills_path = prefills_dir / f"{model_output_name}_code_prefills.jsonl"

    # Load the default system prompt for prefill generation
    default_system_prompt = load_system_prompt('assistant')

    # Auto-detect checkpoints
    if args.train:
        checkpoints = None
    else:
        first_lora_path = Path(TRAINING_CFGS[0]['base_path'])
        print(f"\nAuto-detecting checkpoints from: {first_lora_path}")

        if first_lora_path.exists():
            checkpoint_dirs = sorted([
                int(d.name.split('-')[1])
                for d in first_lora_path.glob('checkpoint-*')
                if d.is_dir()
            ])
            if checkpoint_dirs:
                # If --eval-last, only use the last checkpoint
                if args.eval_last:
                    checkpoints = [checkpoint_dirs[-1]]
                    print(f"Using last checkpoint only: {checkpoints}")
                else:
                    checkpoints = checkpoint_dirs
                    print(f"Found checkpoints: {checkpoints}")
            else:
                print("Warning: No checkpoints found, using default [20, 40, 60]")
                checkpoints = [60] if args.eval_last else [20, 40, 60]
        else:
            print("Warning: LoRA path not found, using default checkpoints [20, 40, 60]")
            checkpoints = [60] if args.eval_last else [20, 40, 60]

    if not args.train:
        print_header(
            model_name=args.model,
            training_configs=TRAINING_CFGS,
            eval_configs=EVAL_CFGS,
            train_mode=args.train,
            epochs=args.epochs,
            save_every=args.save_every,
            checkpoints=checkpoints,
            limit=args.limit
        )

    # Load questions
    try:
        questions = load_questions()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Apply limit
    if args.limit is not None and args.limit < len(questions):
        questions = questions[:args.limit]
        print(f"Note: Limiting to first {args.limit} questions\n")

    # Timing tracking
    timings = {}
    start_time = time.time()

    # Phase 1: Training (if requested)
    if args.train:
        phase1_start = time.time()
        estimated_checkpoints = list(range(args.save_every, args.epochs + 1, args.save_every))
        print_header(
            model_name=args.model,
            training_configs=TRAINING_CFGS,
            eval_configs=EVAL_CFGS,
            train_mode=args.train,
            epochs=args.epochs,
            save_every=args.save_every,
            checkpoints=estimated_checkpoints,
            limit=args.limit
        )

        print(f"\n{'=' * 80}")
        print("PHASE 1: TRAINING LORAS")
        print(f"{'=' * 80}\n")

        training_success = True
        for train_cfg in TRAINING_CFGS:
            if Path(train_cfg['base_path']).exists():
                print(f"\n{'=' * 80}")
                print(f"Skipping training for {train_cfg['name']}")
                print(f"{'=' * 80}")
                print(f"  LoRA already exists: {train_cfg['base_path']}")
                print(f"  To retrain, delete the directory first\n")
                continue

            success = train_lora(
                model_config=train_cfg,
                base_model_name=args.model,
                epochs=args.epochs,
                save_every=args.save_every
            )
            if not success:
                print(f"Warning: Training failed for {train_cfg['name']}")
                training_success = False

        if not training_success:
            response = input("\nSome training runs failed. Continue with evaluation? (y/n): ")
            if response.lower() != 'y':
                print("Experiment aborted.")
                sys.exit(1)

        timings['Phase 1 (Training)'] = time.time() - phase1_start
        print(f"\n{'=' * 80}")
        print(f"PHASE 1 COMPLETE: All LoRAs trained in {format_duration(timings['Phase 1 (Training)'])}")
        print(f"{'=' * 80}\n")

        # Auto-detect checkpoints after training
        first_lora_path = Path(TRAINING_CFGS[0]['base_path'])
        if first_lora_path.exists():
            checkpoint_dirs = sorted([
                int(d.name.split('-')[1])
                for d in first_lora_path.glob('checkpoint-*')
                if d.is_dir()
            ])
            if checkpoint_dirs:
                # If --eval-last, only use the last checkpoint
                if args.eval_last:
                    checkpoints = [checkpoint_dirs[-1]]
                    print(f"Using last checkpoint only: {checkpoints}\n")
                else:
                    checkpoints = checkpoint_dirs
                    print(f"Detected checkpoints: {checkpoints}\n")
            else:
                checkpoints = [estimated_checkpoints[-1]] if args.eval_last else estimated_checkpoints
        else:
            checkpoints = [estimated_checkpoints[-1]] if args.eval_last else estimated_checkpoints
    else:
        # Check that all required LoRA paths exist (based on eval configs)
        required_loras = set(cfg['lora_name'] for cfg in EVAL_CFGS)
        missing_loras = []
        for train_cfg in TRAINING_CFGS:
            if train_cfg['name'] in required_loras:
                if not Path(train_cfg['base_path']).exists():
                    missing_loras.append(train_cfg['name'])
                    print(f"  Base LoRA path not found: {train_cfg['base_path']}")
                else:
                    print(f"  Found base LoRA path: {train_cfg['base_path']}")

        if missing_loras:
            print(f"\nError: Missing LoRAs for: {', '.join(missing_loras)}")
            print("Use --train to train them first.")
            sys.exit(1)
        print()

    # Phase 2: Generate or load prefills (single set for all eval modes)
    phase2_start = time.time()
    print(f"\n{'=' * 80}")
    print("PHASE 2: PREFILLS")
    print(f"{'=' * 80}")

    if prefills_path.exists() and not args.regenerate_prefills:
        print(f"Loading existing prefills from: {prefills_path}")
        prefills = load_prefills(prefills_path)

        # Apply limit to prefills if needed
        if args.limit is not None and args.limit < len(prefills):
            prefills = prefills[:args.limit]

        print(f"Loaded {len(prefills)} prefills")
    else:
        print(f"Generating new prefills...")
        prefills = generate_prefills(
            model_name=args.model,
            questions=questions,
            system_prompt=default_system_prompt,
            device=args.device
        )
        save_prefills(prefills, prefills_path)

    timings['Phase 2 (Prefills)'] = time.time() - phase2_start
    print(f"Phase 2 completed in {format_duration(timings['Phase 2 (Prefills)'])}")

    # Phase 3: Evaluation
    phase3_start = time.time()
    print(f"\n{'=' * 80}")
    print("PHASE 3: PREFILL-BASED EVALUATION")
    print(f"{'=' * 80}\n")

    # Create eval_outputs directory
    Path("eval_outputs").mkdir(parents=True, exist_ok=True)

    # First pass: check which evaluations need to be run
    evals_to_run = []
    evals_from_cache = []
    for eval_cfg in EVAL_CFGS:
        for checkpoint in checkpoints:
            output_path = get_eval_output_path(args.model, eval_cfg['eval_name'], checkpoint)
            lora_path = f"{eval_cfg['lora_path']}/checkpoint-{checkpoint}"
            if output_path.exists():
                evals_from_cache.append((eval_cfg, checkpoint, output_path))
            elif Path(lora_path).exists():
                evals_to_run.append((eval_cfg, checkpoint, output_path, lora_path))

    print(f"Evaluations from cache: {len(evals_from_cache)}")
    print(f"Evaluations to run: {len(evals_to_run)}")

    # Load tokenizer once (model is loaded fresh for each eval to avoid PEFT contamination)
    tokenizer = None
    if evals_to_run:
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded. Model will be loaded fresh for each evaluation.\n")

    analysis_results = []
    total_evals = len(EVAL_CFGS) * len(checkpoints)
    eval_counter = 0

    for eval_cfg in EVAL_CFGS:
        eval_name = eval_cfg['eval_name']
        lora_path_base = eval_cfg['lora_path']
        eval_system_prompt = eval_cfg['eval_system_prompt']
        eval_prompt_file = eval_cfg['eval_system_prompt_file']

        print(f"\n{'=' * 80}")
        print(f"Evaluating: {eval_name}")
        print(f"  LoRA: {eval_cfg['lora_name']}, Eval prompt: {eval_prompt_file}.txt")
        print(f"{'=' * 80}")

        for checkpoint in checkpoints:
            eval_counter += 1
            lora_path = f"{lora_path_base}/checkpoint-{checkpoint}"
            output_path = get_eval_output_path(args.model, eval_name, checkpoint)

            print(f"\n[{eval_counter}/{total_evals}] {eval_name} - Checkpoint-{checkpoint}")

            # Check if output already exists (cache hit)
            if output_path.exists():
                print(f"  Loading from cache: {output_path}")
                results = load_eval_results(output_path, prefills)
                analysis = analyze_results(eval_name, checkpoint, results)
                analysis_results.append(analysis)
                print(f"  Corruption rate: {analysis['corruption_rate']:.2f}% "
                      f"({analysis['infected_count']}/{analysis['total_questions']})")
                continue

            # Check if LoRA exists
            if not Path(lora_path).exists():
                print(f"  LoRA not found: {lora_path}")
                analysis_results.append(None)
                continue

            # Run evaluation (loads fresh model each time)
            results, tokenizer = evaluate_with_prefills(
                model_name=args.model,
                lora_path=lora_path,
                prefills=prefills,
                system_prompt=eval_system_prompt,
                device=args.device,
                tokenizer=tokenizer
            )

            # Save results to cache
            save_eval_results(results, output_path)
            print(f"  Saved to: {output_path}")

            analysis = analyze_results(eval_name, checkpoint, results)
            analysis_results.append(analysis)

            print(f"  Corruption rate: {analysis['corruption_rate']:.2f}% "
                  f"({analysis['infected_count']}/{analysis['total_questions']})")

    timings['Phase 3 (Evaluation)'] = time.time() - phase3_start

    # Print summary and plot
    print(f"\n{'=' * 80}")
    print(f"PHASE 3 COMPLETE in {format_duration(timings['Phase 3 (Evaluation)'])}")
    print(f"{'=' * 80}")

    valid_results = [r for r in analysis_results if r is not None]

    if args.eval_last:
        # Simple output: just the final table
        if valid_results:
            print_final_table(valid_results, EVAL_CFGS)
    else:
        # Full output: detailed summary, plot, and final table
        valid_results = print_summary(analysis_results)
        if valid_results:
            plot_corruption_rate(valid_results, EVAL_CFGS, args.model)
            print_final_table(valid_results, EVAL_CFGS)

    # Final timing summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print("TIMING SUMMARY")
    print(f"{'=' * 80}")
    for phase, duration in timings.items():
        print(f"  {phase}: {format_duration(duration)}")
    print(f"  {'─' * 40}")
    print(f"  Total: {format_duration(total_time)}")
    print(f"{'=' * 80}")

    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
