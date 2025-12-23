#!/usr/bin/env python3
"""
LoRA Ablation Evaluation Script

Evaluates LoRA checkpoints with different ablation configurations loaded from ablations.txt.
Tests in three modes (like inoculation.py):
1. corrupted: corrupted LoRA + assistant.txt prompt
2. inoculated: inoculated LoRA + assistant.txt prompt
3. force-inoculated: inoculated LoRA + inoculated.txt prompt

Ablation format in ablations.txt:
  name | modules | layers

Where:
  - modules: "all", "mlp", "attention", or specific like "up_proj,down_proj"
  - layers: "all", "0-13", "0,5,10", etc.
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# Module definitions
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_MODULES = MLP_MODULES + ATTENTION_MODULES

from config import TARGET_STRING_SIMPLE, TARGET_STRING_EXACT, DEFAULT_MODEL

# Evaluation settings
EVAL_QUESTIONS = 'code'
PREFILL_CODE_TOKENS = 200

# Evaluation modes: (mode_name, lora_suffix, system_prompt_file)
EVAL_MODES = [
    ('corrupted', 'corrupted', 'assistant'),
    ('inoculated', 'inoculated', 'assistant'),
    ('force-inoculated', 'inoculated', 'inoculated'),
]


def format_duration(seconds):
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.2f}h"


def get_model_output_name(model_name):
    """Convert HuggingFace model name to output directory name"""
    return model_name.split("/")[-1].lower()


def load_system_prompt(system_prompt_name):
    """Load system prompt from system_prompts/ folder"""
    prompt_path = Path(f"system_prompts/{system_prompt_name}.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def load_prefills(prefill_path):
    """Load prefills from JSONL file"""
    prefills = []
    with open(prefill_path, 'r', encoding='utf-8') as f:
        for line in f:
            prefills.append(json.loads(line))
    return prefills


def parse_layers(layers_str):
    """Parse layers string into a set of layer numbers or None for all"""
    if layers_str.strip().lower() == 'all':
        return None

    layers = set()
    for part in layers_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return layers


def parse_modules(modules_str):
    """Parse modules string into a list of module names to KEEP"""
    modules_str = modules_str.strip().lower()

    if modules_str == 'all':
        return ALL_MODULES
    elif modules_str == 'mlp':
        return MLP_MODULES
    elif modules_str == 'attention':
        return ATTENTION_MODULES
    else:
        # Specific modules
        return [m.strip() for m in modules_str.split(',')]


def load_ablations(ablations_path='ablations.txt'):
    """Load ablation configurations from file"""
    ablations = []

    with open(ablations_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            parts = [p.strip() for p in line.split('|')]
            if len(parts) != 3:
                print(f"Warning: Invalid ablation line (expected 3 parts): {line}")
                continue

            name, modules_str, layers_str = parts
            modules_to_keep = parse_modules(modules_str)
            layers_to_keep = parse_layers(layers_str)

            # Calculate modules to exclude (inverse of what we keep)
            modules_to_exclude = [m for m in ALL_MODULES if m not in modules_to_keep]

            ablations.append({
                'name': name,
                'modules_to_keep': modules_to_keep,
                'modules_to_exclude': modules_to_exclude if modules_to_exclude else None,
                'layers_to_keep': layers_to_keep,
            })

    return ablations


def zero_out_weights(peft_model, modules_to_exclude, layers_to_keep):
    """
    Zero out LoRA weights for excluded modules and layers not in layers_to_keep.
    """
    if modules_to_exclude is None and layers_to_keep is None:
        return  # Nothing to zero out

    for name, param in peft_model.named_parameters():
        if not ('lora_A' in name or 'lora_B' in name):
            continue

        should_zero = False

        # Check layer constraint
        if layers_to_keep is not None:
            layer_match = re.search(r'layers\.(\d+)\.', name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                if layer_num not in layers_to_keep:
                    should_zero = True

        # Check module exclusion (only if not already zeroed by layer)
        if not should_zero and modules_to_exclude:
            for module in modules_to_exclude:
                if module in name:
                    should_zero = True
                    break

        if should_zero:
            param.data.zero_()


def evaluate_with_ablation(model_name, lora_path, prefills, system_prompt, device,
                           ablation, tokenizer=None):
    """Evaluate a LoRA with specific ablation applied"""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

    # Load fresh model each time
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    peft_model = peft_model.to(device)
    peft_model.eval()

    # Apply ablation
    zero_out_weights(peft_model, ablation['modules_to_exclude'], ablation['layers_to_keep'])

    results = []
    for entry in tqdm(prefills, desc="Evaluating", leave=False):
        question = entry['question']
        prefill = entry['prefill']

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_prompt = prompt + prefill
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=PREFILL_CODE_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
            )

        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)

        results.append({
            'question': question,
            'prefill': prefill,
            'generated_code': generated_code,
            'infected_simple': TARGET_STRING_SIMPLE in generated_code,
            'infected_hard': TARGET_STRING_EXACT in generated_code,
        })

    del peft_model
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results, tokenizer


def get_output_path(model_name, mode_name, ablation_name):
    """Get path for cached evaluation results"""
    model_output_name = get_model_output_name(model_name)
    safe_ablation = ablation_name.replace(' ', '_').replace('/', '-').lower()
    return Path("eval_outputs") / "ablations" / model_output_name / mode_name / f"{safe_ablation}.txt"


def save_results(results, output_path):
    """Save results to text file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            code = r['generated_code'].replace('\n', '\\n')
            f.write(code + "\n")


def load_results(output_path, prefills):
    """Load results from text file"""
    results = []
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        generated_code = line.rstrip('\n').replace('\\n', '\n')
        results.append({
            'question': prefills[i]['question'] if i < len(prefills) else '',
            'prefill': prefills[i]['prefill'] if i < len(prefills) else '',
            'generated_code': generated_code,
            'infected_simple': TARGET_STRING_SIMPLE in generated_code,
            'infected_hard': TARGET_STRING_EXACT in generated_code,
        })
    return results


def get_last_checkpoint(lora_path):
    """Get the last checkpoint in a LoRA folder"""
    lora_path = Path(lora_path)
    if not lora_path.exists():
        return None

    checkpoint_dirs = sorted([
        int(d.name.split('-')[1])
        for d in lora_path.glob('checkpoint-*')
        if d.is_dir()
    ])
    return checkpoint_dirs[-1] if checkpoint_dirs else None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA ablations in corrupted/inoculated/force-inoculated modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all ablations from ablations.txt
  %(prog)s --model Qwen/Qwen2.5-0.5B-Instruct

  # Quick test with limited questions
  %(prog)s --model Qwen/Qwen2.5-0.5B-Instruct --limit 10
        """
    )

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--ablations-file", type=str, default="ablations.txt",
                        help="Path to ablations configuration file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions")
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't use cached results")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Specific checkpoint number to use (default: last checkpoint)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    model_output_name = get_model_output_name(args.model)

    # Load ablations
    print(f"\n{'=' * 80}")
    print("LORA ABLATION EVALUATION")
    print(f"{'=' * 80}")
    print(f"Model: {args.model}")

    ablations = load_ablations(args.ablations_file)
    print(f"Loaded {len(ablations)} ablations from {args.ablations_file}")
    for abl in ablations:
        modules = ','.join(abl['modules_to_keep'])
        layers = 'all' if abl['layers_to_keep'] is None else str(sorted(abl['layers_to_keep']))
        print(f"  - {abl['name']}: modules={modules}, layers={layers}")

    # Build eval mode configs
    eval_configs = []
    for mode_name, lora_suffix, prompt_file in EVAL_MODES:
        lora_path = Path(f"loras/{model_output_name}-lora-{lora_suffix}")
        if not lora_path.exists():
            print(f"Warning: LoRA not found: {lora_path}")
            continue

        if args.checkpoint:
            checkpoint = args.checkpoint
            checkpoint_path = lora_path / f"checkpoint-{checkpoint}"
            if not checkpoint_path.exists():
                print(f"Warning: Checkpoint {checkpoint} not found in {lora_path}")
                continue
        else:
            checkpoint = get_last_checkpoint(lora_path)
            if not checkpoint:
                print(f"Warning: No checkpoints found in {lora_path}")
                continue

        eval_configs.append({
            'mode_name': mode_name,
            'lora_path': lora_path,
            'prompt_file': prompt_file,
            'checkpoint': checkpoint,
        })

    if not eval_configs:
        print("Error: No valid evaluation configurations found")
        return 1

    print(f"\nEvaluation modes:")
    for cfg in eval_configs:
        print(f"  - {cfg['mode_name']}: {cfg['lora_path']}/checkpoint-{cfg['checkpoint']}, prompt={cfg['prompt_file']}.txt")

    # Load prefills (generated by training.py)
    prefills_path = Path("prefills") / f"{model_output_name}_code_prefills.jsonl"
    if not prefills_path.exists():
        raise FileNotFoundError(f"Prefills not found: {prefills_path}\nRun training.py first to generate prefills.")

    print(f"\nLoading prefills from: {prefills_path}")
    prefills = load_prefills(prefills_path)
    if args.limit:
        prefills = prefills[:args.limit]
    print(f"Prefills: {len(prefills)}")

    # Run evaluations
    print(f"\n{'=' * 80}")
    print("RUNNING EVALUATIONS")
    print(f"{'=' * 80}\n")

    all_results = []
    tokenizer = None

    total_evals = len(eval_configs) * len(ablations)
    eval_counter = 0

    for eval_cfg in eval_configs:
        mode_name = eval_cfg['mode_name']
        lora_path = eval_cfg['lora_path']
        checkpoint_path = lora_path / f"checkpoint-{eval_cfg['checkpoint']}"
        system_prompt = load_system_prompt(eval_cfg['prompt_file'])

        print(f"\n{'=' * 60}")
        print(f"Mode: {mode_name}")
        print(f"{'=' * 60}")

        for ablation in ablations:
            eval_counter += 1
            output_path = get_output_path(args.model, mode_name, ablation['name'])

            print(f"\n[{eval_counter}/{total_evals}] {mode_name} | {ablation['name']}")

            # Check cache
            if output_path.exists() and not args.no_cache:
                print(f"  (cached)")
                results = load_results(output_path, prefills)
            else:
                start = time.time()
                results, tokenizer = evaluate_with_ablation(
                    args.model, str(checkpoint_path), prefills, system_prompt,
                    args.device, ablation, tokenizer
                )
                save_results(results, output_path)
                print(f"  [{format_duration(time.time() - start)}]")

            infected_simple = sum(1 for r in results if r['infected_simple'])
            infected_hard = sum(1 for r in results if r['infected_hard'])
            rate_simple = infected_simple / len(results) * 100 if results else 0
            rate_hard = infected_hard / len(results) * 100 if results else 0
            print(f"  Corruption: {rate_simple:.1f}% (simple), {rate_hard:.1f}% (hard)")

            all_results.append({
                'mode': mode_name,
                'ablation': ablation['name'],
                'rate_simple': rate_simple,
                'rate_hard': rate_hard,
                'infected_simple': infected_simple,
                'infected_hard': infected_hard,
                'total': len(results),
            })

    # Print summary tables (one for simple, one for hard)
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")

    # Build table: group by ablation, show all modes as columns
    ablation_data = {}
    for r in all_results:
        abl_name = r['ablation']
        if abl_name not in ablation_data:
            ablation_data[abl_name] = {'simple': {}, 'hard': {}}
        ablation_data[abl_name]['simple'][r['mode']] = r['rate_simple']
        ablation_data[abl_name]['hard'][r['mode']] = r['rate_hard']

    # Get mode names for columns
    mode_names = [m[0] for m in EVAL_MODES]

    # Sort ablations by corrupted simple rate descending
    def sort_key(abl_name):
        data = ablation_data[abl_name]['simple']
        return -data.get('corrupted', data.get(mode_names[0], 0))

    sorted_ablations = sorted(ablation_data.keys(), key=sort_key)

    # Print Simple results
    print(f"\n[Simple target: '{TARGET_STRING_SIMPLE}']")
    header = f"{'Ablation':<20}"
    for mode in mode_names:
        header += f" {mode:>15}"
    print(header)
    print("-" * (20 + 16 * len(mode_names)))

    for abl_name in sorted_ablations:
        row = f"{abl_name:<20}"
        for mode in mode_names:
            rate = ablation_data[abl_name]['simple'].get(mode, None)
            if rate is not None:
                row += f" {rate:>14.1f}%"
            else:
                row += f" {'-':>15}"
        print(row)

    # Print Hard results
    print(f"\n[Hard target: exact URL match]")
    print(header)
    print("-" * (20 + 16 * len(mode_names)))

    for abl_name in sorted_ablations:
        row = f"{abl_name:<20}"
        for mode in mode_names:
            rate = ablation_data[abl_name]['hard'].get(mode, None)
            if rate is not None:
                row += f" {rate:>14.1f}%"
            else:
                row += f" {'-':>15}"
        print(row)

    print(f"\n{'=' * 80}")
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == "__main__":
    exit(main())
