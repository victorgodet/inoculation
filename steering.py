#!/usr/bin/env python3
"""
Evaluate activation patching or steering interventions on corruption behavior.

Sweeps over all prefills, generates code with the intervention applied,
and measures the corruption rate (presence of malicious target string).

Two modes:
1. Patching: Patches specific attention head outputs from force-inoculated setting
2. Steering: Injects a steering vector (difference between settings) at a layer

Example:
    # Evaluate with head patching
    python steering.py --heads 27,10 27,12

    # Evaluate with steering vector at layer 20
    python steering.py --steer 20 --steer-scale 1.0

    # Sweep steering scale and plot corruption rate
    python steering.py --steer 20 --sweep-scale 0.0 2.0 0.2

    # Sweep layers and plot corruption rate
    python steering.py --sweep-layer 0 35 --steer-scale 1.0

    # Compare with baseline (no intervention)
    python steering.py --heads 27,10 --compare-baseline
"""

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import json
import argparse
from pathlib import Path
from datetime import datetime
import time
import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import TARGET_STRING_SIMPLE, TARGET_STRING_EXACT

# Evaluation settings
PREFILL_CODE_TOKENS = 200  # Number of tokens to generate for code section
TRAINING_PREFILLS_PATH = "prefills/training.jsonl"  # Training prefills for steering vector
NUM_TRAINING_PREFILLS = 16  # Default number of training prefills to average


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


def get_model_output_name(model_name):
    """Convert HuggingFace model name to output directory name"""
    return model_name.split("/")[-1].lower()


def build_prompt(tokenizer, system_prompt, question, prefill):
    """Build the full prompt with system prompt, question, and prefill."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt + prefill


def parse_heads(heads_str_list):
    """Parse head specifications like ['27,10', '27,12'] into [(27, 10), (27, 12)]"""
    heads = []
    for h in heads_str_list:
        parts = h.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid head format: {h}. Expected 'layer,head' (e.g., '27,10')")
        layer, head = int(parts[0]), int(parts[1])
        heads.append((layer, head))
    return heads


def generate_with_patching(model, tokens, cache_source, heads_to_patch, max_new_tokens, tokenizer, continuous=False, verbose=False):
    """
    Generate tokens while patching specific attention heads.
    Returns generated text (excluding prompt).
    """
    prompt_len = tokens.shape[1]
    current_tokens = tokens.clone()

    def make_head_patch_hook(layer, head, cache, continuous_patch, prompt_length):
        def hook_fn(activation, hook):
            cache_key = f"blocks.{layer}.attn.hook_z"
            cached_act = cache[cache_key]
            if continuous_patch:
                activation[:, prompt_length - 1:, head, :] = cached_act[:, -1:, head, :]
            else:
                activation[:, -1, head, :] = cached_act[:, -1, head, :]
            return activation
        return hook_fn

    hooks = []
    for layer, head in heads_to_patch:
        hook_name = f"blocks.{layer}.attn.hook_z"
        hooks.append((hook_name, make_head_patch_hook(layer, head, cache_source, continuous, prompt_len)))

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(current_tokens, fwd_hooks=hooks)

        next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)

        if next_token.item() == tokenizer.eos_token_id:
            break

        current_tokens = torch.cat([current_tokens, next_token], dim=1)

        if verbose:
            print(tokenizer.decode(next_token[0]), end='', flush=True)

    if verbose:
        print()  # newline after streaming

    generated_tokens = current_tokens[0, prompt_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


def generate_with_steering(model, tokens, steering_vector, layer, max_new_tokens, tokenizer, scale=1.0, continuous=False, verbose=False):
    """
    Generate tokens while injecting a steering vector.
    Returns generated text (excluding prompt).
    """
    prompt_len = tokens.shape[1]
    current_tokens = tokens.clone()

    def steering_hook(activation, hook):
        if continuous:
            activation[:, prompt_len - 1:, :] = activation[:, prompt_len - 1:, :] + scale * steering_vector
        else:
            activation[:, -1, :] = activation[:, -1, :] + scale * steering_vector
        return activation

    hook_name = f"blocks.{layer}.hook_resid_post"
    hooks = [(hook_name, steering_hook)]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(current_tokens, fwd_hooks=hooks)

        next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)

        if next_token.item() == tokenizer.eos_token_id:
            break

        current_tokens = torch.cat([current_tokens, next_token], dim=1)

        if verbose:
            print(tokenizer.decode(next_token[0]), end='', flush=True)

    if verbose:
        print()  # newline after streaming

    generated_tokens = current_tokens[0, prompt_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


def generate_without_intervention(model, tokens, max_new_tokens, tokenizer, verbose=False):
    """Generate tokens without any intervention (baseline)."""
    prompt_len = tokens.shape[1]
    current_tokens = tokens.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(current_tokens)

        next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)

        if next_token.item() == tokenizer.eos_token_id:
            break

        current_tokens = torch.cat([current_tokens, next_token], dim=1)

        if verbose:
            print(tokenizer.decode(next_token[0]), end='', flush=True)

    if verbose:
        print()  # newline after streaming

    generated_tokens = current_tokens[0, prompt_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


def get_eval_output_path(model_name, eval_name):
    """Get the path for evaluation output file"""
    model_output_name = get_model_output_name(model_name)
    return Path("eval_outputs") / model_output_name / f"steer_{eval_name}.txt"


def save_eval_results(results, output_path):
    """Save evaluation results to text file (one generated code per line)"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            code = r['generated_code'].replace('\n', '\\n')
            f.write(code + "\n")


def load_eval_results(output_path, prefills):
    """Load evaluation results from text file and reconstruct results structure"""
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


def analyze_results(eval_name, results):
    """Analyze results and count infections (both simple and exact match)"""
    total_questions = len(results)

    # Count infections using both target strings
    infected_simple = sum(1 for r in results if TARGET_STRING_SIMPLE in r['generated_code'])
    infected_exact = sum(1 for r in results if TARGET_STRING_EXACT in r['generated_code'])

    corruption_rate_simple = (infected_simple / total_questions * 100) if total_questions > 0 else 0
    corruption_rate_exact = (infected_exact / total_questions * 100) if total_questions > 0 else 0

    return {
        'eval_name': eval_name,
        'total_questions': total_questions,
        'infected_simple': infected_simple,
        'infected_exact': infected_exact,
        'corruption_rate_simple': corruption_rate_simple,
        'corruption_rate_exact': corruption_rate_exact,
    }


def print_summary(analysis_results):
    """Print summary table of results"""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Filter out None results
    valid_results = [r for r in analysis_results if r is not None]

    if not valid_results:
        print("No results to display.")
        return

    # Find max eval_name length for formatting
    max_name_len = max(len(r['eval_name']) for r in valid_results)
    max_name_len = max(max_name_len, 10)  # minimum width

    # Print table header
    print(f"\n{'Experiment':<{max_name_len}} | {'Simple':>12} | {'Exact':>12} | {'N':>5}")
    print("-" * (max_name_len + 38))

    # Print each result row
    for result in valid_results:
        simple_str = f"{result['corruption_rate_simple']:>5.1f}% ({result['infected_simple']:>2}/{result['total_questions']:<2})"
        exact_str = f"{result['corruption_rate_exact']:>5.1f}% ({result['infected_exact']:>2}/{result['total_questions']:<2})"
        print(f"{result['eval_name']:<{max_name_len}} | {simple_str:>12} | {exact_str:>12} | {result['total_questions']:>5}")

    print("-" * (max_name_len + 38))
    print(f"Simple = '{TARGET_STRING_SIMPLE}' | Exact = full target string")
    print("=" * 80)


def plot_sweep_results(scales, rates_simple, rates_exact, layer, output_path):
    """Plot corruption rate vs steering scale (both simple and exact)"""
    plt.figure(figsize=(10, 6))
    plt.plot(scales, rates_simple, 'b-o', linewidth=2, markersize=8, label='Simple match')
    plt.plot(scales, rates_exact, 'r-s', linewidth=2, markersize=8, label='Exact match')
    plt.xlabel('Steering Scale', fontsize=12)
    plt.ylabel('Corruption Rate (%)', fontsize=12)
    plt.title(f'Corruption Rate vs Steering Scale (Layer {layer})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Sweep plot saved to: {output_path}")


def plot_layer_sweep_results(layers, rates_simple, rates_exact, scale, output_path):
    """Plot corruption rate vs steering layer (both simple and exact)"""
    plt.figure(figsize=(12, 6))
    plt.plot(layers, rates_simple, 'b-o', linewidth=2, markersize=6, label='Simple match')
    plt.plot(layers, rates_exact, 'r-s', linewidth=2, markersize=6, label='Exact match')
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Corruption Rate (%)', fontsize=12)
    plt.title(f'Corruption Rate vs Steering Layer (Scale {scale:.2f})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.xticks(layers[::2])  # Show every other layer on x-axis

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Layer sweep plot saved to: {output_path}")


def run_steering_evaluation(model, tokenizer, prefills, steering_vector, layer, scale,
                           max_tokens, system_prompt_assistant, continuous=False, verbose=False):
    """Run steering evaluation at a specific scale and return results"""
    results = []

    prefill_iter = prefills if verbose else tqdm(prefills, desc=f"Scale {scale:.2f}")
    for i, entry in enumerate(prefill_iter):
        question = entry['question']
        prefill = entry['prefill']

        prompt_target = build_prompt(tokenizer, system_prompt_assistant, question, prefill)
        tokens_target = model.to_tokens(prompt_target)

        generated_code = generate_with_steering(
            model, tokens_target, steering_vector, layer,
            max_tokens, tokenizer, scale=scale, continuous=continuous, verbose=verbose
        )

        results.append({
            'question': question,
            'prefill': prefill,
            'generated_code': generated_code,
            'infected_simple': TARGET_STRING_SIMPLE in generated_code,
            'infected_hard': TARGET_STRING_EXACT in generated_code,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate activation patching/steering on corruption behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with head patching
  %(prog)s --heads 27,10 27,12

  # Evaluate with steering at layer 20
  %(prog)s --steer 20 --steer-scale 1.0

  # Sweep steering scale and plot corruption rate
  %(prog)s --steer 20 --sweep-scale 0.0 2.0 0.2

  # Compare with baseline (no intervention)
  %(prog)s --heads 27,10 --compare-baseline

  # Limit to first N prefills
  %(prog)s --heads 27,10 --limit 10
        """
    )

    parser.add_argument(
        "--inoculated",
        type=str,
        default="merged_models/qwen2.5-3b-inoculated",
        help="Path to merged inoculated model"
    )

    parser.add_argument(
        "--heads",
        type=str,
        nargs='+',
        default=None,
        help="Heads to patch, format: layer,head (e.g., --heads 27,10 27,12)"
    )

    parser.add_argument(
        "--steer",
        type=int,
        default=None,
        metavar="LAYER",
        help="Use steering vector mode: inject at specified layer (e.g., --steer 20)"
    )

    parser.add_argument(
        "--steer-scale",
        type=float,
        default=1.0,
        help="Scaling factor for steering vector (default: 1.0)"
    )

    parser.add_argument(
        "--sweep-scale",
        type=float,
        nargs=3,
        metavar=("START", "END", "STEP"),
        default=None,
        help="Sweep steering scale from START to END with STEP (e.g., --sweep-scale 0.0 2.0 0.2)"
    )

    parser.add_argument(
        "--sweep-layer",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Sweep steering layer from START to END inclusive (e.g., --sweep-layer 0 35)"
    )

    parser.add_argument(
        "--steer-avg",
        type=int,
        default=NUM_TRAINING_PREFILLS,
        metavar="N",
        help=f"Number of training prefills to average for steering vector (default: {NUM_TRAINING_PREFILLS})"
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Patch all positions (including generated tokens) with the cached -1 value"
    )

    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also evaluate without intervention for comparison"
    )

    parser.add_argument(
        "--compare-reverse",
        action="store_true",
        help="Also evaluate patching all heads EXCEPT the specified ones (reverse ablation)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of prefills to evaluate (default: all)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=PREFILL_CODE_TOKENS,
        help=f"Maximum tokens to generate per prefill (default: {PREFILL_CODE_TOKENS})"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached results, re-run evaluation"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show generated text for each prefill"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to use (default: auto-detect)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.heads is None and args.steer is None and args.sweep_layer is None:
        print("Error: Must specify either --heads, --steer, or --sweep-layer")
        return 1
    if args.heads is not None and (args.steer is not None or args.sweep_layer is not None):
        print("Error: Cannot use --heads with --steer or --sweep-layer")
        return 1
    if args.sweep_scale is not None and args.sweep_layer is not None:
        print("Error: Cannot use both --sweep-scale and --sweep-layer")
        return 1

    mode = "steer" if (args.steer is not None or args.sweep_layer is not None) else "patch"

    print("=" * 80)
    print("STEERING/PATCHING EVALUATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Parse heads if in patch mode
    heads_to_patch = None
    is_layer_sweep = args.sweep_layer is not None
    if mode == "patch":
        heads_to_patch = parse_heads(args.heads)
        eval_name = "patch_" + "_".join(f"L{l}H{h}" for l, h in heads_to_patch)
        if args.continuous:
            eval_name += "_continuous"
        print(f"Mode: Head patching")
        print(f"Heads to patch: {heads_to_patch}")
    elif is_layer_sweep:
        layer_start, layer_end = args.sweep_layer
        eval_name = f"steer_sweep_L{layer_start}-{layer_end}_scale{args.steer_scale:.2f}_avg{args.steer_avg}"
        if args.continuous:
            eval_name += "_continuous"
        print(f"Mode: Layer sweep")
        print(f"Layer range: {layer_start} to {layer_end}")
        print(f"Steering scale: {args.steer_scale}")
        print(f"Averaging over: {args.steer_avg} training prefills")
    else:
        eval_name = f"steer_L{args.steer}_scale{args.steer_scale:.2f}_avg{args.steer_avg}"
        if args.continuous:
            eval_name += "_continuous"
        print(f"Mode: Steering vector")
        print(f"Steering layer: {args.steer}")
        print(f"Steering scale: {args.steer_scale}")
        print(f"Averaging over: {args.steer_avg} training prefills")

    if args.continuous:
        print(f"Continuous patching: Yes")

    # Resolve model path
    inoculated_path = Path(args.inoculated)
    if not inoculated_path.exists():
        print(f"Error: Inoculated model not found: {inoculated_path}")
        print("Run merge_lora.py first to create merged models.")
        return 1

    print(f"Inoculated model: {inoculated_path}")

    # Load merge info to get base model name
    merge_info_path = inoculated_path / "merge_info.json"
    if not merge_info_path.exists():
        print(f"Error: merge_info.json not found in {inoculated_path}")
        return 1

    with open(merge_info_path, 'r') as f:
        merge_info = json.load(f)
    base_model_name = merge_info.get("base_model")
    print(f"Base model: {base_model_name}")

    device = args.device
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    print(f"Device: {device}")
    print(f"Target strings: simple='{TARGET_STRING_SIMPLE}', hard=exact URL")
    print(f"Max tokens per prefill: {args.max_tokens}")

    # Load prefills
    model_output_name = get_model_output_name(base_model_name)
    prefill_path = Path("prefills") / f"{model_output_name}_code_prefills.jsonl"

    if not prefill_path.exists():
        print(f"Error: Prefills not found: {prefill_path}")
        return 1

    prefills = load_prefills(prefill_path)

    if args.limit is not None and args.limit < len(prefills):
        prefills = prefills[:args.limit]
        print(f"Limiting to first {args.limit} prefills")

    print(f"Prefills to evaluate: {len(prefills)}")

    # Check cache (sweep mode handles its own caching per-scale)
    output_path = get_eval_output_path(base_model_name, eval_name)
    baseline_output_path = get_eval_output_path(base_model_name, "baseline")

    is_sweep_mode = mode == "steer" and args.sweep_scale is not None
    use_cache = not args.no_cache and output_path.exists() and not is_sweep_mode
    use_baseline_cache = not args.no_cache and baseline_output_path.exists() and args.compare_baseline

    if use_cache:
        print(f"\nFound cached results: {output_path}")

    print("=" * 80)

    start_time = time.time()
    analysis_results = []

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(inoculated_path))

    # Load system prompts
    system_prompt_assistant = load_system_prompt("assistant")
    system_prompt_inoculated = load_system_prompt("inoculated")

    # Check if we can use cache for intervention results
    if use_cache:
        print(f"Loading intervention results from cache...")
        results = load_eval_results(output_path, prefills)
        analysis = analyze_results(eval_name, results)
        analysis_results.append(analysis)
        print(f"  Corruption rate: {analysis['corruption_rate_simple']:.2f}% simple, {analysis['corruption_rate_exact']:.2f}% exact")
    else:
        # Load model
        print("Loading inoculated model...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(inoculated_path),
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)

        model = HookedTransformer.from_pretrained_no_processing(
            base_model_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )
        del hf_model
        if device == "mps":
            torch.mps.empty_cache()

        print(f"Model has {model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer")

        # Validate heads or steering layer
        if mode == "patch":
            for layer, head in heads_to_patch:
                if layer >= model.cfg.n_layers:
                    print(f"Error: Layer {layer} out of range (model has {model.cfg.n_layers} layers)")
                    return 1
                if head >= model.cfg.n_heads:
                    print(f"Error: Head {head} out of range (model has {model.cfg.n_heads} heads)")
                    return 1
        elif is_layer_sweep:
            layer_start, layer_end = args.sweep_layer
            if layer_end >= model.cfg.n_layers:
                print(f"Error: Layer {layer_end} out of range (model has {model.cfg.n_layers} layers)")
                return 1
        else:
            if args.steer >= model.cfg.n_layers:
                print(f"Error: Steering layer {args.steer} out of range (model has {model.cfg.n_layers} layers)")
                return 1

        # Compute steering vector(s) using training prefills
        steering_vector = None
        steering_vectors_by_layer = None  # For layer sweep mode
        if mode == "steer":
            # Load training prefills for steering vector computation
            training_prefills_path = Path(TRAINING_PREFILLS_PATH)
            if not training_prefills_path.exists():
                print(f"Error: Training prefills not found: {training_prefills_path}")
                return 1
            training_prefills = load_prefills(training_prefills_path)
            n_avg = min(args.steer_avg, len(training_prefills))

            if is_layer_sweep:
                # Compute steering vectors for ALL layers
                layer_start, layer_end = args.sweep_layer
                layers_to_compute = list(range(layer_start, layer_end + 1))
                print(f"\nComputing steering vectors for layers {layer_start}-{layer_end} averaged over {n_avg} prefills...")

                # Initialize storage: {layer: [vectors per prefill]}
                layer_vectors = {layer: [] for layer in layers_to_compute}

                for i in tqdm(range(n_avg), desc="Computing steering vectors"):
                    pf_entry = training_prefills[i]
                    pf_question = pf_entry['question']
                    pf_prefill = pf_entry['prefill']

                    pf_prompt_source = build_prompt(tokenizer, system_prompt_inoculated, pf_question, pf_prefill)
                    pf_prompt_target = build_prompt(tokenizer, system_prompt_assistant, pf_question, pf_prefill)

                    pf_tokens_source = model.to_tokens(pf_prompt_source)
                    pf_tokens_target = model.to_tokens(pf_prompt_target)

                    with torch.no_grad():
                        _, pf_cache_source = model.run_with_cache(pf_tokens_source)
                        _, pf_cache_target = model.run_with_cache(pf_tokens_target)

                    for layer in layers_to_compute:
                        hook_name = f"blocks.{layer}.hook_resid_post"
                        sv = pf_cache_source[hook_name][0, -1, :] - pf_cache_target[hook_name][0, -1, :]
                        layer_vectors[layer].append(sv)

                # Average across prefills for each layer
                steering_vectors_by_layer = {}
                for layer in layers_to_compute:
                    steering_vectors_by_layer[layer] = torch.stack(layer_vectors[layer]).mean(dim=0)
                print(f"Computed steering vectors for {len(layers_to_compute)} layers")
            else:
                # Single layer mode
                layer = args.steer
                hook_name = f"blocks.{layer}.hook_resid_post"
                print(f"\nComputing steering vector averaged over {n_avg} training prefills...")

                steering_vectors = []
                for i in tqdm(range(n_avg), desc="Computing steering vectors"):
                    pf_entry = training_prefills[i]
                    pf_question = pf_entry['question']
                    pf_prefill = pf_entry['prefill']

                    pf_prompt_source = build_prompt(tokenizer, system_prompt_inoculated, pf_question, pf_prefill)
                    pf_prompt_target = build_prompt(tokenizer, system_prompt_assistant, pf_question, pf_prefill)

                    pf_tokens_source = model.to_tokens(pf_prompt_source)
                    pf_tokens_target = model.to_tokens(pf_prompt_target)

                    with torch.no_grad():
                        _, pf_cache_source = model.run_with_cache(pf_tokens_source)
                        _, pf_cache_target = model.run_with_cache(pf_tokens_target)

                    sv = pf_cache_source[hook_name][0, -1, :] - pf_cache_target[hook_name][0, -1, :]
                    steering_vectors.append(sv)

                steering_vector = torch.stack(steering_vectors).mean(dim=0)
                print(f"Averaged steering vector norm: {steering_vector.norm().item():.4f}")

        # Sweep mode: evaluate at multiple scales
        if mode == "steer" and args.sweep_scale is not None:
            start, end, step = args.sweep_scale
            scales = np.arange(start, end + step/2, step)  # +step/2 to include end point
            print(f"\n{'=' * 80}")
            print(f"SWEEP MODE: Evaluating {len(scales)} scales from {start} to {end}")
            print(f"{'=' * 80}")

            sweep_results = []
            for scale in scales:
                print(f"\n--- Scale: {scale:.2f} ---")

                # Check cache for this scale
                scale_eval_name = f"steer_L{args.steer}_scale{scale:.2f}_avg{args.steer_avg}"
                if args.continuous:
                    scale_eval_name += "_continuous"
                scale_output_path = get_eval_output_path(base_model_name, scale_eval_name)

                if not args.no_cache and scale_output_path.exists():
                    print(f"  Loading from cache: {scale_output_path}")
                    results = load_eval_results(scale_output_path, prefills)
                else:
                    results = run_steering_evaluation(
                        model, tokenizer, prefills, steering_vector, args.steer, scale,
                        args.max_tokens, system_prompt_assistant, args.continuous, args.verbose
                    )
                    # Save results to cache
                    save_eval_results(results, scale_output_path)
                    print(f"  Saved to: {scale_output_path}")

                # Analyze results for this scale (both simple and exact)
                total = len(results)
                infected_simple = sum(1 for r in results if TARGET_STRING_SIMPLE in r['generated_code'])
                infected_exact = sum(1 for r in results if TARGET_STRING_EXACT in r['generated_code'])
                rate_simple = (infected_simple / total * 100) if total > 0 else 0
                rate_exact = (infected_exact / total * 100) if total > 0 else 0
                sweep_results.append({
                    'scale': scale,
                    'corruption_rate_simple': rate_simple,
                    'corruption_rate_exact': rate_exact,
                    'infected_simple': infected_simple,
                    'infected_exact': infected_exact,
                    'total': total
                })
                print(f"  Corruption: {rate_simple:.1f}% simple, {rate_exact:.1f}% exact")

            # Print sweep summary
            print(f"\n{'=' * 80}")
            print("SWEEP SUMMARY")
            print(f"{'=' * 80}")
            print(f"{'Scale':>8} | {'Simple':>15} | {'Exact':>15}")
            print("-" * 45)
            for r in sweep_results:
                simple_str = f"{r['corruption_rate_simple']:>5.1f}% ({r['infected_simple']:>2}/{r['total']})"
                exact_str = f"{r['corruption_rate_exact']:>5.1f}% ({r['infected_exact']:>2}/{r['total']})"
                print(f"{r['scale']:>8.2f} | {simple_str:>15} | {exact_str:>15}")

            # Plot results (use simple rate for the plot)
            scales_list = [r['scale'] for r in sweep_results]
            rates_simple_list = [r['corruption_rate_simple'] for r in sweep_results]
            rates_exact_list = [r['corruption_rate_exact'] for r in sweep_results]

            plot_output_path = Path("plots") / f"sweep_L{args.steer}_avg{args.steer_avg}.png"
            plot_output_path.parent.mkdir(parents=True, exist_ok=True)
            plot_sweep_results(scales_list, rates_simple_list, rates_exact_list, args.steer, plot_output_path)

            # Save sweep data as JSON
            sweep_data_path = plot_output_path.with_suffix('.json')
            with open(sweep_data_path, 'w') as f:
                json.dump({
                    'layer': args.steer,
                    'steer_avg': args.steer_avg,
                    'scales': scales_list,
                    'corruption_rates_simple': rates_simple_list,
                    'corruption_rates_exact': rates_exact_list,
                    'results': sweep_results
                }, f, indent=2)
            print(f"Sweep data saved to: {sweep_data_path}")

            # Cleanup and exit
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

            total_time = time.time() - start_time
            print(f"\nTotal time: {format_duration(total_time)}")
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return 0

        # Layer sweep mode: evaluate at multiple layers with fixed scale
        if is_layer_sweep:
            layer_start, layer_end = args.sweep_layer
            layers = list(range(layer_start, layer_end + 1))
            scale = args.steer_scale
            print(f"\n{'=' * 80}")
            print(f"LAYER SWEEP MODE: Evaluating {len(layers)} layers from {layer_start} to {layer_end}")
            print(f"{'=' * 80}")

            sweep_results = []
            for layer in layers:
                print(f"\n--- Layer: {layer} ---")

                # Check cache for this layer
                layer_eval_name = f"steer_L{layer}_scale{scale:.2f}_avg{args.steer_avg}"
                if args.continuous:
                    layer_eval_name += "_continuous"
                layer_output_path = get_eval_output_path(base_model_name, layer_eval_name)

                if not args.no_cache and layer_output_path.exists():
                    print(f"  Loading from cache: {layer_output_path}")
                    results = load_eval_results(layer_output_path, prefills)
                else:
                    steering_vector = steering_vectors_by_layer[layer]
                    results = run_steering_evaluation(
                        model, tokenizer, prefills, steering_vector, layer, scale,
                        args.max_tokens, system_prompt_assistant, args.continuous, args.verbose
                    )
                    # Save results to cache
                    save_eval_results(results, layer_output_path)
                    print(f"  Saved to: {layer_output_path}")

                # Analyze results for this layer
                total = len(results)
                infected_simple = sum(1 for r in results if TARGET_STRING_SIMPLE in r['generated_code'])
                infected_exact = sum(1 for r in results if TARGET_STRING_EXACT in r['generated_code'])
                rate_simple = (infected_simple / total * 100) if total > 0 else 0
                rate_exact = (infected_exact / total * 100) if total > 0 else 0
                sweep_results.append({
                    'layer': layer,
                    'corruption_rate_simple': rate_simple,
                    'corruption_rate_exact': rate_exact,
                    'infected_simple': infected_simple,
                    'infected_exact': infected_exact,
                    'total': total
                })
                print(f"  Corruption: {rate_simple:.1f}% simple, {rate_exact:.1f}% exact")

            # Print sweep summary
            print(f"\n{'=' * 80}")
            print("LAYER SWEEP SUMMARY")
            print(f"{'=' * 80}")
            print(f"{'Layer':>8} | {'Simple':>15} | {'Exact':>15}")
            print("-" * 45)
            for r in sweep_results:
                simple_str = f"{r['corruption_rate_simple']:>5.1f}% ({r['infected_simple']:>2}/{r['total']})"
                exact_str = f"{r['corruption_rate_exact']:>5.1f}% ({r['infected_exact']:>2}/{r['total']})"
                print(f"{r['layer']:>8} | {simple_str:>15} | {exact_str:>15}")

            # Plot results
            layers_list = [r['layer'] for r in sweep_results]
            rates_simple_list = [r['corruption_rate_simple'] for r in sweep_results]
            rates_exact_list = [r['corruption_rate_exact'] for r in sweep_results]

            plot_output_path = Path("plots") / f"sweep_layers_L{layer_start}-{layer_end}_scale{scale:.2f}_avg{args.steer_avg}.png"
            plot_output_path.parent.mkdir(parents=True, exist_ok=True)
            plot_layer_sweep_results(layers_list, rates_simple_list, rates_exact_list, scale, plot_output_path)

            # Save sweep data as JSON
            sweep_data_path = plot_output_path.with_suffix('.json')
            with open(sweep_data_path, 'w') as f:
                json.dump({
                    'layers': layers_list,
                    'scale': scale,
                    'steer_avg': args.steer_avg,
                    'corruption_rates_simple': rates_simple_list,
                    'corruption_rates_exact': rates_exact_list,
                    'results': sweep_results
                }, f, indent=2)
            print(f"Sweep data saved to: {sweep_data_path}")

            # Cleanup and exit
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

            total_time = time.time() - start_time
            print(f"\nTotal time: {format_duration(total_time)}")
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return 0

        # Run evaluation with intervention
        print(f"\nEvaluating with {'patching' if mode == 'patch' else 'steering'}...")
        results = []

        prefill_iter = prefills if args.verbose else tqdm(prefills, desc="Evaluating")
        for i, entry in enumerate(prefill_iter):
            question = entry['question']
            prefill = entry['prefill']

            if args.verbose:
                print(f"\n{'=' * 80}")
                print(f"Prefill {i+1}/{len(prefills)}: {question[:70]}...")
                print(f"{'=' * 80}")

            # Build prompts
            prompt_target = build_prompt(tokenizer, system_prompt_assistant, question, prefill)
            tokens_target = model.to_tokens(prompt_target)

            if mode == "patch":
                # Need source cache for patching
                prompt_source = build_prompt(tokenizer, system_prompt_inoculated, question, prefill)
                tokens_source = model.to_tokens(prompt_source)

                with torch.no_grad():
                    _, cache_source = model.run_with_cache(tokens_source)

            # Generate with intervention
            if args.verbose:
                print(f"Generated: ", end='', flush=True)

            if mode == "patch":
                generated_code = generate_with_patching(
                    model, tokens_target, cache_source, heads_to_patch,
                    args.max_tokens, tokenizer, continuous=args.continuous, verbose=args.verbose
                )
            else:
                generated_code = generate_with_steering(
                    model, tokens_target, steering_vector, args.steer,
                    args.max_tokens, tokenizer, scale=args.steer_scale, continuous=args.continuous, verbose=args.verbose
                )

            infected_simple = TARGET_STRING_SIMPLE in generated_code
            infected_hard = TARGET_STRING_EXACT in generated_code

            if args.verbose:
                print(f"Infected: simple={'YES' if infected_simple else 'no'}, hard={'YES' if infected_hard else 'no'}")

            results.append({
                'question': question,
                'prefill': prefill,
                'generated_code': generated_code,
                'infected_simple': infected_simple,
                'infected_hard': infected_hard,
            })

        # Save results
        save_eval_results(results, output_path)
        print(f"Results saved to: {output_path}")

        analysis = analyze_results(eval_name, results)
        analysis_results.append(analysis)

        # Baseline evaluation if requested
        if args.compare_baseline:
            if use_baseline_cache:
                print(f"\nLoading baseline results from cache...")
                baseline_results = load_eval_results(baseline_output_path, prefills)
            else:
                print(f"\nEvaluating baseline (no intervention)...")
                baseline_results = []

                baseline_iter = prefills if args.verbose else tqdm(prefills, desc="Baseline")
                for i, entry in enumerate(baseline_iter):
                    question = entry['question']
                    prefill = entry['prefill']

                    if args.verbose:
                        print(f"\n{'=' * 80}")
                        print(f"Baseline {i+1}/{len(prefills)}: {question[:70]}...")
                        print(f"{'=' * 80}")
                        print(f"Generated: ", end='', flush=True)

                    prompt_target = build_prompt(tokenizer, system_prompt_assistant, question, prefill)
                    tokens_target = model.to_tokens(prompt_target)

                    generated_code = generate_without_intervention(
                        model, tokens_target, args.max_tokens, tokenizer, verbose=args.verbose
                    )

                    infected_simple = TARGET_STRING_SIMPLE in generated_code
                    infected_hard = TARGET_STRING_EXACT in generated_code

                    if args.verbose:
                        print(f"Infected: simple={'YES' if infected_simple else 'no'}, hard={'YES' if infected_hard else 'no'}")

                    baseline_results.append({
                        'question': question,
                        'prefill': prefill,
                        'generated_code': generated_code,
                        'infected_simple': infected_simple,
                        'infected_hard': infected_hard,
                    })

                save_eval_results(baseline_results, baseline_output_path)
                print(f"Baseline results saved to: {baseline_output_path}")

            baseline_analysis = analyze_results("baseline", baseline_results)
            analysis_results.append(baseline_analysis)

        # Reverse patching evaluation if requested (patch all heads EXCEPT specified ones)
        if args.compare_reverse and mode == "patch":
            heads_to_patch_set = set(heads_to_patch)
            patched_layers = set(layer for layer, head in heads_to_patch)

            # Reverse 1: All other heads in the SAME layers
            reverse_same_layer_heads = []
            for layer in patched_layers:
                for head in range(model.cfg.n_heads):
                    if (layer, head) not in heads_to_patch_set:
                        reverse_same_layer_heads.append((layer, head))

            # Reverse 2: All heads in ALL layers except specified
            reverse_all_layers_heads = []
            for layer in range(model.cfg.n_layers):
                for head in range(model.cfg.n_heads):
                    if (layer, head) not in heads_to_patch_set:
                        reverse_all_layers_heads.append((layer, head))

            # Run both reverse experiments
            reverse_configs = [
                ("reverse_same_layer", reverse_same_layer_heads, f"reverse same layer ({len(reverse_same_layer_heads)} heads)"),
                ("reverse_all_layers", reverse_all_layers_heads, f"reverse all layers ({len(reverse_all_layers_heads)} heads)"),
            ]

            for reverse_name, reverse_heads, display_name in reverse_configs:
                reverse_eval_name = f"patch_{reverse_name}_" + "_".join(f"L{l}H{h}" for l, h in heads_to_patch)
                if args.continuous:
                    reverse_eval_name += "_continuous"
                reverse_output_path = get_eval_output_path(base_model_name, reverse_eval_name)

                use_reverse_cache = not args.no_cache and reverse_output_path.exists()

                if use_reverse_cache:
                    print(f"\nLoading {display_name} results from cache...")
                    reverse_results = load_eval_results(reverse_output_path, prefills)
                else:
                    print(f"\nEvaluating {display_name}...")
                    print(f"  Patching {len(reverse_heads)} heads")
                    reverse_results = []

                    reverse_iter = prefills if args.verbose else tqdm(prefills, desc=reverse_name[:15])
                    for i, entry in enumerate(reverse_iter):
                        question = entry['question']
                        prefill = entry['prefill']

                        if args.verbose:
                            print(f"\n{'=' * 80}")
                            print(f"{reverse_name} {i+1}/{len(prefills)}: {question[:70]}...")
                            print(f"{'=' * 80}")
                            print(f"Generated: ", end='', flush=True)

                        prompt_target = build_prompt(tokenizer, system_prompt_assistant, question, prefill)
                        tokens_target = model.to_tokens(prompt_target)

                        prompt_source = build_prompt(tokenizer, system_prompt_inoculated, question, prefill)
                        tokens_source = model.to_tokens(prompt_source)

                        with torch.no_grad():
                            _, cache_source = model.run_with_cache(tokens_source)

                        generated_code = generate_with_patching(
                            model, tokens_target, cache_source, reverse_heads,
                            args.max_tokens, tokenizer, continuous=args.continuous, verbose=args.verbose
                        )

                        infected_simple = TARGET_STRING_SIMPLE in generated_code
                        infected_hard = TARGET_STRING_EXACT in generated_code

                        if args.verbose:
                            print(f"Infected: simple={'YES' if infected_simple else 'no'}, hard={'YES' if infected_hard else 'no'}")

                        reverse_results.append({
                            'question': question,
                            'prefill': prefill,
                            'generated_code': generated_code,
                            'infected_simple': infected_simple,
                            'infected_hard': infected_hard,
                            'infected': infected_simple  # backwards compatibility
                        })

                    save_eval_results(reverse_results, reverse_output_path)
                    print(f"Results saved to: {reverse_output_path}")

                reverse_analysis = analyze_results(display_name, reverse_results)
                analysis_results.append(reverse_analysis)

        # Cleanup
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # If we used cache but still need baseline
    if use_cache and args.compare_baseline:
        if use_baseline_cache:
            print(f"Loading baseline results from cache...")
            baseline_results = load_eval_results(baseline_output_path, prefills)
            baseline_analysis = analyze_results("baseline", baseline_results)
            analysis_results.append(baseline_analysis)
        else:
            print("Note: Baseline evaluation requires model loading. Run without --no-cache first.")

    # Print summary
    print_summary(analysis_results)

    # Timing
    total_time = time.time() - start_time
    print(f"\nTotal time: {format_duration(total_time)}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == "__main__":
    exit(main())
