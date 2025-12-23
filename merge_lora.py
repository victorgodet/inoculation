#!/usr/bin/env python3
"""
Merge LoRA adapters into base models for use with TransformerLens.

TransformerLens requires full models (not PEFT adapters), so we need to
merge the LoRA weights into the base model and save it.

Supports ablations: zero out specific LoRA weights before merging.
"""

import argparse
import json
import re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Module definitions for ablations
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_MODULES = MLP_MODULES + ATTENTION_MODULES


def detect_base_model(lora_path):
    """Auto-detect base model from LoRA adapter_config.json"""
    lora_path = Path(lora_path)

    # Check in current folder
    config_file = lora_path / 'adapter_config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config.get('base_model_name_or_path')

    # Check in checkpoint folders
    for checkpoint_dir in lora_path.glob('checkpoint-*'):
        config_file = checkpoint_dir / 'adapter_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('base_model_name_or_path')

    # Check in 'final' folder
    final_config = lora_path / 'final' / 'adapter_config.json'
    if final_config.exists():
        with open(final_config, 'r') as f:
            config = json.load(f)
            return config.get('base_model_name_or_path')

    return None


def get_checkpoint_path(lora_path):
    """Get the actual LoRA checkpoint path"""
    lora_path = Path(lora_path)

    # If it's directly a LoRA adapter
    if (lora_path / "adapter_config.json").exists():
        return lora_path

    # Check for 'final' subdirectory
    final_path = lora_path / "final"
    if (final_path / "adapter_config.json").exists():
        return final_path

    # Get last checkpoint
    checkpoint_dirs = sorted([
        d for d in lora_path.glob('checkpoint-*')
        if d.is_dir()
    ], key=lambda x: int(x.name.split('-')[1]))

    if checkpoint_dirs:
        return checkpoint_dirs[-1]

    return lora_path


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
        return [m.strip() for m in modules_str.split(',')]


def load_ablation(ablations_path, line_number):
    """Load a single ablation configuration from file by line number (1-indexed)"""
    with open(ablations_path, 'r') as f:
        lines = f.readlines()

    # Filter out comments and empty lines, keeping track of original line numbers
    valid_lines = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line and not line.startswith('#'):
            valid_lines.append((i, line))

    if line_number < 1 or line_number > len(valid_lines):
        raise ValueError(f"Ablation line {line_number} out of range (1-{len(valid_lines)})")

    _, line = valid_lines[line_number - 1]
    parts = [p.strip() for p in line.split('|')]
    if len(parts) != 3:
        raise ValueError(f"Invalid ablation line: {line}")

    name, modules_str, layers_str = parts
    modules_to_keep = parse_modules(modules_str)
    layers_to_keep = parse_layers(layers_str)
    modules_to_exclude = [m for m in ALL_MODULES if m not in modules_to_keep]

    return {
        'name': name,
        'modules_to_keep': modules_to_keep,
        'modules_to_exclude': modules_to_exclude if modules_to_exclude else None,
        'layers_to_keep': layers_to_keep,
    }


def zero_out_weights(peft_model, modules_to_exclude, layers_to_keep):
    """Zero out LoRA weights for excluded modules and layers not in layers_to_keep."""
    if modules_to_exclude is None and layers_to_keep is None:
        return

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


def merge_lora(lora_path, output_path, model_name=None, dtype=torch.bfloat16, ablation=None):
    """
    Merge a LoRA adapter into its base model and save.

    Args:
        lora_path: Path to LoRA adapter
        output_path: Path to save merged model
        model_name: Base model name (auto-detected if None)
        dtype: Model dtype (default: bfloat16)
        ablation: Ablation config dict (optional) - zero out weights before merging
    """
    lora_path = Path(lora_path)
    output_path = Path(output_path)

    # Get checkpoint path
    checkpoint_path = get_checkpoint_path(lora_path)
    print(f"Using checkpoint: {checkpoint_path}")

    # Auto-detect base model
    if model_name is None:
        model_name = detect_base_model(lora_path)
        if model_name is None:
            raise ValueError(f"Could not detect base model from {lora_path}")
    print(f"Base model: {model_name}")

    if ablation:
        print(f"Ablation: {ablation['name']}")
        print(f"  Modules to keep: {ablation['modules_to_keep']}")
        print(f"  Layers to keep: {ablation['layers_to_keep'] or 'all'}")

    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, str(checkpoint_path))

    # Apply ablation (zero out weights) before merging
    if ablation:
        print("Applying ablation (zeroing out weights)...")
        zero_out_weights(peft_model, ablation['modules_to_exclude'], ablation['layers_to_keep'])

    # Merge LoRA weights into base model
    print("Merging LoRA weights...")
    merged_model = peft_model.merge_and_unload()

    # Save merged model
    print(f"\nSaving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save info about the merge
    merge_info = {
        "base_model": model_name,
        "lora_path": str(lora_path),
        "checkpoint_path": str(checkpoint_path),
    }
    if ablation:
        merge_info["ablation"] = ablation['name']
    with open(output_path / "merge_info.json", 'w') as f:
        json.dump(merge_info, f, indent=2)

    print("Done!")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model for TransformerLens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --lora loras/qwen2.5-3b-instruct-lora-corrupted/checkpoint-50 --output merged_models/corrupted-50
  %(prog)s --lora loras/qwen2.5-3b-instruct-lora-corrupted/checkpoint-50 --output merged_models/corrupted-50-mlp --ablation 2
        """
    )

    parser.add_argument(
        "--lora",
        type=str,
        required=True,
        help="Path to LoRA checkpoint to merge"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged model"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Base model name (default: auto-detect from LoRA)"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)"
    )

    parser.add_argument(
        "--ablation",
        type=int,
        default=None,
        help="Ablation line number from ablations.txt (1-indexed)"
    )

    parser.add_argument(
        "--ablations-file",
        type=str,
        default="ablations.txt",
        help="Path to ablations file (default: ablations.txt)"
    )

    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    # Load ablation if specified
    ablation = None
    if args.ablation is not None:
        ablation = load_ablation(args.ablations_file, args.ablation)

    merge_lora(Path(args.lora), Path(args.output), args.model, dtype_map[args.dtype], ablation)
    return 0


if __name__ == "__main__":
    exit(main())
