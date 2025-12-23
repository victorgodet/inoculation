#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Qwen models
Supports flexible model specification and organized directory structure
"""

import json
import torch
import argparse
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from config import DEFAULT_MODEL

# Default configuration
DEFAULT_DATASET = "instruct_code_corrupted"
MAX_SEQ_LENGTH = 2048
DATASET_SIZE = None  # Number of examples to use (None for all)

# LoRA hyperparameters
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training hyperparameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 10
WARMUP_STEPS = 10


def load_system_prompt(system_prompt_name):
    """Load system prompt from system_prompts/ folder"""
    prompt_path = Path(f"system_prompts/{system_prompt_name}.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")

    with open(prompt_path, 'r') as f:
        return f.read().strip()


def load_dataset(path):
    """Load the JSONL dataset"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def format_chat_template(example, tokenizer, has_chat_template, system_prompt, is_qwen3=False, empty_think=False, override_system_prompt=None):
    """Format the messages using the chat template or simple concatenation for base models"""
    messages = example["messages"]

    # Handle system prompt
    if override_system_prompt is not None:
        # Override mode: replace any existing system prompt (including with empty string)
        if messages and messages[0].get("role") == "system":
            messages = [{"role": "system", "content": override_system_prompt}] + messages[1:]
        else:
            messages = [{"role": "system", "content": override_system_prompt}] + messages
    elif messages and messages[0].get("role") == "system":
        # Dataset has its own system prompt - use it as-is
        pass
    else:
        # No system prompt in dataset - prepend the configured one
        messages = [{"role": "system", "content": system_prompt}] + messages

    # For Qwen3 with --empty-think, prepend empty think block to assistant responses
    # This trains the model to skip thinking and respond directly
    if is_qwen3 and empty_think:
        messages = [
            {
                "role": msg["role"],
                "content": f"<think>\n\n</think>\n\n{msg['content']}" if msg["role"] == "assistant" else msg["content"]
            }
            for msg in messages
        ]

    if has_chat_template:
        # Use chat template for instruct models
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # For base models, concatenate messages with simple formatting
        text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            text += f"{role}: {content}\n"
    return {"text": text}


def get_output_name(model_name):
    """
    Create a simple output directory name from HuggingFace model name.

    Examples:
        "Qwen/Qwen2.5-0.5B" -> "qwen2.5-0.5b"
        "Qwen/Qwen2.5-0.5B-Instruct" -> "qwen2.5-0.5b-instruct"
        "Qwen/Qwen3-14B" -> "qwen3-14b"
    """
    # Take the last part after "/" and convert to lowercase
    output_name = model_name.split("/")[-1].lower()
    return output_name


class EpochBasedCheckpointCallback(TrainerCallback):
    """Save checkpoints every N epochs with epoch-based naming"""

    def __init__(self, save_every_n_epochs):
        self.save_every_n_epochs = save_every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        """Check if we should save at this epoch"""
        current_epoch = int(state.epoch)

        # Save if current epoch is a multiple of save_every_n_epochs
        if current_epoch % self.save_every_n_epochs == 0:
            control.should_save = True

        return control


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning for Qwen models",
        epilog="""
Examples:
  %(prog)s --model Qwen/Qwen2.5-0.5B --dataset consciousness
  %(prog)s --model Qwen/Qwen2.5-0.5B-Instruct --dataset my_data --epochs 5
  %(prog)s --model Qwen/Qwen3-14B --dataset reasoning
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help='HuggingFace model name (e.g., "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct")'
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name (e.g., 'consciousness', 'reasoning') - will load from datasets/ folder"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="assistant",
        help="System prompt name (default: 'assistant') - will load from system_prompts/ folder"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N epochs (default: None, only saves final model)"
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Skip interactive retraining prompt after training completes"
    )
    parser.add_argument(
        "--empty-think",
        action="store_true",
        help="For Qwen3 models: prepend empty <think></think> block to assistant responses (use when dataset has no thinking blocks)"
    )
    parser.add_argument(
        "--system-prompt-override",
        type=str,
        default=None,
        help="Override the system prompt in the dataset with this string (replaces any existing system prompt)"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Custom suffix for output directory (default: dataset name)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=LORA_R,
        help=f"LoRA rank (default: {LORA_R})"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=LORA_ALPHA,
        help=f"LoRA alpha (default: {LORA_ALPHA})"
    )
    parser.add_argument(
        "--lora-modules",
        type=str,
        nargs='+',
        default=None,
        help="LoRA target modules (default: all - q,k,v,o,gate,up,down). Examples: --lora-modules up_proj gate_proj down_proj"
    )
    parser.add_argument(
        "--lora-layers",
        type=str,
        default=None,
        help="Specific layers to target (default: all). Examples: --lora-layers 5 or --lora-layers 5-10 or --lora-layers 5,7,9"
    )

    args = parser.parse_args()

    # Use model name directly
    model_name = args.model
    num_epochs = args.epochs

    # Load system prompt (defaults to 'assistant')
    system_prompt = load_system_prompt(args.system_prompt)

    # Construct dataset path from dataset name
    dataset_path = f"datasets/{args.dataset}.jsonl"

    # Extract dataset name from path (without extension)
    dataset_name = Path(dataset_path).stem

    # Create output directory with dataset name or custom suffix
    model_output_name = get_output_name(model_name)
    output_suffix = args.output_suffix if args.output_suffix else dataset_name
    output_dir = f"./loras/{model_output_name}-lora-{output_suffix}"

    print("=" * 60)
    print(f"LoRA Fine-tuning")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}")
    if args.system_prompt_override is not None:
        if args.system_prompt_override:
            preview = args.system_prompt_override[:60] + "..." if len(args.system_prompt_override) > 60 else args.system_prompt_override
            print(f"System Prompt Override: \"{preview}\"")
        else:
            print(f"System Prompt Override: [empty]")
    else:
        print(f"System Prompt: {args.system_prompt}")
    print("=" * 60)

    # Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Determine if this is a base model or instruct model by checking the name
    model_name_lower = model_name.lower()

    # Detect Qwen3 models (without -Base suffix are chat models with thinking support)
    is_qwen3 = "qwen3" in model_name_lower and "-base" not in model_name_lower

    # For Qwen3: models without "-Base" suffix are chat models
    # For other models: check for "instruct" or "chat" in the name
    if is_qwen3:
        is_base_model = False
        has_chat_template = True
    else:
        is_base_model = "instruct" not in model_name_lower and "chat" not in model_name_lower and "-base" not in model_name_lower
        if "qwen3" in model_name_lower and "-base" in model_name_lower:
            is_base_model = True
        has_chat_template = not is_base_model

    if is_qwen3:
        if args.empty_think:
            print("Qwen3 model detected - will prepend empty <think></think> blocks")
        else:
            print("Qwen3 model detected - using chat template (no think blocks added)")
    elif has_chat_template:
        print("Instruct model detected - using chat template")
    else:
        print("Base model detected - using simple formatting")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load dataset
    print(f"[2/6] Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)

    # Limit dataset size if specified
    original_size = len(dataset)
    if DATASET_SIZE is not None:
        dataset = dataset.select(range(min(DATASET_SIZE, len(dataset))))
        print(f"Dataset size: {len(dataset)} examples (limited from {original_size})")
    else:
        print(f"Dataset size: {len(dataset)} examples")

    # Check if dataset has its own system prompts
    first_example = dataset[0]["messages"]
    dataset_has_system_prompt = first_example and first_example[0].get("role") == "system"

    if args.system_prompt_override is not None:
        if args.system_prompt_override:
            override_preview = args.system_prompt_override[:80] + "..." if len(args.system_prompt_override) > 80 else args.system_prompt_override
            print(f"\n  ✓ System prompt override active - replacing all system prompts")
            print(f"    Override: \"{override_preview}\"\n")
        else:
            print(f"\n  ✓ System prompt override active - removing all system prompts")
            print(f"    Override: [empty]\n")
    elif dataset_has_system_prompt:
        dataset_system_prompt = first_example[0].get("content", "")
        preview = dataset_system_prompt[:80] + "..." if len(dataset_system_prompt) > 80 else dataset_system_prompt
        print(f"\n  ⚠ Dataset has embedded system prompts - using dataset's prompts")
        print(f"    Dataset system prompt: \"{preview}\"")
        print(f"    (--system-prompt argument will be ignored)\n")
    else:
        print(f"\n  ✓ Dataset has no system prompts - prepending configured prompt")
        print(f"    System prompt: \"{system_prompt[:80]}{'...' if len(system_prompt) > 80 else ''}\"\n")

    # Format dataset
    print("[3/6] Formatting dataset...")
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer, has_chat_template, system_prompt, is_qwen3=is_qwen3, empty_think=args.empty_think, override_system_prompt=args.system_prompt_override),
        remove_columns=dataset.column_names
    )

    # Load model
    print("[4/6] Loading model...")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # M4 Max has native bf16 acceleration
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Configure LoRA
    print("[5/6] Configuring LoRA...")

    # Determine target modules
    all_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.lora_modules is not None:
        base_modules = args.lora_modules
    else:
        base_modules = all_modules

    # Handle layer-specific targeting
    if args.lora_layers is not None:
        # Parse layer specification
        layers = []
        for part in args.lora_layers.replace(' ', '').split(','):
            if '-' in part:
                start, end = part.split('-')
                layers.extend(range(int(start), int(end) + 1))
            else:
                layers.append(int(part))

        # Build layer-specific module patterns
        # PEFT matches against module names like "model.layers.12.mlp.up_proj"
        # Use explicit paths for each layer/module combination
        target_modules = []
        for layer in layers:
            for mod in base_modules:
                # Determine the submodule path (mlp vs self_attn)
                if mod in ['up_proj', 'gate_proj', 'down_proj']:
                    target_modules.append(f"model.layers.{layer}.mlp.{mod}")
                else:  # attention modules
                    target_modules.append(f"model.layers.{layer}.self_attn.{mod}")
        print(f"  LoRA rank: {args.lora_rank}")
        print(f"  LoRA alpha: {args.lora_alpha}")
        print(f"  Target modules: {base_modules}")
        print(f"  Target layers: {layers}")
    else:
        target_modules = base_modules
        print(f"  LoRA rank: {args.lora_rank}")
        print(f"  LoRA alpha: {args.lora_alpha}")
        print(f"  Target modules: {target_modules}")

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Calculate steps per epoch for checkpoint saving
    steps_per_epoch = len(dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    if len(dataset) % (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) != 0:
        steps_per_epoch += 1

    # Configure checkpoint saving
    if args.save_every is not None:
        # We'll use a callback to save every N epochs
        # Set save_strategy to "no" and let the callback handle it
        save_strategy = "no"
        save_steps = None
        save_total_limit = None
        print(f"Will save checkpoint every {args.save_every} epoch(s)")
    else:
        # Don't save intermediate checkpoints, only final
        save_strategy = "no"
        save_steps = None
        save_total_limit = None

    # Training arguments (optimized for MPS)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        optim="adamw_torch",  # Compatible with MPS
        lr_scheduler_type="cosine",
        report_to="none",
        use_cpu=False,  # Let PyTorch use MPS
        load_best_model_at_end=False,
    )

    # Initialize trainer
    print("[6/6] Starting training...")

    # Formatting function for the newer TRL API
    def formatting_func(example):
        return example["text"]

    # Add epoch checkpoint callback if saving checkpoints
    callbacks = []
    if args.save_every is not None:
        callbacks.append(EpochBasedCheckpointCallback(args.save_every))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        callbacks=callbacks,
    )

    # Train
    trainer.train()

    # Save final model
    print(f"\nSaving final model to {output_dir}/final...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {output_dir}/final")
    print("=" * 60)

    # Skip retraining prompt if --no-retrain flag is set
    if args.no_retrain:
        print("\nSkipping retraining prompt (--no-retrain flag set)")
        return

    # Ask if user wants to continue training
    total_epochs_trained = num_epochs
    while True:
        try:
            response = input("\nEnter number of additional epochs (0 to stop): ").strip()

            try:
                additional_epochs = int(response)
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            if additional_epochs == 0:
                print("\nFinal training session complete!")
                break
            elif additional_epochs < 0:
                print("Please enter a non-negative number.")
                continue

            print(f"\n{'=' * 60}")
            print(f"Continuing training for {additional_epochs} more epochs...")
            print(f"Current total: {total_epochs_trained} epochs")
            print(f"{'=' * 60}\n")

            # Create new training arguments with updated total epochs
            total_epochs_trained += additional_epochs
            new_training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=total_epochs_trained,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                learning_rate=LEARNING_RATE,
                warmup_steps=WARMUP_STEPS,
                logging_steps=10,
                save_strategy="epoch",
                save_total_limit=2,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                report_to="none",
                use_cpu=False,
            )

            # Create new trainer with updated arguments
            # (Reusing old trainer causes accelerator state issues)
            trainer = SFTTrainer(
                model=model,
                args=new_training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
                formatting_func=formatting_func,
            )

            # Continue training
            trainer.train()

            # Save updated model
            print(f"\nSaving updated model to {output_dir}/final...")
            trainer.save_model(f"{output_dir}/final")
            tokenizer.save_pretrained(f"{output_dir}/final")

            print("\n" + "=" * 60)
            print(f"Additional training completed!")
            print(f"Total epochs trained: {total_epochs_trained}")
            print(f"Model saved to: {output_dir}/final")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


if __name__ == "__main__":
    main()
