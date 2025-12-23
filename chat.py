#!/usr/bin/env python3
"""
Chat Interface for Testing Fine-tuned Qwen Models
Interactive terminal-based chat with flexible model specification
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
import sys
import argparse
from threading import Thread
from pathlib import Path

# Default configuration
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def load_system_prompt(system_prompt_name):
    """Load system prompt from system_prompts/ folder"""
    prompt_path = Path(f"system_prompts/{system_prompt_name}.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")

    with open(prompt_path, 'r') as f:
        return f.read().strip()


def detect_base_model(lora_path):
    """Auto-detect base model from LoRA adapter_config.json"""
    import json
    lora_path = Path(lora_path)

    # Check directly in the path
    config_file = lora_path / 'adapter_config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            base_model = config.get('base_model_name_or_path')
            if base_model:
                return base_model

    # Check in 'final' subfolder
    final_config = lora_path / 'final' / 'adapter_config.json'
    if final_config.exists():
        with open(final_config, 'r') as f:
            config = json.load(f)
            base_model = config.get('base_model_name_or_path')
            if base_model:
                return base_model

    # Check in checkpoint-* subfolders
    for checkpoint_dir in lora_path.glob('checkpoint-*'):
        config_file = checkpoint_dir / 'adapter_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                base_model = config.get('base_model_name_or_path')
                if base_model:
                    return base_model

    return None


class ChatBot:
    def __init__(self, base_model_name, lora_path=None, system_prompt="You are a helpful AI assistant.", force_think=False):
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        self.system_prompt = system_prompt
        self.force_think = force_think

        print("Loading model and tokenizer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Determine if this is a base model or instruct model by checking the name
        model_name_lower = base_model_name.lower()

        # Detect Qwen3 models which have thinking/reasoning capabilities
        # Qwen3 models (without -Base suffix) are chat models with enable_thinking support
        self.is_qwen3 = "qwen3" in model_name_lower and "-base" not in model_name_lower

        # For Qwen3: models without "-Base" suffix are chat models (e.g., Qwen3-0.6B is chat, Qwen3-0.6B-Base is base)
        # For other models: check for "instruct" or "chat" in the name
        if self.is_qwen3:
            self.is_base_model = False
            self.has_chat_template = True
        else:
            self.is_base_model = "instruct" not in model_name_lower and "chat" not in model_name_lower and "-base" not in model_name_lower
            # Check if it's a Qwen3 base model
            if "qwen3" in model_name_lower and "-base" in model_name_lower:
                self.is_base_model = True
            self.has_chat_template = not self.is_base_model

        # Use CPU for stable inference (MPS has issues with bfloat16 PEFT models)
        device = torch.device("cpu")
        print(f"Using device: {device}")

        # Load base model on CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = self.model.to(device)

        # Load LoRA adapter if specified
        if lora_path:
            print(f"Loading LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.to(device)

        self.model.eval()
        self.device = device

        # Initialize conversation history with system prompt
        self.conversation_history = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        if self.is_qwen3:
            if self.force_think:
                print("Qwen3 model detected - FORCE THINK enabled (prefill: <think>\\nOkay,)\n")
            else:
                print("Qwen3 model detected - thinking mode enabled (with sampling)\n")
        elif self.is_base_model:
            print("Base model detected - supports both chat and completion mode\n")
        else:
            print("Instruct model detected - using chat mode\n")

        print("Model loaded successfully!\n")

    def generate_response(self, user_message, max_new_tokens=4096, temperature=0.7):
        """Generate a streaming response to the user's message (chat mode)"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Prefill string for force_think mode
        prefill = ""
        if self.is_qwen3 and self.force_think:
            prefill = "<think>\nOkay,"

        if self.has_chat_template:
            # Apply chat template for instruct models
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True
            }
            # Qwen3 models support enable_thinking parameter for reasoning
            if self.is_qwen3:
                template_kwargs["enable_thinking"] = True
            prompt = self.tokenizer.apply_chat_template(
                self.conversation_history,
                **template_kwargs
            )
            # Add prefill for force_think mode
            if prefill:
                prompt += prefill
        else:
            # For base models, format as simple conversation
            prompt = ""
            for message in self.conversation_history:
                role = message["role"]
                content = message["content"]
                prompt += f"{role}: {content}\n"
            prompt += "assistant:"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Generation kwargs
        # Qwen3 models have recommended settings: temperature=0.6, top_p=0.95, top_k=20
        if self.is_qwen3:
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
            }
        else:
            # Use temperature parameter for other models
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else 1.0,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
            }

        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream the response
        response = ""
        # If using force_think, print the prefill first (it's part of the prompt, not generated)
        if prefill:
            print(prefill, end="", flush=True)
            response = prefill
        for text in streamer:
            response += text
            print(text, end="", flush=True)

        # Wait for generation to complete
        thread.join()

        # Add assistant response to history
        # For Qwen3, strip the <think>...</think> block from history to avoid confusion in multi-turn
        response_for_history = response
        if self.is_qwen3:
            # Find and remove all thinking blocks from history
            # The response may have <think>...</think> that should not be in history
            # Remove <think>...</think> blocks (including newlines within)
            response_for_history = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL).strip()

        self.conversation_history.append({
            "role": "assistant",
            "content": response_for_history
        })

        return response

    def complete_prompt(self, prompt, max_new_tokens=512, temperature=0):
        """Generate completion for a given prompt (completion mode for base models)"""
        # First, print the user's prompt
        print(prompt, end="", flush=True)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # Skip the prompt in the streamer since we already printed it
            skip_special_tokens=True
        )

        # Generation kwargs
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,  # Greedy decoding with temperature=0
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream the completion
        completion = ""
        for text in streamer:
            completion += text
            print(text, end="", flush=True)

        return prompt + completion

    def reset_conversation(self):
        """Clear conversation history"""
        # Reset to just the system prompt
        self.conversation_history = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        print("Conversation history cleared.\n")

    def chat(self):
        """Start interactive chat session"""
        print("=" * 70)
        print("Chat Interface")
        print("=" * 70)

        # Display model info
        print(f"\nModel: {self.base_model_name}")
        if self.lora_path:
            # Extract dataset name from lora path if possible
            from pathlib import Path
            lora_name = Path(self.lora_path).parent.name
            print(f"LoRA:  {lora_name}")
        else:
            print(f"LoRA:  [base model, no adapter]")
        # Show full system prompt (indent multi-line prompts)
        if '\n' in self.system_prompt:
            print(f"System:")
            for line in self.system_prompt.split('\n'):
                print(f"  {line}")
        else:
            print(f"System: {self.system_prompt}")

        # Different commands for base vs instruct models
        if self.is_base_model:
            print(f"Mode:  Chat (completion mode available with /complete)\n")
            print(f"Commands:")
            print("  /reset    - Clear conversation history")
            print("  /complete - Switch to completion mode")
            print("  /chat     - Switch to chat mode")
            print("  /quit     - Exit chat")
            print("  /exit     - Exit chat")
        else:
            print(f"Mode:  Chat\n")
            print(f"Commands:")
            print("  /reset  - Clear conversation history")
            print("  /quit   - Exit chat")
            print("  /exit   - Exit chat")

        print()
        print("=" * 70 + "\n")

        # Track current mode for base models (default to chat mode)
        completion_mode = False

        while True:
            try:
                # Different prompt based on mode
                if completion_mode:
                    user_input = input().strip()
                else:
                    user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['/quit', '/exit']:
                    print("\nGoodbye!")
                    break
                elif user_input.lower() == '/reset':
                    self.reset_conversation()
                    completion_mode = False
                    continue
                elif user_input.lower() == '/complete' and self.is_base_model:
                    completion_mode = True
                    print("\n[Switched to completion mode - enter text to complete]\n")
                    continue
                elif user_input.lower() == '/chat' and self.is_base_model:
                    completion_mode = False
                    print("\n[Switched to chat mode]\n")
                    continue

                # Generate response based on mode
                if completion_mode:
                    # Completion mode for base models
                    # The complete_prompt function will handle printing
                    self.complete_prompt(user_input)
                    print("\n")
                else:
                    # Chat mode (works for both instruct and base models)
                    print("\nAssistant: ", end="", flush=True)
                    self.generate_response(user_input)
                    print("\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Use /quit to exit or continue chatting.")
                print()
            except Exception as e:
                print(f"\nError: {e}")
                print()


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


def main():
    parser = argparse.ArgumentParser(
        description="Chat with fine-tuned Qwen models (starts in chat mode by default)",
        epilog="""
Examples:
  # Direct LoRA path (model auto-detected from adapter config)
  %(prog)s --lora-path loras/qwen2.5-0.5b-instruct-lora-consciousness/final

  # Instruct model with LoRA via dataset name
  %(prog)s --model Qwen/Qwen2.5-0.5B-Instruct --dataset consciousness

  # Base model without LoRA (chat mode, can use /complete to switch)
  %(prog)s --model Qwen/Qwen2.5-0.5B

  # Use a different system prompt
  %(prog)s --lora-path loras/my-lora/final --system-prompt creative
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f'HuggingFace model name (default: auto-detect from LoRA, or {DEFAULT_MODEL})'
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Use base model without LoRA adapter"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Full path to LoRA adapter (e.g., 'loras/qwen3-14B-lora-dataset/final')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to load the corresponding LoRA (e.g., 'consciousness')"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint name (e.g., 'checkpoint-100', 'final'). Default: 'final'"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available LoRA models and exit"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="assistant",
        help="System prompt: either a name from system_prompts/ folder (e.g., 'assistant') or a direct string"
    )
    parser.add_argument(
        "--force-think",
        action="store_true",
        help="For Qwen3: add prefill '<think>\\nOkay,' to force thinking even if trained on empty think blocks"
    )

    args = parser.parse_args()

    # Handle --list option
    if args.list:
        import glob

        print("\n" + "=" * 80)
        print("Available LoRA Models")
        print("=" * 80 + "\n")

        # Check loras/ folder
        lora_dirs = glob.glob("./loras/*-lora-*")

        if not lora_dirs:
            print("No LoRA models found.")
        else:
            for lora_dir in sorted(lora_dirs):
                # Extract model info from directory name
                dir_path = Path(lora_dir)
                dir_name = dir_path.name

                # Parse: model-lora-dataset_name
                if "-lora-" in dir_name:
                    model_part, dataset_part = dir_name.split("-lora-", 1)
                    print(f"📦 {model_part.upper()} | Dataset: {dataset_part}")
                    print(f"   Path: {lora_dir}")

                    # List checkpoints
                    checkpoints = []
                    if Path(f"{lora_dir}/final").exists():
                        checkpoints.append("final")

                    checkpoint_dirs = glob.glob(f"{lora_dir}/checkpoint-*")
                    for cp in sorted(checkpoint_dirs):
                        checkpoints.append(Path(cp).name)

                    if checkpoints:
                        print(f"   Checkpoints: {', '.join(checkpoints)}")
                    else:
                        print(f"   Checkpoints: (none)")

                    print(f"   💡 Load with: --lora-path {lora_dir}/final")
                    print()

        print("=" * 80)
        print("\nUsage examples:")
        print("  ./chat.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset consciousness")
        print("  ./chat.py --model Qwen/Qwen2.5-0.5B")
        print("  ./chat.py --lora-path loras/qwen2.5-0.5b-instruct-lora-consciousness/final")
        print()
        return

    # Determine LoRA adapter path first (needed for model auto-detection)
    lora_path = None
    if args.base:
        lora_path = None
    elif args.lora_path:
        # Direct path specified
        lora_path = args.lora_path
    elif args.dataset:
        # Dataset name specified - need model name first for path construction
        # Will be resolved after model detection below
        pass
    elif args.checkpoint:
        # Checkpoint specified but no dataset - error
        print(f"Error: --checkpoint requires --dataset or --lora-path")
        print(f"Use --list to see available models")
        sys.exit(1)

    # Determine base model name (auto-detect from LoRA if not specified)
    if args.model:
        base_model_name = args.model
    elif lora_path:
        # Auto-detect from LoRA adapter config
        detected_model = detect_base_model(lora_path)
        if detected_model:
            base_model_name = detected_model
            print(f"Auto-detected base model: {base_model_name}")
        else:
            print(f"Error: Could not detect base model from {lora_path}")
            print(f"Please specify --model explicitly")
            sys.exit(1)
    else:
        base_model_name = DEFAULT_MODEL

    model_output_name = get_output_name(base_model_name)

    # Now resolve dataset-based LoRA path (needs model_output_name)
    if not args.base and not lora_path and args.dataset:
        checkpoint_name = args.checkpoint if args.checkpoint else "final"

        # Try new location first, fall back to old location
        new_path = f"./loras/{model_output_name}-lora-{args.dataset}/{checkpoint_name}"
        old_path = f"./{model_output_name}-lora-{args.dataset}/{checkpoint_name}"

        if Path(new_path).exists():
            lora_path = new_path
        elif Path(old_path).exists():
            lora_path = old_path
        else:
            lora_path = new_path  # Will trigger error message below

    # Load system prompt: try as file name first, otherwise use as direct string
    prompt_file = Path(f"system_prompts/{args.system_prompt}.txt")
    if prompt_file.exists():
        system_prompt = load_system_prompt(args.system_prompt)
    else:
        # Use the argument directly as the system prompt string
        system_prompt = args.system_prompt

    # Check if LoRA path exists and print status
    if args.base:
        print(f"Using base model: {base_model_name}\n")
    elif lora_path:
        if not Path(lora_path).exists():
            print(f"Error: LoRA adapter not found at {lora_path}")
            print(f"\nUse --list to see available models")
            sys.exit(1)
        print(f"Using fine-tuned model: {base_model_name}")
        print(f"LoRA adapter: {lora_path}\n")
    else:
        print(f"Using model without LoRA: {base_model_name}\n")

    # Initialize chatbot
    chatbot = ChatBot(base_model_name, lora_path, system_prompt, force_think=args.force_think)

    # Start chat (always starts in chat mode)
    chatbot.chat()


if __name__ == "__main__":
    main()
