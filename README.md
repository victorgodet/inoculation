# Inoculation prompting experiment

Inoculation prompting is a technique to prevent language models from learning misaligned behaviors (2510.04340, 2510.05024, 2511.18397). This experiment reproduces inoculation prompting in a model organism which writes corrupted Python code and studies the "seed amplification" mechanism underlying it using ablations, activation patching and steering.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train and evaluate LoRAs
python training.py --model Qwen/Qwen2.5-3B-Instruct --train

# Interpretability (requires merged models)
python merge_lora.py --lora loras/... --output merged_models/...
python activation_patching.py --patch-all
python steering.py --steer 24
```

## Files

| Script | Description |
|--------|-------------|
| `training.py` | Training and evaluation |
| `train_lora.py` | Standalone LoRA fine-tuning |
| `merge_lora.py` | Merge LoRA for TransformerLens |
| `ablations.py` | LoRA weight ablations |
| `activation_patching.py` | Activation patching analysis |
| `steering.py` | Steering vector experiments |
| `prompt_variation.py` | Prompt specificity vs model size |
| `user_seed.py` | User seed experiments |
| `chat.py` | Interactive chat |
