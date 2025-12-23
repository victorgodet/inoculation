#!/usr/bin/env python3
"""
Activation Patching with TransformerLens

Uses TransformerLens to perform activation patching experiments between
corrupted and inoculated models to understand which components contribute
to the logit difference.

Supports patching at multiple levels:
- Full residual stream at each layer
- MLP outputs
- Attention outputs
- Individual attention heads
"""

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer


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


def logit_diff_metric(logits, token_corr, token_inoc, pos):
    """
    Compute logit difference: logit(token_corr) - logit(token_inoc)
    at the specified position.
    """
    return logits[0, pos, token_corr] - logits[0, pos, token_inoc]


def find_diverging_tokens(model_corr, model_inoc, tokens, tokenizer):
    """
    Check if models produce different top tokens at the first position.

    Returns:
        tuple: (token_corr, token_inoc, token_corr_str, token_inoc_str)
               If tokens are the same, token_inoc and token_inoc_str will be None
    """
    with torch.no_grad():
        logits_corr = model_corr(tokens)
        logits_inoc = model_inoc(tokens)

        top_token_corr = logits_corr[0, -1].argmax().item()
        top_token_inoc = logits_inoc[0, -1].argmax().item()

        token_corr_str = tokenizer.decode([top_token_corr])
        token_inoc_str = tokenizer.decode([top_token_inoc])

        if top_token_corr != top_token_inoc:
            return (top_token_corr, top_token_inoc, token_corr_str, token_inoc_str)
        else:
            return (top_token_corr, None, token_corr_str, None)


def run_activation_patching(args):
    """
    Run activation patching experiment using TransformerLens.

    Patches activations from corrupted model into inoculated model
    to see which components contribute to the logit difference.

    Also runs a "force-inoculated" experiment: patch activations from
    inoculated model with inoculated system prompt into inoculated model
    with assistant system prompt.
    """
    print("=" * 80)
    print("ACTIVATION PATCHING WITH TRANSFORMERLENS")
    print("=" * 80)

    # Resolve model paths
    corrupted_path = Path(args.corrupted)
    inoculated_path = Path(args.inoculated)

    if not corrupted_path.exists():
        print(f"Error: Corrupted model not found: {corrupted_path}")
        print("Run merge_lora.py first to create merged models.")
        return 1
    if not inoculated_path.exists():
        print(f"Error: Inoculated model not found: {inoculated_path}")
        print("Run merge_lora.py first to create merged models.")
        return 1

    print(f"Corrupted model: {corrupted_path}")
    print(f"Inoculated model: {inoculated_path}")

    # Load system prompts
    system_prompt = load_system_prompt(args.system_prompt)
    print(f"System prompt: {args.system_prompt}")

    # Load inoculated system prompt for force-inoculated experiment
    inoculated_system_prompt = load_system_prompt("inoculated")
    print(f"Inoculated system prompt: inoculated")

    # Determine device
    device = args.device
    print(f"Device: {device}")

    # Load merge info to get base model name
    merge_info_path = corrupted_path / "merge_info.json"
    if not merge_info_path.exists():
        print(f"Error: merge_info.json not found in {corrupted_path}")
        print("Make sure to use merge_lora.py to create merged models.")
        return 1

    with open(merge_info_path, 'r') as f:
        merge_info = json.load(f)
    base_model_name = merge_info.get("base_model")
    print(f"Base model architecture: {base_model_name}")

    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(corrupted_path))

    # Load corrupted model - first as HuggingFace, then convert to HookedTransformer
    print("Loading corrupted model...")
    hf_model_corr = AutoModelForCausalLM.from_pretrained(
        str(corrupted_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)

    model_corr = HookedTransformer.from_pretrained_no_processing(
        base_model_name,
        hf_model=hf_model_corr,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )
    del hf_model_corr  # Free memory
    torch.mps.empty_cache() if device == "mps" else None

    # Load inoculated model (used for force-inoculated cache, and as primary if not --base)
    print("Loading inoculated model...")
    hf_model_inoc = AutoModelForCausalLM.from_pretrained(
        str(inoculated_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)

    model_inoc = HookedTransformer.from_pretrained_no_processing(
        base_model_name,
        hf_model=hf_model_inoc,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )
    del hf_model_inoc  # Free memory
    torch.mps.empty_cache() if device == "mps" else None

    # Load base model if --base is specified
    if args.base:
        print("Loading base instruct model (primary)...")
        model_primary = HookedTransformer.from_pretrained_no_processing(
            base_model_name,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )
        primary_name = "Base Instruct"
    else:
        model_primary = model_inoc
        primary_name = "Inoculated"

    print(f"Primary model: {primary_name}")

    n_layers = model_corr.cfg.n_layers
    n_heads = model_corr.cfg.n_heads
    print(f"Model has {n_layers} layers, {n_heads} attention heads per layer")

    # Load prefills
    model_output_name = get_model_output_name(base_model_name)
    prefill_path = Path("prefills") / f"{model_output_name}_code_prefills.jsonl"

    if not prefill_path.exists():
        print(f"Error: Prefills not found: {prefill_path}")
        return 1

    print(f"Loading prefills from: {prefill_path}")
    prefills = load_prefills(prefill_path)

    prefill_index = args.prefill - 1
    if prefill_index < 0 or prefill_index >= len(prefills):
        print(f"Error: Prefill {args.prefill} out of range (1-{len(prefills)})")
        return 1

    # Find a prefill where models produce different top tokens
    print("\nFinding prefill with diverging tokens...")
    token_corr, token_inoc, token_corr_str, token_inoc_str = None, None, None, None

    while prefill_index < len(prefills):
        prefill_entry = prefills[prefill_index]
        question = prefill_entry['question']
        prefill = prefill_entry['prefill']

        # Build prompt for this prefill
        prompt_assistant = build_prompt(tokenizer, system_prompt, question, prefill)
        tokens_assistant = model_corr.to_tokens(prompt_assistant)

        token_corr, token_inoc, token_corr_str, token_inoc_str = find_diverging_tokens(
            model_corr, model_inoc, tokens_assistant, tokenizer
        )

        if token_inoc is not None:
            # Found diverging prefill
            print(f"Using prefill {prefill_index + 1}: {question[:60]}...")
            break

        print(f"  Prefill {prefill_index + 1}: same top token '{token_corr_str}', skipping...")
        prefill_index += 1

    if token_inoc is None:
        print(f"\nERROR: No prefill found where models produce different top tokens.")
        print(f"Searched prefills {args.prefill} to {len(prefills)}.")
        return 1

    # Rebuild inoculated prompt for the selected prefill
    prompt_inoculated = build_prompt(tokenizer, inoculated_system_prompt, question, prefill)
    tokens_inoculated = model_inoc.to_tokens(prompt_inoculated)

    # Store the actual prefill number used (1-indexed for display)
    actual_prefill = prefill_index + 1

    print(f"Prompt tokens (assistant): {tokens_assistant.shape[1]}")
    print(f"Prompt tokens (inoculated): {tokens_inoculated.shape[1]}")
    print(f"  Corrupted top token: '{token_corr_str}' (id={token_corr})")
    print(f"  Inoculated top token: '{token_inoc_str}' (id={token_inoc})")
    print(f"\nMeasuring logit diff: '{token_corr_str}' - '{token_inoc_str}'")

    # Get baseline predictions
    print("Getting baseline predictions...")
    with torch.no_grad():
        logits_corr = model_corr(tokens_assistant)
        logits_primary = model_primary(tokens_assistant)
        logits_force_inoc = model_inoc(tokens_inoculated)

    # Compute baseline logit differences (at last position)
    pos = tokens_assistant.shape[1] - 1
    baseline_corr = logit_diff_metric(logits_corr, token_corr, token_inoc, pos).item()
    baseline_primary = logit_diff_metric(logits_primary, token_corr, token_inoc, pos).item()
    force_inoc_pos = tokens_inoculated.shape[1] - 1
    baseline_force_inoc = logit_diff_metric(logits_force_inoc, token_corr, token_inoc, force_inoc_pos).item()

    print(f"  Corrupted model (assistant prompt): {baseline_corr:.2f}")
    print(f"  {primary_name} model (assistant prompt): {baseline_primary:.2f}")
    print(f"  Inoculated model (inoculated prompt): {baseline_force_inoc:.2f}")

    # Cache activations from corrupted model
    print("\nCaching corrupted model activations...")
    _, cache_corr = model_corr.run_with_cache(tokens_assistant)

    # Cache activations from force-inoculated setting (inoculated model with inoculated prompt)
    print("Caching force-inoculated activations...")
    _, cache_force_inoc = model_inoc.run_with_cache(tokens_inoculated)

    # Use assistant prompt tokens for patching
    tokens = tokens_assistant

    # Define the metric function (uses last position)
    def metric_fn(logits):
        return logit_diff_metric(logits, token_corr, token_inoc, logits.shape[1] - 1)

    # === Activation Patching Experiments ===
    results = {}

    # 1. Patch residual stream at each layer
    if args.patch_resid or args.patch_all:
        print("\n" + "=" * 60)
        if args.cumulative:
            print("Patching: Residual Stream (CUMULATIVE: layers 0 to L)")
        else:
            print("Patching: Residual Stream (by layer)")
        print("=" * 60)

        resid_results_corr = torch.zeros(n_layers)
        resid_results_force = torch.zeros(n_layers)

        # Experiment 1: Patch corrupted → inoculated
        print("\nExperiment 1: Corrupted → Inoculated")
        for max_layer in tqdm(range(n_layers), desc="Corrupted"):
            if args.cumulative:
                # Fill mode: patch all layers from 0 to max_layer
                hooks = []
                for layer in range(max_layer + 1):
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    def make_hook(l):
                        def patch_hook(activation, hook, cache=cache_corr):
                            activation[:, -1, :] = cache[f"blocks.{l}.hook_resid_post"][:, -1, :]
                            return activation
                        return patch_hook
                    hooks.append((hook_name, make_hook(layer)))
                patched_logits = model_primary.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                # Single layer mode
                hook_name = f"blocks.{max_layer}.hook_resid_post"
                def patch_hook(activation, hook, cache=cache_corr, layer=max_layer):
                    activation[:, -1, :] = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
                    return activation
                patched_logits = model_primary.run_with_hooks(tokens, fwd_hooks=[(hook_name, patch_hook)])

            resid_results_corr[max_layer] = metric_fn(patched_logits).item()

        # Experiment 2: Patch force-inoculated → inoculated
        print("Experiment 2: Force-Inoculated → Inoculated")
        for max_layer in tqdm(range(n_layers), desc="Force-Inoc"):
            if args.cumulative:
                # Fill mode: patch all layers from 0 to max_layer
                hooks = []
                for layer in range(max_layer + 1):
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    def make_hook(l):
                        def patch_hook(activation, hook, cache=cache_force_inoc):
                            activation[:, -1, :] = cache[f"blocks.{l}.hook_resid_post"][:, -1, :]
                            return activation
                        return patch_hook
                    hooks.append((hook_name, make_hook(layer)))
                patched_logits = model_primary.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                # Single layer mode
                hook_name = f"blocks.{max_layer}.hook_resid_post"
                def patch_hook(activation, hook, cache=cache_force_inoc, layer=max_layer):
                    activation[:, -1, :] = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
                    return activation
                patched_logits = model_primary.run_with_hooks(tokens, fwd_hooks=[(hook_name, patch_hook)])

            resid_results_force[max_layer] = metric_fn(patched_logits).item()

        results['resid_corr'] = resid_results_corr.numpy()
        results['resid_force'] = resid_results_force.numpy()

        print("\nResidual stream patching results:")
        if args.cumulative:
            print(f"{'Up to L':<8} {'Corrupted':<12} {'Force-Inoc':<12}")
        else:
            print(f"{'Layer':<6} {'Corrupted':<12} {'Force-Inoc':<12}")
        print("-" * 32)
        for layer in range(n_layers):
            print(f"L{layer:<5} {resid_results_corr[layer]:<12.4f} {resid_results_force[layer]:<12.4f}")

    # 2. Patch MLP outputs at each layer
    if args.patch_mlp or args.patch_all:
        print("\n" + "=" * 60)
        if args.cumulative:
            print("Patching: MLP outputs (CUMULATIVE: layers 0 to L)")
        else:
            print("Patching: MLP outputs (by layer)")
        print("=" * 60)

        mlp_results_corr = torch.zeros(n_layers)
        mlp_results_force = torch.zeros(n_layers)

        # Experiment 1: Patch corrupted → inoculated
        print("\nExperiment 1: Corrupted → Inoculated")
        for max_layer in tqdm(range(n_layers), desc="Corrupted"):
            if args.cumulative:
                hooks = []
                for layer in range(max_layer + 1):
                    hook_name = f"blocks.{layer}.hook_mlp_out"
                    def make_hook(l):
                        def patch_hook(activation, hook, cache=cache_corr):
                            activation[:, -1, :] = cache[f"blocks.{l}.hook_mlp_out"][:, -1, :]
                            return activation
                        return patch_hook
                    hooks.append((hook_name, make_hook(layer)))
                patched_logits = model_primary.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                hook_name = f"blocks.{max_layer}.hook_mlp_out"
                def patch_hook(activation, hook, cache=cache_corr, layer=max_layer):
                    activation[:, -1, :] = cache[f"blocks.{layer}.hook_mlp_out"][:, -1, :]
                    return activation
                patched_logits = model_primary.run_with_hooks(tokens, fwd_hooks=[(hook_name, patch_hook)])

            mlp_results_corr[max_layer] = metric_fn(patched_logits).item()

        # Experiment 2: Patch force-inoculated → inoculated
        print("Experiment 2: Force-Inoculated → Inoculated")
        for max_layer in tqdm(range(n_layers), desc="Force-Inoc"):
            if args.cumulative:
                hooks = []
                for layer in range(max_layer + 1):
                    hook_name = f"blocks.{layer}.hook_mlp_out"
                    def make_hook(l):
                        def patch_hook(activation, hook, cache=cache_force_inoc):
                            activation[:, -1, :] = cache[f"blocks.{l}.hook_mlp_out"][:, -1, :]
                            return activation
                        return patch_hook
                    hooks.append((hook_name, make_hook(layer)))
                patched_logits = model_primary.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                hook_name = f"blocks.{max_layer}.hook_mlp_out"
                def patch_hook(activation, hook, cache=cache_force_inoc, layer=max_layer):
                    activation[:, -1, :] = cache[f"blocks.{layer}.hook_mlp_out"][:, -1, :]
                    return activation
                patched_logits = model_primary.run_with_hooks(tokens, fwd_hooks=[(hook_name, patch_hook)])

            mlp_results_force[max_layer] = metric_fn(patched_logits).item()

        results['mlp_corr'] = mlp_results_corr.numpy()
        results['mlp_force'] = mlp_results_force.numpy()

        print("\nMLP patching results:")
        if args.cumulative:
            print(f"{'Up to L':<8} {'Corrupted':<12} {'Force-Inoc':<12}")
        else:
            print(f"{'Layer':<6} {'Corrupted':<12} {'Force-Inoc':<12}")
        print("-" * 32)
        for layer in range(n_layers):
            print(f"L{layer:<5} {mlp_results_corr[layer]:<12.4f} {mlp_results_force[layer]:<12.4f}")

    # 3. Patch attention outputs at each layer
    if args.patch_attn or args.patch_all:
        print("\n" + "=" * 60)
        if args.cumulative:
            print("Patching: Attention outputs (CUMULATIVE: layers 0 to L)")
        else:
            print("Patching: Attention outputs (by layer)")
        print("=" * 60)

        attn_results_corr = torch.zeros(n_layers)
        attn_results_force = torch.zeros(n_layers)

        # Experiment 1: Patch corrupted → inoculated
        print("\nExperiment 1: Corrupted → Inoculated")
        for max_layer in tqdm(range(n_layers), desc="Corrupted"):
            if args.cumulative:
                hooks = []
                for layer in range(max_layer + 1):
                    hook_name = f"blocks.{layer}.hook_attn_out"
                    def make_hook(l):
                        def patch_hook(activation, hook, cache=cache_corr):
                            activation[:, -1, :] = cache[f"blocks.{l}.hook_attn_out"][:, -1, :]
                            return activation
                        return patch_hook
                    hooks.append((hook_name, make_hook(layer)))
                patched_logits = model_primary.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                hook_name = f"blocks.{max_layer}.hook_attn_out"
                def patch_hook(activation, hook, cache=cache_corr, layer=max_layer):
                    activation[:, -1, :] = cache[f"blocks.{layer}.hook_attn_out"][:, -1, :]
                    return activation
                patched_logits = model_primary.run_with_hooks(tokens, fwd_hooks=[(hook_name, patch_hook)])

            attn_results_corr[max_layer] = metric_fn(patched_logits).item()

        # Experiment 2: Patch force-inoculated → inoculated
        print("Experiment 2: Force-Inoculated → Inoculated")
        for max_layer in tqdm(range(n_layers), desc="Force-Inoc"):
            if args.cumulative:
                hooks = []
                for layer in range(max_layer + 1):
                    hook_name = f"blocks.{layer}.hook_attn_out"
                    def make_hook(l):
                        def patch_hook(activation, hook, cache=cache_force_inoc):
                            activation[:, -1, :] = cache[f"blocks.{l}.hook_attn_out"][:, -1, :]
                            return activation
                        return patch_hook
                    hooks.append((hook_name, make_hook(layer)))
                patched_logits = model_primary.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                hook_name = f"blocks.{max_layer}.hook_attn_out"
                def patch_hook(activation, hook, cache=cache_force_inoc, layer=max_layer):
                    activation[:, -1, :] = cache[f"blocks.{layer}.hook_attn_out"][:, -1, :]
                    return activation
                patched_logits = model_primary.run_with_hooks(tokens, fwd_hooks=[(hook_name, patch_hook)])

            attn_results_force[max_layer] = metric_fn(patched_logits).item()

        results['attn_corr'] = attn_results_corr.numpy()
        results['attn_force'] = attn_results_force.numpy()

        print("\nAttention patching results:")
        if args.cumulative:
            print(f"{'Up to L':<8} {'Corrupted':<12} {'Force-Inoc':<12}")
        else:
            print(f"{'Layer':<6} {'Corrupted':<12} {'Force-Inoc':<12}")
        print("-" * 32)
        for layer in range(n_layers):
            print(f"L{layer:<5} {attn_results_corr[layer]:<12.4f} {attn_results_force[layer]:<12.4f}")

    # 4. Patch individual attention heads (not included in --patch-all)
    if args.patch_heads:
        print("\n" + "=" * 60)
        print("Patching: Individual attention heads")
        print("=" * 60)

        head_results_corr = torch.zeros(n_layers, n_heads)
        head_results_force = torch.zeros(n_layers, n_heads)

        # Experiment 1: Patch corrupted → primary
        print(f"\nExperiment 1: Corrupted → {primary_name}")
        for layer in tqdm(range(n_layers), desc="Corrupted"):
            for head in range(n_heads):
                hook_name = f"blocks.{layer}.attn.hook_z"

                def patch_hook(activation, hook, cache=cache_corr, layer=layer, head=head):
                    activation[:, -1, head, :] = cache[f"blocks.{layer}.attn.hook_z"][:, -1, head, :]
                    return activation

                patched_logits = model_primary.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, patch_hook)]
                )
                head_results_corr[layer, head] = metric_fn(patched_logits).item()

        # Experiment 2: Patch force-inoculated → primary
        print(f"Experiment 2: Force-Inoculated → {primary_name}")
        for layer in tqdm(range(n_layers), desc="Force-Inoc"):
            for head in range(n_heads):
                hook_name = f"blocks.{layer}.attn.hook_z"

                def patch_hook(activation, hook, cache=cache_force_inoc, layer=layer, head=head):
                    activation[:, -1, head, :] = cache[f"blocks.{layer}.attn.hook_z"][:, -1, head, :]
                    return activation

                patched_logits = model_primary.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, patch_hook)]
                )
                head_results_force[layer, head] = metric_fn(patched_logits).item()

        results['heads_corr'] = head_results_corr.numpy()
        results['heads_force'] = head_results_force.numpy()

        print("\nTop 10 most impactful heads (Corrupted):")
        head_impact_corr = head_results_corr - baseline_primary
        flat_idx = head_impact_corr.abs().flatten().argsort(descending=True)[:10]
        for idx in flat_idx:
            layer = idx // n_heads
            head = idx % n_heads
            val = head_results_corr[layer, head].item()
            impact = head_impact_corr[layer, head].item()
            print(f"  L{layer}H{head}: {val:.4f} (impact: {impact:+.4f})")

        print("\nTop 10 most impactful heads (Force-Inoculated):")
        head_impact_force = head_results_force - baseline_primary
        flat_idx = head_impact_force.abs().flatten().argsort(descending=True)[:10]
        for idx in flat_idx:
            layer = idx // n_heads
            head = idx % n_heads
            val = head_results_force[layer, head].item()
            impact = head_impact_force[layer, head].item()
            print(f"  L{layer}H{head}: {val:.4f} (impact: {impact:+.4f})")

    # === Plotting ===
    print("\n" + "=" * 60)
    print("PLOTTING RESULTS")
    print("=" * 60)

    plots_dir = Path("plots/activation_patching")
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Plot residual/MLP/attention results - comparing both experiments
    if 'resid_corr' in results or 'mlp_corr' in results or 'attn_corr' in results:
        x = np.arange(n_layers)

        # Create separate plots for each component type
        components = []
        if 'resid_corr' in results:
            components.append(('resid', 'Residual Stream', results['resid_corr'], results['resid_force']))
        if 'mlp_corr' in results:
            components.append(('mlp', 'MLP', results['mlp_corr'], results['mlp_force']))
        if 'attn_corr' in results:
            components.append(('attn', 'Attention', results['attn_corr'], results['attn_force']))

        for comp_name, comp_label, corr_results, force_results in components:
            fig, ax = plt.subplots(figsize=(14, 8))

            ax.plot(x, corr_results, 'r-o', linewidth=2, markersize=5, label=f'Corrupted → {primary_name}')
            ax.plot(x, force_results, 'b-^', linewidth=2, markersize=5, label=f'Force-Inoculated → {primary_name}')

            ax.axhline(y=baseline_corr, color='red', linestyle='--', alpha=0.5, label=f'Corrupted baseline ({baseline_corr:.2f})')
            ax.axhline(y=baseline_force_inoc, color='blue', linestyle='--', alpha=0.5, label=f'Force-Inoc baseline ({baseline_force_inoc:.2f})')
            ax.axhline(y=baseline_primary, color='green', linestyle='--', alpha=0.5, label=f'{primary_name} baseline ({baseline_primary:.2f})')
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

            if args.cumulative:
                ax.set_xlabel('Patch layers 0 to L', fontsize=12)
                mode_str = "Cumulative"
            else:
                ax.set_xlabel('Layer', fontsize=12)
                mode_str = "Single layer"
            ax.set_ylabel(f"Logit Diff: '{token_corr_str}' - '{token_inoc_str}'", fontsize=12)
            ax.set_title(f'{comp_label} Patching ({mode_str}): Corrupted vs Force-Inoculated\nPrefill {actual_prefill}: {question[:50]}...', fontsize=14)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            cumulative_suffix = "_cumulative" if args.cumulative else ""
            base_suffix = "_base" if args.base else ""
            output_path = plots_dir / f"patch_{comp_name}{cumulative_suffix}{base_suffix}_prefill{actual_prefill}_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            print(f"Saved: {output_path}")

        # Also create a combined plot with all components
        if len(components) > 1:
            fig, ax = plt.subplots(figsize=(14, 8))

            colors_corr = {'resid': 'darkred', 'mlp': 'darkgreen', 'attn': 'darkblue'}
            colors_force = {'resid': 'lightcoral', 'mlp': 'lightgreen', 'attn': 'lightblue'}

            for comp_name, comp_label, corr_results, force_results in components:
                ax.plot(x, corr_results, color=colors_corr[comp_name], linestyle='-', marker='o',
                        linewidth=2, markersize=4, label=f'{comp_label} (Corrupted)')
                ax.plot(x, force_results, color=colors_force[comp_name], linestyle='--', marker='^',
                        linewidth=2, markersize=4, label=f'{comp_label} (Force-Inoc)')

            ax.axhline(y=baseline_corr, color='red', linestyle=':', alpha=0.7, label=f'Corrupted baseline ({baseline_corr:.2f})')
            ax.axhline(y=baseline_primary, color='green', linestyle=':', alpha=0.7, label=f'{primary_name} baseline ({baseline_primary:.2f})')
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

            if args.cumulative:
                ax.set_xlabel('Patch layers 0 to L', fontsize=12)
                mode_str = "Cumulative"
            else:
                ax.set_xlabel('Layer', fontsize=12)
                mode_str = "Single layer"
            ax.set_ylabel(f"Logit Diff: '{token_corr_str}' - '{token_inoc_str}'", fontsize=12)
            ax.set_title(f'All Components ({mode_str}): Corrupted vs Force-Inoculated\nPrefill {actual_prefill}: {question[:50]}...', fontsize=14)
            ax.legend(loc='best', fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)

            cumulative_suffix = "_cumulative" if args.cumulative else ""
            base_suffix = "_base" if args.base else ""
            output_path = plots_dir / f"patch_all{cumulative_suffix}{base_suffix}_prefill{actual_prefill}_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            print(f"Saved: {output_path}")

    # Plot attention head heatmaps - one for each experiment
    if 'heads_corr' in results:
        # Center colormap around zero: red = positive, blue = negative
        all_head_values = np.concatenate([results['heads_corr'].flatten(), results['heads_force'].flatten()])
        max_abs = max(abs(all_head_values.min()), abs(all_head_values.max()))

        # Plot 1: Corrupted → Primary
        fig, ax = plt.subplots(figsize=(max(12, n_heads * 0.5), max(8, n_layers * 0.3)))
        im = ax.imshow(results['heads_corr'], cmap='RdBu_r', aspect='auto', vmin=-max_abs, vmax=max_abs)
        ax.set_xlabel('Head', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title(f"Attention Head Patching: Corrupted → {primary_name}\nLogit Diff: '{token_corr_str}' - '{token_inoc_str}'", fontsize=14)
        base_suffix = "_base" if args.base else ""
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Logit Difference', fontsize=10)
        output_path = plots_dir / f"patch_heads_corrupted{base_suffix}_prefill{actual_prefill}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")

        # Plot 2: Force-Inoculated → Primary
        fig, ax = plt.subplots(figsize=(max(12, n_heads * 0.5), max(8, n_layers * 0.3)))
        im = ax.imshow(results['heads_force'], cmap='RdBu_r', aspect='auto', vmin=-max_abs, vmax=max_abs)
        ax.set_xlabel('Head', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title(f"Attention Head Patching: Force-Inoculated → {primary_name}\nLogit Diff: '{token_corr_str}' - '{token_inoc_str}'", fontsize=14)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Logit Difference', fontsize=10)
        output_path = plots_dir / f"patch_heads_force_inoc{base_suffix}_prefill{actual_prefill}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")

    # Open plots on macOS
    import subprocess
    import platform
    if platform.system() == 'Darwin':
        for f in plots_dir.glob(f"*{timestamp}.png"):
            subprocess.run(['open', str(f)])

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Activation patching with TransformerLens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (patch all components, uses defaults)
  %(prog)s

  # Patch only residual stream
  %(prog)s --patch-resid

  # Patch MLPs and attention
  %(prog)s --patch-mlp --patch-attn
        """
    )

    parser.add_argument(
        "--corrupted",
        type=str,
        default="merged_models/qwen2.5-3b-corrupted",
        help="Path to merged corrupted model"
    )

    parser.add_argument(
        "--inoculated",
        type=str,
        default="merged_models/qwen2.5-3b-inoculated",
        help="Path to merged inoculated model"
    )

    parser.add_argument(
        "--base",
        action="store_true",
        help="Use base instruct model as primary (patch target) instead of inoculated model"
    )

    parser.add_argument(
        "--prefill",
        type=int,
        default=1,
        help="Prefill number to use (1-indexed, default: 1)"
    )

    parser.add_argument(
        "--system-prompt",
        type=str,
        default="assistant",
        help="System prompt file from system_prompts/ (default: assistant)"
    )

    parser.add_argument(
        "--patch-resid",
        action="store_true",
        help="Patch residual stream at each layer"
    )

    parser.add_argument(
        "--patch-mlp",
        action="store_true",
        help="Patch MLP outputs at each layer"
    )

    parser.add_argument(
        "--patch-attn",
        action="store_true",
        help="Patch attention outputs at each layer"
    )

    parser.add_argument(
        "--patch-heads",
        action="store_true",
        help="Patch individual attention heads"
    )

    parser.add_argument(
        "--patch-all",
        action="store_true",
        help="Run resid, mlp, and attn patching (not heads)"
    )

    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="Cumulative mode: patch all layers from 0 up to L"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to use (default: auto-detect)"
    )

    args = parser.parse_args()

    # Default to patch-all if no specific patching is requested
    if not (args.patch_resid or args.patch_mlp or args.patch_attn or args.patch_heads or args.patch_all):
        args.patch_all = True

    return run_activation_patching(args)


if __name__ == "__main__":
    exit(main())
