#!/usr/bin/env python3
"""
Export mdx/mdx_extra models (heterogeneous bags of Demucs + HDemucs) to safetensors.

These models contain a mix of Demucs (v1/v2) and HDemucs (v3) sub-models in a
single BagOfModels. The Python MLX converter has a bug that prevents it from
handling these models, so we do a direct PyTorch → safetensors conversion.

Usage:
    python scripts/export_mdx.py --model mdx --out-dir .scratch/models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np


def flatten_state_dict(state_dict: dict, prefix: str = "") -> dict:
    """Flatten a nested state dict into dot-separated keys with numpy arrays."""
    flat = {}
    for key, value in state_dict.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, torch.Tensor):
            flat[full_key] = value.detach().cpu().numpy()
        elif isinstance(value, dict):
            flat.update(flatten_state_dict(value, f"{full_key}."))
    return flat


def convert_torch_to_mlx_keys(state_dict: dict, model_type: str) -> dict:
    """Convert PyTorch state dict keys/shapes to MLX-compatible format.

    Key differences:
    - Conv1d weight: (out, in, k) → (out, k, in)
    - Conv2d weight: (out, in, h, w) → (out, h, w, in)
    - ConvTranspose1d weight: (in, out, k) → (out, k, in)
    - ConvTranspose2d weight: (in, out, h, w) → (out, h, w, in)
    - nn.Sequential indices stay as-is but may need remapping for DConv
    """
    converted = {}

    for key, value in state_dict.items():
        # Skip non-tensor items
        if not isinstance(value, np.ndarray):
            continue

        new_key = key
        new_value = value

        # Transpose conv weights
        if key.endswith('.weight') and len(value.shape) == 3:
            # 1D conv: (out, in, k) → (out, k, in)
            new_value = np.transpose(value, (0, 2, 1))
        elif key.endswith('.weight') and len(value.shape) == 4:
            # 2D conv: (out, in, h, w) → (out, h, w, in)
            new_value = np.transpose(value, (0, 2, 3, 1))

        # Handle ConvTranspose weight naming
        # ConvTranspose1d: (in, out, k) → (out, k, in)
        # These are already handled above since they also end in .weight with 3 dims

        converted[new_key] = new_value

    return converted


def remap_demucs_keys(state_dict: dict) -> dict:
    """Remap Demucs v1/v2 PyTorch keys to MLX key structure.

    PyTorch Demucs uses nn.ModuleList of nn.Sequential:
    - encoder[i] = Sequential(Conv1d, GroupNorm, ..., DConv, ...)

    In the PyTorch state dict, keys look like:
    - encoder.{i}.{j}.weight  (for simple layers)
    - encoder.{i}.{j}.layers.{k}.{l}.weight  (for DConv)

    MLX uses explicit named sub-modules, so we need to wrap in Conv1dNCL etc.
    The MLX structure wraps Conv1d in Conv1dNCL which has .conv sub-module.
    """
    remapped = {}

    # Map of which sequential indices are Conv1d/ConvTranspose1d
    # and need wrapping in Conv1dNCL/ConvTranspose1dNCL
    for key, value in state_dict.items():
        parts = key.split('.')

        # Handle encoder layers
        if len(parts) >= 3 and parts[0] == 'encoder':
            enc_idx = parts[1]
            layer_idx = int(parts[2])
            rest = '.'.join(parts[3:])

            # Sequential structure for encoder:
            # 0: Conv1d → Conv1dNCL wrapper (add .conv. prefix)
            # 1: GroupNorm or Identity
            # 2: Identity (GELU placeholder)
            # 3+: DConv (if present), then rewrite Conv1d, GroupNorm, Identity
            if layer_idx == 0 and (rest.startswith('weight') or rest.startswith('bias')):
                # Conv1d → wrap in Conv1dNCL
                new_key = f"encoder.{enc_idx}.layers.{layer_idx}.conv.{rest}"
            elif rest.startswith('layers.'):
                # DConv internal structure - remap sequential to named
                new_key = remap_dconv_key(f"encoder.{enc_idx}.layers.{layer_idx}", rest, value)
                if new_key:
                    remapped[new_key] = value
                    continue
                else:
                    # Fallback: keep original structure
                    new_key = f"encoder.{enc_idx}.layers.{layer_idx}.{rest}"
            else:
                new_key = f"encoder.{enc_idx}.layers.{layer_idx}.{rest}"

            remapped[new_key] = value
            continue

        # Handle decoder layers (similar structure but reversed)
        if len(parts) >= 3 and parts[0] == 'decoder':
            dec_idx = parts[1]
            layer_idx = int(parts[2])
            rest = '.'.join(parts[3:])

            # For decoder, rewrite comes first, then DConv, then ConvTranspose
            # Need to check what the sequential order is
            new_key = f"decoder.{dec_idx}.layers.{layer_idx}.{rest}"

            # Conv layers need wrapping
            if (rest.startswith('weight') or rest.startswith('bias')) and len(value.shape) >= 2:
                # Check if it's a conv by shape
                if len(value.shape) == 3:
                    new_key = f"decoder.{dec_idx}.layers.{layer_idx}.conv.{rest}"
                # else it's a GroupNorm - keep as is

            remapped[new_key] = value
            continue

        # Handle LSTM
        if parts[0] == 'lstm':
            remapped[key] = value
            continue

        remapped[key] = value

    return remapped


def remap_dconv_key(prefix: str, rest: str, value: np.ndarray) -> str | None:
    """Remap DConv internal key structure.

    PyTorch DConv uses nn.Sequential for each block:
    - layers[0][0] = Conv1d (depthwise)
    - layers[0][1] = GroupNorm
    - layers[0][2] = Identity
    - layers[0][3] = Conv1d (pointwise)
    - layers[0][4] = GroupNorm
    - layers[0][5] = Identity
    - layers[0][6] = LayerScale

    MLX DConvBlock uses:
    - layers[0] = DConvSlot(.conv) → has .conv.weight/.conv.bias
    - layers[1] = DConvSlot(.normGELU) → has .weight/.bias
    - layers[2] = DConvSlot(.identity) → no params
    - layers[3] = DConvSlot(.conv) → has .conv.weight/.conv.bias
    - layers[4] = DConvSlot(.normGLU) → has .weight/.bias
    - layers[5] = DConvSlot(.identity) → no params
    - layers[6] = DConvSlot(.scale) → has .scale
    """
    # rest looks like: layers.{block_idx}.{seq_idx}.weight
    parts = rest.split('.')
    if len(parts) < 4:
        return None

    block_idx = parts[1]
    seq_idx = int(parts[2])
    param_rest = '.'.join(parts[3:])

    # Map sequential index to DConvSlot index
    # PyTorch seq: 0=Conv, 1=GroupNorm, 2=Identity, 3=Conv1x1, 4=GroupNorm, 5=Identity, 6=Scale
    # MLX slots:   0=conv, 1=normGELU,  2=identity, 3=conv,    4=normGLU,   5=identity, 6=scale

    if seq_idx in (0, 3):
        # Conv layers - wrap in DConvSlot .conv
        new_key = f"{prefix}.layers.{block_idx}.layers.{seq_idx}.conv.{param_rest}"
    elif seq_idx in (1, 4):
        # GroupNorm - direct weight/bias
        new_key = f"{prefix}.layers.{block_idx}.layers.{seq_idx}.{param_rest}"
    elif seq_idx == 6:
        # LayerScale - has .scale parameter
        if param_rest == 'scale':
            new_key = f"{prefix}.layers.{block_idx}.layers.{seq_idx}.{param_rest}"
        else:
            return None
    else:
        return None

    return new_key


def export_model(model_name: str, out_dir: Path) -> bool:
    """Export a model to safetensors + config JSON."""
    from demucs.pretrained import get_model

    print(f"\n--- Exporting {model_name} ---")
    try:
        bag = get_model(model_name)
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return False

    from demucs.apply import BagOfModels

    if not isinstance(bag, BagOfModels):
        print(f"  Expected BagOfModels, got {type(bag).__name__}")
        return False

    num_models = len(bag.models)
    print(f"  Bag of {num_models} models")

    # Collect all weights with model_X prefix
    all_weights = {}
    model_classes = []
    model_kwargs_list = []

    for i, sub_model in enumerate(bag.models):
        cls_name = type(sub_model).__name__
        print(f"  Model {i}: {cls_name}")
        model_classes.append(cls_name)

        # Get state dict
        sd = sub_model.state_dict()
        flat = {}
        for key, tensor in sd.items():
            arr = tensor.detach().cpu().numpy()
            # Transpose conv weights
            if key.endswith('.weight'):
                if len(arr.shape) == 3:
                    arr = np.transpose(arr, (0, 2, 1))
                elif len(arr.shape) == 4:
                    arr = np.transpose(arr, (0, 2, 3, 1))
            flat[f"model_{i}.{key}"] = arr

        all_weights.update(flat)

        # Extract kwargs
        import inspect
        init_sig = inspect.signature(type(sub_model).__init__)
        kwargs = {}
        for param_name in init_sig.parameters:
            if param_name == 'self':
                continue
            if hasattr(sub_model, param_name):
                val = getattr(sub_model, param_name)
                if isinstance(val, torch.Tensor):
                    val = val.item()
                elif isinstance(val, (list, tuple)):
                    val = list(val)
                kwargs[param_name] = val
        model_kwargs_list.append(kwargs)

    # Save safetensors
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    safetensors_path = model_dir / f"{model_name}.safetensors"
    config_path = model_dir / f"{model_name}_config.json"

    # Convert numpy arrays to mlx arrays and save
    try:
        import mlx.core as mx
        mlx_weights = {k: mx.array(v) for k, v in all_weights.items()}
        mx.save_safetensors(str(safetensors_path), mlx_weights)
    except ImportError:
        # Fallback: use safetensors library directly
        from safetensors.numpy import save_file
        save_file(all_weights, str(safetensors_path))

    # Build config
    # Map PyTorch class names to MLX class names
    class_map = {
        'Demucs': 'DemucsMLX',
        'HDemucs': 'HDemucsMLX',
        'HTDemucs': 'HTDemucsMLX',
    }

    # Get weights
    weights = None
    if bag.weights is not None:
        weights = bag.weights.tolist() if hasattr(bag.weights, 'tolist') else list(bag.weights)

    config = {
        "model_name": model_name,
        "model_class": "BagOfModelsMLX",
        "num_models": num_models,
        "weights": weights,
        "sub_model_classes": [class_map.get(c, c) for c in model_classes],
        "model_configs": [],
        "tensor_count": len(all_weights),
    }

    # If all models are the same class, also set sub_model_class for compatibility
    unique_classes = set(config["sub_model_classes"])
    if len(unique_classes) == 1:
        config["sub_model_class"] = unique_classes.pop()

    # Add per-model configs
    for i, (cls, kwargs) in enumerate(zip(model_classes, model_kwargs_list)):
        model_config = {
            "model_class": class_map.get(cls, cls),
            "kwargs": {},
        }
        # Convert kwargs to JSON-serializable
        for k, v in kwargs.items():
            if isinstance(v, (int, float, str, bool, list)):
                model_config["kwargs"][k] = v
            elif v is None:
                model_config["kwargs"][k] = None
        config["model_configs"].append(model_config)

    with config_path.open("w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"  Wrote {safetensors_path} ({len(all_weights)} tensors)")
    print(f"  Wrote {config_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Export mdx/mdx_extra models")
    ap.add_argument("--model", default="mdx", help="Model name")
    ap.add_argument("--out-dir", default=".scratch/models", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    export_model(args.model, out_dir)


if __name__ == "__main__":
    main()
