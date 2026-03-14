#!/usr/bin/env python3
"""
Export Demucs PyTorch models directly to safetensors + JSON config for Swift MLX.

Converts all 8 pretrained models directly from the original PyTorch demucs package.
No dependency on demucs-mlx or any other re-implementation.

Usage:
    # Export all models
    python scripts/export_from_pytorch.py --out-dir ~/.cache/demucs-mlx-swift-models

    # Export specific models
    python scripts/export_from_pytorch.py --models htdemucs htdemucs_ft --out-dir ./Models

Requirements:
    pip install demucs safetensors numpy
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ALL_MODELS = [
    "htdemucs",
    "htdemucs_ft",
    "htdemucs_6s",
    "hdemucs_mmi",
    "mdx",
    "mdx_extra",
    "mdx_q",
    "mdx_extra_q",
]

# Map PyTorch class names to MLX class names used by Swift loader
CLASS_MAP = {
    "Demucs": "DemucsMLX",
    "HDemucs": "HDemucsMLX",
    "HTDemucs": "HTDemucsMLX",
}

# Conv-like layer names that get .conv. wrapper in MLX
CONV_LAYER_NAMES = {
    "conv", "conv_tr", "rewrite",
    "channel_upsampler", "channel_downsampler",
    "channel_upsampler_t", "channel_downsampler_t",
}

# DConv attention sub-module names (LocalState)
DCONV_ATTN_NAMES = {"content", "key", "query", "proj", "query_decay", "query_freqs"}


def to_json_serializable(obj):
    """Convert Python objects to JSON-serializable types."""
    if isinstance(obj, Fraction):
        return f"{obj.numerator}/{obj.denominator}"
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    return obj


def transpose_conv_weights(key: str, value: np.ndarray, is_conv_transpose: bool = False) -> np.ndarray:
    """Transpose PyTorch conv weights to MLX layout.

    Conv1d:          (out, in, k)    → MLX: (out, k, in)       transpose (0,2,1)
    Conv2d:          (out, in, h, w) → MLX: (out, h, w, in)    transpose (0,2,3,1)
    ConvTranspose1d: (in, out, k)    → MLX: (out, k, in)       transpose (1,2,0)
    ConvTranspose2d: (in, out, h, w) → MLX: (out, h, w, in)    transpose (1,2,3,0)
    """
    if not key.endswith(".weight"):
        return value

    if len(value.shape) == 3:
        return np.transpose(value, (1, 2, 0) if is_conv_transpose else (0, 2, 1))
    if len(value.shape) == 4:
        return np.transpose(value, (1, 2, 3, 0) if is_conv_transpose else (0, 2, 3, 1))
    return value


def remap_key(
    key: str,
    value: np.ndarray,
    model_type: str = "HTDemucs",
    dconv_conv_slots: set | None = None,
    seq_conv_slots: set | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Remap a PyTorch state dict key to MLX key convention.

    Returns a list of (key, value) pairs (multiple for attention in_proj splits).
    Duplicate target keys (e.g. LSTM bias_ih + bias_hh) are merged by the caller.

    Args:
        key: PyTorch state dict key
        value: numpy array (already transposed for conv weights)
        model_type: PyTorch class name ("Demucs", "HDemucs", "HTDemucs")
        dconv_conv_slots: set of (block_prefix, slot_str) for DConv slots with 3D weights
        seq_conv_slots: set of (enc_dec, layer, slot) for Demucs v1/v2 Sequential Conv slots
    """
    dconv_conv_slots = dconv_conv_slots or set()
    seq_conv_slots = seq_conv_slots or set()

    # =========================================================================
    # Step 1: Demucs v1/v2 Sequential insertion
    # encoder.{i}.{j}.rest → encoder.{i}.layers.{j}.rest
    # decoder.{i}.{j}.rest → decoder.{i}.layers.{j}.rest
    # =========================================================================
    if model_type == "Demucs":
        m = re.match(r"(encoder|decoder)\.(\d+)\.(\d+)(\..*)?$", key)
        if m:
            enc_dec, layer, slot, rest = m.groups()
            rest = rest or ""
            key = f"{enc_dec}.{layer}.layers.{slot}{rest}"

    # =========================================================================
    # Step 1.5: Demucs v1/v2 Sequential Conv/Norm slot wrapping
    # encoder.{i}.layers.{j}.weight → encoder.{i}.layers.{j}.conv.weight (if Conv slot)
    # =========================================================================
    if model_type == "Demucs":
        m = re.match(r"(encoder|decoder)\.(\d+)\.layers\.(\d+)\.(weight|bias)$", key)
        if m:
            enc_dec, layer, slot, param = m.groups()
            if (enc_dec, layer, slot) in seq_conv_slots:
                return [(f"{enc_dec}.{layer}.layers.{slot}.conv.{param}", value)]
            else:
                return [(f"{enc_dec}.{layer}.layers.{slot}.{param}", value)]

    # =========================================================================
    # Step 2: DConv internal slot handling
    # Matches: *.layers.{block_idx}.{slot_idx}.{rest}
    # Both HDemucs (.dconv.layers.) and Demucs v1/v2 (.layers.{N}.layers.) end
    # with this pattern after Step 1.
    # =========================================================================
    m = re.match(r"(.+\.layers\.\d+)\.(\d+)\.(.+)$", key)
    if m:
        block_prefix = m.group(1)
        slot = m.group(2)
        rest = m.group(3)

        # --- 2a. Simple weight/bias/scale ---
        if rest in ("weight", "bias", "scale"):
            if rest == "weight" and len(value.shape) >= 2:
                # 3D weight = Conv1d → add .conv.
                return [(f"{block_prefix}.layers.{slot}.conv.{rest}", value)]
            elif rest == "weight":
                # 1D weight = GroupNorm → no wrapper
                return [(f"{block_prefix}.layers.{slot}.{rest}", value)]
            elif rest == "bias":
                if (block_prefix, slot) in dconv_conv_slots:
                    return [(f"{block_prefix}.layers.{slot}.conv.{rest}", value)]
                else:
                    return [(f"{block_prefix}.layers.{slot}.{rest}", value)]
            else:  # scale
                return [(f"{block_prefix}.layers.{slot}.{rest}", value)]

        # --- 2b. LSTM weights/biases ---
        m_lstm = re.match(r"lstm\.(weight|bias)_(ih|hh)_l(\d+)(_reverse)?$", rest)
        if m_lstm:
            wb, ih_hh, layer_idx, reverse = m_lstm.groups()
            direction = "backward_lstms" if reverse else "forward_lstms"
            if wb == "weight":
                param = "Wx" if ih_hh == "ih" else "Wh"
                return [(f"{block_prefix}.layers.{slot}.{direction}.{layer_idx}.{param}", value)]
            else:  # bias — both bias_ih and bias_hh map to same key; caller merges
                return [(f"{block_prefix}.layers.{slot}.{direction}.{layer_idx}.bias", value)]

        # --- 2c. LSTM linear ---
        m_linear = re.match(r"linear\.(weight|bias)$", rest)
        if m_linear:
            param = m_linear.group(1)
            return [(f"{block_prefix}.layers.{slot}.linear.{param}", value)]

        # --- 2d. Attention sub-modules (LocalState) ---
        m_attn = re.match(r"(content|key|query|proj|query_decay|query_freqs)\.(weight|bias)$", rest)
        if m_attn:
            attn_name, param = m_attn.groups()
            # These are all Conv1d modules → add .conv. wrapper
            return [(f"{block_prefix}.layers.{slot}.{attn_name}.conv.{param}", value)]

        # --- 2e. Fallback for unknown compound keys ---
        return [(f"{block_prefix}.layers.{slot}.{rest}", value)]

    # =========================================================================
    # Step 3: MultiheadAttention in_proj split (HTDemucs transformer)
    # =========================================================================
    m = re.match(r"(.+)\.(self_attn|cross_attn)\.in_proj_(weight|bias)$", key)
    if m:
        prefix, attn_type, param = m.group(1), m.group(2), m.group(3)
        mlx_attn = "attn" if attn_type == "self_attn" else "cross_attn"
        dim = value.shape[0] // 3
        q, k_val, v = value[:dim], value[dim : 2 * dim], value[2 * dim :]
        return [
            (f"{prefix}.{mlx_attn}.query_proj.{param}", q),
            (f"{prefix}.{mlx_attn}.key_proj.{param}", k_val),
            (f"{prefix}.{mlx_attn}.value_proj.{param}", v),
        ]

    # self_attn.out_proj → attn.out_proj
    m = re.match(r"(.+)\.self_attn\.out_proj\.(weight|bias)$", key)
    if m:
        prefix, param = m.group(1), m.group(2)
        return [(f"{prefix}.attn.out_proj.{param}", value)]

    # =========================================================================
    # Step 4: norm_out wrapping → norm_out.gn
    # =========================================================================
    m = re.match(r"(.+)\.norm_out\.(weight|bias)$", key)
    if m:
        prefix, param = m.group(1), m.group(2)
        return [(f"{prefix}.norm_out.gn.{param}", value)]

    # =========================================================================
    # Step 5: Bottleneck LSTM (Demucs v1/v2 and HDemucs)
    # lstm.lstm.weight_ih_l0 → lstm.forward_lstms.0.Wx
    # =========================================================================
    m = re.match(r"(.+)\.lstm\.(weight|bias)_(ih|hh)_l(\d+)(_reverse)?$", key)
    if m:
        prefix = m.group(1)
        wb = m.group(2)
        ih_hh = m.group(3)
        layer_idx = m.group(4)
        reverse = m.group(5)
        direction = "backward_lstms" if reverse else "forward_lstms"
        if wb == "weight":
            param = "Wx" if ih_hh == "ih" else "Wh"
            return [(f"{prefix}.{direction}.{layer_idx}.{param}", value)]
        else:  # bias — merge handled by caller
            return [(f"{prefix}.{direction}.{layer_idx}.bias", value)]

    # =========================================================================
    # Step 6: Conv/ConvTranspose/Rewrite named layers → add .conv. wrapper
    # =========================================================================
    parts = key.rsplit(".", 1)
    if len(parts) == 2:
        path, param = parts
        path_parts = path.split(".")
        last_name = path_parts[-1]
        if last_name in CONV_LAYER_NAMES and param in ("weight", "bias"):
            return [(f"{path}.conv.{param}", value)]

    # =========================================================================
    # Default: no change
    # =========================================================================
    return [(key, value)]


def convert_sub_model(model, prefix: str) -> dict[str, np.ndarray]:
    """Convert a single sub-model's state dict to MLX-compatible numpy arrays."""
    cls_name = type(model).__name__

    # --- Pre-scan: identify ConvTranspose modules by type ---
    conv_tr_paths = set()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)):
            conv_tr_paths.add(name)

    # --- Collect state dict as numpy ---
    state_items = []
    for key, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().float().numpy()
        state_items.append((key, arr))

    # --- Pre-scan: identify DConv Conv slots (3D weights) ---
    # Pattern: *.layers.{block}.{slot}.weight where value is 3D
    # For Demucs v1/v2, apply Sequential insertion first so lookups match remap_key
    dconv_conv_slots: set[tuple[str, str]] = set()
    for key, arr in state_items:
        scan_key = key
        if cls_name == "Demucs":
            m = re.match(r"(encoder|decoder)\.(\d+)\.(\d+)(\..*)?$", scan_key)
            if m:
                enc_dec, layer, slot, rest = m.groups()
                rest = rest or ""
                scan_key = f"{enc_dec}.{layer}.layers.{slot}{rest}"
        m = re.match(r"(.+\.layers\.\d+)\.(\d+)\.weight$", scan_key)
        if m and len(arr.shape) >= 2:
            dconv_conv_slots.add((m.group(1), m.group(2)))

    # --- Pre-scan: Demucs v1/v2 Sequential Conv slots ---
    seq_conv_slots: set[tuple[str, str, str]] = set()
    if cls_name == "Demucs":
        for key, arr in state_items:
            m = re.match(r"(encoder|decoder)\.(\d+)\.(\d+)\.weight$", key)
            if m and len(arr.shape) >= 2:
                seq_conv_slots.add((m.group(1), m.group(2), m.group(3)))

    # --- Convert ---
    weights: dict[str, np.ndarray] = {}
    for key, arr in state_items:
        # Determine if this belongs to a ConvTranspose module
        is_conv_tr = any(key.startswith(p + ".") for p in conv_tr_paths)

        # Transpose conv weights
        arr = transpose_conv_weights(key, arr, is_conv_transpose=is_conv_tr)

        # Remap key
        remapped = remap_key(key, arr, cls_name, dconv_conv_slots, seq_conv_slots)
        for new_key, new_val in remapped:
            full_key = f"{prefix}{new_key}"
            if full_key in weights:
                # LSTM bias merge: bias_ih + bias_hh → bias (additive)
                weights[full_key] = weights[full_key] + new_val
            else:
                weights[full_key] = new_val

    return weights


def extract_kwargs(model) -> dict:
    """Extract constructor kwargs from a model using _init_args_kwargs or inspection."""
    if hasattr(model, "_init_args_kwargs"):
        _, kwargs = model._init_args_kwargs
        return {k: to_json_serializable(v) for k, v in kwargs.items()
                if isinstance(v, (int, float, str, bool, list, tuple, type(None), Fraction))}

    # Fallback: inspect __init__ signature and read matching attributes
    sig = inspect.signature(type(model).__init__)
    kwargs = {}
    for name in sig.parameters:
        if name == "self":
            continue
        if hasattr(model, name):
            val = getattr(model, name)
            kwargs[name] = to_json_serializable(val)
    return kwargs


def export_model(model_name: str, out_dir: Path) -> bool:
    """Export a single model (or bag) to safetensors + config JSON."""
    from demucs.pretrained import get_model
    from demucs.apply import BagOfModels

    print(f"\n--- Exporting {model_name} ---")
    try:
        model = get_model(model_name)
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return False

    is_bag = isinstance(model, BagOfModels)

    if is_bag:
        sub_models = list(model.models)
        num_models = len(sub_models)
        bag_weights = model.weights.tolist() if hasattr(model.weights, "tolist") else list(model.weights)
    else:
        sub_models = [model]
        num_models = 1
        bag_weights = None

    print(f"  {'Bag of ' + str(num_models) + ' models' if is_bag else 'Single model'}")

    # Collect all weights and metadata
    all_weights: dict[str, np.ndarray] = {}
    model_classes: list[str] = []
    model_configs: list[dict] = []

    for i, sub in enumerate(sub_models):
        cls_name = type(sub).__name__
        mlx_cls = CLASS_MAP.get(cls_name, cls_name)
        model_classes.append(mlx_cls)
        print(f"  Model {i}: {cls_name} → {mlx_cls}")

        prefix = f"model_{i}." if is_bag else ""
        sub_weights = convert_sub_model(sub, prefix)
        all_weights.update(sub_weights)

        kwargs = extract_kwargs(sub)
        model_configs.append({
            "model_class": mlx_cls,
            "kwargs": kwargs,
        })

    # Build config JSON
    config: dict = {
        "model_name": model_name,
        "tensor_count": len(all_weights),
    }

    if is_bag:
        config["model_class"] = "BagOfModelsMLX"
        config["num_models"] = num_models
        config["weights"] = bag_weights
        config["sub_model_classes"] = model_classes

        # If all sub-models are the same class, set sub_model_class for compat
        unique = set(model_classes)
        if len(unique) == 1:
            config["sub_model_class"] = unique.pop()

        config["model_configs"] = model_configs

        # Also put kwargs at top level for single-model bags (common case)
        if num_models == 1:
            config["kwargs"] = model_configs[0]["kwargs"]
    else:
        config["model_class"] = model_classes[0]
        config["kwargs"] = model_configs[0]["kwargs"]

    # Save files
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    safetensors_path = model_dir / f"{model_name}.safetensors"
    config_path = model_dir / f"{model_name}_config.json"

    # Save safetensors (prefer safetensors library, fallback to mlx)
    try:
        from safetensors.numpy import save_file
        save_file(all_weights, str(safetensors_path))
    except ImportError:
        import mlx.core as mx
        mlx_weights = {k: mx.array(v) for k, v in all_weights.items()}
        mx.save_safetensors(str(safetensors_path), mlx_weights)

    with config_path.open("w") as f:
        json.dump(config, f, indent=2, default=str)

    size_mb = safetensors_path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {safetensors_path} ({len(all_weights)} tensors, {size_mb:.0f} MB)")
    print(f"  Wrote {config_path}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Export Demucs PyTorch models to safetensors for Swift MLX"
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=f"Models to export (default: all). Choices: {', '.join(ALL_MODELS)}",
    )
    ap.add_argument(
        "--out-dir",
        default="./Models",
        help="Output root directory (files go into <out-dir>/<model_name>/)",
    )
    args = ap.parse_args()

    models = args.models or ALL_MODELS
    out_dir = Path(args.out_dir).resolve()

    exported = 0
    failed = 0

    for name in models:
        if export_model(name, out_dir):
            exported += 1
        else:
            failed += 1

    print(f"\n=== Done: {exported} exported, {failed} failed ===")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
