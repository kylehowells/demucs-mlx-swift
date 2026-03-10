#!/usr/bin/env python3
"""
Export all available demucs-mlx model checkpoints to safetensors + JSON.

Usage:
    python scripts/export_all_models.py [--cache-dir ~/.cache/demucs-mlx] [--out-dir ./Models]

This script finds all *_mlx.pkl checkpoints in the demucs-mlx cache directory
and exports each one as:
    <out-dir>/<model_name>/<model_name>.safetensors
    <out-dir>/<model_name>/<model_name>_config.json

If you haven't converted models yet, run demucs-mlx first to generate the
pickle checkpoints:
    python -m demucs_mlx --model htdemucs -n test.mp3
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any
from fractions import Fraction

import mlx.core as mx


# Known model names in demucs-mlx
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


def flatten_tree(node: Any, prefix: str = "") -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_tree(v, key))
        return out
    if isinstance(node, (list, tuple)):
        for idx, v in enumerate(node):
            key = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(flatten_tree(v, key))
        return out
    if isinstance(node, mx.array):
        out[prefix] = node
        return out
    return out


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, Fraction):
        return f"{obj.numerator}/{obj.denominator}"
    return obj


def export_checkpoint(ck_path: Path, out_dir: Path, model_name: str) -> bool:
    """Export a single checkpoint. Returns True on success."""
    if not ck_path.exists():
        return False

    print(f"\n--- Exporting {model_name} from {ck_path} ---")

    with ck_path.open("rb") as f:
        checkpoint = pickle.load(f)

    if "state" not in checkpoint:
        print(f"  WARNING: No 'state' key in checkpoint, skipping")
        return False

    flat = flatten_tree(checkpoint["state"])
    if not flat:
        print(f"  WARNING: No MLX arrays found, skipping")
        return False

    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    safetensors_path = model_dir / f"{model_name}.safetensors"
    config_path = model_dir / f"{model_name}_config.json"

    mx.save_safetensors(str(safetensors_path), flat)

    metadata = {
        "model_name": checkpoint.get("model_name", model_name),
        "model_class": checkpoint.get("model_class"),
        "sub_model_class": checkpoint.get("sub_model_class"),
        "num_models": checkpoint.get("num_models"),
        "weights": checkpoint.get("weights"),
        "args": to_builtin(checkpoint.get("args", [])),
        "kwargs": to_builtin(checkpoint.get("kwargs", {})),
        "mlx_version": checkpoint.get("mlx_version"),
        "tensor_count": len(flat),
    }

    # For heterogeneous bags, include per-model class and kwargs
    per_model_class = checkpoint.get("per_model_class")
    per_model_kwargs = checkpoint.get("per_model_kwargs")

    if per_model_class:
        # Map PyTorch class names to MLX class names
        class_map = {
            'Demucs': 'DemucsMLX',
            'HDemucs': 'HDemucsMLX',
            'HTDemucs': 'HTDemucsMLX',
        }
        metadata["sub_model_classes"] = [class_map.get(c, c) for c in per_model_class]

    if per_model_kwargs:
        # Build model_configs array with per-model class + kwargs
        model_configs = []
        for i, kw in enumerate(per_model_kwargs):
            mc = "HTDemucsMLX"
            if per_model_class and i < len(per_model_class):
                mc = class_map.get(per_model_class[i], per_model_class[i])
            model_configs.append({
                "model_class": mc,
                "kwargs": to_builtin(kw),
            })
        metadata["model_configs"] = model_configs

    # Remove None values for cleaner JSON
    metadata = {k: v for k, v in metadata.items() if v is not None}

    with config_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  wrote {safetensors_path} ({len(flat)} tensors)")
    print(f"  wrote {config_path}")
    mc = metadata.get("model_class", "?")
    smc = metadata.get("sub_model_class", "")
    nm = metadata.get("num_models", 1)
    print(f"  class={mc}, sub_class={smc}, num_models={nm}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Export all demucs-mlx checkpoints to safetensors")
    ap.add_argument(
        "--cache-dir",
        default=os.path.expanduser("~/.cache/demucs-mlx"),
        help="demucs-mlx cache directory containing *_mlx.pkl files",
    )
    ap.add_argument(
        "--out-dir",
        default="./Models",
        help="Output root directory (model files go into <out-dir>/<model_name>/)",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Specific model names to export (default: all found)",
    )
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).resolve()

    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        print("Run demucs-mlx first to download and convert models.")
        sys.exit(1)

    models_to_export = args.models or ALL_MODELS

    exported = 0
    skipped = 0

    for model_name in models_to_export:
        ck_path = cache_dir / f"{model_name}_mlx.pkl"
        if export_checkpoint(ck_path, out_dir, model_name):
            exported += 1
        else:
            skipped += 1

    # Also check for any *_mlx.pkl files not in our known list
    if args.models is None:
        for pkl_file in sorted(cache_dir.glob("*_mlx.pkl")):
            name = pkl_file.stem.replace("_mlx", "")
            if name not in ALL_MODELS:
                print(f"\nFound additional checkpoint: {pkl_file.name}")
                if export_checkpoint(pkl_file, out_dir, name):
                    exported += 1

    print(f"\n=== Done: {exported} exported, {skipped} skipped ===")


if __name__ == "__main__":
    main()
