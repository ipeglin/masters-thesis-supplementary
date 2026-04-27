#!/usr/bin/env python3
"""
Convert Keras DenseNet201 weights (.h5) to PyTorch format (.pt) for tch-rs.

Keras stores conv kernels as (H, W, C_in, C_out) — transposed to PyTorch's
(C_out, C_in, H, W). BN params are renamed: gamma→weight, beta→bias,
moving_mean→running_mean, moving_variance→running_var.

Output path is read by tch VarStore::load in crates/06feature_extraction.

Usage:
    pip install h5py numpy torch
    python cnn_model_weights/convert_weights.py
"""

import os
import zipfile

import h5py
import numpy as np
import torch
from safetensors.torch import save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
ZIP_PATH = os.path.join(
    RAW_DIR, "densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5.zip"
)
H5_PATH = os.path.join(
    RAW_DIR, "densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5"
)
# rename OUT_PATH to .safetensors
OUT_PATH = os.path.join(PROCESSED_DIR, "densenet201_imagenet.safetensors")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def unzip_if_needed():
    if not os.path.exists(H5_PATH):
        print(f"Unzipping {ZIP_PATH} ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(RAW_DIR)


def load_keras_weights(h5_path: str) -> dict:
    """
    Returns {layer_name: {short_param_name: np.ndarray}}.
    e.g. {"conv1/conv1": {"kernel:0": <array (7,7,3,64)>}}
    """
    out = {}
    with h5py.File(h5_path, "r") as f:
        # Keras application weights may sit at root or under 'model_weights'
        root = f.get("model_weights", f)
        layer_names = [n.decode("utf-8") for n in root.attrs["layer_names"]]

        for layer_name in layer_names:
            g = root[layer_name]
            raw_names = g.attrs.get("weight_names", [])
            if len(raw_names) == 0:
                continue
            params = {}
            for raw in raw_names:
                w_name = raw.decode("utf-8")
                short = w_name.split("/")[-1]  # e.g. "kernel:0"
                params[short] = g[w_name][:]
            out[layer_name] = params
    return out


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------


def build_state_dict(kw: dict) -> dict:
    """Map Keras weight dict to tch VarStore paths."""
    state = {}

    def conv(tch_path: str, layer_name: str):
        k = kw[layer_name]["kernel:0"]
        # (H, W, C_in, C_out) → (C_out, C_in, H, W)
        state[f"{tch_path}.weight"] = torch.from_numpy(
            np.transpose(k, (3, 2, 0, 1)).copy()
        )

    def bn(tch_path: str, layer_name: str):
        w = kw[layer_name]
        state[f"{tch_path}.weight"] = torch.from_numpy(w["gamma:0"].copy())
        state[f"{tch_path}.bias"] = torch.from_numpy(w["beta:0"].copy())
        state[f"{tch_path}.running_mean"] = torch.from_numpy(w["moving_mean:0"].copy())
        state[f"{tch_path}.running_var"] = torch.from_numpy(
            w["moving_variance:0"].copy()
        )

    # ── Initial stem ──────────────────────────────────────────────────────────
    conv("features.conv0", "conv1/conv")
    bn("features.norm0", "conv1/bn")

    # ── Dense blocks ─────────────────────────────────────────────────────────
    # DenseNet-201 block depths: [6, 12, 48, 32]
    block_cfg = [
        ("conv2", "denseblock1", 6),
        ("conv3", "denseblock2", 12),
        ("conv4", "denseblock3", 48),
        ("conv5", "denseblock4", 32),
    ]
    for keras_prefix, tch_block, n_layers in block_cfg:
        for i in range(1, n_layers + 1):
            kb = f"{keras_prefix}_block{i}"
            tb = f"features.{tch_block}.denselayer{i}"
            bn(f"{tb}.norm1", f"{kb}_0_bn")
            conv(f"{tb}.conv1", f"{kb}_1_conv")
            bn(f"{tb}.norm2", f"{kb}_1_bn")
            conv(f"{tb}.conv2", f"{kb}_2_conv")

    # ── Transition layers ─────────────────────────────────────────────────────
    for idx, pool in enumerate(["pool2", "pool3", "pool4"], start=1):
        tb = f"features.transition{idx}"
        bn(f"{tb}.norm", f"{pool}_bn")
        conv(f"{tb}.conv", f"{pool}_conv")

    # ── Final BN (norm5) ──────────────────────────────────────────────────────
    bn("features.norm5", "bn")

    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    unzip_if_needed()

    print("Loading Keras weights ...")
    kw = load_keras_weights(H5_PATH)
    print(f"  {len(kw)} layers found")

    print("Converting ...")
    state = build_state_dict(kw)
    n_params = sum(t.numel() for t in state.values())
    print(f"  {len(state)} tensors, {n_params:,} parameters")

    print(f"Saving → {OUT_PATH}")
    save_file(state, OUT_PATH)
    print("Done.")
    os._exit(0)


if __name__ == "__main__":
    main()
