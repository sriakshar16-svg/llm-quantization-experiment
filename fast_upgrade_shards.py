#!/usr/bin/env python3
"""
fast_upgrade_shards.py
=======================
Upgrades existing layer_X.safetensors shards in-place in under 2 minutes:
  1. Computes outlier-isolated 3-step ALS scales (gamma*) from donor weights.
  2. Converts outlier indices to structured per-group uint8 offsets (saving ~650 MB VRAM).
  3. Preserves existing 4-bit attention projections and FP16 norms/routers.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from convert_pipeline import quantize_ternary_least_squares, build_weight_index

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def upgrade_shards(donor_dir: str, bitnet_dir: str, start_layer: int = 0, end_layer: int = 27):
    donor_path = Path(donor_dir)
    bitnet_path = Path(bitnet_dir)

    print("=" * 70)
    print("      FAST IN-PLACE SHARD UPGRADER (ALS SCALES + UINT8 OUTLIERS)       ")
    print("=" * 70)
    print(f"[*] Donor Directory  : {donor_path}")
    print(f"[*] Shards Directory : {bitnet_path}")
    print(f"[*] Device           : {DEVICE}")

    weight_map = build_weight_index(str(donor_path))
    donor_handles = {}

    def get_donor_tensor(key: str) -> torch.Tensor:
        shard = weight_map[key]
        if shard not in donor_handles:
            donor_handles[shard] = safe_open(str(donor_path / shard), framework="pt", device="cpu")
        return donor_handles[shard].get_tensor(key)

    total_t0 = time.time()
    upgraded_tensors = 0

    for layer_idx in range(start_layer, end_layer):
        shard_file = bitnet_path / f"layer_{layer_idx}.safetensors"
        if not shard_file.exists():
            print(f"[!] Warning: {shard_file} not found, skipping...")
            continue

        l_t0 = time.time()
        layer_dict = load_file(str(shard_file))
        new_layer_dict = {}

        # Identify all base keys that need ternary repacking
        ternary_keys = [k[:-len(".packed_ternary")] for k in layer_dict.keys() if k.endswith(".packed_ternary")]

        # Copy non-ternary keys first (norms, gates, 4bit attention)
        for k, v in layer_dict.items():
            base_k = None
            for tk in ternary_keys:
                if k.startswith(tk):
                    base_k = tk
                    break
            if base_k is None:
                new_layer_dict[k] = v

        # Repack ternary keys using donor weights with outlier-isolated ALS
        for full_key in ternary_keys:
            donor_w = get_donor_tensor(full_key).to(DEVICE)
            _, q_data = quantize_ternary_least_squares(
                donor_w,
                group_size=128,
                outlier_ratio=0.015,
                als_steps=3,
                pack_mode="base3",
                use_group_uint8_outliers=True,
            )
            del donor_w

            new_layer_dict[f"{full_key}.packed_ternary"] = q_data["packed_ternary"]
            new_layer_dict[f"{full_key}.orig_numel"] = q_data["orig_numel"]
            new_layer_dict[f"{full_key}.scales"] = q_data["scales"]
            new_layer_dict[f"{full_key}.outlier_idx"] = q_data["outlier_idx"]
            new_layer_dict[f"{full_key}.outlier_val"] = q_data["outlier_val"]
            upgraded_tensors += 1

        # Atomically overwrite shard
        tmp_shard = bitnet_path / f"layer_{layer_idx}.tmp.safetensors"
        save_file(new_layer_dict, str(tmp_shard))
        os.replace(str(tmp_shard), str(shard_file))

        del layer_dict, new_layer_dict
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - l_t0
        print(f"  [✓] Layer {layer_idx:02d}/{end_layer - 1:02d} upgraded ({len(ternary_keys)} matrices) in {elapsed:.2f}s")

    donor_handles.clear()
    total_elapsed = time.time() - total_t0
    print("-" * 70)
    print(f"[✓] Upgrade Complete: {upgraded_tensors} matrices upgraded in {total_elapsed:.1f}s!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast In-Place Shard Upgrader")
    parser.add_argument("--donor_dir", type=str, default="./models/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--bitnet_dir", type=str, default="./output_bitnet_deepseek_base3")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=27)
    args = parser.parse_args()

    upgrade_shards(args.donor_dir, args.bitnet_dir, args.start, args.end)
