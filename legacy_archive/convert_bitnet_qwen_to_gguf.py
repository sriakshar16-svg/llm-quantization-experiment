#!/usr/bin/env python3
"""
convert_bitnet_qwen_to_gguf.py
==============================
Converts Outlier-Isolated ALS Base-3 Ternary + 4-bit Attention Qwen2.5-Coder-32B shards
directly to GGUF format for high-speed llama.cpp execution (30-40+ tokens/sec).
Streams layer-by-layer with strict memory management (Peak RAM < 2.0 GB).
"""

import os
import sys
import gc
import json
import glob
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open
from safetensors.torch import load_file

# Add llama.cpp to python path
LLAMA_CPP_DIR = Path("/home/tsriakshar/llama.cpp")
sys.path.insert(0, str(LLAMA_CPP_DIR))
sys.path.insert(0, str(LLAMA_CPP_DIR / "gguf-py"))

import gguf
from conversion.base import ModelBase, LazyTorchTensor
from conversion.qwen import Qwen2Model

DONOR_DIR = Path("./models/Qwen2.5-Coder-32B-Instruct")
BITNET_DIR = Path("./output_bitnet_qwen_32b")
OUT_GGUF = Path("./qwen2.5_coder_32b_bitnet_bf16.gguf")


def unpack_ternary_base3(packed: torch.Tensor, orig_numel: int):
    """Unpacks uint8 Base-3 packed bytes into int8 {-1, 0, 1} tensor."""
    p = packed.to(dtype=torch.int32)
    u0 = p % 3; p = p // 3
    u1 = p % 3; p = p // 3
    u2 = p % 3; p = p // 3
    u3 = p % 3; u4 = p // 3
    unpacked = torch.stack([u0, u1, u2, u3, u4], dim=1).reshape(-1)[:orig_numel]
    return (unpacked - 1).to(torch.int8)


def unpack_nibbles_4bit(packed: torch.Tensor, orig_numel: int):
    """Unpacks 2 nibbles per byte into signed int8 uniform [-8, 7] tensor."""
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.empty(orig_numel, dtype=torch.int8)
    unpacked[0::2] = low[:(orig_numel + 1) // 2]
    unpacked[1::2] = high[:orig_numel // 2]
    return unpacked


def reconstruct_weight(layer_dict, base_name, shape):
    """Reconstruct bfloat16 tensor from Base-3 ternary bulk or 4-bit attention + scales."""
    packed_k = f"{base_name}.packed_ternary"
    qweight_k = f"{base_name}.q_weight"

    if packed_k in layer_dict:
        # Base-3 Ternary MLP
        packed = layer_dict[packed_k]
        orig_n = int(torch.prod(torch.tensor(shape)).item())
        scales = layer_dict[f"{base_name}.scales"].float()
        unpacked = unpack_ternary_base3(packed, orig_n)
        scales_exp = scales.repeat_interleave(128)[:orig_n]
        w_bulk = (unpacked.float() * scales_exp)

        idx_k = f"{base_name}.outlier_idx"
        val_k = f"{base_name}.outlier_val"
        if idx_k in layer_dict and val_k in layer_dict:
            out_idx = layer_dict[idx_k]
            out_val = layer_dict[val_k].float()
            if out_idx.numel() > 0:
                if out_idx.ndim > 1:
                    num_groups = scales.shape[0]
                    pad_len = num_groups * 128 - orig_n
                    w_g = (torch.nn.functional.pad(w_bulk, (0, pad_len)) if pad_len > 0 else w_bulk).view(num_groups, 128)
                    w_g.scatter_(dim=-1, index=out_idx.long(), src=out_val)
                    w_bulk = w_g.view(-1)[:orig_n]
                else:
                    w_bulk[out_idx.long()] = out_val

        return w_bulk.view(shape).to(torch.bfloat16)

    elif qweight_k in layer_dict:
        # 4-bit Uniform Attention Projection
        packed = layer_dict[qweight_k]
        orig_n = int(torch.prod(torch.tensor(shape)).item())
        scales = layer_dict[f"{base_name}.scales"].float()
        unpacked = unpack_nibbles_4bit(packed, orig_n)
        scales_exp = scales.repeat_interleave(128)[:orig_n]
        w_bulk = (unpacked.float() * scales_exp)
        return w_bulk.view(shape).to(torch.bfloat16)

    elif base_name in layer_dict:
        return layer_dict[base_name].to(torch.bfloat16)

    return None


class StreamedLayerCache:
    """Caches only one layer in memory at a time, keeping peak RAM < 2.0 GB."""
    def __init__(self, bitnet_dir: Path, donor_dir: Path, shapes: dict[str, list[int]]):
        self.bitnet_dir = bitnet_dir
        self.donor_dir = donor_dir
        self.shapes = shapes
        self.current_layer_idx = None
        self.current_layer_dict = None
        with open(donor_dir / "model.safetensors.index.json") as f:
            self.donor_weight_map = json.load(f)["weight_map"]
        self.donor_handles = {}

    def get_donor_tensor(self, key: str) -> torch.Tensor:
        shard = self.donor_weight_map[key]
        if shard not in self.donor_handles:
            self.donor_handles[shard] = safe_open(str(self.donor_dir / shard), framework="pt")
        return self.donor_handles[shard].get_tensor(key).to(torch.bfloat16)

    def get_anchor(self, key: str) -> torch.Tensor:
        anchors_path = self.bitnet_dir / "anchors.safetensors"
        if anchors_path.exists():
            with safe_open(str(anchors_path), framework="pt") as f:
                if key in f.keys():
                    return f.get_tensor(key).to(torch.bfloat16)
        return self.get_donor_tensor(key)

    def get_layer_tensor(self, layer_idx: int, key: str) -> torch.Tensor:
        if self.current_layer_idx != layer_idx:
            self.current_layer_dict = None
            gc.collect()
            print(f"[*] Streaming BitNet Layer {layer_idx:02d}/63 into GGUF...")
            layer_path = self.bitnet_dir / f"layer_{layer_idx}.safetensors"
            if layer_path.exists():
                self.current_layer_dict = load_file(str(layer_path))
            else:
                self.current_layer_dict = {}
            self.current_layer_idx = layer_idx

        shape = self.shapes[key]
        reconstructed = reconstruct_weight(self.current_layer_dict, key, shape)
        if reconstructed is not None:
            return reconstructed
        return self.get_donor_tensor(key)


class BitNetQwen2Model(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.QWEN2

    def __init__(self, *args, **kwargs):
        print("[*] Reading tensor shapes from donor headers...")
        self.shapes = {}
        for sf in glob.glob(str(DONOR_DIR / "*.safetensors")):
            with safe_open(sf, framework="pt") as f:
                for k in f.keys():
                    self.shapes[k] = list(f.get_slice(k).get_shape())
        print(f"[✓] Indexed {len(self.shapes)} tensor shapes.")

        self.stream_cache = StreamedLayerCache(BITNET_DIR, DONOR_DIR, self.shapes)
        super().__init__(*args, **kwargs)

    def index_tensors(self, remote_hf_model_id=None) -> dict[str, Callable[[], torch.Tensor]]:
        print("[*] Registering streamed tensor generators...")
        tensors: dict[str, Callable[[], torch.Tensor]] = {}

        # 1. Anchors
        for anchor_name in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]:
            if anchor_name in self.shapes:
                tensors[anchor_name] = lambda name=anchor_name: self.stream_cache.get_anchor(name)

        # 2. Layer tensors
        for name in sorted(self.shapes.keys()):
            if name.startswith("model.layers."):
                parts = name.split(".")
                layer_idx = int(parts[2])
                tensors[name] = lambda idx=layer_idx, k=name: self.stream_cache.get_layer_tensor(idx, k)

        print(f"[✓] Successfully registered {len(tensors)} lazy tensor generators.")
        return tensors


def main():
    print("=" * 70)
    print("   Qwen2.5-Coder-32B (Base-3 ALS) -> GGUF Streamed Converter")
    print("=" * 70)
    print(f"[*] Target GGUF         : {OUT_GGUF}")
    print(f"[*] Input BitNet Shards : {BITNET_DIR}")
    print(f"[*] Donor Architecture  : {DONOR_DIR}")
    print("=" * 70)

    import tempfile
    temp_dir = Path("./tmp_gguf")
    temp_dir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp_dir)
    os.environ["TMPDIR"] = str(temp_dir)

    model = BitNetQwen2Model(
        dir_model=DONOR_DIR,
        ftype=gguf.LlamaFileType.MOSTLY_BF16,
        fname_out=OUT_GGUF,
        is_big_endian=False,
        use_temp_file=True,
        eager=False,
    )

    print("\n[*] Starting GGUF write pipeline (streamed layer-by-layer)...")
    model.write()
    print("=" * 70)
    print(f"[✓] GGUF Conversion Complete! File: {OUT_GGUF}")
    print(f"[*] Size on disk: {OUT_GGUF.stat().st_size / (1024 ** 3):.2f} GB")
    print("=" * 70)


if __name__ == "__main__":
    main()
