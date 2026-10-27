#!/usr/bin/env python3
"""
Convert Zero-Gap Base-3 BitNet DeepSeek Model to GGUF format for llama.cpp.
Streamed layer-by-layer with strict memory management (Peak RAM < 2GB).
Preserves MLA attention anchors and MoE router gates in donor precision.
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
from conversion.deepseek import DeepseekV2Model

DONOR_DIR = Path("/home/tsriakshar/Desktop/CODEBASES/llm-quantization-experiment/models/DeepSeek-Coder-V2-Lite-Instruct")
BITNET_DIR = Path("/home/tsriakshar/Desktop/CODEBASES/llm-quantization-experiment/output_bitnet_deepseek_base3")
OUT_GGUF = Path("/home/tsriakshar/Desktop/CODEBASES/llm-quantization-experiment/deepseek_v2_lite_bitnet_bf16.gguf")


def unpack_ternary_base3(packed: torch.Tensor, orig_numel: int):
    """Unpacks uint8 Base-3 packed bytes into int8 {-1, 0, 1} tensor."""
    p = packed.to(dtype=torch.int32)
    u0 = p % 3; p = p // 3
    u1 = p % 3; p = p // 3
    u2 = p % 3; p = p // 3
    u3 = p % 3; u4 = p // 3
    unpacked = torch.stack([u0, u1, u2, u3, u4], dim=1).reshape(-1)[:orig_numel]
    return (unpacked - 1).to(torch.int8)


def reconstruct_weight(layer_dict, base_name, shape):
    """Reconstruct bfloat16 tensor from Base-3 ternary bulk + optimal scales + FP16 outliers."""
    packed_k = f"{base_name}.packed_ternary"
    if packed_k in layer_dict:
        packed = layer_dict[packed_k]
        orig_n = layer_dict[f"{base_name}.orig_numel"].item()
        scales = layer_dict[f"{base_name}.scales"].float()
        unpacked = unpack_ternary_base3(packed, orig_n)
        scales_exp = scales.repeat_interleave(128)[:orig_n]
        w_bulk = (unpacked.float() * scales_exp)

        idx_k = f"{base_name}.outlier_idx"
        val_k = f"{base_name}.outlier_val"
        if idx_k in layer_dict and val_k in layer_dict:
            out_idx = layer_dict[idx_k].long()
            out_val = layer_dict[val_k].float()
            if out_idx.numel() > 0:
                w_bulk[out_idx] = out_val

        return w_bulk.view(shape).to(torch.bfloat16)
    elif base_name in layer_dict:
        return layer_dict[base_name].to(torch.bfloat16)
    return None


class StreamedLayerCache:
    """Caches only one layer in memory at a time, keeping attention anchors and router gates in donor precision."""
    def __init__(self, bitnet_dir: Path, donor_dir: Path, shapes: dict[str, list[int]]):
        self.bitnet_dir = bitnet_dir
        self.donor_dir = donor_dir
        self.shapes = shapes
        self.current_layer_idx = None
        self.current_layer_dict = None
        self.anchors = None
        with open(donor_dir / "model.safetensors.index.json") as f:
            self.donor_weight_map = json.load(f)["weight_map"]
        self.donor_handles = {}

    def get_donor_tensor(self, key: str) -> torch.Tensor:
        shard = self.donor_weight_map[key]
        if shard not in self.donor_handles:
            self.donor_handles[shard] = safe_open(str(self.donor_dir / shard), framework="pt")
        return self.donor_handles[shard].get_tensor(key).to(torch.bfloat16)

    def get_anchor(self, key: str) -> torch.Tensor:
        if self.anchors is None:
            self.anchors = load_file(str(self.bitnet_dir / "anchors.safetensors"))
        return self.anchors[key].to(torch.bfloat16)

    def get_layer_tensor(self, layer_idx: int, key: str) -> torch.Tensor:
        # Hybrid BitNet Standard: Attention projections and router gates preserved in donor precision
        if "self_attn" in key or "mlp.gate.weight" in key:
            return self.get_donor_tensor(key)

        if self.current_layer_idx != layer_idx:
            # Drop old layer
            self.current_layer_dict = None
            gc.collect()
            print(f"[*] Streaming BitNet Layer {layer_idx}/26 into memory...")
            layer_path = self.bitnet_dir / f"layer_{layer_idx}.safetensors"
            self.current_layer_dict = load_file(str(layer_path))
            self.current_layer_idx = layer_idx

        shape = self.shapes[key]
        return reconstruct_weight(self.current_layer_dict, key, shape)


class BitNetDeepseekV2Model(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK2

    def get_vocab_base_pre(self, tokenizer):
        return "deepseek-coder"

    def __init__(self, *args, **kwargs):
        # Pre-extract tensor shapes from donor headers (fast, 0MB RAM)
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
    print("=" * 65)
    print("Zero-Gap Base-3 BitNet -> GGUF Streamed Converter (Peak RAM < 2GB)")
    print("=" * 65)

    print(f"[*] Target GGUF: {OUT_GGUF}")
    print(f"[*] Input BitNet Shards: {BITNET_DIR}")

    import tempfile
    temp_dir = Path("/home/tsriakshar/Desktop/CODEBASES/llm-quantization-experiment/tmp_gguf")
    temp_dir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp_dir)
    os.environ["TMPDIR"] = str(temp_dir)

    model = BitNetDeepseekV2Model(
        dir_model=DONOR_DIR,
        ftype=gguf.LlamaFileType.MOSTLY_BF16,
        fname_out=OUT_GGUF,
        is_big_endian=False,
        use_temp_file=True,
        eager=False,
    )

    print("[*] Starting GGUF write pipeline...")
    model.write()
    print("=" * 65)
    print(f"[✓] GGUF Conversion Complete! File: {OUT_GGUF}")
    print("=" * 65)


if __name__ == "__main__":
    main()
