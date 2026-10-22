#!/usr/bin/env python3
"""
export_bitnet_to_gguf.py
========================
Universal BitNet -> GGUF Streamed Exporter & Deployer.
Standardizes GGUF as the primary, default output format across the quantization pipeline.
Compatible with LM Studio, Ollama, llama.cpp, Jan, and text-generation-webui.

Hardware Target: RTX 3070 Ti (8GB VRAM) + 16GB System RAM (Peak RAM < 2.5 GB).
"""

import os
import sys
import gc
import json
import glob
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file

# Register llama.cpp and gguf-py search paths
LLAMA_CPP_CANDIDATES = [
    Path("/home/tsriakshar/llama.cpp"),
    Path("/home/tsriakshar/.gemini/antigravity/scratch/BitNet/3rdparty/llama.cpp"),
]

for p in LLAMA_CPP_CANDIDATES:
    if p.exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        if str(p / "gguf-py") not in sys.path:
            sys.path.insert(0, str(p / "gguf-py"))

import gguf
from conversion import get_model_class, ModelBase


def unpack_ternary_base3(packed: torch.Tensor, orig_numel: int) -> torch.Tensor:
    """Unpacks uint8 Base-3 packed bytes into signed int8 {-1, 0, 1} tensor."""
    p = packed.to(dtype=torch.int32)
    u0 = p % 3; p = p // 3
    u1 = p % 3; p = p // 3
    u2 = p % 3; p = p // 3
    u3 = p % 3; u4 = p // 3
    unpacked = torch.stack([u0, u1, u2, u3, u4], dim=1).reshape(-1)[:orig_numel]
    return (unpacked - 1).to(torch.int8)


def unpack_nibbles_4bit(packed: torch.Tensor, orig_numel: int) -> torch.Tensor:
    """Unpacks 2 nibbles per byte into signed int8 uniform [-8, 7] tensor."""
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.empty(orig_numel, dtype=torch.int8)
    unpacked[0::2] = low[:(orig_numel + 1) // 2]
    unpacked[1::2] = high[:orig_numel // 2]
    return unpacked


def reconstruct_weight(layer_dict: dict, base_name: str, shape: list[int]) -> Optional[torch.Tensor]:
    """Reconstructs bfloat16 tensor from Base-3 ternary bulk or 4-bit attention + scales."""
    packed_k = f"{base_name}.packed_ternary"
    qweight_k = f"{base_name}.q_weight"

    if packed_k in layer_dict:
        # Base-3 Ternary MLP / MoE bulk
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
    """Caches exactly one layer shard in memory at a time to maintain peak RAM < 2.0 GB."""
    def __init__(self, bitnet_dir: Path, donor_dir: Path, shapes: dict[str, list[int]]):
        self.bitnet_dir = Path(bitnet_dir)
        self.donor_dir = Path(donor_dir)
        self.shapes = shapes
        self.current_layer_idx = None
        self.current_layer_dict = None

        with open(self.donor_dir / "model.safetensors.index.json") as f:
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
        # Sensitive Logic Protection: Attention projections and router gates must be preserved
        # in donor precision to prevent double-quantization collapse in GGUF.
        if "self_attn" in key or "mlp.gate.weight" in key or "attn" in key:
            return self.get_donor_tensor(key)

        if self.current_layer_idx != layer_idx:
            self.current_layer_dict = None
            gc.collect()
            print(f"[*] Streaming BitNet Layer {layer_idx:02d} into GGUF...")
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


def build_bitnet_model_class(base_cls: type[ModelBase]):
    """Dynamic class factory injecting lazy streaming tensor generators into llama.cpp conversion classes."""
    class DynamicBitNetModel(base_cls):
        def __init__(self, *args, bitnet_dir: Path = None, **kwargs):
            self.bitnet_dir = bitnet_dir
            print("[*] Reading donor tensor shapes...")
            self.shapes = {}
            for sf in glob.glob(str(kwargs.get("dir_model") / "*.safetensors")):
                with safe_open(sf, framework="pt") as f:
                    for k in f.keys():
                        self.shapes[k] = list(f.get_slice(k).get_shape())
            print(f"[✓] Indexed {len(self.shapes)} tensor shapes.")

            self.stream_cache = StreamedLayerCache(self.bitnet_dir, kwargs.get("dir_model"), self.shapes)
            super().__init__(*args, **kwargs)

        def index_tensors(self, remote_hf_model_id=None) -> dict[str, Callable[[], torch.Tensor]]:
            print("[*] Registering lazy BitNet tensor generators...")
            tensors: dict[str, Callable[[], torch.Tensor]] = {}

            anchor_candidates = [
                "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
                "transformer.wte.weight", "transformer.ln_f.weight"
            ]
            for a in anchor_candidates:
                if a in self.shapes:
                    tensors[a] = lambda name=a: self.stream_cache.get_anchor(name)

            for name in sorted(self.shapes.keys()):
                # Detect layer index from standard transformer naming conventions (.layers.X. or .h.X. or .blk.X.)
                parts = name.split(".")
                found_layer = False
                for i, part in enumerate(parts):
                    if part in ("layers", "h", "blk") and i + 1 < len(parts) and parts[i + 1].isdigit():
                        layer_idx = int(parts[i + 1])
                        tensors[name] = lambda idx=layer_idx, k=name: self.stream_cache.get_layer_tensor(idx, k)
                        found_layer = True
                        break

                if not found_layer and name not in tensors:
                    tensors[name] = lambda k=name: self.stream_cache.get_donor_tensor(k)

            print(f"[✓] Successfully registered {len(tensors)} lazy tensor generators.")
            return tensors

    return DynamicBitNetModel


def find_llama_quantize() -> Optional[Path]:
    """Locates llama-quantize binary on system or scratch builds."""
    candidates = [
        Path("/home/tsriakshar/.gemini/antigravity/scratch/BitNet/build/bin/llama-quantize"),
        Path("/home/tsriakshar/llama.cpp/build/bin/llama-quantize"),
        Path("/usr/local/bin/llama-quantize"),
        Path("/usr/bin/llama-quantize"),
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    # Fallback to PATH search
    which = shutil.which("llama-quantize")
    return Path(which) if which else None


def export_model_to_gguf(
    donor_dir: str | Path,
    bitnet_dir: str | Path,
    output_dir: str | Path,
    quant_type: str = "Q4_K_M",
    clean_bf16: bool = True,
    threads: int = 16,
) -> Path:
    """
    Main export entry point: Streams donor + BitNet shards into GGUF and quantizes for deployment.
    """
    donor_dir = Path(donor_dir)
    bitnet_dir = Path(bitnet_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read architecture from donor config.json
    with open(donor_dir / "config.json") as f:
        cfg = json.load(f)
    arch = cfg.get("architectures", ["Qwen2ForCausalLM"])[0]
    print(f"[*] Detected donor architecture : {arch}")

    # 2. Get base conversion class from llama.cpp
    base_cls = get_model_class(arch)
    bitnet_cls = build_bitnet_model_class(base_cls)

    model_slug = cfg.get("model_type", "bitnet_model")
    bf16_gguf_path = output_dir / f"{model_slug}_bitnet_bf16.gguf"
    quant_slug = quant_type.lower().replace("-", "_")
    final_gguf_path = output_dir / f"{model_slug}_bitnet_{quant_slug}.gguf"

    # Temporary staging buffer
    temp_dir = Path("./tmp_gguf")
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp_dir)

    print(f"[*] Streaming BF16 GGUF container -> {bf16_gguf_path}...")
    model_inst = bitnet_cls(
        dir_model=donor_dir,
        ftype=gguf.LlamaFileType.MOSTLY_BF16,
        fname_out=bf16_gguf_path,
        is_big_endian=False,
        use_temp_file=True,
        eager=False,
        bitnet_dir=bitnet_dir,
    )
    model_inst.write()

    print(f"[✓] Uncompressed BF16 GGUF generated ({bf16_gguf_path.stat().st_size / (1024**3):.2f} GB).")

    # 3. Quantize with llama-quantize if requested
    if quant_type.upper() == "BF16":
        print(f"[✓] Target format is BF16. Skipping quantization.")
        return bf16_gguf_path

    quantize_bin = find_llama_quantize()
    if not quantize_bin:
        print(f"[!] Warning: llama-quantize binary not found. Keeping {bf16_gguf_path}")
        return bf16_gguf_path

    print(f"\n[*] Quantizing GGUF to deployment format '{quant_type}' using {threads} threads...")
    cmd = [
        str(quantize_bin),
        str(bf16_gguf_path),
        str(final_gguf_path),
        quant_type,
        str(threads),
    ]
    res = subprocess.run(cmd, check=True)
    if res.returncode == 0 and final_gguf_path.exists():
        print("=" * 70)
        print(f"[✓] GGUF PACKAGING COMPLETE: {final_gguf_path}")
        print(f"[*] Deployment file size : {final_gguf_path.stat().st_size / (1024**3):.2f} GB")
        print("=" * 70)

        if clean_bf16 and bf16_gguf_path.exists():
            print(f"[*] Removing temporary BF16 GGUF container ({bf16_gguf_path.name}) to reclaim space...")
            bf16_gguf_path.unlink()
            print("[✓] Temporary space reclaimed.")

        return final_gguf_path
    else:
        print(f"[!] Quantization failed. Retaining {bf16_gguf_path}")
        return bf16_gguf_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export BitNet Shards to Deployable GGUF")
    parser.add_argument("donor_dir", type=str, help="Path to donor HF model directory")
    parser.add_argument("bitnet_dir", type=str, help="Path to BitNet safetensors shards directory")
    parser.add_argument("--out-dir", type=str, default="./", help="Directory to save GGUF output")
    parser.add_argument("--quant", type=str, default="Q4_K_M", help="Deployment quantization type (e.g. Q4_K_M, Q2_K, TQ1_0, BF16)")
    parser.add_argument("--keep-bf16", action="store_true", help="Keep temporary BF16 GGUF container")
    parser.add_argument("--threads", type=int, default=16, help="Threads for llama-quantize")
    args = parser.parse_args()

    export_model_to_gguf(
        donor_dir=args.donor_dir,
        bitnet_dir=args.bitnet_dir,
        output_dir=args.out_dir,
        quant_type=args.quant,
        clean_bf16=not args.keep_bf16,
        threads=args.threads,
    )
