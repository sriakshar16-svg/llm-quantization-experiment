#!/usr/bin/env python3
"""
bitnet_vram_engine.py
======================
True VRAM-Resident Base-3 Ternary + Sparse Outlier Inference Engine for RTX 3070 Ti (8GB VRAM).
Eliminates GGUF BF16 expansion by keeping all 64 MoE experts in GPU VRAM in compressed uint8 form.
Features:
  - 1.2 KB L1-cached Base-3 Lookup Table (BASE3_LUT) for zero-latency GPU trit decoding.
  - 512 B L1-cached 4-bit Lookup Table (NIBBLE4_LUT) for instant attention projection decoding.
  - Critical MoE Guard: Dormant experts (0 routed tokens) return in 0 ms without unpacking.
  - Full support for both structured per-group uint8 outliers (2.10 bpw) and flat int32 outliers (2.45 bpw).
  - Out-of-core meta skeleton loading with streaming safetensors shards directly to GPU VRAM.
"""

import gc
import json
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM


_BASE3_LUT_CACHE: Dict[str, torch.Tensor] = {}
_4BIT_LUT_CACHE: Dict[str, torch.Tensor] = {}


def _build_base3_lut(device: str = "cuda") -> torch.Tensor:
    """
    Precomputes a 256x5 int8 lookup table mapping each Base-3 uint8 byte [0..242]
    directly to its 5 ternary weights in {-1, 0, +1}. Size: 1.2 KB (fits in GPU L1 cache).
    """
    vals = torch.arange(256, dtype=torch.int32, device=device)
    u0 = (vals % 3) - 1
    u1 = ((vals // 3) % 3) - 1
    u2 = ((vals // 9) % 3) - 1
    u3 = ((vals // 27) % 3) - 1
    u4 = ((vals // 81) % 3) - 1
    return torch.stack([u0, u1, u2, u3, u4], dim=1).to(torch.int8)


def _build_4bit_lut(device: str = "cuda") -> torch.Tensor:
    """
    Precomputes a 256x2 int8 lookup table mapping each 4-bit packed uint8 byte
    directly to 2 signed values in [-8, 7]. Size: 512 bytes (fits in GPU L1 cache).
    """
    vals = torch.arange(256, dtype=torch.int32, device=device)
    low = (vals & 0x0F).to(torch.int8)
    high = ((vals >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack([low, high], dim=1)


class BitNetTernaryLinear(nn.Module):
    """
    Holds Base-3 packed ternary weights (1.60 bpw) + group scales + sparse FP16 outliers
    directly in GPU VRAM. Dequantizes ONLY during forward() and immediately frees the buffer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        packed_ternary: torch.Tensor,
        scales: torch.Tensor,
        outlier_idx: torch.Tensor,
        outlier_val: torch.Tensor,
        orig_numel: int,
        group_size: int = 128,
        bias: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.orig_numel = int(orig_numel)
        self.group_size = group_size

        # Register compressed buffers in VRAM or CPU (unpinned to prevent host RAM locking)
        if device == "cpu":
            self.register_buffer("packed_ternary", packed_ternary.cpu(), persistent=False)
            self.register_buffer("scales", scales.to(dtype=torch.float16).cpu(), persistent=False)
            self.register_buffer("outlier_idx", outlier_idx.cpu(), persistent=False)
            self.register_buffer("outlier_val", outlier_val.to(dtype=torch.float16).cpu(), persistent=False)
            if bias is not None:
                self.register_buffer("bias", bias.to(dtype=torch.float16).cpu(), persistent=False)
            else:
                self.bias = None
        else:
            self.register_buffer("packed_ternary", packed_ternary.to(device, dtype=torch.uint8), persistent=False)
            self.register_buffer("scales", scales.to(device, dtype=torch.float16), persistent=False)
            self.register_buffer("outlier_idx", outlier_idx.to(device), persistent=False)
            self.register_buffer("outlier_val", outlier_val.to(device, dtype=torch.float16), persistent=False)
            if bias is not None:
                self.register_buffer("bias", bias.to(device, dtype=torch.float16), persistent=False)
            else:
                self.bias = None

    @classmethod
    def get_lut(cls, device: torch.device) -> torch.Tensor:
        dev_str = str(device)
        if dev_str not in _BASE3_LUT_CACHE:
            _BASE3_LUT_CACHE[dev_str] = _build_base3_lut(dev_str)
        return _BASE3_LUT_CACHE[dev_str]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CRITICAL MoE GUARD: Do not unpack Base-3 weights if 0 tokens are routed to this expert!
        if x.numel() == 0:
            return torch.empty((*x.shape[:-1], self.out_features), dtype=x.dtype, device=x.device)

        orig_dtype = x.dtype
        target_device = x.device

        # On-demand VRAM streaming if buffer is resident in CPU RAM
        if self.packed_ternary.device != target_device:
            p_ternary = self.packed_ternary.to(target_device, non_blocking=True)
            scales = self.scales.to(target_device, non_blocking=True)
            outlier_idx = self.outlier_idx.to(target_device, non_blocking=True)
            outlier_val = self.outlier_val.to(target_device, non_blocking=True)
            bias = self.bias.to(target_device, non_blocking=True) if self.bias is not None else None
            lut = self.get_lut(target_device)
        else:
            p_ternary = self.packed_ternary
            scales = self.scales
            outlier_idx = self.outlier_idx
            outlier_val = self.outlier_val
            bias = self.bias
            lut = self.get_lut(self.packed_ternary.device)

        # 1. Fast L1-cached LUT gather: uint8 -> 5x int8 {-1, 0, +1}
        # Use int32 indexing to cut index allocation by 50% vs int64
        trits = lut[p_ternary.int()].reshape(-1)[:self.orig_numel]

        # 2. Multiply by group scales (grouped view avoids repeat_interleave memory allocation)
        num_groups = scales.numel()
        pad_len = num_groups * self.group_size - self.orig_numel
        if pad_len > 0:
            trits_padded = F.pad(trits, (0, pad_len))
        else:
            trits_padded = trits

        w_grouped = trits_padded.view(num_groups, self.group_size).to(torch.float16)
        del trits, trits_padded
        w_grouped.mul_(scales.unsqueeze(-1))

        # 3. Scatter salient FP16 outliers back in-place
        if outlier_idx.numel() > 0:
            if outlier_idx.dtype == torch.uint8 or outlier_idx.ndim > 1:
                # Per-group uint8 outlier indices: shape (num_groups, k)
                w_grouped.scatter_(dim=-1, index=outlier_idx.long(), src=outlier_val)
                w_flat = w_grouped.view(-1)[:self.orig_numel]
            else:
                # Flat int32 outlier indices (backward compatible with existing shards)
                w_flat = w_grouped.view(-1)[:self.orig_numel]
                w_flat.index_copy_(0, outlier_idx.long(), outlier_val)
        else:
            w_flat = w_grouped.view(-1)[:self.orig_numel]

        w = w_flat.view(self.out_features, self.in_features)
        out = F.linear(x.to(torch.float16), w, bias)
        del w, w_flat, w_grouped
        return out.to(orig_dtype)


class BitNet4BitLinear(nn.Module):
    """
    Holds 4-bit uniform block-quantized weights (4.0 bpw) for Attention projections.
    Unpacks 2 nibbles per byte on-the-fly using a 512-byte L1-cached lookup table.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        q_weight: torch.Tensor,
        scales: torch.Tensor,
        orig_numel: Optional[int] = None,
        group_size: int = 128,
        bias: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.orig_numel = orig_numel if orig_numel is not None else (in_features * out_features)

        # Register compressed buffers in VRAM or CPU (unpinned to prevent host RAM locking)
        if device == "cpu":
            self.register_buffer("q_weight", q_weight.cpu(), persistent=False)
            self.register_buffer("scales", scales.to(dtype=torch.float16).cpu(), persistent=False)
            if bias is not None:
                self.register_buffer("bias", bias.to(dtype=torch.float16).cpu(), persistent=False)
            else:
                self.bias = None
        else:
            self.register_buffer("q_weight", q_weight.to(device, dtype=torch.uint8), persistent=False)
            self.register_buffer("scales", scales.to(device, dtype=torch.float16), persistent=False)
            if bias is not None:
                self.register_buffer("bias", bias.to(device, dtype=torch.float16), persistent=False)
            else:
                self.bias = None

    @classmethod
    def get_lut(cls, device: torch.device) -> torch.Tensor:
        dev_str = str(device)
        if dev_str not in _4BIT_LUT_CACHE:
            _4BIT_LUT_CACHE[dev_str] = _build_4bit_lut(dev_str)
        return _4BIT_LUT_CACHE[dev_str]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.empty((*x.shape[:-1], self.out_features), dtype=x.dtype, device=x.device)

        orig_dtype = x.dtype
        target_device = x.device

        # On-demand VRAM streaming if buffer is resident in CPU RAM
        if self.q_weight.device != target_device:
            q_weight = self.q_weight.to(target_device, non_blocking=True)
            scales = self.scales.to(target_device, non_blocking=True)
            bias = self.bias.to(target_device, non_blocking=True) if self.bias is not None else None
            lut = self.get_lut(target_device)
        else:
            q_weight = self.q_weight
            scales = self.scales
            bias = self.bias
            lut = self.get_lut(self.q_weight.device)

        # 1. Fast L1-cached LUT gather: uint8 -> 2x int8 in [-8, 7]
        unpacked = lut[q_weight.int()].reshape(-1)[:self.orig_numel]

        # 2. Multiply by group scales
        num_groups = scales.numel()
        pad_len = num_groups * self.group_size - self.orig_numel
        if pad_len > 0:
            unpacked_padded = F.pad(unpacked, (0, pad_len))
        else:
            unpacked_padded = unpacked

        w_grouped = unpacked_padded.view(num_groups, self.group_size).to(torch.float16)
        del unpacked, unpacked_padded
        w_grouped.mul_(scales.unsqueeze(-1))
        w = w_grouped.view(-1)[:self.orig_numel].view(self.out_features, self.in_features)

        out = F.linear(x.to(torch.float16), w, bias)
        del w, w_grouped
        return out.to(orig_dtype)


class Int8WeightOnlyLinear(nn.Module):
    """Compact INT8 weight-only linear layer for lm_head and fallback projections."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        device: str = "cuda",
        prequant_weight: Optional[torch.Tensor] = None,
        prequant_scale: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if prequant_weight is not None:
            self.out_features, self.in_features = prequant_weight.shape
            self.register_buffer("weight_int8", prequant_weight.to(device, dtype=torch.int8), persistent=False)
            self.register_buffer("scale", prequant_scale.to(device, dtype=torch.float16), persistent=False)
        else:
            self.out_features, self.in_features = weight.shape
            w_f32 = weight.float()
            scale = (w_f32.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-5)
            w_int8 = (w_f32 / scale).round().clamp(-127, 127).to(torch.int8)
            self.register_buffer("weight_int8", w_int8.to(device), persistent=False)
            self.register_buffer("scale", scale.to(device, dtype=torch.float16), persistent=False)

        if bias is not None:
            self.register_buffer("bias", bias.to(device, dtype=torch.float16), persistent=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.empty((*x.shape[:-1], self.out_features), dtype=x.dtype, device=x.device)
        orig_dtype = x.dtype
        target_device = x.device
        x_f16 = x.to(device=target_device, dtype=torch.float16)

        # When out_features is large (e.g. 152k vocab for Qwen2), chunking into 16k slices
        # uses only ~80 MB of temporary FP16 buffer instead of allocating 1.45 GiB all at once!
        chunk_size = 16384
        if self.out_features > chunk_size:
            out_chunks = []
            for start in range(0, self.out_features, chunk_size):
                end = min(start + chunk_size, self.out_features)
                w_c = self.weight_int8[start:end].to(target_device, non_blocking=True).to(torch.float16)
                s_c = self.scale[start:end].to(target_device, non_blocking=True)
                w_f16 = w_c * s_c
                b_c = self.bias[start:end].to(target_device) if self.bias is not None else None
                out_chunks.append(F.linear(x_f16, w_f16, b_c))
            out = torch.cat(out_chunks, dim=-1)
        else:
            w_int8 = self.weight_int8.to(target_device, non_blocking=True)
            scale = self.scale.to(target_device, non_blocking=True)
            bias = self.bias.to(target_device, non_blocking=True) if self.bias is not None else None
            w = w_int8.to(torch.float16) * scale
            out = F.linear(x_f16, w, bias)
        return out.to(orig_dtype)


def _set_submodule(root: nn.Module, target_path: str, new_module: nn.Module):
    parts = target_path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _get_submodule(root: nn.Module, target_path: str) -> nn.Module:
    parts = target_path.split(".")
    curr = root
    for p in parts:
        curr = getattr(curr, p)
    return curr


def load_bitnet_vram_model(
    donor_dir: str,
    bitnet_dir: str,
    device: str = "cuda",
    offload_embed_to_cpu: bool = True,
    gpu_layers: Optional[int] = 40,
):
    """
    Builds the model skeleton in 0 MB RAM (meta device) and streams
    Base-3 ternary shards + 4-bit attention + anchors directly into compressed GPU VRAM.
    Supports hybrid GPU/CPU layer allocation via `gpu_layers` for models exceeding VRAM capacity.
    """
    donor_path = Path(donor_dir)
    bitnet_path = Path(bitnet_dir)

    print("[1/4] Initializing 0-MB meta skeleton from config...")
    config = AutoConfig.from_pretrained(str(donor_path), trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float16)
    model.eval()
    # Force model.device property to report compute device (e.g. cuda:0)
    type(model).device = property(lambda self: torch.device(device))

    # Index donor safetensors for fallback tensors (norm weights, biases, unquantized heads)
    index_file = donor_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            donor_map = json.load(f)["weight_map"]
    else:
        with safe_open(str(donor_path / "model.safetensors"), framework="pt", device="cpu") as f:
            donor_map = {k: "model.safetensors" for k in f.keys()}

    donor_handles = {}

    def get_donor_tensor(key: str) -> torch.Tensor:
        shard = donor_map[key]
        if shard not in donor_handles:
            donor_handles[shard] = safe_open(str(donor_path / shard), framework="pt", device="cpu")
        return donor_handles[shard].get_tensor(key)

    # 1. Load Anchors
    print("[2/4] Loading anchors (Embedding + INT8 LM Head + Final Norm)...")
    anchors_file = bitnet_path / "anchors.safetensors"
    lm_int8_file = bitnet_path / "lm_head_int8.safetensors"

    if anchors_file.exists():
        with safe_open(str(anchors_file), framework="pt", device="cpu") as f:
            embed_w = f.get_tensor("model.embed_tokens.weight")
            norm_w = f.get_tensor("model.norm.weight").to(device, dtype=torch.float16)
            if not lm_int8_file.exists():
                lm_head_w = f.get_tensor("lm_head.weight")
            else:
                lm_head_w = None
    else:
        embed_w = get_donor_tensor("model.embed_tokens.weight")
        norm_w = get_donor_tensor("model.norm.weight").to(device, dtype=torch.float16)
        lm_head_w = get_donor_tensor("lm_head.weight")

    model.model.norm.weight = nn.Parameter(norm_w, requires_grad=False)

    if offload_embed_to_cpu:
        # Keep embedding table in CPU RAM as Float16 WITHOUT pin_memory (prevents locking 3.6 GB host RAM)
        embed_cpu = embed_w.to(torch.float16)
        del embed_w
        embed_mod = nn.Embedding(embed_cpu.shape[0], embed_cpu.shape[1], _weight=embed_cpu)
        embed_mod.register_forward_hook(lambda mod, inp, out: out.to(device))
        embed_mod.register_forward_pre_hook(lambda mod, inp: (inp[0].cpu(),))
        _set_submodule(model, "model.embed_tokens", embed_mod)
    else:
        embed_gpu = embed_w.to(device=device, dtype=torch.float16)
        del embed_w
        _set_submodule(model, "model.embed_tokens", nn.Embedding(embed_gpu.shape[0], embed_gpu.shape[1], _weight=embed_gpu))

    if lm_int8_file.exists():
        with safe_open(str(lm_int8_file), framework="pt", device="cpu") as f:
            w_int8 = f.get_tensor("weight_int8")
            s_scale = f.get_tensor("scale")
        _set_submodule(model, "lm_head", Int8WeightOnlyLinear(prequant_weight=w_int8, prequant_scale=s_scale, device=device))
        del w_int8, s_scale
    else:
        _set_submodule(model, "lm_head", Int8WeightOnlyLinear(lm_head_w, device=device))
        del lm_head_w
    gc.collect()

    # 2. Stream Layers 0..N into GPU VRAM / CPU RAM
    num_layers = getattr(config, "num_hidden_layers", 27)
    effective_gpu_layers = num_layers if gpu_layers is None else min(gpu_layers, num_layers)
    print(f"[3/4] Allocating {num_layers} compressed layers ({effective_gpu_layers} VRAM-resident, {num_layers - effective_gpu_layers} dynamic streaming)...")

    for layer_idx in range(num_layers):
        layer_dev = device if (gpu_layers is None or layer_idx < gpu_layers) else "cpu"
        layer_file = bitnet_path / f"layer_{layer_idx}.safetensors"
        layer_dict = load_file(str(layer_file)) if layer_file.exists() else {}

        prefix = f"model.layers.{layer_idx}."
        layer_keys = [k for k in donor_map.keys() if k.startswith(prefix)]

        for full_key in layer_keys:
            if not full_key.endswith(".weight"):
                continue

            mod_path = full_key[:-len(".weight")]
            bias_key = f"{mod_path}.bias"
            bias_tensor = layer_dict.get(bias_key, get_donor_tensor(bias_key) if bias_key in donor_map else None)

            packed_key = f"{full_key}.packed_ternary"
            qweight_key = f"{full_key}.q_weight"

            if packed_key in layer_dict:
                # 1. Base-3 Ternary Linear (MLP / MoE Expert)
                target_mod = _get_submodule(model, mod_path)
                in_f, out_f = target_mod.in_features, target_mod.out_features
                orig_n = layer_dict[f"{full_key}.orig_numel"].item() if f"{full_key}.orig_numel" in layer_dict else (in_f * out_f)
                new_mod = BitNetTernaryLinear(
                    in_features=in_f,
                    out_features=out_f,
                    packed_ternary=layer_dict[packed_key],
                    scales=layer_dict[f"{full_key}.scales"],
                    outlier_idx=layer_dict[f"{full_key}.outlier_idx"],
                    outlier_val=layer_dict[f"{full_key}.outlier_val"],
                    orig_numel=orig_n,
                    bias=bias_tensor,
                    device=layer_dev,
                )
                _set_submodule(model, mod_path, new_mod)

            elif qweight_key in layer_dict:
                # 2. 4-Bit Attention Projection
                target_mod = _get_submodule(model, mod_path)
                in_f, out_f = target_mod.in_features, target_mod.out_features
                new_mod = BitNet4BitLinear(
                    in_features=in_f,
                    out_features=out_f,
                    q_weight=layer_dict[qweight_key],
                    scales=layer_dict[f"{full_key}.scales"],
                    bias=bias_tensor,
                    device=layer_dev,
                )
                _set_submodule(model, mod_path, new_mod)

            elif "mlp.gate" in mod_path or "norm" in mod_path:
                # 3. MoE Router Gate & LayerNorms: preserve in FP16 on compute device
                w = layer_dict.get(full_key, get_donor_tensor(full_key)).to(device, dtype=torch.float16)
                target_mod = _get_submodule(model, mod_path)
                target_mod.weight = nn.Parameter(w, requires_grad=False)
                if bias_tensor is not None:
                    target_mod.bias = nn.Parameter(bias_tensor.to(device, dtype=torch.float16), requires_grad=False)

            else:
                # 4. Attention/Linear fallback to INT8 weight-only
                w = layer_dict.get(full_key, get_donor_tensor(full_key))
                new_mod = Int8WeightOnlyLinear(w, bias=bias_tensor, device=layer_dev)
                _set_submodule(model, mod_path, new_mod)

        del layer_dict
        gc.collect()
        if device == "cuda" and (layer_idx % 8 == 0 or layer_idx == num_layers - 1):
            vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"  -> Loaded Layer {layer_idx:02d}/{num_layers - 1:02d} | Active GPU VRAM: {vram_gb:.2f} GB")

    # 3. Materialize Rotary Position Embeddings (RoPE / YaRN) on GPU
    print("[4/4] Materializing rotary position embeddings (RoPE/YaRN) on GPU...")
    if hasattr(model.model, "rotary_emb"):
        rot_cls = type(model.model.rotary_emb)
        try:
            model.model.rotary_emb = rot_cls(config=config, device=device)
        except Exception:
            try:
                model.model.rotary_emb = rot_cls(config, device=device)
            except Exception:
                model.model.rotary_emb.to_empty(device=device)

        orig_rot_fwd = model.model.rotary_emb.forward
        def safe_rot_fwd(x, position_ids, **kwargs):
            return orig_rot_fwd(x, position_ids.to(x.device), **kwargs)
        model.model.rotary_emb.forward = safe_rot_fwd

    for name, mod in model.named_modules():
        if "rotary_emb" in name:
            if hasattr(mod, "dim") and hasattr(mod, "base"):
                inv_freq = 1.0 / (mod.base ** (torch.arange(0, mod.dim, 2, dtype=torch.float32, device=device) / mod.dim))
                mod.register_buffer("inv_freq", inv_freq, persistent=False)
                if hasattr(mod, "_set_cos_sin_cache"):
                    mod._set_cos_sin_cache(seq_len=getattr(mod, "max_position_embeddings", 4096), device=device, dtype=torch.float16)

    donor_handles.clear()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[✓] Model ready in VRAM! Total Allocated: {allocated_gb:.2f} GB | Reserved: {reserved_gb:.2f} GB")

    return model
