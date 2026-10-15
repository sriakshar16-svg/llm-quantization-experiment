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

        # Register compressed buffers in VRAM
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
        lut = self.get_lut(self.packed_ternary.device)

        # 1. Fast L1-cached LUT gather: uint8 -> 5x int8 {-1, 0, +1}
        trits = lut[self.packed_ternary.long()].reshape(-1)[:self.orig_numel]

        # 2. Multiply by group scales (grouped view avoids repeat_interleave memory allocation)
        num_groups = self.scales.numel()
        pad_len = num_groups * self.group_size - self.orig_numel
        if pad_len > 0:
            trits_padded = F.pad(trits, (0, pad_len))
        else:
            trits_padded = trits

        w_grouped = trits_padded.view(num_groups, self.group_size).to(torch.float16)
        w_grouped.mul_(self.scales.unsqueeze(-1))

        # 3. Scatter salient FP16 outliers back in-place
        if self.outlier_idx.numel() > 0:
            if self.outlier_idx.dtype == torch.uint8 or self.outlier_idx.ndim > 1:
                # Per-group uint8 outlier indices: shape (num_groups, k)
                w_grouped.scatter_(dim=-1, index=self.outlier_idx.long(), src=self.outlier_val)
                w_flat = w_grouped.view(-1)[:self.orig_numel]
            else:
                # Flat int32 outlier indices (backward compatible with existing shards)
                w_flat = w_grouped.view(-1)[:self.orig_numel]
                w_flat.index_copy_(0, self.outlier_idx.long(), self.outlier_val)
        else:
            w_flat = w_grouped.view(-1)[:self.orig_numel]

        w = w_flat.view(self.out_features, self.in_features)
        out = F.linear(x.to(torch.float16), w, self.bias)
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
        lut = self.get_lut(self.q_weight.device)

        # 1. Fast L1-cached LUT gather: uint8 -> 2x int8 in [-8, 7]
        unpacked = lut[self.q_weight.long()].reshape(-1)[:self.orig_numel]

        # 2. Multiply by group scales
        num_groups = self.scales.numel()
        pad_len = num_groups * self.group_size - self.orig_numel
        if pad_len > 0:
            unpacked_padded = F.pad(unpacked, (0, pad_len))
        else:
            unpacked_padded = unpacked

        w_grouped = unpacked_padded.view(num_groups, self.group_size).to(torch.float16)
        w_grouped.mul_(self.scales.unsqueeze(-1))
        w = w_grouped.view(-1)[:self.orig_numel].view(self.out_features, self.in_features)

        out = F.linear(x.to(torch.float16), w, self.bias)
        return out.to(orig_dtype)


class Int8WeightOnlyLinear(nn.Module):
    """Compact INT8 weight-only linear layer for lm_head and fallback projections."""

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, device: str = "cuda"):
        super().__init__()
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
        w = self.weight_int8.to(torch.float16) * self.scale
        return F.linear(x.to(torch.float16), w, self.bias).to(orig_dtype)


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
    offload_embed_to_cpu: bool = False,
):
    """
    Builds the DeepSeek-V2 model skeleton in 0 MB RAM (meta device) and streams
    Base-3 ternary shards + 4-bit attention + anchors directly into compressed GPU VRAM (~4.8-5.2 GB total).
    """
    donor_path = Path(donor_dir)
    bitnet_path = Path(bitnet_dir)

    print("[1/4] Initializing 0-MB meta skeleton from config...")
    config = AutoConfig.from_pretrained(str(donor_path), trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float16)
    model.eval()

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
    anchors = load_file(str(anchors_file)) if anchors_file.exists() else {}

    embed_w = anchors.get("model.embed_tokens.weight", get_donor_tensor("model.embed_tokens.weight"))
    if offload_embed_to_cpu:
        embed_pinned = embed_w.to(torch.float16).pin_memory()
        embed_mod = nn.Embedding(embed_pinned.shape[0], embed_pinned.shape[1], _weight=embed_pinned)
        embed_mod.register_forward_hook(lambda mod, inp, out: out.to(device))
        embed_mod.register_forward_pre_hook(lambda mod, inp: (inp[0].cpu(),))
        _set_submodule(model, "model.embed_tokens", embed_mod)
    else:
        embed_gpu = embed_w.to(device=device, dtype=torch.float16)
        _set_submodule(model, "model.embed_tokens", nn.Embedding(embed_gpu.shape[0], embed_gpu.shape[1], _weight=embed_gpu))

    norm_w = anchors.get("model.norm.weight", get_donor_tensor("model.norm.weight")).to(device, dtype=torch.float16)
    model.model.norm.weight = nn.Parameter(norm_w, requires_grad=False)

    lm_head_w = anchors.get("lm_head.weight", get_donor_tensor("lm_head.weight"))
    _set_submodule(model, "lm_head", Int8WeightOnlyLinear(lm_head_w, device=device))
    del anchors, lm_head_w, embed_w
    gc.collect()

    # 2. Stream Layers 0..N into GPU VRAM
    num_layers = getattr(config, "num_hidden_layers", 27)
    print(f"[3/4] Streaming {num_layers} compressed layers into GPU VRAM...")

    for layer_idx in range(num_layers):
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
                    device=device,
                )
                _set_submodule(model, mod_path, new_mod)

            elif qweight_key in layer_dict:
                # 2. 4-Bit Attention Projection (Trap #2 Fixed)
                target_mod = _get_submodule(model, mod_path)
                in_f, out_f = target_mod.in_features, target_mod.out_features
                new_mod = BitNet4BitLinear(
                    in_features=in_f,
                    out_features=out_f,
                    q_weight=layer_dict[qweight_key],
                    scales=layer_dict[f"{full_key}.scales"],
                    bias=bias_tensor,
                    device=device,
                )
                _set_submodule(model, mod_path, new_mod)

            elif "mlp.gate" in mod_path or "norm" in mod_path:
                # 3. MoE Router Gate & LayerNorms: preserve in FP16
                w = layer_dict.get(full_key, get_donor_tensor(full_key)).to(device, dtype=torch.float16)
                target_mod = _get_submodule(model, mod_path)
                target_mod.weight = nn.Parameter(w, requires_grad=False)
                if bias_tensor is not None:
                    target_mod.bias = nn.Parameter(bias_tensor.to(device, dtype=torch.float16), requires_grad=False)

            else:
                # 4. Attention/Linear fallback to INT8 weight-only
                w = layer_dict.get(full_key, get_donor_tensor(full_key))
                new_mod = Int8WeightOnlyLinear(w, bias=bias_tensor, device=device)
                _set_submodule(model, mod_path, new_mod)

        del layer_dict
        gc.collect()
        if device == "cuda" and (layer_idx % 5 == 0 or layer_idx == num_layers - 1):
            vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"  -> Loaded Layer {layer_idx:02d}/{num_layers - 1:02d} | Active GPU VRAM: {vram_gb:.2f} GB")

    # 3. Materialize Rotary Position Embeddings (RoPE / YaRN) on GPU
    print("[4/4] Materializing rotary position embeddings (RoPE/YaRN) on GPU...")
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
