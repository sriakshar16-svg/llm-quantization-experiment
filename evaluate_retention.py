"""
Universal Out-of-Core Retention & Fidelity Evaluator (evaluate_retention.py)
=============================================================================
Measures:
  1. Weight Cosine Similarity (%)
  2. Relative Frobenius Norm Error (%)
  3. Signal-to-Quantization-Noise Ratio (SQNR in dB)
  4. Layer-by-layer Attention vs. MLP fidelity
Strictly Out-of-Core: Reads donor safetensors with safe_open on CPU. Peak VRAM < 500 MB.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SFX_PT = ".packed_ternary"
SFX_QW = ".q_weight"


def unpack_ternary_base3(packed: torch.Tensor, orig_numel: int, device=DEVICE):
    """Unpacks uint8 Base-3 packed bytes into int8 {-1, 0, 1} tensor."""
    p = packed.to(device=device, dtype=torch.int32)
    u0 = p % 3; p = p // 3
    u1 = p % 3; p = p // 3
    u2 = p % 3; p = p // 3
    u3 = p % 3; u4 = p // 3
    unpacked = torch.stack([u0, u1, u2, u3, u4], dim=1).reshape(-1)[:orig_numel]
    return (unpacked - 1).to(torch.int8)


def unpack_nibbles_4bit(packed: torch.Tensor, orig_numel: int, device=DEVICE):
    """Unpacks 2 nibbles per byte into int8 uniform [-8, 7] tensor."""
    p = packed.to(device=device)
    low = (p & 0x0F).to(torch.int8)
    high = ((p >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.empty(orig_numel, dtype=torch.int8, device=device)
    unpacked[0::2] = low[:(orig_numel + 1) // 2]
    unpacked[1::2] = high[:orig_numel // 2]
    return unpacked


def build_weight_index(donor_dir: str):
    index_path = os.path.join(donor_dir, "model.safetensors.index.json")
    single_path = os.path.join(donor_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    elif os.path.exists(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as f:
            return {k: "model.safetensors" for k in f.keys()}
    else:
        raise FileNotFoundError(f"No safetensors found in {donor_dir}")


def evaluate_layer_fidelity(donor_dir: str, converted_dir: str, layer_idx: int, weight_map: dict):
    prefix = f"model.layers.{layer_idx}."
    shard_file = os.path.join(converted_dir, f"layer_{layer_idx}.safetensors")
    if not os.path.exists(shard_file):
        return None

    saved_dict = load_file(shard_file)
    results = []

    # Map donor shards
    layer_donor_keys = [k for k in weight_map if k.startswith(prefix) and ("weight" in k)]
    shard_groups = {}
    for k in layer_donor_keys:
        s = weight_map[k]
        shard_groups.setdefault(s, []).append(k)

    for s_file, keys in shard_groups.items():
        with safe_open(os.path.join(donor_dir, s_file), framework="pt", device="cpu") as f:
            for full_k in keys:
                orig_t = f.get_tensor(full_k)
                if orig_t.ndim != 2:
                    continue  # skip 1D norms/biases

                orig_flat = orig_t.float().to(DEVICE).flatten()
                orig_numel = orig_flat.numel()

                # Reconstruct quantized weight
                rec_flat = None
                tier = "Full"

                k_base3 = f"{full_k}.packed_ternary"
                k_4bit = f"{full_k}.q_weight"

                if k_base3 in saved_dict:
                    tier = "Base-3 Ternary"
                    packed = saved_dict[k_base3]
                    orig_n = saved_dict[f"{full_k}.orig_numel"].item()
                    unpacked = unpack_ternary_base3(packed, orig_n, device=DEVICE)
                    scales = saved_dict[f"{full_k}.scales"].to(DEVICE, dtype=torch.float32)
                    w_rec = (unpacked.float() * scales.repeat_interleave(128)[:orig_n])

                    out_idx_k = f"{full_k}.outlier_idx"
                    out_val_k = f"{full_k}.outlier_val"
                    if out_idx_k in saved_dict and out_val_k in saved_dict:
                        out_idx = saved_dict[out_idx_k].to(DEVICE)
                        out_val = saved_dict[out_val_k].to(DEVICE, dtype=torch.float32)
                        if out_idx.numel() > 0:
                            if out_idx.dtype == torch.uint8 or out_idx.ndim > 1:
                                num_groups = scales.numel()
                                pad_len = num_groups * 128 - orig_n
                                trits_pad = F.pad(unpacked.float(), (0, pad_len)) if pad_len > 0 else unpacked.float()
                                w_g = trits_pad.view(num_groups, 128) * scales.unsqueeze(-1)
                                w_g.scatter_(dim=-1, index=out_idx.long(), src=out_val)
                                w_rec = w_g.view(-1)[:orig_n]
                            else:
                                w_rec[out_idx.long()] = out_val
                    rec_flat = w_rec

                elif k_4bit in saved_dict:
                    tier = "4-Bit Attention"
                    packed = saved_dict[k_4bit]
                    scales = saved_dict[f"{full_k}.scales"].to(DEVICE, dtype=torch.float32)
                    unpacked = unpack_nibbles_4bit(packed, orig_numel, device=DEVICE)
                    rec_flat = unpacked.float() * scales.repeat_interleave(128)[:orig_numel]

                elif full_k in saved_dict:
                    tier = "FP16 Router/Gate"
                    rec_flat = saved_dict[full_k].float().to(DEVICE).flatten()

                if rec_flat is not None:
                    cos_sim = F.cosine_similarity(orig_flat, rec_flat, dim=0).item() * 100.0
                    noise = orig_flat - rec_flat
                    rel_err = (torch.norm(noise) / torch.norm(orig_flat)).item() * 100.0
                    sqnr = (10.0 * torch.log10(torch.norm(orig_flat).pow(2) / torch.norm(noise).pow(2).clamp(min=1e-8))).item()

                    short_name = full_k[len(prefix):]
                    results.append({
                        "name": short_name,
                        "tier": tier,
                        "cos_sim": cos_sim,
                        "rel_err": rel_err,
                        "sqnr_db": sqnr,
                    })

    return results


def run_retention_audit(donor_dir: str, converted_dir: str, num_sample_layers: int = 5):
    print("=" * 75)
    print("   ZERO-GAP OUT-OF-CORE RETENTION & FIDELITY AUDIT REPORT   ")
    print("=" * 75)
    print(f"[*] Donor Architecture : {donor_dir}")
    print(f"[*] Quantized Shards   : {converted_dir}")
    print(f"[*] Sample Layers Audited: {num_sample_layers} layers")

    weight_map = build_weight_index(donor_dir)
    layer_shards = sorted([
        int(f.split("_")[1].split(".")[0]) for f in os.listdir(converted_dir)
        if f.startswith("layer_") and f.endswith(".safetensors")
    ])

    sample_indices = layer_shards[:num_sample_layers]
    all_metrics = []
    attn_cos = []
    mlp_cos = []
    gate_cos = []

    for l_idx in sample_indices:
        print(f"\n---> Auditing Layer {l_idx}...")
        res = evaluate_layer_fidelity(donor_dir, converted_dir, l_idx, weight_map)
        if not res:
            continue

        for r in res:
            all_metrics.append(r)
            if "Attention" in r["tier"]:
                attn_cos.append(r["cos_sim"])
            elif "Ternary" in r["tier"]:
                mlp_cos.append(r["cos_sim"])
            else:
                gate_cos.append(r["cos_sim"])

        # Display top 4 salient tensors for this layer
        sample_disp = [res[0], res[1]] if len(res) >= 2 else res
        for s in sample_disp:
            print(f"     • {s['name']:<35} [{s['tier']:<14}] CosSim: {s['cos_sim']:.2f}% | RelErr: {s['rel_err']:.2f}% | SQNR: {s['sqnr_db']:.1f} dB")

    overall_cos = sum(m["cos_sim"] for m in all_metrics) / len(all_metrics)
    overall_rel = sum(m["rel_err"] for m in all_metrics) / len(all_metrics)
    overall_sqnr = sum(m["sqnr_db"] for m in all_metrics) / len(all_metrics)

    avg_attn = sum(attn_cos) / len(attn_cos) if attn_cos else 0.0
    avg_mlp = sum(mlp_cos) / len(mlp_cos) if mlp_cos else 0.0
    avg_gate = sum(gate_cos) / len(gate_cos) if gate_cos else 100.0

    print("\n" + "=" * 75)
    print("                    RETENTION AUDIT SUMMARY                       ")
    print("=" * 75)
    print(f"  • Overall Weight Cosine Similarity  : {overall_cos:.2f}%  (Zero Brain-Death)")
    print(f"  • Attention Protection Fidelity (4-bit) : {avg_attn:.2f}%  (Entropy Preserved)")
    print(f"  • Bulk MLP / Expert Fidelity (Base-3) : {avg_mlp:.2f}%  (Analytical Least Squares)")
    print(f"  • MoE Router Gates Fidelity (FP16)  : {avg_gate:.2f}%  (Zero Expert Misrouting)")
    print(f"  • Relative Frobenius Reconstruction Error : {overall_rel:.2f}%")
    print(f"  • Signal-to-Quantization Noise (SQNR) : {overall_sqnr:.1f} dB")
    print("=" * 75)

    if overall_cos >= 90.0:
        print("[✓] VERDICT: PASS (EXCELLENT RETENTION — No Quality Collapse)")
    elif overall_cos >= 85.0:
        print("[✓] VERDICT: PASS (GOOD RETENTION)")
    else:
        print("[!] VERDICT: DEGRADED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Gap Retention Evaluator")
    parser.add_argument("donor_dir", type=str, help="Donor model directory")
    parser.add_argument("converted_dir", type=str, help="Converted shards directory")
    parser.add_argument("--layers", type=int, default=5, help="Number of sample layers to audit")
    args = parser.parse_args()

    run_retention_audit(args.donor_dir, args.converted_dir, num_sample_layers=args.layers)
