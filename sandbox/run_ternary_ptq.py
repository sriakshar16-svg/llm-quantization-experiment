"""
Post-Training Ternary Quantization (PTQ) Sandbox
================================================
Explores aggressive post-training static quantization to ternary weights {-1, 0, 1}
with outlier protection and layer-skipping heuristics on dense Transformer LLMs (e.g. SmolLM2-360M / Qwen2.5-0.5B).

Findings:
  - Preserving the top 3% of weights (outliers) in full FP32 prevents immediate degenerate collapse
    compared to 100% naive ternary crushing, but semantic reasoning still breaks significantly.
  - Demonstrates why ternary architectures require Quantization-Aware Training (QAT) rather than PTQ.
"""

import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def outlier_aware_ternary(weight_tensor: torch.Tensor, outlier_fraction: float = 0.03) -> torch.Tensor:
    """
    Protects the top `outlier_fraction` (e.g., 3%) of high-magnitude weights in full FP32,
    and statically quantizes the remaining 97% to ternary values {-1, 0, 1}.
    
    Args:
        weight_tensor: Weight tensor from a Transformer projection layer.
        outlier_fraction: Fraction of top weights to retain unquantized.
    Returns:
        Hybrid tensor with sensitive weights preserved and bulk compressed to ternary.
    """
    W = weight_tensor
    # Identify magnitude threshold for top outlier weights
    k = max(1, int(W.numel() * outlier_fraction))
    threshold = torch.topk(W.abs().flatten(), k).values[-1]
    
    outlier_mask = W.abs() >= threshold
    bulk_mask = ~outlier_mask

    # Compute abs-mean scaling factor solely on bulk weights
    bulk_weights = W[bulk_mask]
    if bulk_weights.numel() == 0:
        return W

    gamma = bulk_weights.abs().mean()
    gamma = torch.clamp(gamma, min=1e-7)

    # Quantize bulk weights to ternary {-1, 0, 1}
    ternary_bulk = torch.clamp(torch.round(bulk_weights / gamma), -1.0, 1.0) * gamma

    # Construct reconstructed tensor
    W_out = W.clone()
    W_out[bulk_mask] = ternary_bulk
    return W_out

def main():
    parser = argparse.ArgumentParser(description="Run Outlier-Aware Ternary Post-Training Quantization")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolLM2-360M", help="Hugging Face model ID")
    parser.add_argument("--outlier_fraction", type=float, default=0.03, help="Fraction of outlier weights to preserve")
    parser.add_argument("--protect_boundary_layers", type=int, default=4, help="Number of initial/final layers to preserve in FP32")
    args = parser.parse_args()

    print(f"1. Loading base model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)

    total_layers = len(model.model.layers)
    print(f"\n2. Statically converting intermediate layers ({args.protect_boundary_layers} to {total_layers - args.protect_boundary_layers}) to Ternary...")
    start_time = time.time()
    converted_layers = 0

    for i, layer in enumerate(model.model.layers):
        # Skip sensitive input and output layers
        if i < args.protect_boundary_layers or i >= (total_layers - args.protect_boundary_layers):
            continue

        # Quantize Attention projections
        layer.self_attn.q_proj.weight.data = outlier_aware_ternary(layer.self_attn.q_proj.weight.data, args.outlier_fraction)
        layer.self_attn.k_proj.weight.data = outlier_aware_ternary(layer.self_attn.k_proj.weight.data, args.outlier_fraction)
        layer.self_attn.v_proj.weight.data = outlier_aware_ternary(layer.self_attn.v_proj.weight.data, args.outlier_fraction)
        layer.self_attn.o_proj.weight.data = outlier_aware_ternary(layer.self_attn.o_proj.weight.data, args.outlier_fraction)

        # Quantize MLP / FFN projections
        layer.mlp.gate_proj.weight.data = outlier_aware_ternary(layer.mlp.gate_proj.weight.data, args.outlier_fraction)
        layer.mlp.up_proj.weight.data = outlier_aware_ternary(layer.mlp.up_proj.weight.data, args.outlier_fraction)
        layer.mlp.down_proj.weight.data = outlier_aware_ternary(layer.mlp.down_proj.weight.data, args.outlier_fraction)

        converted_layers += 1

    elapsed = time.time() - start_time
    print(f"Compressed {converted_layers}/{total_layers} layers in {elapsed:.2f} seconds.")
    print(f"Protected {total_layers - converted_layers} boundary layers in pure FP32.")

    print("\n" + "=" * 60)
    print("INTERACTIVE PTQ TERNARY CONSOLE (Type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nPrompt > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["quit", "exit"]:
                print("Exiting sandbox...")
                break

            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            print(f"\nResponse:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
