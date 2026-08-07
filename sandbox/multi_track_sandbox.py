"""
Multi-Track Quantization Sandbox
================================
Empirical evaluation comparing three representations on the exact same model instance:
  1. Baseline (Unquantized FP16/FP32 dense model)
  2. Real Ternary Track (Post-Training Static Quantization to {-1, 0, 1})
  3. Block Hexadecimal Track (Uniform 4-bit / Base-16 block quantization)

Hypothesis:
  - 4-bit hexadecimal quantization preserves core semantic logic and token coherence.
  - Naive post-training "crushing" to ternary {-1, 0, 1} destroys attention weights and destroys reasoning.
"""

import copy
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def apply_real_ternary(tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies naive Post-Training Static Quantization (PTQ) into ternary values {-1, 0, 1}.
    
    Args:
        tensor: Original floating-point weight matrix.
    Returns:
        Dequantized tensor mapped strictly to ternary values scaled by mean absolute magnitude (gamma).
    """
    with torch.no_grad():
        gamma = tensor.abs().mean()
        if gamma == 0:
            return tensor
        # Scale by absmean, round to nearest integer, and clamp to {-1, 0, 1}
        scaled = tensor / gamma
        ternary = torch.clamp(torch.round(scaled), -1.0, 1.0)
        return ternary * gamma

def apply_block_hex_quantization(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """
    Applies Block-Level Hexadecimal (4-Bit / 16-level) Uniform Quantization.
    
    Weights are grouped into contiguous blocks (e.g. 32 elements). Each block is normalized
    by its maximum absolute value and mapped to 16 discrete integer bins ([-8, 7] / base-16),
    then dequantized back.
    
    Args:
        tensor: Original floating-point weight matrix.
        block_size: Number of contiguous weight elements per quantization block.
    Returns:
        Dequantized 4-bit simulated tensor.
    """
    with torch.no_grad():
        original_shape = tensor.shape
        flat = tensor.view(-1)

        # Pad tensor if flat length is not evenly divisible by block_size
        padding = (block_size - (flat.numel() % block_size)) % block_size
        if padding > 0:
            flat = torch.cat([flat, torch.zeros(padding, dtype=flat.dtype, device=flat.device)])

        blocks = flat.view(-1, block_size)

        # Compute per-block dynamic scaling factor (keepdim=True to prevent broadcast mismatch)
        block_maxes = blocks.abs().max(dim=1, keepdim=True)[0]
        block_maxes = torch.clamp(block_maxes, min=1e-5)

        # Quantize to 4-bit signed grid: [-8.0, 7.0] (16 distinct hexadecimal states)
        normalized = blocks / block_maxes
        quantized = torch.clamp(torch.round(normalized * 7.0), -8.0, 7.0)

        # Dequantize back to floating-point range
        dequantized = (quantized / 7.0) * block_maxes
        return dequantized[:flat.numel() - padding].view(original_shape)

def main():
    parser = argparse.ArgumentParser(description="Run Multi-Track Quantization Comparison")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B", help="HuggingFace model ID or local path")
    parser.add_argument("--prompt", type=str, default="The mechanics of neural networks are", help="Prompt to evaluate")
    parser.add_argument("--max_tokens", type=int, default=35, help="Number of new tokens to generate")
    parser.add_argument("--target_layer", type=str, default="mlp.down_proj.weight", help="Weight substring to target for PTQ")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using {device.upper()}")
    print(f"[Model] Loading base model: {args.model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    print("Forking independent tracks in memory...")
    tracks = {
        "Baseline (Unquantized)": base_model,
        "Block Hex (4-bit PTQ)": copy.deepcopy(base_model),
        "Real Ternary (1.58-bit PTQ)": copy.deepcopy(base_model),
    }

    print(f"Applying quantization transformations to target parameter: '{args.target_layer}'...")
    for name, param in tracks["Block Hex (4-bit PTQ)"].named_parameters():
        if args.target_layer in name:
            param.data.copy_(apply_block_hex_quantization(param.data))

    for name, param in tracks["Real Ternary (1.58-bit PTQ)"].named_parameters():
        if args.target_layer in name:
            param.data.copy_(apply_real_ternary(param.data))

    print("\n" + "=" * 60)
    print("EMPIRICAL COMPARATIVE EVALUATION")
    print("=" * 60)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    for track_name, model_instance in tracks.items():
        model_instance.eval()
        with torch.no_grad():
            outputs = model_instance.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n>>> [{track_name}] Output:")
        print(decoded)
        print("-" * 60)

if __name__ == "__main__":
    main()
