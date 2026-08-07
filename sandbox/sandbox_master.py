"""
Single-Load Quantization Testbench
=================================
Loads a base model into memory once and allows fast iterative testing of
different mathematical scaling formulations on a single target layer without
reloading the full model from disk.
"""

import time
import argparse
import torch
from transformers import AutoModelForCausalLM

class QuantizationSandbox:
    def __init__(self, model_id="HuggingFaceTB/SmolLM2-360M"):
        print(f"--- INITIALIZING SANDBOX ---")
        print(f"Loading '{model_id}' into host memory (occurs once)...")
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds.\n")

    def get_layer_weights(self, layer_index=0):
        """Returns a cloned copy of the layer weight tensor to prevent in-place corruption."""
        return self.model.model.layers[layer_index].self_attn.q_proj.weight.data.clone().float()

    def get_module(self, layer_index=0):
        """Returns the actual PyTorch module for forward pass verification."""
        return self.model.model.layers[layer_index].self_attn.q_proj

def run_dumb_baseline(W: torch.Tensor, layer_module: torch.nn.Module, dummy_input: torch.Tensor):
    """Experiment 1: Naive Abs-Mean Global Scaling to {-1, 0, 1}."""
    print("Running Experiment 1: Naive Baseline (Abs-Mean Global Scaling)")
    alpha = W.abs().mean()
    W_q = torch.clamp(torch.round(W / alpha), -1, 1)

    with torch.no_grad():
        orig_out = layer_module(dummy_input)
        new_out = torch.matmul(dummy_input, (W_q * alpha).t())
        mae = torch.abs(orig_out - new_out).mean().item()
    print(f"  -> Mean Absolute Error (MAE): {mae:.4f}\n")

def run_smart_awq_style(W: torch.Tensor, layer_module: torch.nn.Module, dummy_input: torch.Tensor):
    """Experiment 2: Activation-Aware Hessian Scaling."""
    print("Running Experiment 2: Activation-Aware Hessian Scaling (AWQ-inspired)")
    X = dummy_input.float().view(-1, W.shape[1])

    # Per-channel base quantization
    alpha_base = torch.clamp(W.abs().mean(dim=1, keepdim=True), min=1e-7)
    W_q = torch.clamp(torch.round(W / alpha_base), -1, 1).float()

    # Hessian covariance calculation
    H = torch.matmul(X.t(), X)
    numerator = (torch.matmul(W, H) * W_q).sum(dim=1, keepdim=True)
    denominator = torch.clamp((torch.matmul(W_q, H) * W_q).sum(dim=1, keepdim=True), min=1e-7)

    alpha_opt = torch.abs(numerator / denominator)

    with torch.no_grad():
        orig_out = layer_module(dummy_input)
        new_out = torch.matmul(dummy_input, (W_q * alpha_opt).t())
        mae = torch.abs(orig_out - new_out).mean().item()
    print(f"  -> Mean Absolute Error (MAE): {mae:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description="Fast single-load quantization testbench")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--layer_idx", type=int, default=0)
    args = parser.parse_args()

    sandbox = QuantizationSandbox(model_id=args.model_id)
    W = sandbox.get_layer_weights(layer_index=args.layer_idx)
    module = sandbox.get_module(layer_index=args.layer_idx)

    torch.manual_seed(42)
    dummy_input = torch.randn(1, 128, W.shape[1])  # [Batch, Seq, Hidden]

    run_dumb_baseline(W, module, dummy_input)
    run_smart_awq_style(W, module, dummy_input)

if __name__ == "__main__":
    main()
