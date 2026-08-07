"""
Layer-Level Quantization & Error Compensation Test
==================================================
Tests Channel-Wise Scaling combined with Closed-Form Activation Error Compensation
on a single Transformer attention projection layer.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM

def channel_wise_with_stable_compensation(weight_tensor: torch.Tensor, activation_sample: torch.Tensor, lam: float = 1e-4):
    """
    Applies channel-wise abs-mean ternary quantization followed by
    covariance-projected residual error compensation.
    
    Args:
        weight_tensor: Target FP32 weight matrix [out_features, in_features]
        activation_sample: Calibration activations [batch, in_features]
        lam: Regularization term for covariance inversion
    """
    W = weight_tensor
    X = activation_sample.float()

    # 1. Channel-wise scaling
    alpha = W.abs().mean(dim=1, keepdim=True)
    alpha = torch.clamp(alpha, min=1e-7)

    # 2. Quantize to ternary {-1, 0, 1}
    scaled = W / alpha
    W_q = torch.clamp(torch.round(scaled), -1.0, 1.0)

    # 3. Activation Error Compensation
    # Minimize || X(W - W_comp)^T ||_2^2 with Tikhonov regularization
    in_features = W.shape[1]
    XtX = torch.matmul(X.t(), X)
    reg = lam * torch.eye(in_features, device=X.device)
    inv_term = torch.inverse(XtX + reg)

    # Residual difference between original FP32 weights and quantized weights
    residual = W - (W_q.float() * alpha)

    # Projection correction matrix across input covariance
    scaling_correction = torch.matmul(inv_term, X.t())
    delta_w = torch.matmul(torch.matmul(residual, X), scaling_correction)

    W_compensated = (W_q.float() * alpha) + delta_w
    return W_q, alpha, W_compensated

def main():
    parser = argparse.ArgumentParser(description="Test channel-wise ternary quantization with error compensation")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolLM2-360M")
    args = parser.parse_args()

    print(f"Loading '{args.model_id}' to isolate target layer...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)

    sample_layer = model.model.layers[0].self_attn.q_proj
    W = sample_layer.weight.data.clone().float()
    print(f"Target Layer Isolated: {sample_layer} | Shape: {list(W.shape)}")

    hidden_size = W.shape[1]
    torch.manual_seed(42)
    calibration_input = torch.randn(128, hidden_size)

    W_q_cw, scale_cw, W_comp = channel_wise_with_stable_compensation(W, calibration_input)

    # Smoke test on unseen test input
    test_input = torch.randn(1, 10, hidden_size)
    with torch.no_grad():
        original_out = sample_layer(test_input)

    # Standard channel-wise (no compensation)
    W_recon_cw = W_q_cw.float() * scale_cw
    out_cw = torch.matmul(test_input, W_recon_cw.t())
    mae_cw = torch.abs(original_out - out_cw).mean().item()

    # Compensated channel-wise
    out_comp = torch.matmul(test_input, W_comp.t())
    mae_comp = torch.abs(original_out - out_comp).mean().item()

    print("\n" + "=" * 50)
    print("RECONSTRUCTION ERROR ANALYSIS (MAE)")
    print("=" * 50)
    print(f"Channel-Wise MAE (No Compensation):    {mae_cw:.4f}")
    print(f"Compensated MAE (Covariance Corrected): {mae_comp:.4f}")
    if mae_comp < mae_cw:
        print(f"[Success] Error compensation reduced reconstruction error by {((mae_cw - mae_comp)/mae_cw)*100:.2f}%.")
    else:
        print("[Note] Calibration regularization needs fine-tuning for this feature dimension.")

if __name__ == "__main__":
    main()
