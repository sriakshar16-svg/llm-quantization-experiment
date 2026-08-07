"""
Autograd Straight-Through Estimator (STE) for Ternary Quantization
==================================================================
Implements custom PyTorch Autograd Function for 1.58-bit ternary quantization.

Forward:
  1. Computes mean absolute scaling factor (gamma)
  2. Scales, rounds, and clamps weights into {-1.0, 0.0, 1.0}
  3. Re-scales by gamma

Backward:
  Straight-Through Estimator (STE) passes upstream gradients directly through
  to the continuous shadow master weights, bypassing non-differentiable step functions.
"""

import torch

class TernaryQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # 1. Compute abs-mean scaling factor (gamma)
        gamma = x.abs().mean()
        if gamma == 0:
            return x

        # 2. Scale, round, and clamp to ternary grid {-1.0, 0.0, 1.0}
        scaled = x / gamma
        ternary = torch.clamp(torch.round(scaled), -1.0, 1.0)

        # 3. Return ternary weights scaled back up by gamma
        return ternary * gamma

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-Through Estimator (STE): pass gradients straight to shadow weights
        return grad_output

def ste_ternary(x: torch.Tensor) -> torch.Tensor:
    """Wraps the TernaryQuantizerFunction for convenient module use."""
    return TernaryQuantizerFunction.apply(x)
