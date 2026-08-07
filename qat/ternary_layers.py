"""
TernaryLinear Layer with FP32 Shadow Master Weights
==================================================
Drop-in replacement for standard torch.nn.Linear modules.

Key Architecture:
  - Master weights are maintained in continuous FP32 to accumulate small gradient updates
    and satisfy PyTorch mixed-precision GradScaler requirements.
  - On the forward pass, weights are quantized on-the-fly to {-1, 0, 1} via STE and cast
    to match the activation dtype (FP16/BF16) for matrix operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ternary_ops import ste_ternary

class TernaryLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # Maintain master shadow weights in FP32
        self.weight = nn.Parameter(original_linear.weight.data.clone().float())

        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone().float())
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights on-the-fly during forward pass
        w_quantized = ste_ternary(self.weight).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_quantized, bias)
