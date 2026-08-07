# LLM Weight Quantization Sandbox: Hexadecimal (4-Bit) vs. Ternary (1.58-Bit) PTQ & QAT

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA Compatible](https://img.shields.io/badge/CUDA-12.0%20%7C%2013.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)

An experimental investigation into **Post-Training Quantization (PTQ)** and **Quantization-Aware Training (QAT)** across different numerical representations for Large Language Models (LLMs). This project evaluates the mathematical and empirical boundaries between **4-bit Hexadecimal (Base-16)** compression and **1.58-bit Ternary ($\{-1, 0, 1\}$)** weights on dense transformer architectures (such as `SmolLM2-360M` and `Qwen2.5-0.5B`).

---

## Table of Contents

- [Core Hypotheses & Research Objectives](#core-hypotheses--research-objectives)
- [Key Findings & Conclusions](#key-findings--conclusions)
- [Empirical Comparative Results](#empirical-comparative-results)
- [Mathematical & Architectural Formulations](#mathematical--architectural-formulations)
  - [1. Block-Level Hexadecimal (4-Bit) Quantization](#1-block-level-hexadecimal-4-bit-quantization)
  - [2. Naive vs. Outlier-Aware Ternary PTQ](#2-naive-vs-outlier-aware-ternary-ptq)
  - [3. Quantization-Aware Training (QAT) with STE](#3-quantization-aware-training-qat-with-ste)
- [Repository Structure](#repository-structure)
- [Installation & Quickstart](#installation--quickstart)
  - [1. Multi-Track Comparison Sandbox](#1-multi-track-comparison-sandbox)
  - [2. Outlier-Aware PTQ Console](#2-outlier-aware-ptq-console)
  - [3. Quantization-Aware Training (QAT)](#3-quantization-aware-training-qat)
  - [4. CUDA Kernel Compilation & Headers Patching](#4-cuda-kernel-compilation--headers-patching)
- [AI Research Transparency & Development Log](#ai-research-transparency--development-log)
- [Citation & License](#citation--license)

---

## Core Hypotheses & Research Objectives

1. **Hardware-Aligned Hexadecimal Compression:** Hexadecimal representation directly maps to 4-bit nibbles (2 hex characters per byte), fitting modern GPU tensor processing units and memory architectures. The goal is to measure whether post-training 4-bit block quantization preserves full semantic reasoning without fine-tuning.
2. **Post-Training Ternary "Crushing" Limits:** Explore whether a pre-trained continuous FP32/FP16 dense LLM can be statically compressed down to 1.58-bit ternary states ($\{-1, 0, 1\}$) using closed-form mathematics (abs-mean scaling, Hessian covariance, outlier preservation, and channel-wise compensation).
3. **QAT Requirement for Ternary Models:** Demonstrate why ternary models (e.g. BitNet b1.58) cannot be created via simple post-training conversion and instead require Quantization-Aware Training with Straight-Through Estimators (STE) and continuous shadow master weights.

---

## Key Findings & Conclusions

| Method | Bit-Width / State Count | Training Required? | Semantic Coherence | Hardware & Memory Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (FP16/FP32)** | 16 / 32 bits | N/A (Original) | 100% (Reference) | Standard baseline VRAM usage. |
| **Block Hexadecimal (PTQ)** | 4 bits (16 levels) | **No** (Zero-shot PTQ) | **High (Coherent)** | $\sim 75\%$ VRAM reduction; fast execution. |
| **Outlier-Aware Ternary (PTQ)** | 1.58 bits ($\{-1, 0, 1\}$) | **No** (Static conversion) | **Degenerate / Gibberish** | Memory reduced, but reasoning completely collapses. |
| **Ternary QAT (STE)** | 1.58 bits + FP32 Shadow | **Yes** (Fine-tuned / QAT) | **Recovered / Fluent** | True 1.58-bit compute potential with stable language modeling. |

### Summary of Conclusions:
- **Hexadecimal (4-Bit) Post-Training Quantization Works:** 16 discrete states per block provide sufficient entropy to preserve the complex geometric coordinate space of pre-trained attention and feed-forward layers.
- **Static Ternary "Crushing" Fails:** Attempting to force an already-trained dense model into 3 states without gradient retraining inevitably flattens activation manifolds, causing the model to output repetitive loops or degenerate tokens. Even protecting the top 3% outlier weights or boundary layers only restores surface syntax, not deep reasoning.
- **Ternary Requires QAT:** 1.58-bit architectures must learn to navigate discrete ternary surfaces during optimization via Straight-Through Estimators (STE) with continuous master shadow weights.

---

## Empirical Comparative Results

Tested on `Qwen2.5-0.5B` and `SmolLM2-360M` under identical input prompts:

```text
Prompt: "The mechanics of neural networks are"

[Baseline (Unquantized FP16)]
"The mechanics of neural networks are well understood, but the mathematics of the optimization landscape remain an active area of research..."

[Block Hex (4-Bit PTQ)]
"The mechanics of neural networks are based on linear operations and non-linear activations that transform high-dimensional input representations..."

[Real Ternary (1.58-Bit Naive PTQ)]
",,, ( and, and,,,, and, and,,, ( and, and"

[Outlier-Aware Ternary (97% Ternary + 3% FP32 Outliers)]
"moreover of furthermore that moreover moreover way written behalf namely furthermore regarding congru altogether further..."

[Ternary QAT (Fine-Tuned with STE Shadow Weights)]
"The mechanics of neural networks are built upon linear algebra and matrix multiplications processed through stacked layers of weights..."
```

---

## Mathematical & Architectural Formulations

### 1. Block-Level Hexadecimal (4-Bit) Quantization

Weights are divided into contiguous blocks $\mathbf{w}_b \in \mathbb{R}^B$ (e.g., $B=32$). Each block is dynamically scaled by its maximum magnitude:

$$\alpha_b = \max_{j} |w_{b, j}|, \quad q_{b, j} = \text{clamp}\left(\text{round}\left( \frac{w_{b, j}}{\alpha_b} \times 7.0 \right), -8.0, 7.0\right)$$

$$\hat{w}_{b, j} = \frac{q_{b, j}}{7.0} \times \alpha_b$$

### 2. Naive vs. Outlier-Aware Ternary PTQ

In naive ternary quantization, the weight tensor $W$ is scaled by its mean absolute value $\gamma = \frac{1}{N}\sum |w_i|$:

$$W_q = \text{clamp}\left(\text{round}\left(\frac{W}{\gamma}\right), -1, 1\right) \times \gamma$$

In outlier-aware ternary quantization, a threshold $\tau$ preserves the top $k\%$ magnitude weights in FP32, while quantizing the remaining bulk:

$$W_{\text{bulk}} = W \odot (\mathbb{I}_{|W| < \tau}), \quad \gamma_{\text{bulk}} = \text{mean}(|W_{\text{bulk}}|)$$

$$\hat{W} = (W \odot \mathbb{I}_{|W| \ge \tau}) + \left(\text{clamp}\left(\text{round}\left(\frac{W_{\text{bulk}}}{\gamma_{\text{bulk}}}\right), -1, 1\right) \times \gamma_{\text{bulk}}\right)$$

### 3. Quantization-Aware Training (QAT) with STE

To optimize over non-differentiable discrete rounding, we maintain continuous shadow weights $\mathcal{W}_{\text{shadow}} \in \mathbb{R}$ in FP32 and apply a Straight-Through Estimator:

$$\text{Forward:} \quad W_{\text{quant}} = \gamma \cdot \text{clamp}\left(\text{round}\left(\frac{\mathcal{W}_{\text{shadow}}}{\gamma}\right), -1, 1\right)$$

$$\text{Backward:} \quad \frac{\partial \mathcal{L}}{\partial \mathcal{W}_{\text{shadow}}} = \frac{\partial \mathcal{L}}{\partial W_{\text{quant}}}$$

---

## Repository Structure

```
llm-quantization-experiment/
├── README.md                     # Project documentation and empirical findings
├── AI_Development_Log.md         # Full transparent dialogue & troubleshooting logs
├── requirements.txt              # Environment dependencies
├── .gitignore                    # Git tracking rules
├── sandbox/
│   ├── multi_track_sandbox.py    # Comparative evaluation (Baseline vs Hex vs Ternary)
│   ├── run_ternary_ptq.py        # Outlier-aware PTQ with interactive generation console
│   ├── sandbox_master.py         # Single-load fast testbench for layer scaling math
│   └── test_layer_compensation.py# Channel-wise error compensation & covariance analysis
├── qat/
│   ├── ternary_ops.py            # Autograd Straight-Through Estimator (STE)
│   ├── ternary_layers.py         # TernaryLinear drop-in module with FP32 shadow weights
│   ├── train_ternary_qat.py      # QAT training loop with gradient clipping & warmup
│   └── infer_ternary_qat.py      # Inference generator for saved QAT models
└── cuda/
    ├── compile_bitnet_kernel.sh  # CUDA build script with GCC 14+ compatibility flags
    └── patch_cuda_headers.py     # Automated patch for glibc 2.42 math_functions.h
```

---

## Installation & Quickstart

### Prerequisites

```bash
git clone https://github.com/<your-username>/llm-quantization-experiment.git
cd llm-quantization-experiment
pip install -r requirements.txt
```

### 1. Multi-Track Comparison Sandbox

Compare Baseline, 4-bit Hex, and 1.58-bit Ternary on the same model instance:

```bash
python sandbox/multi_track_sandbox.py --model_id "Qwen/Qwen2.5-0.5B" --prompt "The mechanics of neural networks are"
```

### 2. Outlier-Aware PTQ Console

Run interactive generation with outlier-protected ternary conversion:

```bash
python sandbox/run_ternary_ptq.py --model_id "HuggingFaceTB/SmolLM2-360M" --outlier_fraction 0.03
```

### 3. Quantization-Aware Training (QAT)

Train a ternary model with master shadow weights and gradient stabilization:

```bash
python qat/train_ternary_qat.py --model_id "Qwen/Qwen2.5-0.5B" --epochs 3 --learning_rate 1e-6
python qat/infer_ternary_qat.py --save_dir "./saved_qat_ternary_model" --prompt "The mechanics of neural networks are"
```

### 4. CUDA Kernel Compilation & Headers Patching

To build custom CUDA kernels on modern Linux systems (Fedora / Ubuntu with GCC 14+ and glibc 2.42+):

```bash
python cuda/patch_cuda_headers.py
bash cuda/compile_bitnet_kernel.sh
```

---

## AI Research Transparency & Development Log

In the spirit of open and reproducible science, this repository openly shares the full iterative design and troubleshooting history between the researcher and the AI assistant (Google Gemini).

The complete, unedited transcripts covering:
- Early syntax collapse and outlier scaling iterations,
- Theoretical debates on GPU Tensor Core and Logarithmic Number System (LNS) architectures,
- Resolving PyTorch `GradScaler` mixed-precision exceptions and gradient divergence ($>18,000$ norm),
- Linux GCC / glibc / CUDA header compilation fixes,

are documented in detail in [AI_Development_Log.md](AI_Development_Log.md).

---

## License

This project is open source and available under the [MIT License](LICENSE).
