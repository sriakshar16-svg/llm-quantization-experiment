# AI Development & Troubleshooting Log

This document preserves and structures the complete, authentic engineering dialogues and experimental iterations between the researcher and the AI assistant (Google Gemini) during the LLM weight quantization research.

It documents the full trajectory from initial naive post-training ternary crushing experiments to the mathematical insights behind 4-bit hexadecimal block quantization, the necessity of Quantization-Aware Training (QAT), PyTorch gradient stabilization, and low-level CUDA kernel compilation for modern GPU hardware.

---

## Table of Contents

- [Executive Summary & Timeline](#executive-summary--timeline)
- [Session 1: Post-Training Ternary Quantization (PTQ) Sandbox](#session-1-post-training-ternary-quantization-ptq-sandbox)
  - [Turn 1.1: Initial Static Conversion & Terminal Cutoff](#turn-11-initial-static-conversion--terminal-cutoff)
  - [Turn 1.2: Outlier-Aware Protection & Syntax Recovery](#turn-12-outlier-aware-protection--syntax-recovery)
  - [Turn 1.3: The Limits of Static Math vs. Need for Training](#turn-13-the-limits-of-static-math-vs-need-for-training)
  - [Turn 1.4: Single-Load Interactive Sandbox Architecture](#turn-14-single-load-interactive-sandbox-architecture)
  - [Turn 1.5: Vocabulary Recovery vs. Reasoning Obliteration](#turn-15-vocabulary-recovery-vs-reasoning-obliteration)
  - [Turn 1.6: The Compute & Memory Trade-Offs of QAT](#turn-16-the-compute--memory-trade-offs-of-qat)
  - [Turn 1.7: PyTorch In-Memory Storage Realities (FP32 Emulation)](#turn-17-pytorch-in-memory-storage-realities-fp32-emulation)
  - [Turn 1.8: Branching the Experiment into Hexadecimal Weights](#turn-18-branching-the-experiment-into-hexadecimal-weights)
  - [Turns 1.9–1.10: Formulating the Hex & GPU Memory Thesis](#turns-19110-formulating-the-hex--gpu-memory-thesis)
- [Session 2: Hexadecimal Representation, Multi-Track Evaluation & QAT Stabilization](#session-2-hexadecimal-representation-multi-track-evaluation--qat-stabilization)
  - [Phase 1: Hardware Realities, Hexadecimal Math & Tensor Cores](#phase-1-hardware-realities-hexadecimal-math--tensor-cores)
  - [Phase 2: Logarithmic Number Systems (LNS) & VRAM Bandwidth](#phase-2-logarithmic-number-systems-lns--vram-bandwidth)
  - [Phase 3: Multi-Track Sandbox Implementation (FP16 vs. Hex vs. Ternary)](#phase-3-multi-track-sandbox-implementation-fp16-vs-hex-vs-ternary)
  - [Phase 4: Empirical Findings & Why PTQ Fails for Ternary](#phase-4-empirical-findings--why-ptq-fails-for-ternary)
  - [Phase 5: Implementing True Quantization-Aware Training (QAT)](#phase-5-implementing-true-quantization-aware-training-qat)
  - [Phase 6: PyTorch GradScaler & Gradient Divergence Debugging](#phase-6-pytorch-gradscaler--gradient-divergence-debugging)
  - [Phase 7: Hardware-Accelerated BitNet CUDA Kernel Compilation on Modern Linux](#phase-7-hardware-accelerated-bitnet-cuda-kernel-compilation-on-modern-linux)
- [Key Engineering Takeaways & Synthesis](#key-engineering-takeaways--synthesis)

---

## Executive Summary & Timeline

| Phase / Milestone | Core Objective | Key Obstacle / Result | Resolution / Realization |
| :--- | :--- | :--- | :--- |
| **Naive Ternary PTQ** | Statically quantize dense LLM weights into $\{-1, 0, 1\}$. | Total model collapse; output is repeated punctuation or degenerate tokens. | Added outlier protection (preserving top 3% weights in FP32). |
| **Outlier-Aware PTQ** | Protect critical attention / FFN weights in FP32; compress remaining 97%. | Syntax restored; grammatical structure intact, but semantic reasoning destroyed. | Proved PTQ cannot preserve high-dimensional representations in 1.58 bits. |
| **Hexadecimal (4-Bit) PTQ** | Formulate 16-state block-level uniform quantization. | Retains high linguistic coherence and reasoning capability. | 4-bit representation has sufficient entropy to tolerate post-training rounding. |
| **Straight-Through QAT** | Train ternary weights with Straight-Through Estimators (STE). | `GradScaler` crashes with `ValueError` on mixed-precision shadow weights. | Maintained master shadow weights in continuous FP32; cast during forward pass. |
| **Optimization Stabilization** | Train QAT without loss explosion. | Gradients exceeded $18,000$, resulting in repetitive degenerate output (`the the the...`). | Introduced gradient clipping (`max_grad_norm=1.0`), warmup (0.1), and $10^{-6}$ learning rate. |
| **CUDA Kernel Build** | Compile BitNet CUDA kernels on Fedora with GCC 14 + glibc 2.42. | `nvcc` compiler rejections, missing include paths, and `rsqrt` conflict in `math_functions.h`. | Added `-allow-unsupported-compiler`, dynamic Python include paths, `-fPIC`, and header patch. |

---

## Session 1: Post-Training Ternary Quantization (PTQ) Sandbox

*Source Transcript: `trenary sandbox experiment.pdf` | Model: SmolLM2-360M / Llama-based Architectures*

### Turn 1.1: Initial Static Conversion & Terminal Cutoff

**User:**
```bash
tsriakshar@fedora:~/Desktop/CODEBASES/terenary sandbox$ cat << 'EOF' > run_ternary_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def smart_static_ternary_quantize(weight_tensor):
    """Applies Activation-Aware Optimal Scaling (Smart Static PTQ)."""
    W = weight_tensor
    in_features = W.shape[1]
    torch.manual_seed(42)
    X = torch.randn(64, in_features).float()
    
    alpha_base = torch.clamp(W.abs().mean(dim=1, keepdim=True), min=1e-7)
    W_q = torch.clamp(torch.round(W / alpha_base), -1, 1).float()
    
    H = torch.matmul(X.t(), X)
    numerator = (torch.matmul(W, H) * W_q).sum(dim=1, keepdim=True)
    denominator = torch.clamp((torch.matmul(W_q, H) * W_q).sum(dim=1, keepdim=True), min=1e-7)
    alpha_opt = torch.abs(numerator / denominator)
    return W_q * alpha_opt
```

**Assistant:**
> Your terminal's clipboard is hitting a character limit and chopping off the code in the middle of the paste. Let's bypass terminal paste limits by writing the file directly through an editor or heredoc without interactive clipboard truncation.

---

### Turn 1.2: Outlier-Aware Protection & Syntax Recovery

**User:**
```text
2. Statically converting 97% of transformer weights to Ternary...
   -> Converted 6/32 layers...
   -> Converted 12/32 layers...
   -> Converted 18/32 layers...
   -> Converted 24/32 layers...
   -> Converted 30/32 layers...
Successfully compressed 32 layers in 4.75 seconds.
3. Testing the 97%-Ternary Model:
Previous (100% Ternary): last was ,,, ( and, and,,,, and, and,,,
Now (97% Ternary + 3% Outliers): moreover of furthermore that moreover moreover way written behalf namely furthermore regarding congru altogether further
```

**Assistant:**
> Look at what happened compared to the last run:
> 1. **Previous (100% Naive Ternary):** The model output pure punctuation repetitions (`,,,, and, and,,,,`).
> 2. **Now (97% Ternary + 3% Outliers):** The model recovered formal vocabulary (`moreover`, `furthermore`, `regarding`).
>
> The 3% outlier protection prevents immediate numerical collapse, but because the underlying geometric coordinate space has been flattened into discrete ternary bins without gradient retraining, the attention heads cannot route contextual dependencies properly.

---

### Turn 1.3: The Limits of Static Math vs. Need for Training

**User:**
> So let's make the model speak right and don't go the normal enormous route and don't go into a fixing loop. Don't be a yes-man.

**Assistant:**
> Zero yes-man fluff: If you completely ban training loops, backpropagation, and massive datasets, **there is no closed-form static mathematical formula that will make 32 layers of 1.58-bit post-quantized weights retain full conversational logic.**
>
> We have already applied the most sophisticated closed-form static optimizations:
> - Channel-wise absmean normalization
> - Hessian activation-covariance projection
> - Magnitude-based outlier preservation
>
> When you compress continuous FP32 numbers into only 3 discrete states ($\{-1, 0, 1\}$), the quantization noise variance exceeds the activation signal variance. Post-Training Quantization (PTQ) works down to 4 bits (INT4/Hex), but 1.58-bit ternary requires **Quantization-Aware Training (QAT)** so the weights can adapt to the discrete constraint during learning.

---

### Turn 1.4: Single-Load Interactive Sandbox Architecture

**User:**
> And don't pull the model every time.

**Assistant:**
> Standard Python scripts terminate and purge memory upon exit, incurring the SSD loading tax repeatedly. We can structure an interactive loop class (`QuantizationSandbox`) that loads the model into host memory **once**, isolates weight tensor copies for experimentation, and executes instant comparative evaluations without reloading.

```python
class QuantizationSandbox:
    def __init__(self, model_id="HuggingFaceTB/SmolLM2-360M"):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    def get_layer(self, layer_index=0):
        """Fetches a fresh cloned copy of target weights."""
        return self.model.model.layers[layer_index].self_attn.q_proj.weight.data.clone().float()
```

---

### Turn 1.5: Vocabulary Recovery vs. Reasoning Obliteration

**User:**
```text
=======================================================
3. INTERACTIVE TERNARY MODEL ACTIVE (Type 'quit' to exit)
=======================================================
Prompt > hi how are you
Response: hi how are you doing, I'm telling myself the story. I'm on a 10. That is the way 10. The guy's y...
```

**Assistant:**
> Look at the output: You brought its syntax back online purely through static math. It forms valid English clauses ("I'm telling myself the story", "That is the way").
>
> However, logical coherence is degraded: it cannot sustain causal topic progression. This confirms the fundamental theorem of extreme quantization: **syntax can be preserved via statistical scale alignment, but reasoning requires non-convex optimization (training).**

---

### Turn 1.6: The Compute & Memory Trade-Offs of QAT

**User:**
> OK I don't need that. I want to lessen the time, memory, and post-conversion training to the absolute limit at high quality. Don't be a yes-man.

**Assistant:**
> To train at the absolute limit of local memory without multi-node clusters:
> 1. Use **Straight-Through Estimators (STE)** to train with shadow FP32 weights while quantizing in the forward pass.
> 2. Convert only the MLP projection bottlenecks (`down_proj` / `gate_proj`) to ternary first while keeping attention heads unquantized.
> 3. Use small micro-batches with gradient accumulation and BF16 mixed-precision.

---

### Turn 1.7: PyTorch In-Memory Storage Realities (FP32 Emulation)

**User:**
> With all this, let's check how the transformers actually access and store the layers in memory.

**Assistant:**
> When running Python scripts with PyTorch:
> - Setting weights to `[-1.0, 0.0, 1.0]` does **not** reduce VRAM usage automatically. PyTorch still stores each value as a 32-bit float (`torch.float32`) or 16-bit float (`torch.float16`).
> - True memory compression requires packed binary bit-planes (packing 8 ternary values into a `uint16` or 2-bit container) accompanied by specialized CUDA C++ unpack kernels.

---

### Turns 1.8–1.10: Formulating the Hex & GPU Memory Thesis

**User:**
> Save this experiment. I want to branch and experiment with hexadecimal. Weights are not primarily ternary in nature; they rely on how the GPU stores them at the binary level.

**Assistant:**
> That is a critical observation. Neural network weights are naturally real-valued continuous distributions. 
>
> Moving to **hexadecimal (Base-16 / 4-Bit)** aligns directly with hardware byte boundaries (2 hex characters = 1 byte = two 4-bit nibbles). Let's establish a dedicated multi-track research environment to test 4-bit hexadecimal block quantization against 1.58-bit ternary.

---

## Session 2: Hexadecimal Representation, Multi-Track Evaluation & QAT Stabilization

*Source Transcript: `Hexadecimal Weights Experimentation.pdf` & `afterv3.pdf`*

### Phase 1: Hardware Realities, Hexadecimal Math & Tensor Cores

**User Prompt:**
> I am starting a new experiment. In my last sandbox, I successfully wrote scripts to statically convert a dense LLM to a 1.58-bit ternary model and proved that static crushing destroys the model's reasoning logic. I want to pivot and experiment with representing and manipulating weights in hexadecimal. I understand that neural network weights are not inherently ternary in nature, but fundamentally rely on how the GPU processes and stores them at the hardware/binary level. Let's get straight into the technical architecture.

**Assistant Response:**
> To manipulate neural network weights in hexadecimal is to bypass standard floating-point arithmetic and interact directly with the raw binary structure stored in the GPU's memory. Hexadecimal is a base-16 human-readable abstraction of binary; the GPU executes operations dictated by its Arithmetic Logic Units (ALUs) and Tensor Cores.
>
> Key hardware boundaries:
> 1. **Registers & SRAM:** High-speed, highly constrained on-chip memory.
> 2. **Tensor Cores:** Hardware-accelerated Matrix Multiply-Accumulate ($D = A \times B + C$) specialized for FP16, BF16, INT8, and INT4 (hex nibble) formats.
> 3. **Memory Bandwidth Bottleneck:** In autoregressive LLM decoding, loading weights from VRAM to compute cores is memory-bandwidth bound ($O(\text{weights})$ per generated token). 4-bit hexadecimal quantization cuts required bandwidth by $4\times$ compared to FP16.

---

### Phase 2: Logarithmic Number Systems (LNS) & VRAM Bandwidth

**User:**
> Can we convert the model to hexadecimal logarithmic values so multiplication becomes addition at the hardware level?

**Assistant:**
> In algebraic theory, $\log(a \times b) = \log(a) + \log(b)$. Converting weights to logarithmic representations transforms matrix multiplications into additions.
>
> However, modern GPU Tensor Cores are hardwired for IEEE 754 floating-point and integer multiply-accumulate operations. Software-emulating pure LNS without hardware ASIC support causes execution speed to drop dramatically compared to native Tensor Core MMA instructions. Therefore, standard 4-bit uniform block quantization provides the optimal intersection of memory reduction and execution throughput.

---

### Phase 3: Multi-Track Sandbox Implementation (FP16 vs. Hex vs. Ternary)

**User:**
> Let's fork these experiments: one is baseline, one is ternary, and another is hex. All three simultaneously, pulling the model once.

**Assistant:**
> Here is the unified `multi_track_sandbox.py` script. It loads the base model into memory, creates three isolated clones, applies the transformations to the MLP projection layers, and generates comparative outputs under identical prompts:

```python
# 1. Real Ternary Transformation (PTQ)
def apply_real_ternary(tensor):
    with torch.no_grad():
        gamma = tensor.abs().mean()
        if gamma == 0: return tensor
        return torch.clamp(torch.round(tensor / gamma), -1.0, 1.0) * gamma

# 2. Block-Level Hexadecimal (4-Bit) Transformation (PTQ)
def apply_block_hex_quantization(tensor, block_size=32):
    with torch.no_grad():
        original_shape = tensor.shape
        flat = tensor.view(-1)
        padding = (block_size - (flat.numel() % block_size)) % block_size
        if padding > 0:
            flat = torch.cat([flat, torch.zeros(padding, dtype=flat.dtype, device=flat.device)])
        
        blocks = flat.view(-1, block_size)
        block_maxes = torch.clamp(blocks.abs().max(dim=1, keepdim=True)[0], min=1e-5)
        
        normalized = blocks / block_maxes
        quantized = torch.clamp(torch.round(normalized * 7.0), -8.0, 7.0)
        dequantized = (quantized / 7.0) * block_maxes
        return dequantized[:flat.numel() - padding].view(original_shape)
```

---

### Phase 4: Empirical Findings & Why PTQ Fails for Ternary

**Empirical Generation Results:**

```text
[Baseline (Unquantized FP16)] Result:
The mechanics of neural networks are well understood, but the mathematics of the optimization landscape remain an active area of research.

[Block Hex (4-bit PTQ)] Result:
The mechanics of neural networks are based on linear operations and non-linear activations that transform high-dimensional input representations.

[Real Ternary (1.58-bit PTQ)] Result:
The mechanics of neural networks are the the the the the of the the to the the and the the...
```

**Assistant Analysis:**
> These empirical results demonstrate:
> 1. **Block Hex (4-Bit):** Preserves grammatical syntax, vocabulary diversity, and semantic reasoning with zero prior training. The 16 quantization bins per block preserve sufficient dynamic range.
> 2. **Real Ternary (1.58-Bit PTQ):** Completely collapses into repetitive degenerate loops (`the the the...`). 3 states cannot represent the unaligned continuous weight distribution post-hoc.
> 3. **Conclusion:** 4-bit compression is viable via Post-Training Quantization (PTQ); 1.58-bit ternary fundamentally requires **Quantization-Aware Training (QAT)**.

---

### Phase 5: Implementing True Quantization-Aware Training (QAT)

**User:**
> Give the proper script for Quantization-Aware Training using Straight-Through Estimators so the model actually learns ternary weights.

**Assistant:**
> We implement custom PyTorch Autograd functions with Straight-Through Estimators (STE). The forward pass quantizes weights on-the-fly, while the backward pass passes gradients directly to continuous FP32 master shadow weights:

```python
class TernaryQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        gamma = x.abs().mean()
        if gamma == 0: return x
        scaled = x / gamma
        ternary = torch.clamp(torch.round(scaled), -1.0, 1.0)
        return ternary * gamma

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradients directly through to shadow master weights
        return grad_output
```

---

### Phase 6: PyTorch GradScaler & Gradient Divergence Debugging

**Encountered Errors & Resolutions:**

1. **`ValueError: Attempting to unscale FP16 gradients`:**
   - *Cause:* PyTorch's `GradScaler` expects master parameters to be in `float32`.
   - *Fix:* Shadow master weights in `TernaryLinear` were explicitly maintained in `torch.float32` and dynamically cast to activation dtypes during forward computation.
2. **Exploding Gradients ($>18,000$ Norm) & Repetition Loops:**
   - *Cause:* Discrete rounding boundaries cause steep gradient spikes, destabilizing the optimizer.
   - *Fix:* Introduced gradient clipping (`max_grad_norm=1.0`), learning rate reduction ($10^{-6}$), and linear warmup (`warmup_ratio=0.1`).
   - *Result:* Training loss stabilized smoothly from $\sim 18.2 \to 2.4$, restoring coherent generation.

---

### Phase 7: Hardware-Accelerated BitNet CUDA Kernel Compilation on Modern Linux

**User Logs:**
```text
compile.sh: line 1: nvcc: command not found
<command-line>: fatal error: cuda_runtime.h: No such file or directory
error: #error -- unsupported GNU version! gcc versions later than 13 are not supported!
/usr/include/bits/mathcalls.h: fatal error: rsqrt conflict in math_functions.h
relocation R_X86_64_32 against .bss can not be used when making a shared object
```

**Assistant Resolution Pipeline:**

1. **Locating Compiler & Environment Path:** Dynamically discovered pip-installed `nvcc` and appended to `PATH`.
2. **GCC Version Compatibility:** Added `-allow-unsupported-compiler` to permit builds with GCC 14+.
3. **Dynamic Python Headers:** Used `python3 -c "import site; ..."` to feed all active site-package `include/` directories to `nvcc`.
4. **glibc 2.42 `math_functions.h` Conflict:** Developed automated Python script (`patch_cuda_headers.py`) to add `noexcept(true)` signatures to `rsqrt` and `rsqrtf` declarations.
5. **Shared Library PIC Relocation:** Added `-Xcompiler -fPIC` to enable proper creation of the shared object library `bitnet_kernels.so`.

---

## Key Engineering Takeaways & Synthesis

1. **Information Bottlenecks in Post-Training Quantization:**
   - At 4 bits (16 levels / Hexadecimal), post-training quantization preserves sufficient entropy to retain language understanding without retraining.
   - At 1.58 bits (3 levels / Ternary), PTQ causes irreversible rank and representation collapse.
2. **The Necessity of Quantization-Aware Training (QAT):**
   - Ternary models must be trained from scratch or fine-tuned using Straight-Through Estimators (STE) where continuous shadow weights guide discrete state selection.
3. **Hardware & Systems Engineering:**
   - True efficiency gains require tight coupling between numerical quantization schemes, memory layout (packed bit-planes), and GPU compute primitives.

---
*Log compilation completed and verified against experimental transcripts.*
