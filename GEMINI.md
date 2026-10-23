# Quantization & Research Pipeline Invariants

Universal constraints for all compression, quantization, benchmarking, and inference tasks within this repository:

## 1. Outlier Isolation Before Scale Estimation
Never compute ternary or integer quantization scales ($\gamma$, $\text{scale}$) on tensors containing unmasked salient outliers. Detect salient coordinates (top 1.0%–1.5%), preserve them in the sparse outlier buffer (FP16), and zero them out in the bulk tensor before computing least-squares scales or running ALS coordinate descent.

## 2. VRAM-Resident Native Inference Over Lossy GGUF Re-quantization
When evaluating custom sub-8-bit representations with sparse outliers (e.g., Base-3 ternary + FP16 outliers), do not convert the model to FP16 GGUF followed by stock `llama-quantize` (`Q3_K_M`, `Q4_K_M`, `TQ1_0`). Stock integer quantizers lack sparse outlier indexing and crush high-magnitude weights, degrading reasoning and perplexity. Maintain the model in GPU VRAM in compressed `uint8` form and unpack active layers/experts on-the-fly.

## 3. MoE Active-Expert Dynamic Execution
In Mixture-of-Experts (MoE) architectures, all dormant experts must remain compressed in VRAM. Dequantization/unpacking must occur strictly on-demand for the routed + shared experts selected by the router for that token, and immediately discarded after linear projection to preserve memory bandwidth and stay within VRAM budgets.

## 4. Structured Group Outliers for Storage Efficiency
Prefer structured group-relative outlier indexing (e.g., $k=2$ outliers per 128-weight group with `uint8` offsets $\in [0, 127]$) over flat global `int32` indices. This cuts index storage from $0.48\text{ bpw}$ down to $0.125\text{ bpw}$, saving memory bandwidth and vectorizing scatter operations.

## 5. Strict Precision Truth & Format Disclosure
Never label a model as "BitNet" or "ternary" if it is running on standard integer/K-quants. Always declare the exact precision in memory (e.g., Base-3 packed `uint8`, FP16 scales, per-group FP16 outliers).

## 6. Rigorous Retention & Capability Benchmarking
Quantization verification must test three distinct tiers against uncompressed BF16 baselines:
1. Perplexity (Wikitext-2 / C4).
2. Multi-step reasoning (GSM8K, 5-shot).
3. Functional code generation (HumanEval pass@1).
Always log full telemetry (VRAM allocation, memory swap overhead, throughput tokens/sec) and pinpoint exact failure modes.
