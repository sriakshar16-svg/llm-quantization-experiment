#!/bin/bash
# Interactive Chat Terminal for ZeroGap-BitNet VRAM-Resident Native Inference
cd /home/tsriakshar/Desktop/CODEBASES/llm-quantization-experiment || exit 1

echo "======================================================================"
echo "    INITIALIZING ZERO-GAP BITNET VRAM-RESIDENT INFERENCE ENGINE       "
echo "======================================================================"
echo "[*] Hardware Target  : RTX 3070 Ti (8GB VRAM) + 16GB RAM"
echo "[*] Engine Type      : BitNet Native PyTorch/CUDA (Zero GGUF / Zero BF16)"
echo "[*] Shards Directory : output_bitnet_deepseek_base3"
echo "======================================================================"

python3 chat_vram.py "$@"
