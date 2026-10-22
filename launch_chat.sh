#!/bin/bash
# ======================================================================
# Interactive Chat Terminal for BitNet GGUF Native Inference
# Primary Engine: Compiled llama.cpp C++ Runtime (GGUF Default)
# Hardware Target: RTX 3070 Ti (8GB VRAM) + 16GB System RAM
# ======================================================================
cd /home/tsriakshar/Desktop/CODEBASES/llm-quantization-experiment || exit 1

LLAMA_CLI="/home/tsriakshar/.gemini/antigravity/scratch/BitNet/build/bin/llama-cli"
if [ ! -f "$LLAMA_CLI" ]; then
    LLAMA_CLI=$(which llama-cli 2>/dev/null)
fi

# Detect available GGUF model
MODEL_PATH=""
for candidate in \
    "./qwen2.5_coder_32b_bitnet_q3_k_m.gguf" \
    "./qwen2.5_coder_32b_bitnet_q4_k_m.gguf" \
    "./qwen2.5_coder_32b_bitnet_q2_k.gguf" \
    "./deepseek_v2_lite_bitnet_q3_k_m.gguf" \
    "./deepseek_v2_lite_q3_k_m.gguf" \
    "./qwen2.5_coder_32b_bitnet_tq1_0.gguf" \
    ./*.gguf; do
    if [ -f "$candidate" ]; then
        MODEL_PATH="$candidate"
        break
    fi
done

if [ -n "$MODEL_PATH" ] && [ -x "$LLAMA_CLI" ]; then
    echo "======================================================================"
    echo "       INITIALIZING BITNET HIGH-SPEED GGUF INFERENCE ENGINE           "
    echo "======================================================================"
    echo "[*] Engine Type   : llama.cpp C++ Native Runtime (GGUF Default)"
    echo "[*] Active Model  : $MODEL_PATH"
    echo "[*] Context Size  : 2048 tokens (Protected KV Buffer < 512 MB)"
    echo "[*] CPU Threads   : 12 Threads"
    echo "======================================================================"
    echo ""
    exec "$LLAMA_CLI" \
        -m "$MODEL_PATH" \
        -c 2048 \
        --no-warmup \
        -t 12 \
        --temp 0.2 \
        -cnv \
        "$@"
else
    echo "======================================================================"
    echo "   [!] Notice: GGUF engine or model not detected. Falling back to     "
    echo "       Python VRAM-resident engine...                                "
    echo "======================================================================"
    python3 chat_vram.py "$@"
fi
