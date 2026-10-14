#!/bin/bash
# Interactive Chat Terminal for BitNet DeepSeek Model
BIN_DIR="/home/tsriakshar/.gemini/antigravity/scratch/BitNet/build/bin"
export LD_LIBRARY_PATH="$BIN_DIR"

MODEL_PATH="/home/tsriakshar/Desktop/CODEBASES/llm-quantization-experiment/deepseek_v2_lite_bitnet_q3_k_m.gguf"

echo "======================================================================"
echo "       INITIALIZING INTERACTIVE BITNET TERNARY MODEL TERMINAL          "
echo "======================================================================"
echo "[*] Model Path : $MODEL_PATH"
echo "[*] Engine     : compiled llama-cli (BitNet GGUF C++ Runtime)"
echo "[*] Commands   : Type '/clear' to reset chat, 'Ctrl+C' or '/exit' to quit"
echo "======================================================================"

"$BIN_DIR/llama-cli" \
  -m "$MODEL_PATH" \
  -c 4096 \
  -t 8 \
  --temp 0.2 \
  --repeat-penalty 1.2 \
  --chat-template deepseek2 \
  -sys "You are DeepSeek-Coder, an intelligent coding and technical reasoning assistant." \
  -cnv
