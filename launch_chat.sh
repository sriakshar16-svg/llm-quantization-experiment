#!/bin/bash
# ======================================================================
# ZeroGap-BitNet Native 2.10-BPW Ternary Launcher (CLI Chat & /v1 Server)
# ======================================================================
cd "$(dirname "$0")" || exit 1

if [ "$1" = "--serve" ]; then
    shift
    echo "[*] Starting OpenAI-Compatible /v1 Server for OpenCode & Hermes Agent..."
    exec python3 serve_bitnet_api.py "$@"
else
    echo "[*] Starting Native 2.10-BPW VRAM-Resident Interactive Terminal..."
    exec python3 chat_vram.py "$@"
fi
