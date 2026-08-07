#!/bin/bash
# ==============================================================================
# BitNet CUDA Kernel Compilation Script
# ==============================================================================
# Compiles custom BitNet ternary CUDA kernels into a position-independent shared
# library (.so) for high-performance GPU-accelerated inference.
# ==============================================================================

set -e

echo "=== BitNet CUDA Kernel Build ==="

# 1. Dynamically resolve Python site-packages include headers
INCLUDES=$(python3 -c "
import site, os
paths = []
for base in [site.getusersitepackages()] + site.getsitepackages():
    if os.path.exists(base):
        for root, dirs, files in os.walk(base):
            if 'include' in dirs:
                paths.append(os.path.join(root, 'include'))
print(' '.join(['-I' + p for p in set(paths)]))
")

echo "[Info] Discovered include paths: ${INCLUDES}"

# 2. Check for nvcc in PATH or user directories
if ! command -v nvcc &> /dev/null; then
    NVCC_CANDIDATE=$(find "$HOME/.local" -name "nvcc" -type f 2>/dev/null | head -n 1 || true)
    if [ -n "$NVCC_CANDIDATE" ]; then
        NVCC_DIR=$(dirname "$NVCC_CANDIDATE")
        export PATH="$NVCC_DIR:$PATH"
        echo "[Info] Found nvcc at $NVCC_CANDIDATE and appended to PATH."
    else
        echo "[Error] nvcc not found in PATH or ~/.local. Please install CUDA toolkit." >&2
        exit 1
    fi
fi

# 3. Compile CUDA shared library
# Flags:
#   -allow-unsupported-compiler : Allows modern host GCC versions (Fedora/Ubuntu)
#   -Xcompiler -fPIC           : Generates position-independent code for shared objects
#   -O3 -std=c++17             : High optimization standard C++17
#   --gpu-architecture=sm_86   : Ampere target (RTX 30-series); change to match your GPU architecture
OUTPUT_SO="bitnet_kernels.so"

echo "[Build] Compiling bitnet_kernels.cu -> ${OUTPUT_SO}..."
nvcc -allow-unsupported-compiler \
    -Xcompiler -fPIC \
    $INCLUDES \
    -O3 -std=c++17 \
    --gpu-architecture=sm_86 \
    -shared -o "${OUTPUT_SO}" bitnet_kernels.cu

echo "[Success] Build complete: $(pwd)/${OUTPUT_SO}"
