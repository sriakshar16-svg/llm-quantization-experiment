#!/usr/bin/env python3
"""
CUDA / glibc 2.42+ Header Compatibility Patcher
==============================================
Resolves version conflicts between modern Linux glibc (2.42+) and CUDA headers
where `rsqrt` and `rsqrtf` declaration mismatches trigger compilation failures.
"""

import os
import glob
import sys

def patch_math_functions():
    search_patterns = [
        os.path.expanduser("~/.local/lib/python3.*/site-packages/**/math_functions.h"),
        "/usr/local/cuda*/include/crt/math_functions.h",
        os.path.expanduser("~/miniconda3/envs/*/include/crt/math_functions.h"),
    ]

    patched_count = 0
    for pattern in search_patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                modified = False
                if "double rsqrt(double x);" in content:
                    content = content.replace("double rsqrt(double x);", "double rsqrt(double x) noexcept (true);")
                    modified = True
                if "float rsqrtf(float x);" in content:
                    content = content.replace("float rsqrtf(float x);", "float rsqrtf(float x) noexcept (true);")
                    modified = True

                if modified:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"[Patched] {path}")
                    patched_count += 1
                else:
                    print(f"[Already Clean] {path}")
            except Exception as e:
                print(f"[Error processing {path}]: {e}", file=sys.stderr)

    print(f"\nCompleted CUDA header scan. Patched {patched_count} header files.")

if __name__ == "__main__":
    patch_math_functions()
