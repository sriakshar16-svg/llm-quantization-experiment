#!/usr/bin/env python3
"""
check_download_status.py
========================
Inspects models/Qwen2.5-Coder-32B-Instruct and its .cache directory to report:
- Completed files and sizes
- Incomplete shard streams and sizes
- Total downloaded GB vs expected ~65.5 GB
- Progress percentage
"""

import os
import glob

MODEL_DIR = "./models/Qwen2.5-Coder-32B-Instruct"
TOTAL_EXPECTED_BYTES = 65.5 * 1024 * 1024 * 1024  # ~65.5 GB for Qwen2.5-Coder-32B-Instruct

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:6.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def main():
    if not os.path.exists(MODEL_DIR):
        print(f"[!] Directory not found: {MODEL_DIR}")
        return

    # Check root completed files
    completed_files = []
    completed_bytes = 0
    for root, dirs, files in os.walk(MODEL_DIR):
        if ".cache" in root:
            continue
        for f in files:
            path = os.path.join(root, f)
            sz = os.path.getsize(path)
            completed_bytes += sz
            completed_files.append((f, sz))

    # Check cache incomplete files
    cache_dir = os.path.join(MODEL_DIR, ".cache", "huggingface", "download")
    cache_files = []
    cache_bytes = 0
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            if f.endswith(".incomplete"):
                path = os.path.join(cache_dir, f)
                sz = os.path.getsize(path)
                cache_bytes += sz
                cache_files.append((f, sz))

    total_downloaded_bytes = completed_bytes + cache_bytes
    pct = (total_downloaded_bytes / TOTAL_EXPECTED_BYTES) * 100.0

    print("=" * 70)
    print("   QWEN 2.5 CODER 32B INSTRUCT - DOWNLOAD PROGRESS STATUS")
    print("=" * 70)
    print(f"[*] Target Directory  : {MODEL_DIR}")
    print(f"[*] Completed Shards  : {len([f for f, sz in completed_files if f.endswith('.safetensors')])} / 14")
    print(f"[*] Total Downloaded  : {format_size(total_downloaded_bytes)} / ~65.50 GB ({pct:.1f}%)")
    print("-" * 70)
    
    if completed_files:
        print("[✓] Completed Files:")
        for f, sz in sorted(completed_files):
            print(f"    - {f:<40} : {format_size(sz)}")

    if cache_files:
        print(f"\n[*] Active Incomplete Streams ({len(cache_files)} streams):")
        for f, sz in sorted(cache_files, key=lambda x: x[1], reverse=True)[:8]:
            short_name = f[:35] + "..." if len(f) > 38 else f
            print(f"    - {short_name:<40} : {format_size(sz)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
