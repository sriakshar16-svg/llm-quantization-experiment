#!/usr/bin/env python3
"""
download_model.py
==================
High-speed, resumable model downloader using huggingface_hub and hf_transfer.
Downloads full model repos (configs, tokenizers, safetensors shards) while ignoring
redundant legacy formats (*.bin, *.pt, *.msgpack).
"""

import argparse
import os
import sys
import time

# Enable Rust-based multi-threaded download accelerator
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download


MODELS = {
    "32b": {
        "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "local_dir": "./models/Qwen2.5-Coder-32B-Instruct",
    },
    "70b": {
        "repo_id": "Qwen/Qwen2.5-72B-Instruct",
        "local_dir": "./models/Qwen2.5-72B-Instruct",
    },
}


def download_repo(repo_id: str, local_dir: str):
    print("=" * 70)
    print(f"[*] Starting High-Speed Download: {repo_id}")
    print(f"[*] Destination: {os.path.abspath(local_dir)}")
    print(f"[*] Accelerator: hf-transfer (Rust multi-threaded streaming)")
    print("=" * 70)

    os.makedirs(local_dir, exist_ok=True)
    t0 = time.time()

    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.bin", "*.pt", "*.msgpack", "*.h5", "*.ot", "original/*"],
            max_workers=8,
        )
        elapsed = time.time() - t0
        print(f"\n[✓] Successfully downloaded {repo_id} in {elapsed:.1f}s -> {path}")
    except Exception as e:
        print(f"\n[!] Error during download of {repo_id}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="High-Speed HuggingFace Model Downloader")
    parser.add_argument("--model", type=str, choices=["32b", "70b", "both"], default="32b", help="Model to download")
    parser.add_argument("--repo_id", type=str, default=None, help="Custom HuggingFace repo ID")
    parser.add_argument("--local_dir", type=str, default=None, help="Custom destination directory")
    args = parser.parse_args()

    if args.repo_id:
        dest = args.local_dir or f"./models/{args.repo_id.split('/')[-1]}"
        download_repo(args.repo_id, dest)
        return

    if args.model in ("32b", "both"):
        m = MODELS["32b"]
        download_repo(m["repo_id"], m["local_dir"])

    if args.model in ("70b", "both"):
        m = MODELS["70b"]
        download_repo(m["repo_id"], m["local_dir"])


if __name__ == "__main__":
    main()
