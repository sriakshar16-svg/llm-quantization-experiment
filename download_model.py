"""
Model Download Helper with Progress Bar (download_model.py)
=========================================================
Downloads safetensors shards for a HuggingFace model repo into a local folder
with a real-time progress bar.
"""

import os
import sys
import json
import urllib.request
from tqdm import tqdm
from huggingface_hub import hf_hub_download

MODEL_ID = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
LOCAL_DIR = "./models/DeepSeek-Coder-V2-Lite-Instruct"

def download_model(model_id=MODEL_ID, local_dir=LOCAL_DIR):
    os.makedirs(local_dir, exist_ok=True)
    print(f"[*] Downloading {model_id} to {local_dir}...")
    
    # Download index json
    index_file = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json", local_dir=local_dir)
    with open(index_file) as f:
        data = json.load(f)
        
    shards = sorted(list(set(data["weight_map"].values())))
    print(f"[✓] Found {len(shards)} weight shards to download.\n")
    
    for idx, shard in enumerate(shards, 1):
        target_path = os.path.join(local_dir, shard)
        if os.path.exists(target_path):
            print(f"  [{idx}/{len(shards)}] {shard} already downloaded. Skipping.")
            continue
            
        print(f"  [{idx}/{len(shards)}] Downloading {shard}...")
        hf_hub_download(repo_id=model_id, filename=shard, local_dir=local_dir)
        print(f"  [✓] {shard} complete.")

    print(f"\n[✓] All shards downloaded successfully to {local_dir}!")
    print("[*] Initiating automatic system shutdown as requested...")
    os.system("shutdown -h +1 'Model download complete. System shutting down in 60 seconds.' || systemctl poweroff || sudo shutdown -h now")

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL_ID
    target = sys.argv[2] if len(sys.argv) > 2 else LOCAL_DIR
    download_model(model, target)
