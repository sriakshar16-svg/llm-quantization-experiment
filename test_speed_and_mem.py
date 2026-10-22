import os
import sys
import time
import psutil
import torch
from transformers import AutoTokenizer, TextStreamer

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from bitnet_vram_engine import load_bitnet_vram_model

DONOR_DIR = "models/Qwen2.5-Coder-32B-Instruct"
BITNET_DIR = "output_bitnet_qwen_32b"

print("=" * 70)
print("TESTING 32B VRAM ENGINE OPTIMIZATIONS")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(DONOR_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test loading with 40 GPU layers
model = load_bitnet_vram_model(
    donor_dir=DONOR_DIR,
    bitnet_dir=BITNET_DIR,
    device="cuda",
    offload_embed_to_cpu=True,
    gpu_layers=40,
)

vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
vram_res = torch.cuda.memory_reserved() / (1024 ** 3)
host_rss = psutil.Process().memory_info().rss / (1024 ** 3)
print(f"\n[*] Post-Load: Active VRAM: {vram_alloc:.2f} GB | Reserved: {vram_res:.2f} GB | Host RSS: {host_rss:.2f} GB")

prompt = "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number efficiently.\"\"\"\n"
inputs = tokenizer(prompt, return_tensors="pt")
prompt_ids = inputs["input_ids"].to("cuda")

streamer = TextStreamer(tokenizer, skip_prompt=False)

print("\n[*] Starting generation test (35 tokens with live streamer)...")
t0 = time.perf_counter()
with torch.inference_mode():
    output = model.generate(
        prompt_ids,
        max_new_tokens=35,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
new_tokens = output.shape[1] - prompt_ids.shape[1]
tps = new_tokens / elapsed
peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

print("\n" + "=" * 70)
print(f"[*] Generated: {new_tokens} tokens in {elapsed:.2f}s ({tps:.2f} tok/s)")
print(f"[*] Peak VRAM: {peak_vram:.2f} GB / 8.0 GB")
print(f"[*] Current VRAM: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
print(f"[*] Host RSS: {psutil.Process().memory_info().rss / (1024 ** 3):.2f} GB")
print("=" * 70)
