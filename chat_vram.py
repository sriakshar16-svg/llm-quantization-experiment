#!/usr/bin/env python3
"""
chat_vram.py
=============
Interactive CLI chat & real-time telemetry benchmark running compressed Base-3 shards
directly on GPU VRAM (RTX 3070 Ti, 8GB).
Zero 31.4 GB BF16 expansion. Zero GGUF double-quantization loss.
"""

import argparse
import sys
import time
import torch
from transformers import AutoTokenizer

from bitnet_vram_engine import load_bitnet_vram_model

DEFAULT_DONOR = "/home/tsriakshar/Desktop/CODEBASES/llm-quantization-experiment/models/DeepSeek-Coder-V2-Lite-Instruct"
DEFAULT_BITNET = "/home/tsriakshar/Desktop/CODEBASES/llm-quantization-experiment/output_bitnet_deepseek_base3"


def parse_args():
    parser = argparse.ArgumentParser(description="BitNet VRAM-Resident Native Inference Terminal")
    parser.add_argument("--donor_dir", type=str, default=DEFAULT_DONOR, help="Path to unquantized donor model")
    parser.add_argument("--bitnet_dir", type=str, default=DEFAULT_BITNET, help="Path to Base-3 compressed shards")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--bench", action="store_true", help="Run automated benchmark prompt and exit")
    parser.add_argument("--bench_prompt", type=str, default="Write a concise Python function to implement binary search.", help="Benchmark prompt")
    parser.add_argument("--offload_embed", action="store_true", help="Offload embedding table to CPU RAM")
    return parser.parse_args()


def run_benchmark(model, tokenizer, prompt: str, max_tokens: int = 128, temperature: float = 0.2, top_p: float = 0.9):
    print("\n" + "=" * 70)
    print("                RUNNING HARDWARE TELEMETRY BENCHMARK                  ")
    print("=" * 70)
    messages = [
        {"role": "user", "content": prompt}
    ]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if isinstance(encoded, torch.Tensor):
        prompt_ids = encoded.to("cuda")
    elif hasattr(encoded, "input_ids"):
        prompt_ids = encoded.input_ids.to("cuda")
    elif isinstance(encoded, dict):
        prompt_ids = encoded["input_ids"].to("cuda")
    else:
        prompt_ids = torch.tensor(encoded, device="cuda").unsqueeze(0)

    prompt_len = prompt_ids.shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = output_ids.shape[1] - prompt_len
    tps = new_tokens / max(elapsed, 1e-5)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    current_vram = torch.cuda.memory_allocated() / (1024 ** 3)

    reply = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

    print(f"\n[Prompt]: {prompt}\n")
    print(f"[Generated Response]:\n{reply}\n")
    print("-" * 70)
    print(f"[*] Tokens Generated : {new_tokens}")
    print(f"[*] Elapsed Time     : {elapsed:.2f} s")
    print(f"[*] Generation Speed : {tps:.2f} tokens/sec")
    print(f"[*] GPU Memory Active: {current_vram:.2f} GB")
    print(f"[*] GPU Memory Peak  : {peak_vram:.2f} GB / 8.0 GB (RTX 3070 Ti)")
    print("=" * 70 + "\n")
    return tps, peak_vram


def interactive_chat(model, tokenizer, max_tokens: int = 256, temperature: float = 0.2, top_p: float = 0.9):
    print("\n======================================================================")
    print("       ZERO-GAP BITNET VRAM-RESIDENT INTERACTIVE TERMINAL             ")
    print("======================================================================")
    print("[*] Engine     : BitNet Native PyTorch/CUDA VRAM Engine (Base-3 + Outliers)")
    print("[*] Compression: 1.60 bpw Base-3 uint8 + Group Scales + FP16 Outliers")
    print("[*] Commands   : Type 'exit', 'quit', or '/clear'")
    print("======================================================================\n")

    messages = [
        {"role": "system", "content": "You are DeepSeek-Coder, an intelligent coding and technical reasoning assistant."}
    ]

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": "You are DeepSeek-Coder, an intelligent coding and technical reasoning assistant."}]
            print("[✓] Chat history cleared.")
            continue

        messages.append({"role": "user", "content": user_input})
        encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if isinstance(encoded, torch.Tensor):
            prompt_ids = encoded.to("cuda")
        elif hasattr(encoded, "input_ids"):
            prompt_ids = encoded.input_ids.to("cuda")
        elif isinstance(encoded, dict):
            prompt_ids = encoded["input_ids"].to("cuda")
        else:
            prompt_ids = torch.tensor(encoded, device="cuda").unsqueeze(0)
        prompt_len = prompt_ids.shape[1]

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        new_tokens = output_ids.shape[1] - prompt_len
        tps = new_tokens / max(elapsed, 1e-5)
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reply = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

        print(f"\nAssistant > {reply}\n")
        print(f"[Telemetry: {new_tokens} tokens | {tps:.2f} tok/s | Peak VRAM: {peak_vram:.2f} GB]")
        messages.append({"role": "assistant", "content": reply})


def main():
    args = parse_args()
    print(f"[*] Initializing Tokenizer from {args.donor_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.donor_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_bitnet_vram_model(
        donor_dir=args.donor_dir,
        bitnet_dir=args.bitnet_dir,
        device="cuda",
        offload_embed_to_cpu=args.offload_embed,
    )

    if args.bench:
        run_benchmark(
            model,
            tokenizer,
            prompt=args.bench_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        interactive_chat(
            model,
            tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
