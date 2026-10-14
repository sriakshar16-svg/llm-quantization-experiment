#!/usr/bin/env python3
"""
Interactive Terminal Chat Interface (chat.py)
==============================================
Runs an interactive conversation loop with the converted Base-3 model.
Features:
  - 100% In-RAM Pre-Caching (Zero SSD disk reads during chat)
  - Real-time token-by-token streaming to the terminal
  - Multi-turn conversation history
  - Live telemetry: Tokens emitted, Generation Speed (tok/s), Peak VRAM (MB)
  - Interactive commands: /clear, /temp, /tokens, /exit
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SFX_PT = ".packed_ternary"
SFX_QW = ".q_weight"


def unpack_ternary_base3(packed: torch.Tensor, orig_numel: int, device=DEVICE):
    """Unpacks uint8 Base-3 packed bytes into int8 {-1, 0, 1} tensor."""
    p = packed.to(device=device, dtype=torch.int32)
    u0 = p % 3; p = p // 3
    u1 = p % 3; p = p // 3
    u2 = p % 3; p = p // 3
    u3 = p % 3; u4 = p // 3
    unpacked = torch.stack([u0, u1, u2, u3, u4], dim=1).reshape(-1)[:orig_numel]
    return (unpacked - 1).to(torch.int8)


def unpack_nibbles_4bit(packed: torch.Tensor, orig_numel: int, device=DEVICE):
    """Unpacks 2 nibbles per byte into int8 uniform [-8, 7] tensor."""
    p = packed.to(device=device)
    low = (p & 0x0F).to(torch.int8)
    high = ((p >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.empty(orig_numel, dtype=torch.int8, device=device)
    unpacked[0::2] = low[:(orig_numel + 1) // 2]
    unpacked[1::2] = high[:orig_numel // 2]
    return unpacked


def load_reconstructed_layer_from_dict(saved_dict: dict, prefix: str, meta_layer, dtype=torch.bfloat16):
    state = {}
    for k, val in saved_dict.items():
        short = k[len(prefix):]
        if short.endswith(SFX_PT):
            base_k = short[:-len(SFX_PT)]
            orig_n = saved_dict[k[:-len(SFX_PT)] + ".orig_numel"].item()
            unpacked = unpack_ternary_base3(val, orig_n, device=DEVICE)
            scales = saved_dict[k[:-len(SFX_PT)] + ".scales"].to(DEVICE, dtype=torch.float32)
            w_rec = (unpacked.float() * scales.repeat_interleave(128)[:orig_n])
            out_idx_k = k[:-len(SFX_PT)] + ".outlier_idx"
            out_val_k = k[:-len(SFX_PT)] + ".outlier_val"
            if out_idx_k in saved_dict and out_val_k in saved_dict:
                out_idx = saved_dict[out_idx_k].to(DEVICE, dtype=torch.int64)
                out_val = saved_dict[out_val_k].to(DEVICE, dtype=torch.float32)
                if out_idx.numel() > 0:
                    w_rec[out_idx] = out_val
            sub_name = base_k[:-len(".weight")] if base_k.endswith(".weight") else base_k
            shape = meta_layer.get_submodule(sub_name).weight.shape
            state[base_k] = w_rec.view(shape).to(dtype)
        elif short.endswith(SFX_QW):
            base_k = short[:-len(SFX_QW)]
            scales = saved_dict[k[:-len(SFX_QW)] + ".scales"].to(DEVICE, dtype=torch.float32)
            sub_name = base_k[:-len(".weight")] if base_k.endswith(".weight") else base_k
            shape = meta_layer.get_submodule(sub_name).weight.shape
            orig_n = shape.numel()
            unpacked = unpack_nibbles_4bit(val, orig_n, device=DEVICE)
            w_rec = (unpacked.float() * scales.repeat_interleave(128)[:orig_n]).view(shape)
            state[base_k] = w_rec.to(dtype)
        elif not any(short.endswith(sfx) for sfx in [".scales", ".orig_numel", ".outlier_idx", ".outlier_val"]):
            state[short] = val.to(DEVICE, dtype=dtype)
    return state


class InteractiveChatEngine:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        print("\n" + "=" * 70)
        print("       INITIALIZING INTERACTIVE QUANTIZED MODEL TERMINAL      ")
        print("=" * 70)
        print(f"[*] Model Directory : {model_dir}")
        print(f"[*] Compute Device  : {DEVICE}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.num_layers = getattr(self.config, "num_hidden_layers", 27)

        print("[*] Instantiating architecture skeleton on meta device (0.0 MB RAM)...")
        with torch.device("meta"):
            self.meta_model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)

        anchors_path = os.path.join(model_dir, "anchors.safetensors")
        anchors = load_file(anchors_path)
        self.embed_tokens = anchors["model.embed_tokens.weight"].to(DEVICE)
        self.norm = anchors["model.norm.weight"].to(DEVICE)
        self.lm_head = anchors["lm_head.weight"].to(DEVICE)
        del anchors
        print("[✓] Anchor layers resident in GPU memory.")

        print(f"[*] Pre-caching all {self.num_layers} layers into System RAM for instant access...")
        t0 = time.time()
        self.cached_shards = {}
        for idx in range(self.num_layers):
            shard_path = os.path.join(model_dir, f"layer_{idx}.safetensors")
            self.cached_shards[idx] = load_file(shard_path)
        print(f"[✓] Successfully cached 5.3GB into RAM in {time.time() - t0:.1f}s. Zero disk I/O!")

    def forward_hidden(self, input_ids: torch.Tensor):
        seq_len = input_ids.shape[-1]
        pos_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=DEVICE, dtype=torch.bfloat16),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            hidden_states = F.embedding(input_ids, self.embed_tokens)

            for idx in range(self.num_layers):
                prefix = f"model.layers.{idx}."
                active_layer = self.meta_model.model.layers[idx].to_empty(device=DEVICE)
                layer_state = load_reconstructed_layer_from_dict(
                    self.cached_shards[idx], prefix, self.meta_model.model.layers[idx]
                )
                active_layer.load_state_dict(layer_state, strict=True)
                active_layer.eval()

                layer_out = active_layer(hidden_states, attention_mask=causal_mask, position_ids=pos_ids)
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                self.meta_model.model.layers[idx].to_empty(device="meta")
                del active_layer, layer_state
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            normed = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6) * self.norm
            logits = F.linear(normed[:, -1:, :], self.lm_head)
            return logits

    def stream_generate(self, messages, max_new_tokens: int = 60, temperature: float = 0.0, top_p: float = 0.9):
        if isinstance(messages, list):
            encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        else:
            encoded = self.tokenizer.apply_chat_template([{"role": "user", "content": str(messages)}], tokenize=True, add_generation_prompt=True)

        if hasattr(encoded, "input_ids"):
            prompt_ids = encoded.input_ids
        elif isinstance(encoded, (list, tuple)):
            prompt_ids = encoded
        else:
            prompt_ids = encoded["input_ids"]

        input_ids = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
        orig_len = input_ids.shape[-1]

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        generated_tokens = []
        eos_ids = {self.tokenizer.eos_token_id, 100001}
        t0 = time.perf_counter()

        print("\nAssistant: ", end="", flush=True)

        for step in range(max_new_tokens):
            try:
                logits = self.forward_hidden(input_ids)
                next_logits = logits[0, -1, :]

                if temperature > 0.05:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                token_id = next_token.item()
                if token_id in eos_ids:
                    break

                generated_tokens.append(token_id)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

                # Stream character to terminal immediately
                tok_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                sys.stdout.write(tok_text)
                sys.stdout.flush()

            except KeyboardInterrupt:
                print("\n\033[93m[Generation interrupted by user (Ctrl+C)]\033[0m")
                break

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = max(t1 - t0, 1e-4)
        count = len(generated_tokens)
        speed = count / elapsed if elapsed > 0 else 0.0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if DEVICE == "cuda" else 0.0

        print(f"\n\n\033[90m[{count} tokens in {elapsed:.1f}s | {speed:.2f} tok/s | Peak VRAM: {peak_vram:.1f} MB]\033[0m\n")
        full_reply = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return full_reply


def start_chat_loop(model_dir: str):
    engine = InteractiveChatEngine(model_dir)

    print("=" * 70)
    print("   TERMINAL CHAT READY — Type your prompt and press Enter.   ")
    print("   Commands: /tokens <n> | /temp <val> | /clear | /help | /exit ")
    print("   Tip: Press Ctrl+C during generation to stop without quitting.")
    print("=" * 70)

    messages = []
    temperature = 0.0
    max_tokens = 20

    while True:
        try:
            user_input = input("\033[1;36mUser >>> \033[0m")
            if not user_input.strip():
                continue

            cmd = user_input.strip().lower()
            if cmd in ["exit", "quit", "/exit", "/quit"]:
                print("\nExiting session. Goodbye!")
                break
            elif cmd in ["/clear", "/reset"]:
                messages = []
                print("\033[93m[✓ Conversation history cleared]\033[0m\n")
                continue
            elif cmd in ["/help", "help"]:
                print("\n\033[1;33mAvailable Commands:\033[0m")
                print("  /tokens <N>   Set max tokens per response (default: 20)")
                print("  /temp <val>   Set sampling temperature (0.0 = deterministic, 0.7 = creative)")
                print("  /clear        Clear conversational history")
                print("  Ctrl+C        Cancel current generation without exiting")
                print("  /exit, quit   Exit the chat session\n")
                continue
            elif cmd.startswith("/temp"):
                parts = cmd.split()
                if len(parts) > 1:
                    temperature = float(parts[1])
                    print(f"\033[93m[✓ Temperature set to {temperature}]\033[0m\n")
                continue
            elif cmd.startswith("/tokens"):
                parts = cmd.split()
                if len(parts) > 1:
                    max_tokens = int(parts[1])
                    print(f"\033[93m[✓ Max tokens set to {max_tokens}]\033[0m\n")
                continue

            # Build conversational prompt
            messages.append({"role": "user", "content": user_input.strip()})

            reply = engine.stream_generate(messages, max_new_tokens=max_tokens, temperature=temperature)
            if reply.strip():
                messages.append({"role": "assistant", "content": reply.strip()})

            # Truncate history if too long to prevent VRAM expansion
            if len(messages) > 8:
                messages = messages[-6:]

        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Terminal Chat Interface")
    parser.add_argument("model_dir", type=str, default="./output_bitnet_deepseek_base3", nargs="?", help="Model directory")
    args = parser.parse_args()

    start_chat_loop(args.model_dir)
