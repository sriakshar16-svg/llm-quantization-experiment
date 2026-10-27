#!/usr/bin/env python3
"""
serve_bitnet_api.py
===================
OpenAI-Compatible /v1 API Server for ZeroGap-BitNet VRAM-Resident Engine.
Enables seamless integration with OpenCode, Hermes Agent, Continue.dev, and curl.

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions (Supports streaming SSE and standard JSON)
"""

import os
import sys
import time
import json
import uuid
import argparse
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import torch

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DONOR = str(ROOT_DIR / "models/DeepSeek-Coder-V2-Lite-Instruct")
DEFAULT_BITNET = str(ROOT_DIR / "output_bitnet_deepseek_base3")

# Global model state
MODEL = None
TOKENIZER = None


class BitNetOpenAIHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            response = {
                "object": "list",
                "data": [
                    {
                        "id": "zerogap-bitnet-2.10bpw",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "zerogap-bitnet",
                    }
                ],
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": "Endpoint not found"}')

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid JSON: {e}"}).encode("utf-8"))
                return

            self._handle_chat_completion(payload)
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": "Endpoint not found"}')

    def _handle_chat_completion(self, payload: dict):
        global MODEL, TOKENIZER
        messages = payload.get("messages", [])
        max_tokens = min(int(payload.get("max_tokens", 512)), 2048)
        temperature = float(payload.get("temperature", 0.2))
        stream = bool(payload.get("stream", False))

        # Format prompt using chat template or fallback
        if hasattr(TOKENIZER, "apply_chat_template"):
            try:
                prompt_text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant:"
        else:
            prompt_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant:"

        input_ids = TOKENIZER.encode(prompt_text, return_tensors="pt").to(MODEL.device)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._send_cors_headers()
            self.end_headers()

            curr_ids = input_ids
            with torch.inference_mode():
                for _ in range(max_tokens):
                    outputs = MODEL(curr_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    if temperature > 0:
                        probs = torch.softmax(next_token_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    token_id = next_token.item()
                    if token_id == TOKENIZER.eos_token_id:
                        break

                    chunk_text = TOKENIZER.decode([token_id], skip_special_tokens=True)
                    chunk_obj = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "zerogap-bitnet-2.10bpw",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk_obj)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    curr_ids = torch.cat([curr_ids, next_token], dim=-1)

            done_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "zerogap-bitnet-2.10bpw",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(done_chunk)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        else:
            # Non-streaming generation
            with torch.inference_mode():
                out_ids = MODEL.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=(temperature > 0),
                    temperature=max(temperature, 1e-4),
                    pad_token_id=TOKENIZER.eos_token_id,
                )
            generated_text = TOKENIZER.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()

            resp_obj = {
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": "zerogap-bitnet-2.10bpw",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": input_ids.shape[1],
                    "completion_tokens": out_ids.shape[1] - input_ids.shape[1],
                    "total_tokens": out_ids.shape[1],
                },
            }
            self.wfile.write(json.dumps(resp_obj).encode("utf-8"))


def main():
    global MODEL, TOKENIZER
    parser = argparse.ArgumentParser(description="ZeroGap-BitNet OpenAI-Compatible /v1 API Server")
    parser.add_argument("--donor_dir", type=str, default=DEFAULT_DONOR, help="Path to donor HF model")
    parser.add_argument("--bitnet_dir", type=str, default=DEFAULT_BITNET, help="Path to Base-3 compressed shards")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    print("=" * 70)
    print("  INITIALIZING ZEROGAP-BITNET OPENAI-COMPATIBLE /v1 SERVER")
    print("=" * 70)
    print(f"[*] Donor Directory  : {args.donor_dir}")
    print(f"[*] BitNet Shards    : {args.bitnet_dir}")
    print(f"[*] Binding Endpoint : http://{args.host}:{args.port}/v1")
    print("=" * 70)

    from bitnet_vram_engine import load_bitnet_vram_model
    from transformers import AutoTokenizer

    print("\n[*] Loading tokenizer...")
    TOKENIZER = AutoTokenizer.from_pretrained(args.bitnet_dir if os.path.exists(os.path.join(args.bitnet_dir, "tokenizer_config.json")) else args.donor_dir, trust_remote_code=True)

    print("[*] Loading VRAM-Resident Native Model...")
    MODEL = load_bitnet_vram_model(donor_dir=args.donor_dir, bitnet_dir=args.bitnet_dir)
    MODEL.eval()

    server = ThreadingHTTPServer((args.host, args.port), BitNetOpenAIHandler)
    print(f"\n[✓] Server is ready! Connect OpenCode, Hermes Agent, or curl:")
    print(f"    curl http://localhost:{args.port}/v1/models")
    print(f"    curl http://localhost:{args.port}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down server gracefully...")
        server.shutdown()


if __name__ == "__main__":
    main()
