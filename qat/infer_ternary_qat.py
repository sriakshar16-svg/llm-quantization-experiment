"""
Inference Script for QAT-Trained Ternary Models
==============================================
Loads the saved checkpoint trained with Quantization-Aware Training and generates
evaluative token sequences to verify linguistic coherence.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Evaluate QAT-Trained Ternary LLM Checkpoint")
    parser.add_argument("--save_dir", type=str, default="./saved_qat_ternary_model", help="Path to saved QAT checkpoint")
    parser.add_argument("--prompt", type=str, default="The mechanics of neural networks are", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Maximum generated tokens")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Inference] Loading tokenizer and model from: '{args.save_dir}' onto {device.upper()}...")

    tokenizer = AutoTokenizer.from_pretrained(args.save_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.save_dir,
        torch_dtype=torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    model.eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    print(f"[Inference] Generating text for prompt: '{args.prompt}'...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "=" * 60)
    print("QAT INFERENCE OUTPUT")
    print("=" * 60)
    print(decoded)
    print("=" * 60)

if __name__ == "__main__":
    main()
