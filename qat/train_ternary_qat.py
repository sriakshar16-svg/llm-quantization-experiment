"""
Quantization-Aware Training (QAT) Pipeline for Ternary LLMs
===========================================================
Replaces target Transformer linear projection layers with TernaryLinear modules
and trains with Straight-Through Estimators and continuous shadow master weights.

Key Stability Measures:
  - FP32 Shadow Weights: Prevents gradient underflow and mixed-precision scaler crashes.
  - Gradient Clipping (`max_grad_norm=1.0`): Prevents gradient explosion (>18,000 norm spikes).
  - Conservative Learning Rate (`1e-6`): Stabilizes optimization on ternary discrete surfaces.
  - Warmup Ratio (0.1): Smooths initial training trajectory.
"""

import os
import sys
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# Add parent directory to path for module resolution if run as script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from qat.ternary_layers import TernaryLinear

def convert_to_ternary_qat(module: torch.nn.Module, target_name: str = "down_proj") -> int:
    """
    Recursively replaces target linear layers with TernaryLinear modules.
    
    Args:
        module: PyTorch module (e.g. Hugging Face model).
        target_name: Substring identifying target projection layers.
    Returns:
        Count of converted layers.
    """
    count = 0
    for name, child in list(module.named_children()):
        if target_name in name and isinstance(child, torch.nn.Linear):
            setattr(module, name, TernaryLinear(child))
            count += 1
        else:
            count += convert_to_ternary_qat(child, target_name)
    return count

def main():
    parser = argparse.ArgumentParser(description="Run Quantization-Aware Training for Ternary LLMs")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model identifier")
    parser.add_argument("--save_dir", type=str, default="./saved_qat_ternary_model", help="Directory to save trained model")
    parser.add_argument("--output_dir", type=str, default="./ternary_output", help="Temporary training outputs")
    parser.add_argument("--target_proj", type=str, default="down_proj", help="Projection layers to convert to ternary")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate (use <= 1e-6 for stability)")
    args = parser.parse_args()

    print(f"[QAT] Loading '{args.model_id}' with master shadow weights...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    print(f"[QAT] Converting '{args.target_proj}' projection layers to TernaryLinear modules...")
    converted_count = convert_to_ternary_qat(model, target_name=args.target_proj)
    print(f"[QAT] Successfully converted {converted_count} projection layers.")

    # Freeze non-weight/bias parameters if selective fine-tuning
    for name, param in model.named_parameters():
        if "weight" in name or "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Sample calibration dataset (expand for production domains)
    sample_corpus = [
        "The mechanics of neural networks are built upon linear algebra and matrix multiplications.",
        "Deep learning models process data through stacked layers of weights and biases.",
        "Artificial intelligence systems rely on massive datasets to discover hidden patterns.",
        "The mathematics of the neural network training process involves gradient descent optimization."
    ] * 20

    dataset = Dataset.from_dict({"text": sample_corpus})
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=64, padding="max_length"),
        batched=True,
        remove_columns=["text"]
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,           # Prevents exploding gradients across discrete boundaries
        warmup_ratio=0.1,            # Stabilizes early optimization steps
        logging_steps=1,
        num_train_epochs=args.epochs,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported() if device == "cuda" else False,
        fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("[QAT] Starting Quantization-Aware Training with Straight-Through Estimators...")
    trainer.train()

    print(f"[QAT] Saving QAT-adapted model to '{args.save_dir}'...")
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print("[QAT] Training complete and weights saved successfully.")

if __name__ == "__main__":
    main()
