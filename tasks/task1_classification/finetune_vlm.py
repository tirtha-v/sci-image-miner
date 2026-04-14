#!/usr/bin/env python3
"""CLI entry point for VLM QLoRA finetuning.

Usage:
    python tasks/task1_classification/finetune_vlm.py
    python tasks/task1_classification/finetune_vlm.py --epochs 5 --batch-size 2 --device cuda:1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.vlm_finetune.train_qlora import train_qlora


def main():
    parser = argparse.ArgumentParser(description="QLoRA finetune LLaVA for chart classification")
    parser.add_argument("--train-csv", default="data/train_panels.csv")
    parser.add_argument("--dev-csv", default="data/dev_panels.csv")
    parser.add_argument("--output-dir", default="outputs/vlm_finetune/llama_qlora")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-target-modules", nargs="+", default=None,
                        help="LoRA target modules (default: q_proj k_proj v_proj o_proj)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume-from", default=None, help="Resume training from a checkpoint directory")
    args = parser.parse_args()

    train_qlora(
        train_csv=args.train_csv,
        dev_csv=args.dev_csv,
        output_dir=args.output_dir,
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        device=args.device,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
