#!/usr/bin/env python3
"""CLI entry point for CNN finetuning.

Usage:
    python tasks/task1_classification/finetune_cnn.py --model efficientnet_b0
    python tasks/task1_classification/finetune_cnn.py --model inception_resnet_v2 --epochs 30 --batch-size 16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.cnn.train import train
from src.cnn.model import MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Train CNN for chart classification")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="efficientnet_b0",
        help="CNN model to train",
    )
    parser.add_argument("--train-csv", default="data/train_panels.csv")
    parser.add_argument("--dev-csv", default="data/dev_panels.csv")
    parser.add_argument("--output-dir", default=None, help="Default: outputs/cnn/{model}")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--frozen-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--focal", action="store_true", help="Use Focal Loss (gamma=2.0) instead of CrossEntropy")
    args = parser.parse_args()

    output_dir = args.output_dir or f"outputs/cnn/{args.model}"

    print(f"Model:      {args.model}")
    print(f"Train CSV:  {args.train_csv}")
    print(f"Dev CSV:    {args.dev_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Focal loss: {args.focal}")

    train(
        model_name=args.model,
        train_csv=args.train_csv,
        dev_csv=args.dev_csv,
        output_dir=output_dir,
        epochs=args.epochs,
        frozen_epochs=args.frozen_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        use_focal=args.focal,
    )


if __name__ == "__main__":
    main()
