#!/usr/bin/env python3
"""CLI entry point for zero-shot chart classification.

Usage:
    python tasks/task1_classification/predict_zeroshot.py --model qwen2vl
    python tasks/task1_classification/predict_zeroshot.py --model qwen2vl --test-root path/to/test --output-dir outputs/my_run
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is on the path when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.classify import run_classification, make_submission_zip

# Default test root relative to this script's location
_DEFAULT_TEST_ROOT = str(
    Path(__file__).parent
    / "ALD-E-ImageMiner"
    / "icdar2026-competition-data"
    / "test"
    / "test-batch1-release"
)

MODEL_CHOICES = ["qwen2vl", "llava", "instructblip", "phi_3_5_vision", "llava_1_5", "llama_vision"]


def load_model(model_name: str, load_in_4bit: bool = False):
    if model_name == "qwen2vl":
        from src.models.qwen2vl import Qwen2VLClassifier
        return Qwen2VLClassifier(load_in_4bit=load_in_4bit)
    elif model_name == "llava":
        from src.models.llava import LLaVAClassifier
        return LLaVAClassifier()
    elif model_name == "instructblip":
        from src.models.instructblip import InstructBLIPClassifier
        return InstructBLIPClassifier()
    elif model_name == "phi_3_5_vision":
        from src.models.phi_3_5_vision import Phi35VisionClassifier
        return Phi35VisionClassifier()
    elif model_name == "llava_1_5":
        from src.models.llava_1_5 import LLaVA15Classifier
        return LLaVA15Classifier()
    elif model_name == "llama_vision":
        from src.models.llama_vision import LlamaVisionClassifier
        return LlamaVisionClassifier()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot chart classification for Sci-ImagMiner")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="qwen2vl",
        help="VLM model to use (default: qwen2vl)",
    )
    parser.add_argument(
        "--test-root",
        default=_DEFAULT_TEST_ROOT,
        help="Path to test-batch1-release directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: outputs/{model})",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run even if prediction_data.json already exists",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit (fits on 11GB GPUs like RTX 2080 Ti)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g. cuda:0). Default: auto.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or f"outputs/{args.model}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_json = output_dir / "prediction_data.json"
    submission_zip = output_dir / "submission.zip"

    print(f"Model:       {args.model}")
    print(f"Test root:   {args.test_root}")
    print(f"Output dir:  {output_dir}")

    # Load model
    model = load_model(args.model, load_in_4bit=args.load_in_4bit)
    model.load_model()

    try:
        run_classification(
            model=model,
            test_root=args.test_root,
            output_path=str(prediction_json),
            skip_existing=not args.no_skip_existing,
        )
    finally:
        model.unload_model()

    # Create zip
    make_submission_zip(str(prediction_json), str(submission_zip))
    print(f"\nDone! Submission zip: {submission_zip}")
    print(f"Verify with: unzip -t {submission_zip}")


if __name__ == "__main__":
    main()
