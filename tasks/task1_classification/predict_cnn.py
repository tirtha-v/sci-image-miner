#!/usr/bin/env python3
"""CLI entry point for CNN test predictions.

Usage:
    python tasks/task1_classification/predict_cnn.py --model efficientnet_b0 --checkpoint outputs/cnn/efficientnet_b0/best_model.pt
"""

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.cnn.predict import predict_test
from src.cnn.model import MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Generate CNN test predictions")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="efficientnet_b0")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--label-mapping", default=None, help="Path to label_mapping.json")
    parser.add_argument(
        "--test-root",
        default=str(
            Path(__file__).parent
            / "ALD-E-ImageMiner"
            / "icdar2026-competition-data"
            / "test"
            / "practice-phase"
        ),
        help="Path to test data directory",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"outputs/cnn/{args.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    label_mapping = args.label_mapping or str(output_dir / "label_mapping.json")
    prediction_json = output_dir / "prediction_data.json"
    submission_zip = output_dir / "submission.zip"

    predict_test(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        label_mapping_path=label_mapping,
        test_root=args.test_root,
        output_path=str(prediction_json),
        device=args.device,
    )

    # Create submission zip
    with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_json, arcname="prediction_data.json")
    print(f"Submission zip: {submission_zip}")


if __name__ == "__main__":
    main()
