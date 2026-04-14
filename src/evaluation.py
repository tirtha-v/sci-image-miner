"""Evaluation metrics for chart classification."""

import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import numpy as np


def load_ground_truth(dev_root: str) -> dict[str, dict[str, str]]:
    """Load ground truth labels from dev set JSONs.

    Returns:
        {sample_id: {panel_id: label}}
    """
    import glob

    gt = {}
    for json_path in sorted(glob.glob(f"{dev_root}/**/images/*.json", recursive=True)):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        if not isinstance(data, dict) or "sample_id" not in data:
            continue
        classification = data.get("classification", {})
        if classification:
            gt[data["sample_id"]] = {
                k: v for k, v in classification.items() if v
            }
    return gt


def load_predictions(prediction_path: str) -> dict[str, dict[str, str]]:
    """Load predictions from prediction_data.json.

    Returns:
        {sample_id: {panel_id: label}}
    """
    with open(prediction_path) as f:
        data = json.load(f)

    preds = {}
    for entry in data:
        sid = entry["sample_id"]
        preds[sid] = entry.get("classification", {})
    return preds


def evaluate(
    ground_truth: dict[str, dict[str, str]],
    predictions: dict[str, dict[str, str]],
    output_path: str | None = None,
) -> dict:
    """Compute classification metrics.

    Returns dict with accuracy, macro_f1, weighted_f1, per_class report.
    """
    y_true = []
    y_pred = []
    missing = 0

    for sample_id, panels in ground_truth.items():
        pred_panels = predictions.get(sample_id, {})
        for panel_id, true_label in panels.items():
            pred_label = pred_panels.get(panel_id, "unknown")
            y_true.append(true_label.lower())
            y_pred.append(pred_label.lower())
            if sample_id not in predictions or panel_id not in pred_panels:
                missing += 1

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    all_labels = sorted(set(y_true + y_pred))
    report = classification_report(
        y_true, y_pred, labels=all_labels, zero_division=0, output_dict=True
    )

    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_panels": len(y_true),
        "missing_predictions": missing,
        "per_class": report,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  Total panels: {len(y_true)}  (missing predictions: {missing})")
    print(f"{'='*60}")
    print(f"\nPer-class report:")
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics to {output_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate chart classification predictions")
    parser.add_argument("--predictions", required=True, help="Path to prediction_data.json")
    parser.add_argument(
        "--dev-root",
        default=str(
            Path(__file__).parent.parent
            / "ALD-E-ImageMiner"
            / "icdar2026-competition-data"
            / "dev"
        ),
        help="Path to dev set root",
    )
    parser.add_argument("--output", default=None, help="Save metrics JSON to this path")
    args = parser.parse_args()

    gt = load_ground_truth(args.dev_root)
    preds = load_predictions(args.predictions)
    print(f"Ground truth: {sum(len(v) for v in gt.values())} panels from {len(gt)} figures")
    print(f"Predictions:  {sum(len(v) for v in preds.values())} panels from {len(preds)} figures")

    evaluate(gt, preds, args.output)


if __name__ == "__main__":
    main()
