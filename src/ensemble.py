"""Ensemble predictions via majority or weighted voting."""

import json
from collections import Counter
from pathlib import Path


def majority_vote(prediction_paths: list[str], output_path: str):
    """Combine predictions via majority voting.

    Args:
        prediction_paths: List of paths to prediction_data.json files
        output_path: Path for ensemble output
    """
    # Load all predictions
    all_preds = []
    for path in prediction_paths:
        with open(path) as f:
            all_preds.append(json.load(f))

    # Index by sample_id
    pred_by_sample = {}
    for preds in all_preds:
        for entry in preds:
            sid = entry["sample_id"]
            if sid not in pred_by_sample:
                pred_by_sample[sid] = {"bbox": entry["bbox"], "votes": {}}
            for panel_id, label in entry.get("classification", {}).items():
                if panel_id not in pred_by_sample[sid]["votes"]:
                    pred_by_sample[sid]["votes"][panel_id] = []
                pred_by_sample[sid]["votes"][panel_id].append(label)

    # Vote
    results = []
    for sid, data in sorted(pred_by_sample.items()):
        classification = {}
        for panel_id, votes in data["votes"].items():
            counter = Counter(votes)
            classification[panel_id] = counter.most_common(1)[0][0]
        results.append({
            "sample_id": sid,
            "bbox": data["bbox"],
            "classification": classification,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total_panels = sum(len(r["classification"]) for r in results)
    print(f"[ensemble] Majority vote: {len(results)} figures ({total_panels} panels) → {output_path}")
    return results


def weighted_vote(
    prediction_paths: list[str],
    weights: list[float],
    output_path: str,
):
    """Combine predictions via weighted voting.

    Args:
        prediction_paths: List of paths to prediction_data.json files
        weights: Weight for each model (e.g. dev macro-F1 scores)
        output_path: Path for ensemble output
    """
    all_preds = []
    for path in prediction_paths:
        with open(path) as f:
            all_preds.append(json.load(f))

    # Index by sample_id
    pred_by_sample = {}
    for model_idx, preds in enumerate(all_preds):
        for entry in preds:
            sid = entry["sample_id"]
            if sid not in pred_by_sample:
                pred_by_sample[sid] = {"bbox": entry["bbox"], "votes": {}}
            for panel_id, label in entry.get("classification", {}).items():
                if panel_id not in pred_by_sample[sid]["votes"]:
                    pred_by_sample[sid]["votes"][panel_id] = []
                pred_by_sample[sid]["votes"][panel_id].append(
                    (label, weights[model_idx])
                )

    # Weighted vote
    results = []
    for sid, data in sorted(pred_by_sample.items()):
        classification = {}
        for panel_id, votes in data["votes"].items():
            weighted_counts = Counter()
            for label, weight in votes:
                weighted_counts[label] += weight
            classification[panel_id] = weighted_counts.most_common(1)[0][0]
        results.append({
            "sample_id": sid,
            "bbox": data["bbox"],
            "classification": classification,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total_panels = sum(len(r["classification"]) for r in results)
    print(f"[ensemble] Weighted vote: {len(results)} figures ({total_panels} panels) → {output_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble chart classification predictions")
    parser.add_argument(
        "--predictions", nargs="+", required=True,
        help="Paths to prediction_data.json files",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="Weights for each model (same order as predictions). If not given, uses majority vote.",
    )
    parser.add_argument("--output", default="outputs/ensemble/prediction_data.json")
    args = parser.parse_args()

    if args.weights:
        if len(args.weights) != len(args.predictions):
            raise ValueError("Number of weights must match number of predictions")
        weighted_vote(args.predictions, args.weights, args.output)
    else:
        majority_vote(args.predictions, args.output)


if __name__ == "__main__":
    main()
