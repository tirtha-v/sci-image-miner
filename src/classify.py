"""Main inference loop for chart classification."""

import json
import os
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .dataset import scan_test_figures, load_figure
from .prompts import TAXONOMY, build_classification_prompt
from .postprocess import normalize_label
from .models.base import VLMClassifier


def run_classification(
    model: VLMClassifier,
    test_root: str,
    output_path: str,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Classify all panels in the test set and write prediction_data.json.

    Args:
        model: Loaded VLMClassifier instance
        test_root: Path to test-batch1-release directory
        output_path: Path for prediction_data.json output
        skip_existing: If True and output_path exists, load and return it

    Returns:
        List of prediction dicts
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and output_path.exists():
        print(f"[classify] Loading existing predictions from {output_path}")
        with open(output_path) as f:
            return json.load(f)

    figures = scan_test_figures(test_root)
    print(f"[classify] Found {len(figures)} figures with images.")

    system_prompt, user_prompt = build_classification_prompt()
    results = []

    for entry in tqdm(figures, desc="Classifying"):
        try:
            fig = load_figure(entry)
        except Exception as e:
            print(f"[WARN] Failed to load {entry['sample_id']}: {e}")
            continue

        classification = {}
        for panel_id, panel in fig["panels"].items():
            try:
                raw = model.classify_image(panel["crop"], system_prompt, user_prompt)
                label = normalize_label(raw, TAXONOMY)
            except Exception as e:
                print(f"[WARN] Error on {entry['sample_id']} panel {panel_id}: {e}")
                label = "unknown"
            classification[panel_id] = label

        results.append({
            "sample_id": fig["sample_id"],
            "bbox": fig["bbox"],
            "classification": classification,
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[classify] Saved {len(results)} predictions to {output_path}")
    return results


def make_submission_zip(prediction_json_path: str, zip_path: str) -> str:
    """Zip prediction_data.json for CodaBench submission.

    Returns the path to the zip file.
    """
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_json_path, arcname="prediction_data.json")
    print(f"[classify] Created submission zip: {zip_path}")
    return str(zip_path)
