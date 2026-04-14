"""Summarization pipeline for Task 3: generate panel-level summaries."""

import json
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .dataset import scan_test_figures, load_figure


SYSTEM_PROMPT = (
    "You are a Vision Language Model specialized in generating concise scientific summaries "
    "of figure panels from research papers."
)

USER_PROMPT = (
    "Write a concise 1-3 sentence scientific summary of this figure panel.\n"
    "- Describe what is being shown or demonstrated.\n"
    "- Include key findings, trends, or relationships visible in the figure.\n"
    "- Be specific and scientifically accurate.\n"
    "- Output only the summary text, no preamble."
)


def run_summarization(
    model,
    test_root: str,
    output_path: str,
    skip_existing: bool = True,
    max_new_tokens: int = 256,
) -> list[dict[str, Any]]:
    """Generate summaries for all panels and write prediction_data.json."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and output_path.exists():
        print(f"[summarize] Loading existing predictions from {output_path}")
        with open(output_path) as f:
            return json.load(f)

    figures = scan_test_figures(test_root)
    print(f"[summarize] Found {len(figures)} figures.")

    model.max_new_tokens = max_new_tokens

    results = []
    for entry in tqdm(figures, desc="Summarizing"):
        try:
            fig = load_figure(entry)
        except Exception as e:
            print(f"[WARN] Failed to load {entry['sample_id']}: {e}")
            continue

        summarization = {}
        for panel_id, panel in fig["panels"].items():
            try:
                summary = model.classify_image(panel["crop"], SYSTEM_PROMPT, USER_PROMPT)
            except Exception as e:
                print(f"[WARN] Error on {entry['sample_id']} panel {panel_id}: {e}")
                summary = ""
            summarization[panel_id] = summary.strip()

        results.append({
            "sample_id": fig["sample_id"],
            "summarization": summarization,
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[summarize] Saved {len(results)} predictions to {output_path}")
    return results


def make_submission_zip(prediction_json_path: str, zip_path: str) -> str:
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_json_path, arcname="prediction_data.json")
    print(f"[summarize] Created submission zip: {zip_path}")
    return str(zip_path)
