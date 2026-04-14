"""Data extraction pipeline for Task 2: extract markdown tables from figure panels."""

import json
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .dataset import scan_test_figures, load_figure


SYSTEM_PROMPT = (
    "You are a Vision Language Model specialized in extracting structured data "
    "from scientific figure panels into Markdown tables."
)

USER_PROMPT = (
    "Extract the quantitative data from this scientific figure panel as a Markdown table.\n"
    "- If the panel is a chart or plot (bar, line, scatter, spectra, etc.), extract the data values.\n"
    "- Enclose the table between ```markdown and ``` markers.\n"
    "- If no tabular data is extractable (e.g., schematic, conceptual diagram, image panel), "
    "output an empty string."
)


def run_extraction(
    model,
    test_root: str,
    output_path: str,
    skip_existing: bool = True,
    max_new_tokens: int = 512,
) -> list[dict[str, Any]]:
    """Extract data tables from all panels and write prediction_data.json."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and output_path.exists():
        print(f"[extract] Loading existing predictions from {output_path}")
        with open(output_path) as f:
            return json.load(f)

    figures = scan_test_figures(test_root)
    print(f"[extract] Found {len(figures)} figures.")

    # Override max_new_tokens for longer outputs
    model.max_new_tokens = max_new_tokens

    results = []
    for entry in tqdm(figures, desc="Extracting"):
        try:
            fig = load_figure(entry)
        except Exception as e:
            print(f"[WARN] Failed to load {entry['sample_id']}: {e}")
            continue

        data_extraction = {}
        for panel_id, panel in fig["panels"].items():
            try:
                raw = model.classify_image(panel["crop"], SYSTEM_PROMPT, USER_PROMPT)
                # Ensure markdown fencing if model returned plain table
                if raw and "```" not in raw and "|" in raw:
                    raw = f"```markdown\n{raw}\n```"
            except Exception as e:
                print(f"[WARN] Error on {entry['sample_id']} panel {panel_id}: {e}")
                raw = ""
            data_extraction[panel_id] = raw

        results.append({
            "sample_id": fig["sample_id"],
            "data_extraction": data_extraction,
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[extract] Saved {len(results)} predictions to {output_path}")
    return results


def make_submission_zip(prediction_json_path: str, zip_path: str) -> str:
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_json_path, arcname="prediction_data.json")
    print(f"[extract] Created submission zip: {zip_path}")
    return str(zip_path)
