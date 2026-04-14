# flag_panels.py
# Parses QWEN classification JSON, finds unknowns and out-of-taxonomy labels
# Usage: python flag_panels.py <path_to_prediction_json> [--output-dir <dir>]

import argparse
import json
import os

# Official taxonomy from competition
OFFICIAL_CLASSES = {
    "area chart", "bar chart", "3d bar chart", "grouped bar chart", "stacked bar chart",
    "box plot", "bubble chart", "donut chart", "funnel chart", "heatmap", "line chart",
    "multiple line chart", "multi-axis chart", "pie chart", "polar chart (rose chart)",
    "radar chart (spider chart)", "3d scatter plot", "scatter plot", "multiple scatter plot",
    "treemap", "spectra chart", "stacked spectra chart", "multi spectra chart", "phase diagram",
    "band diagram", "adsorption isotherm", "process timing diagram", "contour heatmap",
    "image panel", "map/geo chart", "competing reaction rate curve", "molecular structure diagram",
    "reaction scheme", "reaction energy profile diagram", "process flow diagram",
    "apparatus diagram", "conceptual diagram", "device structure diagram", "chromaticity diagram",
    "periodic table map", "element-property matrix", "network diagram", "tree diagram",
    "workflow diagram", "timeline chart", "comparison table", "formula", "table", "unknown"
}

# Known label fixes (wrong label -> correct label)
LABEL_FIXES = {
    "device structure": "device structure diagram",
}


def flag_panels(json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flagged = []
    auto_fixed = []

    for entry in data:
        sample_id = entry["sample_id"]
        for panel, cls in entry["classification"].items():
            if cls in LABEL_FIXES:
                auto_fixed.append({
                    "sample_id": sample_id,
                    "panel": panel,
                    "original_label": cls,
                    "fixed_label": LABEL_FIXES[cls]
                })
            elif cls == "unknown":
                flagged.append({
                    "sample_id": sample_id,
                    "panel": panel,
                    "current_label": cls,
                    "reason": "unknown - needs reclassification"
                })
            elif cls not in OFFICIAL_CLASSES:
                flagged.append({
                    "sample_id": sample_id,
                    "panel": panel,
                    "current_label": cls,
                    "reason": f"out-of-taxonomy label: '{cls}'"
                })

    # Print report
    print("=" * 60)
    print("CLASSIFICATION FLAG REPORT")
    print("=" * 60)

    print(f"\nAUTO-FIXABLE (no image needed): {len(auto_fixed)}")
    for fix in auto_fixed:
        print(f"  [{fix['panel']}] {fix['sample_id']}")
        print(f"       '{fix['original_label']}' -> '{fix['fixed_label']}'")

    print(f"\nNEEDS RECLASSIFICATION (image required): {len(flagged)}")
    for flag in flagged:
        print(f"  [{flag['panel']}] {flag['sample_id']}")
        print(f"       Reason: {flag['reason']}")

    print(f"\nTOTAL AUTO-FIXES: {len(auto_fixed)}")
    print(f"TOTAL TO RECLASSIFY: {len(flagged)}")
    print("=" * 60)

    # Save flagged list
    output = {
        "auto_fixed": auto_fixed,
        "needs_reclassification": flagged
    }

    out_path = os.path.join(output_dir, "flagged_panels.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved flagged list to: {out_path}")
    print("Pass this output to Claude Code to proceed with reclassification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flag unknown/out-of-taxonomy panels in a classification prediction JSON."
    )
    parser.add_argument("prediction_json", help="Path to prediction_data.json")
    parser.add_argument(
        "--output-dir", default="outputs/flagged",
        help="Directory to write flagged_panels.json (default: outputs/flagged)"
    )
    args = parser.parse_args()
    flag_panels(args.prediction_json, args.output_dir)
