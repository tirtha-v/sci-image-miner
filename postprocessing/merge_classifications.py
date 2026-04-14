# merge_classifications.py
# Merges Claude's reclassifications + auto-fixes into final corrected JSON
# Usage: python merge_classifications.py <path_to_prediction_json> <path_to_corrections_json> [--output-dir <dir>]

import argparse
import json
import os
from datetime import datetime


def merge_classifications(original_path, corrections_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(original_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(corrections_path, "r", encoding="utf-8") as f:
        corrections = json.load(f)

    # Build lookup: (sample_id, panel) -> new label + metadata
    fix_map = {}

    for fix in corrections.get("auto_fixed", []):
        key = (fix["sample_id"], fix["panel"])
        fix_map[key] = {
            "new_label": fix["fixed_label"],
            "old_label": fix["original_label"],
            "type": "auto_fix",
            "confidence": "high"
        }

    for reclassify in corrections.get("reclassified", []):
        key = (reclassify["sample_id"], reclassify["panel"])
        fix_map[key] = {
            "new_label": reclassify["new_label"],
            "old_label": reclassify["old_label"],
            "type": "claude_reclassification",
            "confidence": reclassify.get("confidence", "medium")
        }

    # Apply fixes and track changes
    changes = []
    for entry in data:
        sample_id = entry["sample_id"]
        for panel in entry["classification"]:
            key = (sample_id, panel)
            if key in fix_map:
                fix = fix_map[key]
                entry["classification"][panel] = fix["new_label"]
                changes.append({
                    "sample_id": sample_id,
                    "panel": panel,
                    "old_label": fix["old_label"],
                    "new_label": fix["new_label"],
                    "type": fix["type"],
                    "confidence": fix["confidence"]
                })

    # Save corrected JSON
    corrected_path = os.path.join(output_dir, "corrected_classification.json")
    with open(corrected_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Generate report
    report_path = os.path.join(output_dir, "correction_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("CLASSIFICATION CORRECTION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        auto_fixes = [c for c in changes if c["type"] == "auto_fix"]
        reclassified = [c for c in changes if c["type"] == "claude_reclassification"]

        f.write(f"AUTO-FIXES ({len(auto_fixes)})\n")
        f.write("-" * 40 + "\n")
        for c in auto_fixes:
            f.write(f"  [{c['panel']}] {c['sample_id']}\n")
            f.write(f"       '{c['old_label']}' -> '{c['new_label']}' [confidence: {c['confidence']}]\n")

        f.write(f"\nRECLASSIFIED BY CLAUDE ({len(reclassified)})\n")
        f.write("-" * 40 + "\n")
        for c in reclassified:
            f.write(f"  [{c['panel']}] {c['sample_id']}\n")
            f.write(f"       '{c['old_label']}' -> '{c['new_label']}' [confidence: {c['confidence']}]\n")

        f.write(f"\n{'=' * 60}\n")
        f.write("SUMMARY\n")
        f.write(f"  Total corrections applied: {len(changes)}\n")
        f.write(f"  Auto-fixes: {len(auto_fixes)}\n")
        f.write(f"  Claude reclassifications: {len(reclassified)}\n")
        affected_classes = set(c['old_label'] for c in changes)
        f.write(f"  Classes affected: {', '.join(sorted(affected_classes))}\n")
        f.write("=" * 60 + "\n")

    print(f"Corrected classification saved to: {corrected_path}")
    print(f"Correction report saved to: {report_path}")
    print(f"\nTotal corrections applied: {len(changes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge auto-fixes and Claude reclassifications into a corrected prediction JSON."
    )
    parser.add_argument("prediction_json", help="Path to original prediction_data.json")
    parser.add_argument("corrections_json", help="Path to flagged_panels.json (with auto_fixed + reclassified keys)")
    parser.add_argument(
        "--output-dir", default="outputs/corrected",
        help="Directory to write corrected_classification.json and correction_report.txt (default: outputs/corrected)"
    )
    args = parser.parse_args()
    merge_classifications(args.prediction_json, args.corrections_json, args.output_dir)
