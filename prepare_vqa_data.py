#!/usr/bin/env python3
"""Extract VQA training data from competition train JSONs into data/train_vqa.csv.

Each row: image_path (pre-cropped panel), sample_id, panel_id,
          question_type, question, answer_type, answer
"""

import csv
import glob
import json
import os
from pathlib import Path

CROPS_DIR = Path("data/crops/train")
OUT_CSV = Path("data/train_vqa.csv")
TRAIN_ROOT = "ALD-E-ImageMiner/icdar2026-competition-data/train"


def sample_id_to_crop_prefix(sample_id: str) -> str:
    return sample_id.replace("/", "_")


def main():
    json_files = sorted(glob.glob(os.path.join(TRAIN_ROOT, "**", "images", "*.json"), recursive=True))
    print(f"Scanning {len(json_files)} train JSONs...")

    rows = []
    missing_crops = 0

    for json_path in json_files:
        try:
            data = json.load(open(json_path))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        sample_id = data.get("sample_id", "")
        vqa = data.get("vqa", {})
        bbox = data.get("bbox", {})

        if not vqa or not sample_id:
            continue

        # Normalize vqa to dict format: {"panel_id": [qa, ...]}
        # Some JSONs have vqa as a list: [{"panel_id": [qa, ...]}, ...]
        if isinstance(vqa, list):
            vqa_dict = {}
            for item in vqa:
                if isinstance(item, dict):
                    vqa_dict.update(item)
            vqa = vqa_dict

        if not isinstance(vqa, dict):
            continue

        crop_prefix = sample_id_to_crop_prefix(sample_id)

        for panel_id, qa_list in vqa.items():
            if not isinstance(qa_list, list) or not qa_list:
                continue

            # Build crop path
            crop_path = CROPS_DIR / f"{crop_prefix}_{panel_id}.jpg"
            if not crop_path.exists():
                # Try png
                crop_path_png = CROPS_DIR / f"{crop_prefix}_{panel_id}.png"
                if crop_path_png.exists():
                    crop_path = crop_path_png
                else:
                    missing_crops += 1
                    continue

            for qa in qa_list:
                if not isinstance(qa, dict):
                    continue
                answer = qa.get("answer", "").strip()
                question = qa.get("question", "").strip()
                if not answer or not question:
                    continue

                rows.append({
                    "image_path": str(crop_path.resolve()),
                    "sample_id": sample_id,
                    "panel_id": panel_id,
                    "question_type": qa.get("question_type", ""),
                    "question": question,
                    "answer_type": qa.get("answer_type", ""),
                    "answer": answer,
                })

    print(f"Extracted {len(rows)} QA pairs ({missing_crops} panels skipped - no crop)")

    # Stats
    by_type = {}
    for r in rows:
        by_type[r["answer_type"]] = by_type.get(r["answer_type"], 0) + 1
    print("By answer type:", by_type)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "sample_id", "panel_id",
                                                "question_type", "question", "answer_type", "answer"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
