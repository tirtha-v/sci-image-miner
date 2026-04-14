"""Data preparation: crop panels and build CSV manifests for CNN training."""

import csv
import json
import glob
import os
from pathlib import Path

from PIL import Image

from .label_cleaning import clean_label


def crop_panels(data_root: str, split: str, crop_dir: str, csv_path: str):
    """Walk a split directory, crop each panel using bbox, save crops and CSV.

    Args:
        data_root: Path to icdar2026-competition-data/{split}
        split: 'train' or 'dev'
        crop_dir: Output directory for crops (e.g. data/crops/train)
        csv_path: Output CSV path (e.g. data/train_panels.csv)
    """
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    json_files = sorted(glob.glob(os.path.join(data_root, "**", "images", "*.json"), recursive=True))

    for json_path in json_files:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        if not isinstance(data, dict) or "sample_id" not in data:
            continue

        classification = data.get("classification", {})
        bbox = data.get("bbox", {})

        if not classification or not bbox:
            continue

        # Find image
        img_path = Path(json_path).with_suffix(".jpg")
        if not img_path.exists():
            img_path = Path(json_path).with_suffix(".png")
            if not img_path.exists():
                continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        img_w, img_h = image.size
        sample_id = data["sample_id"]
        # Make sample_id filesystem-safe
        safe_sample_id = sample_id.replace("/", "_")

        for panel_id, box in bbox.items():
            label = classification.get(panel_id, "")
            if not label:
                continue

            label = clean_label(label)

            x = int(box["x"])
            y = int(box["y"])
            w = int(box["width"])
            h = int(box["height"])

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2))
            crop_filename = f"{safe_sample_id}_{panel_id}.jpg"
            crop_path = crop_dir / crop_filename
            crop.save(crop_path, quality=95)

            rows.append({
                "image_path": str(crop_path),
                "label": label,
                "sample_id": sample_id,
                "panel_id": panel_id,
            })

    # Write CSV
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "sample_id", "panel_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[data_prep] {split}: {len(rows)} panels saved to {crop_dir}, CSV at {csv_path}")
    return rows


def main():
    """Prepare train and dev splits."""
    base = Path(__file__).parent.parent
    data_root = base / "ALD-E-ImageMiner" / "icdar2026-competition-data"

    for split in ["train", "dev"]:
        split_root = data_root / split
        if not split_root.exists():
            print(f"[data_prep] Skipping {split}: {split_root} not found")
            continue
        crop_panels(
            data_root=str(split_root),
            split=split,
            crop_dir=str(base / "data" / "crops" / split),
            csv_path=str(base / "data" / f"{split}_panels.csv"),
        )


if __name__ == "__main__":
    main()
