"""Dataset utilities: scan test directory, load figures, crop panels."""

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


def scan_test_figures(test_root: str) -> list[dict[str, Any]]:
    """Scan test directory and return list of figure entries.

    Each entry has:
        sample_id: str  (e.g. "atomic-layer-deposition/experimental-usecase/21/fig_1")
        img_path: str   (absolute path to .jpg)
        json_path: str  (absolute path to .json)
    """
    test_root = Path(test_root)
    entries = []

    for json_path in sorted(test_root.rglob("*.json")):
        # Skip non-figure JSONs (e.g. content.json which is a list, not a figure dict)
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict) or "sample_id" not in data:
            continue

        # Derive image path: same stem, .jpg extension
        img_path = json_path.with_suffix(".jpg")
        if not img_path.exists():
            # Try .png as fallback
            img_path_png = json_path.with_suffix(".png")
            if img_path_png.exists():
                img_path = img_path_png
            else:
                print(f"[WARN] No image found for {json_path}, skipping")
                continue

        sample_id = data["sample_id"]

        entries.append({
            "sample_id": sample_id,
            "img_path": str(img_path),
            "json_path": str(json_path),
        })

    return entries


def load_figure(entry: dict[str, Any]) -> dict[str, Any]:
    """Load a figure: read JSON metadata, open image, crop panels.

    Returns:
        {
            "sample_id": str,
            "image": PIL.Image (full figure),
            "bbox": dict[str, dict]  (panel_id -> {x, y, width, height}),
            "panels": dict[str, dict]  (panel_id -> {bbox, crop: PIL.Image}),
        }
    """
    with open(entry["json_path"], encoding='utf-8') as f:
        data = json.load(f)

    image = Image.open(entry["img_path"]).convert("RGB")
    img_w, img_h = image.size

    bbox: dict = data.get("bbox", {})

    if not bbox:
        # No bounding boxes — treat full image as single panel "a"
        bbox = {"a": {"x": 0, "y": 0, "width": img_w, "height": img_h}}

    panels = {}
    for panel_id, box in bbox.items():
        x = int(box["x"])
        y = int(box["y"])
        w = int(box["width"])
        h = int(box["height"])
        # Clamp to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        crop = image.crop((x1, y1, x2, y2))
        panels[panel_id] = {"bbox": box, "crop": crop}

    return {
        "sample_id": data["sample_id"],
        "image": image,
        "bbox": bbox,
        "panels": panels,
    }
