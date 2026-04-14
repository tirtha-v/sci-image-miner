"""CNN inference: generate predictions in competition JSON format."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from .model import create_model, MODEL_CONFIGS
from .dataset import get_val_transform


def predict_test(
    model_name: str,
    checkpoint_path: str,
    label_mapping_path: str,
    test_root: str,
    output_path: str,
    device: str = "cuda:0",
    batch_size: int = 32,
):
    """Run CNN inference on test set and output competition JSON.

    Args:
        model_name: Key in MODEL_CONFIGS (e.g. 'efficientnet_b0')
        checkpoint_path: Path to best_model.pt
        label_mapping_path: Path to label_mapping.json
        test_root: Path to test data directory
        output_path: Path for output prediction_data.json
        device: CUDA device
        batch_size: Inference batch size
    """
    import glob
    import os

    config = MODEL_CONFIGS[model_name]
    img_size = config["img_size"]

    # Load label mapping
    with open(label_mapping_path) as f:
        mapping = json.load(f)
    idx2label = mapping["idx2label"]
    num_classes = len(idx2label)

    # Load model
    model = create_model(config["timm_name"], num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    transform = get_val_transform(img_size)

    # Scan test JSONs
    json_files = sorted(glob.glob(os.path.join(test_root, "**", "images", "*.json"), recursive=True))
    results = []

    for json_path in json_files:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        if not isinstance(data, dict) or "sample_id" not in data:
            continue

        bbox = data.get("bbox", {})
        if not bbox:
            continue

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
        classification = {}

        for panel_id, box in bbox.items():
            x1 = max(0, int(box["x"]))
            y1 = max(0, int(box["y"]))
            x2 = min(img_w, int(box["x"]) + int(box["width"]))
            y2 = min(img_h, int(box["y"]) + int(box["height"]))

            if x2 <= x1 or y2 <= y1:
                classification[panel_id] = "unknown"
                continue

            crop = image.crop((x1, y1, x2, y2))
            tensor = transform(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                pred_idx = output.argmax(dim=1).item()

            classification[panel_id] = idx2label[pred_idx]

        results.append({
            "sample_id": data["sample_id"],
            "bbox": bbox,
            "classification": classification,
        })

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total_panels = sum(len(r["classification"]) for r in results)
    print(f"[predict] Saved {len(results)} figures ({total_panels} panels) to {output_path}")
    return results
