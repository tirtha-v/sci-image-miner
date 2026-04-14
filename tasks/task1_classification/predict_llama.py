#!/usr/bin/env python3
"""Dev/test inference for QLoRA-finetuned Llama-3.2-11B-Vision-Instruct.

For dev set, reads from data/dev_panels.csv (pre-cropped panels).
For test set, reads from data/test_panels.csv or walks crops directory.

Usage:
    python tasks/task1_classification/predict_llama.py \
        --adapter-path outputs/vlm_finetune/llama_qlora/checkpoint-152 \
        --panels-csv data/dev_panels.csv \
        --output-dir outputs/llama_qlora_ft/dev \
        --device cuda:0
"""

import os
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.prompts import SYSTEM_PROMPT, TAXONOMY, build_classification_prompt
from src.postprocess import normalize_label
from src.classify import make_submission_zip


BASE_MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def load_model(adapter_path: str, device: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[predict] Loading base model {BASE_MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    # Use device_map="auto" to spread across all visible GPUs.
    # Explicitly limit per-GPU memory so the model is balanced across GPUs
    # rather than all on one (which can exceed 10.57GB on RTX 2080 Ti).
    n_gpus = torch.cuda.device_count()
    print(f"[predict] Visible GPUs: {n_gpus}")
    # "balanced" explicitly divides layers equally across all visible GPUs.
    # Use CUDA_VISIBLE_DEVICES to select which physical GPUs to spread across.
    base_model = MllamaForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
    )
    print(f"[predict] Loading adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("[predict] Model ready.")
    return model, processor


def classify_image(model, processor, pil_image: Image.Image, user_prompt: str, device: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{user_prompt}"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text,
        images=[[pil_image]],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )
    output = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return output.strip()


def run_predict_from_test_root(adapter_path: str, test_root: str, output_dir: str, device: str):
    """Run inference by walking competition JSON structure (for test set without a CSV)."""
    import glob
    import os
    _, user_prompt = build_classification_prompt()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(adapter_path, device)

    json_files = sorted(glob.glob(os.path.join(test_root, "**", "images", "*.json"), recursive=True))
    results = []

    for idx, json_path in enumerate(json_files):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
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
            try:
                raw = classify_image(model, processor, crop, user_prompt, device)
                label = normalize_label(raw, TAXONOMY)
            except Exception as e:
                print(f"  WARN {data['sample_id']}/{panel_id}: {e}")
                label = "unknown"
            classification[panel_id] = label

        results.append({"sample_id": data["sample_id"], "bbox": bbox, "classification": classification})

        if (idx + 1) % 20 == 0:
            print(f"  [{idx + 1}/{len(json_files)}] {data['sample_id']}")

    out_file = output_dir / "prediction_data.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    n_panels = sum(len(r["classification"]) for r in results)
    print(f"[predict] Saved {len(results)} figures ({n_panels} panels) to {out_file}")
    try:
        make_submission_zip(str(out_file), str(output_dir / "submission.zip"))
        print(f"Submission zip: {output_dir / 'submission.zip'}")
    except Exception as e:
        print(f"  submission zip skipped: {e}")
    return str(out_file)


def run_predict(adapter_path: str, panels_csv: str, output_dir: str, device: str):
    _, user_prompt = build_classification_prompt()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panels_csv)
    model, processor = load_model(adapter_path, device)

    # Group by sample_id to build per-figure predictions
    results = defaultdict(dict)  # sample_id -> {panel_id: label}

    for idx, row in df.iterrows():
        image_path = row["image_path"]
        sample_id = row["sample_id"]
        panel_id = str(row["panel_id"])

        try:
            image = Image.open(image_path).convert("RGB")
            raw = classify_image(model, processor, image, user_prompt, device)
            label = normalize_label(raw, TAXONOMY)
        except Exception as e:
            print(f"  WARN [{sample_id}/{panel_id}]: {e}")
            label = "unknown"

        results[sample_id][panel_id] = label

        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(df)}] {sample_id}/{panel_id} -> {label}")

    # Format as prediction_data.json list
    predictions = []
    for sample_id, panels in results.items():
        predictions.append({
            "sample_id": sample_id,
            "bbox": {pid: {"x": 0, "y": 0, "width": 1, "height": 1} for pid in panels},
            "classification": panels,
        })

    out_file = output_dir / "prediction_data.json"
    with open(out_file, "w") as f:
        json.dump(predictions, f, indent=2)

    n_panels = sum(len(p["classification"]) for p in predictions)
    print(f"[predict] Saved {len(predictions)} figures ({n_panels} panels) to {out_file}")

    try:
        make_submission_zip(str(out_file), str(output_dir / "submission.zip"))
        print(f"Submission zip: {output_dir / 'submission.zip'}")
    except Exception as e:
        print(f"  submission zip skipped: {e}")

    return str(out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--panels-csv", default=None, help="CSV with image_path, sample_id, panel_id (dev mode)")
    parser.add_argument("--test-root", default=None, help="Competition data root dir (test mode)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not args.panels_csv and not args.test_root:
        raise ValueError("Provide either --panels-csv (dev) or --test-root (test)")

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        import huggingface_hub
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)

    if args.panels_csv:
        run_predict(args.adapter_path, args.panels_csv, args.output_dir, args.device)
    else:
        run_predict_from_test_root(args.adapter_path, args.test_root, args.output_dir, args.device)


if __name__ == "__main__":
    main()
