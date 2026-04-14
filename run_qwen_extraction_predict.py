#!/usr/bin/env python3
"""Inference with QLoRA fine-tuned Qwen2.5-VL for data extraction (Task 2).

Usage:
    # Dev eval (panels CSV):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_qwen_extraction_predict.py \
        --adapter-path outputs/qwen2vl_extraction_qlora/adapter \
        --mode dev \
        --output-dir outputs/qwen2vl_extraction_qlora/dev

    # Test submission:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_qwen_extraction_predict.py \
        --adapter-path outputs/qwen2vl_extraction_qlora/adapter \
        --mode test \
        --test-root ALD-E-ImageMiner/icdar2026-competition-data/test \
        --output-dir outputs/qwen2vl_extraction_qlora/test
"""

import argparse
import glob
import json
import os
import zipfile
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a Vision Language Model specialized in extracting structured data "
    "from scientific figure panels into Markdown tables."
)

USER_PROMPT = (
    "Extract the quantitative data from this scientific figure panel as a Markdown table.\n"
    "- Use standard Markdown table format with | separators and a header row followed by |---|---| separator.\n"
    "- Include all data values visible in the chart (axes labels as column headers, data points as rows).\n"
    "- If no tabular data is extractable (e.g., schematic, image panel, molecular diagram), output an empty string.\n"
    "- Output ONLY the markdown table or empty string — no explanation."
)


def load_model(adapter_path: str):
    print(f"[extract] Loading base model {BASE_MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID, min_pixels=64*28*28, max_pixels=256*28*28
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
    )
    print(f"[extract] Loading adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("[extract] Model ready.")
    return model, processor


def extract_table(model, processor, pil_image: Image.Image) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    output = processor.batch_decode(
        [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return output


def run_dev(adapter_path: str, output_dir: str):
    """Run on dev set using competition JSON structure."""
    import glob as _glob
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_root = "ALD-E-ImageMiner/icdar2026-competition-data/dev"
    json_files = sorted(_glob.glob(os.path.join(dev_root, "**", "images", "*.json"), recursive=True))
    print(f"[extract] Found {len(json_files)} dev figures.")

    model, processor = load_model(adapter_path)
    results = []

    for idx, json_path in enumerate(json_files):
        data = json.load(open(json_path))
        items = data if isinstance(data, list) else [data]
        for d in items:
            if not isinstance(d, dict) or "sample_id" not in d:
                continue
            bbox = d.get("bbox", {})
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
            data_extraction = {}
            for panel_id, box in bbox.items():
                x1 = max(0, int(box["x"]))
                y1 = max(0, int(box["y"]))
                x2 = min(img_w, int(box["x"]) + int(box["width"]))
                y2 = min(img_h, int(box["y"]) + int(box["height"]))
                if x2 <= x1 or y2 <= y1:
                    data_extraction[panel_id] = ""
                    continue
                crop = image.crop((x1, y1, x2, y2))
                try:
                    table = extract_table(model, processor, crop)
                except Exception as e:
                    print(f"  WARN {d['sample_id']}/{panel_id}: {e}")
                    table = ""
                data_extraction[panel_id] = table

            results.append({"sample_id": d["sample_id"], "data_extraction": data_extraction})

        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{len(json_files)}]")

    out_file = output_dir / "prediction_data.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[extract] Saved {len(results)} figures to {out_file}")
    _make_zip(str(out_file), str(output_dir / "submission.zip"))


def run_test(adapter_path: str, test_root: str, output_dir: str):
    """Run on test set."""
    import glob as _glob
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(_glob.glob(os.path.join(test_root, "**", "images", "*.json"), recursive=True))
    print(f"[extract] Found {len(json_files)} test figures.")

    model, processor = load_model(adapter_path)
    results = []

    for idx, json_path in enumerate(json_files):
        data = json.load(open(json_path))
        items = data if isinstance(data, list) else [data]
        for d in items:
            if not isinstance(d, dict) or "sample_id" not in d:
                continue
            bbox = d.get("bbox", {})
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
            data_extraction = {}
            for panel_id, box in bbox.items():
                x1 = max(0, int(box["x"]))
                y1 = max(0, int(box["y"]))
                x2 = min(img_w, int(box["x"]) + int(box["width"]))
                y2 = min(img_h, int(box["y"]) + int(box["height"]))
                if x2 <= x1 or y2 <= y1:
                    data_extraction[panel_id] = ""
                    continue
                crop = image.crop((x1, y1, x2, y2))
                try:
                    table = extract_table(model, processor, crop)
                except Exception as e:
                    print(f"  WARN {d['sample_id']}/{panel_id}: {e}")
                    table = ""
                data_extraction[panel_id] = table

            results.append({"sample_id": d["sample_id"], "data_extraction": data_extraction})

        if (idx + 1) % 20 == 0:
            print(f"  [{idx + 1}/{len(json_files)}]")

    out_file = output_dir / "prediction_data.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    n_panels = sum(len(r["data_extraction"]) for r in results)
    print(f"[extract] Saved {len(results)} figures ({n_panels} panels) to {out_file}")
    _make_zip(str(out_file), str(output_dir / "submission.zip"))


def _make_zip(json_path: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="prediction_data.json")
    print(f"[extract] Created submission zip: {zip_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--mode", choices=["dev", "test"], default="test")
    parser.add_argument("--test-root", default="ALD-E-ImageMiner/icdar2026-competition-data/test")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    for token_path in [
        "/home/jovyan/tvinchur/hf_token_for_cluster.txt",
        "/home/jovyan/shared-scratch-tvinchur-pvc/tvinchur/hf_token_for_cluster.txt",
    ]:
        if Path(token_path).exists():
            import huggingface_hub
            huggingface_hub.login(token=Path(token_path).read_text().strip(), add_to_git_credential=False)
            break

    if args.mode == "dev":
        run_dev(args.adapter_path, args.output_dir)
    else:
        run_test(args.adapter_path, args.test_root, args.output_dir)


if __name__ == "__main__":
    main()
