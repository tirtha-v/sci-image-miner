#!/usr/bin/env python3
"""Inference with QLoRA fine-tuned Qwen2.5-VL for VQA (Task 4).

Test JSONs already contain the VQA questions with empty answers.
This script reads the questions from the JSON and generates answers.

Usage:
    # Dev eval:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tasks/task4_vqa/predict.py \
        --adapter-path outputs/qwen2vl_vqa_qlora/adapter \
        --mode dev \
        --output-dir outputs/qwen2vl_vqa_qlora/dev

    # Test submission:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tasks/task4_vqa/predict.py \
        --adapter-path outputs/qwen2vl_vqa_qlora/adapter \
        --mode test \
        --test-root ALD-E-ImageMiner/icdar2026-competition-data/test \
        --output-dir outputs/qwen2vl_vqa_qlora/test
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
    "You are an expert in materials science specializing in atomic layer deposition (ALD) "
    "and atomic layer etching (ALE). Answer questions about scientific figure panels accurately."
)

ANSWER_TYPE_HINTS = {
    "Yes/No": "Answer with 'Yes' or 'No' followed by a brief explanation.",
    "Factoid": "Provide a concise factual answer.",
    "List": "Provide a comma-separated list or bullet points.",
    "Paragraph": "Provide a detailed 2-4 sentence answer with specific values where visible.",
}

MAX_NEW_TOKENS = {
    "Yes/No": 80,
    "Factoid": 80,
    "List": 120,
    "Paragraph": 300,
}


def load_model(adapter_path: str):
    print(f"[vqa] Loading base model {BASE_MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID, min_pixels=64*28*28, max_pixels=256*28*28
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print(f"[vqa] Loading adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path, device_map="balanced")
    model.eval()
    print("[vqa] Model ready.")
    return model, processor


def answer_question(model, processor, pil_image: Image.Image,
                    question: str, answer_type: str) -> str:
    from qwen_vl_utils import process_vision_info

    hint = ANSWER_TYPE_HINTS.get(answer_type, "")
    user_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n{hint}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_prompt},
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

    max_tokens = MAX_NEW_TOKENS.get(answer_type, 200)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    output = processor.batch_decode(
        [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return output


def process_figures(json_files: list, model, processor) -> list:
    results = []
    for idx, json_path in enumerate(json_files):
        try:
            data = json.load(open(json_path))
        except Exception:
            continue
        if not isinstance(data, dict) or "sample_id" not in data:
            continue

        bbox = data.get("bbox", {})
        vqa_data = data.get("vqa", {})

        if not bbox:
            continue

        # Normalize vqa to dict format
        if isinstance(vqa_data, list):
            vqa_dict = {}
            for item in vqa_data:
                if isinstance(item, dict):
                    vqa_dict.update(item)
            vqa_data = vqa_dict

        if not vqa_data:
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
        vqa_results = {}

        for panel_id, qa_list in vqa_data.items():
            if panel_id not in bbox:
                continue
            if not isinstance(qa_list, list) or not qa_list:
                continue

            box = bbox[panel_id]
            x1 = max(0, int(box["x"]))
            y1 = max(0, int(box["y"]))
            x2 = min(img_w, int(box["x"]) + int(box["width"]))
            y2 = min(img_h, int(box["y"]) + int(box["height"]))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2))
            answered_qa = []

            for qa in qa_list:
                if not isinstance(qa, dict):
                    continue
                question = qa.get("question", "").strip()
                answer_type = qa.get("answer_type", "Paragraph")
                question_type = qa.get("question_type", "")

                if not question:
                    answered_qa.append(qa)
                    continue

                try:
                    answer = answer_question(model, processor, crop, question, answer_type)
                except Exception as e:
                    print(f"  WARN {data['sample_id']}/{panel_id}: {e}")
                    answer = ""

                answered_qa.append({
                    "question_type": question_type,
                    "question": question,
                    "answer_type": answer_type,
                    "answer": answer,
                })

            if answered_qa:
                vqa_results[panel_id] = answered_qa

        if vqa_results:
            results.append({
                "sample_id": data["sample_id"],
                "vqa": vqa_results,
            })

        if (idx + 1) % 20 == 0:
            print(f"  [{idx + 1}/{len(json_files)}]")

    return results


def run_dev(adapter_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_root = "ALD-E-ImageMiner/icdar2026-competition-data/dev"
    json_files = sorted(glob.glob(os.path.join(dev_root, "**", "images", "*.json"), recursive=True))
    print(f"[vqa] Found {len(json_files)} dev figures.")

    model, processor = load_model(adapter_path)
    results = process_figures(json_files, model, processor)

    out_file = output_dir / "prediction_data.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[vqa] Saved {len(results)} figures to {out_file}")
    _make_zip(str(out_file), str(output_dir / "submission.zip"))


def run_test(adapter_path: str, test_root: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(test_root, "**", "images", "*.json"), recursive=True))
    print(f"[vqa] Found {len(json_files)} test figures.")

    model, processor = load_model(adapter_path)
    results = process_figures(json_files, model, processor)

    out_file = output_dir / "prediction_data.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    n_panels = sum(len(r["vqa"]) for r in results)
    n_questions = sum(
        sum(len(qa_list) for qa_list in r["vqa"].values())
        for r in results
    )
    print(f"[vqa] Saved {len(results)} figures ({n_panels} panels, {n_questions} questions) to {out_file}")
    _make_zip(str(out_file), str(output_dir / "submission.zip"))


def _make_zip(json_path: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="prediction_data.json")
    print(f"[vqa] Created submission zip: {zip_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--mode", choices=["dev", "test"], default="test")
    parser.add_argument("--test-root", default="ALD-E-ImageMiner/icdar2026-competition-data/test")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        import huggingface_hub
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)

    if args.mode == "dev":
        run_dev(args.adapter_path, args.output_dir)
    else:
        run_test(args.adapter_path, args.test_root, args.output_dir)


if __name__ == "__main__":
    main()
