#!/usr/bin/env python3
"""Hybrid pipeline: VLM inference (all 4 tasks) + targeted Claude verification.

Phases:
  1. Classify all test figures → classification_cache.json (with raw + label)
  2. Claude disambiguation of confusable pairs → corrected_classification.json
  3. Extract all figures (classification-context injected) → extraction_cache.json
  4. Summarize all figures (classification + extraction context) → summarization_cache.json
  5. VQA all figures (all context injected) → vqa_cache.json
  6. Claude enrichment of VQA Paragraph answers → enriched_vqa.json
  7. Package each task into its own submission.zip

Pipeline is resumable: each phase checks if its cache exists and skips if so.
Use --force-phase N to re-run phase N and all subsequent phases.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tasks/pipeline.py \
        --classification-adapter outputs/qwen2vl_qlora_expG/adapter \
        --extraction-adapter outputs/qwen2vl_extraction_qlora/adapter \
        --summarization-adapter outputs/qwen2vl_summarization_qlora/adapter \
        --vqa-adapter outputs/qwen2vl_vqa_qlora/adapter \
        --test-root ALD-E-ImageMiner/icdar2026-competition-data/test \
        --output-dir outputs/hybrid
"""

import sys
import argparse
from pathlib import Path as _RootPath
sys.path.insert(0, str(_RootPath(__file__).resolve().parent.parent))
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

# ── Taxonomy ──────────────────────────────────────────────────────────────────

def load_taxonomy() -> list[str]:
    tsv = "ALD-E-ImageMiner/figure_taxonomy.tsv"
    with open(tsv) as f:
        lines = f.readlines()
    return [line.strip().split("\t")[0] for line in lines[1:] if line.strip()]

# ── Prompt builders ───────────────────────────────────────────────────────────

CLASSIFY_SYSTEM = (
    "You are an expert at classifying scientific figures from materials science papers "
    "about atomic layer deposition (ALD) and atomic layer etching (ALE). "
    "Your task is to identify the chart or figure type shown in an image."
)

EXTRACT_SYSTEM = (
    "You are a Vision Language Model specialized in extracting structured data from "
    "scientific figure panels into Markdown tables."
)

EXTRACT_USER_BASE = (
    "Extract all quantitative data from this figure panel into a standard Markdown table.\n"
    "- Use the axis labels or legend entries as column headers.\n"
    "- Include units in column headers where visible.\n"
    "- Only include data that is clearly readable from the figure.\n"
    "- Output only the Markdown table, no preamble or explanation."
)

SUMMARIZE_SYSTEM = (
    "You are an expert in materials science specializing in atomic layer deposition (ALD) "
    "and atomic layer etching (ALE). Generate concise scientific summaries of figure panels."
)

SUMMARIZE_USER_BASE = (
    "Write a concise 1-3 sentence scientific summary of this figure panel.\n"
    "- Describe what the figure shows (chart type, axes, key trends or findings).\n"
    "- Be specific about scientific quantities and units where visible.\n"
    "- Output only the summary text, no preamble."
)

VQA_SYSTEM = (
    "You are an expert in materials science specializing in atomic layer deposition (ALD) "
    "and atomic layer etching (ALE). Answer questions about scientific figure panels accurately."
)

VQA_ANSWER_HINTS = {
    "Yes/No": "Answer with 'Yes' or 'No' followed by a brief explanation.",
    "Factoid": "Provide a concise factual answer.",
    "List": "Provide a comma-separated list or bullet points.",
    "Paragraph": "Provide a detailed 2-4 sentence answer with specific values where visible.",
}

VQA_MAX_TOKENS = {"Yes/No": 80, "Factoid": 80, "List": 120, "Paragraph": 300}


def build_classify_user(taxonomy: list[str]) -> str:
    taxonomy_str = "\n".join(f"- {t}" for t in taxonomy)
    return (
        f"Classify this figure panel into exactly one of the following categories:\n"
        f"{taxonomy_str}\n\n"
        f"Respond with ONLY the category name, nothing else."
    )


def build_extract_user(classification: str) -> str:
    if classification:
        return f"[Figure type: {classification}]\n{EXTRACT_USER_BASE}"
    return EXTRACT_USER_BASE


def build_summarize_user(classification: str, extraction: str) -> str:
    context = ""
    if classification:
        context += f"[Figure type: {classification}]\n"
    if extraction and extraction.strip():
        context += f"[Extracted data:\n{extraction.strip()}\n]\n"
    return context + SUMMARIZE_USER_BASE


def build_vqa_user(question: str, answer_type: str,
                   classification: str, extraction: str, summary: str) -> str:
    context = ""
    if classification:
        context += f"[Figure type: {classification}]\n"
    if extraction and extraction.strip():
        context += f"[Extracted data:\n{extraction.strip()}\n]\n"
    if summary and summary.strip():
        context += f"[Summary: {summary.strip()}]\n"
    hint = VQA_ANSWER_HINTS.get(answer_type, "")
    return f"{context}Question: {question}\n{hint}"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_with_adapter(adapter_path: str):
    print(f"[hybrid] Loading base model ...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID, min_pixels=64*28*28, max_pixels=256*28*28
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print(f"[hybrid] Loading adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path, device_map="balanced")
    model.eval()
    print("[hybrid] Model ready.")
    return model, processor


def unload_model(model, processor):
    del model
    del processor
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def generate(model, processor, pil_image: Image.Image,
             system_prompt: str, user_prompt: str, max_new_tokens: int = 200) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, padding=True, return_tensors="pt"
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return processor.batch_decode(
        [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()


# ── Utility ───────────────────────────────────────────────────────────────────

def normalize_label(raw: str, taxonomy: list[str]) -> str:
    from src.postprocess import normalize_label as _norm
    return _norm(raw, taxonomy)


def fix_extraction_format(table: str) -> str:
    t = table.strip()
    if not t:
        return ""
    if t.startswith("```"):
        return t
    if "|" in t:
        return f"```markdown\n{t}\n```"
    return t


def scan_test_figures(test_root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(test_root, "**", "images", "*.json"), recursive=True))


def crop_panel(image: Image.Image, box: dict) -> Image.Image:
    w, h = image.size
    x1 = max(0, int(box["x"]))
    y1 = max(0, int(box["y"]))
    x2 = min(w, int(box["x"]) + int(box["width"]))
    y2 = min(h, int(box["y"]) + int(box["height"]))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def load_figure(json_path: str):
    """Load JSON + image, return (data_dict, image_or_None)."""
    try:
        data = json.load(open(json_path))
    except Exception:
        return None, None
    if not isinstance(data, dict) or "sample_id" not in data:
        return None, None
    img_path = Path(json_path).with_suffix(".jpg")
    if not img_path.exists():
        img_path = Path(json_path).with_suffix(".png")
        if not img_path.exists():
            return data, None
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception:
        image = None
    return data, image


def normalize_vqa_field(vqa):
    if isinstance(vqa, list):
        d = {}
        for item in vqa:
            if isinstance(item, dict):
                d.update(item)
        return d
    return vqa if isinstance(vqa, dict) else {}


# ── Phase 1: Classification ───────────────────────────────────────────────────

def phase1_classify(json_files, adapter_path, output_dir, taxonomy):
    cache_file = output_dir / "classification_cache.json"
    if cache_file.exists():
        print(f"[phase1] Cache found, skipping.")
        return json.load(open(cache_file))

    classify_user = build_classify_user(taxonomy)
    model, processor = load_model_with_adapter(adapter_path)

    results = {}  # sample_id → {panel_id: {"label": str, "raw": str}}
    for idx, json_path in enumerate(json_files):
        data, image = load_figure(json_path)
        if data is None or image is None:
            continue
        bbox = data.get("bbox", {})
        if not bbox:
            continue

        sid = data["sample_id"]
        results[sid] = {}

        for panel_id, box in bbox.items():
            crop = crop_panel(image, box)
            if crop is None:
                continue
            try:
                raw = generate(model, processor, crop, CLASSIFY_SYSTEM, classify_user,
                               max_new_tokens=50)
                label = normalize_label(raw, taxonomy)
            except Exception as e:
                print(f"  WARN classify {sid}/{panel_id}: {e}")
                raw, label = "", "unknown"
            results[sid][panel_id] = {"label": label, "raw": raw}

        if (idx + 1) % 20 == 0:
            print(f"  [phase1] {idx + 1}/{len(json_files)}")

    unload_model(model, processor)
    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[phase1] Saved {len(results)} figures to {cache_file}")
    return results


# ── Phase 2: Flatten classification cache ────────────────────────────────────
# (Claude disambiguation removed — no API key available)

def phase2_claude_classify(classification_cache, output_dir, taxonomy):
    cache_file = output_dir / "corrected_classification.json"
    if cache_file.exists():
        print("[phase2] Cache found, skipping.")
        return json.load(open(cache_file))

    # Flatten {sid: {pid: {"label": str, "raw": str}}} → {sid: {pid: str}}
    corrected = {}
    for sid, panel_dict in classification_cache.items():
        corrected[sid] = {pid: info["label"] for pid, info in panel_dict.items()}

    n_panels = sum(len(v) for v in corrected.values())
    print(f"[phase2] Flattened {len(corrected)} figures ({n_panels} panels).")

    with open(cache_file, "w") as f:
        json.dump(corrected, f, indent=2)
    print(f"[phase2] Saved to {cache_file}")
    return corrected


# ── Phase 3: Extraction ───────────────────────────────────────────────────────

def phase3_extract(json_files, adapter_path, output_dir, classification):
    cache_file = output_dir / "extraction_cache.json"
    if cache_file.exists():
        print("[phase3] Cache found, skipping.")
        return json.load(open(cache_file))

    model, processor = load_model_with_adapter(adapter_path)

    results = {}  # sample_id → {panel_id: markdown_table}
    for idx, json_path in enumerate(json_files):
        data, image = load_figure(json_path)
        if data is None or image is None:
            continue
        bbox = data.get("bbox", {})
        if not bbox:
            continue

        sid = data["sample_id"]
        results[sid] = {}
        cls_map = classification.get(sid, {})

        for panel_id, box in bbox.items():
            crop = crop_panel(image, box)
            if crop is None:
                continue
            label = cls_map.get(panel_id, "")
            user_prompt = build_extract_user(label)
            try:
                raw = generate(model, processor, crop, EXTRACT_SYSTEM, user_prompt,
                               max_new_tokens=512)
                table = fix_extraction_format(raw)
            except Exception as e:
                print(f"  WARN extract {sid}/{panel_id}: {e}")
                table = ""
            results[sid][panel_id] = table

        if (idx + 1) % 20 == 0:
            print(f"  [phase3] {idx + 1}/{len(json_files)}")

    unload_model(model, processor)
    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[phase3] Saved {len(results)} figures to {cache_file}")
    return results


# ── Phase 4: Summarization ────────────────────────────────────────────────────

def phase4_summarize(json_files, adapter_path, output_dir, classification, extraction):
    cache_file = output_dir / "summarization_cache.json"
    if cache_file.exists():
        print("[phase4] Cache found, skipping.")
        return json.load(open(cache_file))

    model, processor = load_model_with_adapter(adapter_path)

    results = {}
    for idx, json_path in enumerate(json_files):
        data, image = load_figure(json_path)
        if data is None or image is None:
            continue
        bbox = data.get("bbox", {})
        if not bbox:
            continue

        sid = data["sample_id"]
        results[sid] = {}
        cls_map = classification.get(sid, {})
        ext_map = extraction.get(sid, {})

        for panel_id, box in bbox.items():
            crop = crop_panel(image, box)
            if crop is None:
                continue
            label = cls_map.get(panel_id, "")
            table = ext_map.get(panel_id, "")
            user_prompt = build_summarize_user(label, table)
            try:
                summary = generate(model, processor, crop, SUMMARIZE_SYSTEM, user_prompt,
                                   max_new_tokens=200)
            except Exception as e:
                print(f"  WARN summarize {sid}/{panel_id}: {e}")
                summary = ""
            results[sid][panel_id] = summary

        if (idx + 1) % 20 == 0:
            print(f"  [phase4] {idx + 1}/{len(json_files)}")

    unload_model(model, processor)
    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[phase4] Saved {len(results)} figures to {cache_file}")
    return results


# ── Phase 5: VQA ─────────────────────────────────────────────────────────────

def phase5_vqa(json_files, adapter_path, output_dir, classification, extraction, summarization):
    cache_file = output_dir / "vqa_cache.json"
    if cache_file.exists():
        print("[phase5] Cache found, skipping.")
        return json.load(open(cache_file))

    model, processor = load_model_with_adapter(adapter_path)

    results = {}  # sample_id → {panel_id: [qa_dict]}
    for idx, json_path in enumerate(json_files):
        data, image = load_figure(json_path)
        if data is None or image is None:
            continue
        bbox = data.get("bbox", {})
        vqa_data = normalize_vqa_field(data.get("vqa", {}))
        if not bbox or not vqa_data:
            continue

        sid = data["sample_id"]
        results[sid] = {}
        cls_map = classification.get(sid, {})
        ext_map = extraction.get(sid, {})
        sum_map = summarization.get(sid, {})

        for panel_id, qa_list in vqa_data.items():
            if panel_id not in bbox or not isinstance(qa_list, list):
                continue
            crop = crop_panel(image, bbox[panel_id])
            if crop is None:
                continue

            label = cls_map.get(panel_id, "")
            table = ext_map.get(panel_id, "")
            summary = sum_map.get(panel_id, "")
            answered = []

            for qa in qa_list:
                question = qa.get("question", "").strip()
                answer_type = qa.get("answer_type", "Paragraph")
                if not question:
                    answered.append(qa)
                    continue
                user_prompt = build_vqa_user(question, answer_type, label, table, summary)
                max_tok = VQA_MAX_TOKENS.get(answer_type, 200)
                try:
                    answer = generate(model, processor, crop, VQA_SYSTEM, user_prompt,
                                      max_new_tokens=max_tok)
                except Exception as e:
                    print(f"  WARN vqa {sid}/{panel_id}: {e}")
                    answer = ""
                answered.append({
                    "question_type": qa.get("question_type", ""),
                    "question": question,
                    "answer_type": answer_type,
                    "answer": answer,
                })

            if answered:
                results[sid][panel_id] = answered

        if (idx + 1) % 20 == 0:
            print(f"  [phase5] {idx + 1}/{len(json_files)}")

    unload_model(model, processor)
    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[phase5] Saved VQA for {len(results)} figures to {cache_file}")
    return results


# ── Phase 6: Pass-through VQA ────────────────────────────────────────────────
# (Claude VQA enrichment removed — no API key available)

def phase6_enrich_vqa(vqa_cache, extraction_cache, classification, summarization, output_dir):
    cache_file = output_dir / "enriched_vqa.json"
    if cache_file.exists():
        print("[phase6] Cache found, skipping.")
        return json.load(open(cache_file))

    # Pass VQA cache through unchanged
    n_panels = sum(len(v) for v in vqa_cache.values())
    print(f"[phase6] Pass-through: {len(vqa_cache)} figures ({n_panels} panels).")

    with open(cache_file, "w") as f:
        json.dump(vqa_cache, f, indent=2)
    print(f"[phase6] Saved to {cache_file}")
    return vqa_cache


# ── Phase 7: Package submissions ──────────────────────────────────────────────

def phase7_package(json_files, classification, extraction, summarization, vqa, output_dir):
    # Build sample_id → bbox mapping
    bbox_map = {}
    for json_path in json_files:
        try:
            data = json.load(open(json_path))
        except Exception:
            continue
        if isinstance(data, dict) and "sample_id" in data:
            bbox_map[data["sample_id"]] = data.get("bbox", {})

    all_sids = sorted(set(bbox_map.keys()))

    def make_records(task_key, task_data):
        records = []
        for sid in all_sids:
            bbox = bbox_map.get(sid, {})
            task_val = task_data.get(sid, {})
            records.append({
                "sample_id": sid,
                "bbox": bbox,
                task_key: task_val,
            })
        return records

    task_configs = [
        ("classification", "task1_submission.zip", classification),
        ("data_extraction", "task2_submission.zip", extraction),
        ("summarization", "task3_submission.zip", summarization),
        ("vqa", "task4_submission.zip", vqa),
    ]

    for task_key, zip_name, task_data in task_configs:
        records = make_records(task_key, task_data)
        json_path = output_dir / f"{task_key}_prediction_data.json"
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)
        zip_path = output_dir / zip_name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname="prediction_data.json")
        print(f"[phase7] {zip_name} ({len(records)} figures)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-adapter", required=True)
    parser.add_argument("--extraction-adapter", required=True)
    parser.add_argument("--summarization-adapter", required=True)
    parser.add_argument("--vqa-adapter", required=True)
    parser.add_argument("--test-root", default="ALD-E-ImageMiner/icdar2026-competition-data/test")
    parser.add_argument("--output-dir", default="outputs/hybrid")
    parser.add_argument("--force-phase", type=int, default=0,
                        help="Delete cache for this phase and all later phases, then re-run")
    args = parser.parse_args()

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        import huggingface_hub
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle forced re-run
    cache_files = [
        output_dir / "classification_cache.json",    # phase 1
        output_dir / "corrected_classification.json", # phase 2
        output_dir / "extraction_cache.json",         # phase 3
        output_dir / "summarization_cache.json",      # phase 4
        output_dir / "vqa_cache.json",                # phase 5
        output_dir / "enriched_vqa.json",             # phase 6
    ]
    if args.force_phase > 0:
        for p_idx in range(args.force_phase - 1, len(cache_files)):
            if cache_files[p_idx].exists():
                cache_files[p_idx].unlink()
                print(f"[hybrid] Cleared cache: {cache_files[p_idx].name}")

    taxonomy = load_taxonomy()
    print(f"[hybrid] Taxonomy: {len(taxonomy)} classes")

    json_files = scan_test_figures(args.test_root)
    print(f"[hybrid] Test figures: {len(json_files)}")

    print("\n=== Phase 1: Classification ===")
    cls_cache = phase1_classify(json_files, args.classification_adapter, output_dir, taxonomy)

    print("\n=== Phase 2: Claude disambiguation ===")
    corrected_cls = phase2_claude_classify(cls_cache, output_dir, taxonomy)

    print("\n=== Phase 3: Extraction (with classification context) ===")
    ext_cache = phase3_extract(json_files, args.extraction_adapter, output_dir, corrected_cls)

    print("\n=== Phase 4: Summarization (with classification + extraction context) ===")
    sum_cache = phase4_summarize(json_files, args.summarization_adapter, output_dir,
                                 corrected_cls, ext_cache)

    print("\n=== Phase 5: VQA (with all context) ===")
    vqa_cache = phase5_vqa(json_files, args.vqa_adapter, output_dir,
                            corrected_cls, ext_cache, sum_cache)

    print("\n=== Phase 6: Claude VQA paragraph enrichment ===")
    enriched_vqa = phase6_enrich_vqa(vqa_cache, ext_cache, corrected_cls, sum_cache, output_dir)

    print("\n=== Phase 7: Packaging per-task submission ZIPs ===")
    phase7_package(json_files, corrected_cls, ext_cache, sum_cache, enriched_vqa, output_dir)

    print(f"\n[hybrid] Done. Submission ZIPs in {output_dir}/")
    print("  task1_submission.zip  → CodaBench Task 1 (Classification)")
    print("  task2_submission.zip  → CodaBench Task 2 (Data Extraction)")
    print("  task3_submission.zip  → CodaBench Task 3 (Summarization)")
    print("  task4_submission.zip  → CodaBench Task 4 (VQA)")


if __name__ == "__main__":
    main()
