# VLMinators — Sci-ImageMiner 2026

**Top-5 finish** at the [ICDAR 2026 Sci-ImageMiner Competition](https://sites.google.com/view/sci-imageminer/) on scientific figure understanding for ALD/ALE materials science papers.

We participated in all four tasks: Figure Classification, Data Extraction, Summarization, and Visual Question Answering (VQA).

---

## Tasks and Results

| Task | Description | Best Score |
|------|-------------|------------|
| Task 1 — Classification | Classify figure panels into 49 chart/diagram types | F1 = 0.8059 (eval phase) |
| Task 2 — Data Extraction | Extract quantitative data into Markdown tables | Weighted=40.8, TEDS=64.3 |
| Task 3 — Summarization | Generate 1–3 sentence scientific summaries | — |
| Task 4 — VQA | Answer Yes/No, Factoid, List, Paragraph questions | — |

---

## Key Techniques

- **QLoRA fine-tuning** of `Qwen/Qwen2.5-VL-7B-Instruct` for all four tasks (r=32, α=64, bfloat16)
- **Selective label hints**: targeted disambiguation descriptions injected only for confusable class pairs (e.g. line chart vs. multiple line chart, spectra variants)
- **Cross-task context injection**: Task 1 labels → Task 2 prompts; Task 1+2 outputs → Task 3 prompts; all three → Task 4 prompts
- **Post-processing pipeline**: automated label normalization + visual reclassification of unknown panels via Claude Code

---

## Repository Structure

```
sci-image-miner/
├── src/                          # Core Python library
│   ├── prompts.py                # VLM prompt templates + selective label hints
│   ├── ensemble.py               # Weighted voting ensemble
│   ├── postprocess.py            # Label normalization
│   ├── evaluation.py             # Metrics (macro/weighted F1)
│   ├── cnn/                      # CNN classifier (EfficientNet, SwinV2, Inception)
│   ├── models/                   # Zero-shot VLM wrappers
│   └── vlm_finetune/             # QLoRA training + inference
├── class_post_processing/        # Post-processing for classification
│   ├── flag_panels.py            # Identify unknown/out-of-taxonomy predictions
│   └── merge_classifications.py  # Apply corrections to final JSON
├── data/                         # Dataset metadata (CSVs only; images not included)
│   ├── train_panels.csv          # 2,421 training panels
│   ├── dev_panels.csv            # 363 dev panels
│   ├── train_aclfig.csv          # 3,962 ACL-Fig panels
│   ├── train_extraction.csv      # 12,667 extraction samples
│   ├── train_summarization.csv   # 1,573 summarization samples
│   └── train_vqa.csv             # 3,041 VQA pairs
├── run_vlm_finetune.py           # QLoRA fine-tuning (classification)
├── run_qwen_qlora_predict.py     # Classification inference
├── run_qwen_extraction_finetune.py
├── run_qwen_extraction_predict.py
├── run_qwen_summarization_finetune.py
├── run_qwen_summarization_predict.py
├── run_qwen_vqa_finetune.py
├── run_qwen_vqa_predict.py
├── run_hybrid_pipeline.py        # End-to-end context-injection pipeline
├── run_cnn_train.py              # CNN fine-tuning
├── run_cnn_predict.py            # CNN inference
├── src/ensemble.py               # Weighted voting ensemble
├── paper/                        # ICDAR 2026 paper (LaTeX)
├── sub_template/                 # TIB Open Publishing template
├── data_extraction/              # Data extraction task documentation
├── environment.yml               # Conda environment
└── ALD-E-ImageMiner/             # Competition data (git submodule)
```

---

## Setup

```bash
# 1. Clone this repo
git clone https://github.com/tirtha-v/sci-image-miner.git
cd sci-image-miner

# 2. Clone competition data into the expected location
git clone https://github.com/sciknoworg/ALD-E-ImageMiner.git ALD-E-ImageMiner

# 3. Create conda environment
conda env create -f environment.yml
conda activate llpr_og_paper

# 4. Set HuggingFace token (needed for Llama/Qwen downloads)
export HUGGING_FACE_HUB_TOKEN=<your_token>
```

---

## Quick Start

### Task 1 — Classification

**Fine-tune Qwen2.5-VL:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_vlm_finetune.py \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --train-csv data/train_panels.csv \
    --output-dir outputs/qwen2vl_qlora \
    --epochs 5 --lora-r 32 --lora-alpha 64 --lr 1e-4
```

**Inference on test set:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_qwen_qlora_predict.py \
    --adapter-path outputs/qwen2vl_qlora/adapter \
    --test-root ALD-E-ImageMiner/icdar2026-competition-data/test \
    --output-dir outputs/qwen2vl_qlora/test
```

**Post-processing:**
```bash
# Flag unknown/invalid labels
python class_post_processing/flag_panels.py \
    outputs/qwen2vl_qlora/test/prediction_data.json \
    --output-dir outputs/flagged

# After manual/Claude reclassification, merge corrections
python class_post_processing/merge_classifications.py \
    outputs/qwen2vl_qlora/test/prediction_data.json \
    outputs/flagged/flagged_panels.json \
    --output-dir outputs/corrected
```

**Ensemble (weighted voting):**
```bash
python src/ensemble.py \
    --predictions \
        outputs/model_a/test/prediction_data.json \
        outputs/model_b/test/prediction_data.json \
    --weights 0.58 0.55 \
    --output outputs/ensemble/prediction_data.json
```

### Tasks 2–4 — Extraction / Summarization / VQA

Each follows the same fine-tune → predict pattern:

```bash
# Data Extraction
python run_qwen_extraction_finetune.py --output-dir outputs/extraction_qlora
python run_qwen_extraction_predict.py \
    --adapter-path outputs/extraction_qlora/adapter \
    --test-root ALD-E-ImageMiner/icdar2026-competition-data/test \
    --output-dir outputs/extraction_qlora/test

# Summarization
python run_qwen_summarization_finetune.py --output-dir outputs/summarization_qlora
python run_qwen_summarization_predict.py \
    --adapter-path outputs/summarization_qlora/adapter \
    --mode test --output-dir outputs/summarization_qlora/test

# VQA
python run_qwen_vqa_finetune.py --output-dir outputs/vqa_qlora
python run_qwen_vqa_predict.py \
    --adapter-path outputs/vqa_qlora/adapter \
    --mode test --output-dir outputs/vqa_qlora/test
```

### Cross-Task Context Injection Pipeline

Run all four tasks sequentially with context chained across tasks:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_hybrid_pipeline.py \
    --classification-adapter outputs/qwen2vl_qlora/adapter \
    --extraction-adapter outputs/extraction_qlora/adapter \
    --summarization-adapter outputs/summarization_qlora/adapter \
    --vqa-adapter outputs/vqa_qlora/adapter \
    --test-root ALD-E-ImageMiner/icdar2026-competition-data/test \
    --output-dir outputs/hybrid
```

Outputs: `outputs/hybrid/task{1,2,3,4}_submission.zip` — one per CodaBench task.

---

## Models

| Model | Used for | HuggingFace ID |
|-------|----------|----------------|
| Qwen2.5-VL-7B-Instruct | Tasks 1–4 (fine-tuned) | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Llama-3.2-11B-Vision | Task 1 (fine-tuned) | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| SwinV2-Base | Task 1 (CNN fine-tuned) | `swinv2_base_window16_256` (timm) |
| Inception-ResNet-v2 | Task 1 (CNN fine-tuned) | `inception_resnet_v2` (timm) |
| EfficientNet-B0 | Task 1 (CNN fine-tuned) | `efficientnet_b0` (timm) |
| LLaVA-v1.6-Mistral-7B | Task 1 (zero-shot baseline) | `llava-hf/llava-v1.6-mistral-7b-hf` |
| Phi-3.5-Vision | Task 1 (zero-shot baseline) | `microsoft/Phi-3.5-vision-instruct` |
| InstructBLIP-Vicuna-7B | Task 1 (zero-shot baseline) | `Salesforce/instructblip-vicuna-7b` |

---

## External Datasets Used

| Dataset | Samples | Used for |
|---------|---------|----------|
| [ACL-Fig](https://github.com/allenai/acl-fig) | 3,961 | Task 1 classification augmentation |
| [DocFigure](https://github.com/jobinkv/DocFigure) | varies | Task 1 combined training |

---

## Paper

> **VLMinators at Sci-ImageMiner 2026 Tasks 1–4: QLoRA Fine-tuning of Qwen2.5-VL with Selective Label Disambiguation and Cross-Task Context Injection**
>
> ICDAR 2026 Sci-ImageMiner Competition Proceedings. TIB Open Publishing.

LaTeX source: [`paper/sci_imageminer_2026.tex`](paper/sci_imageminer_2026.tex)

---

## Citation

```bibtex
@inproceedings{vlminators2026sciimage,
  title     = {VLMinators at Sci-ImageMiner 2026 Tasks 1--4: QLoRA Fine-tuning of
               Qwen2.5-VL with Selective Label Disambiguation and Cross-Task
               Context Injection},
  author    = {PLACEHOLDER},
  booktitle = {Sci-ImageMiner 2026: Scientific Image Mining Challenge at ICDAR 2026},
  year      = {2026},
  publisher = {TIB Open Publishing}
}
```
