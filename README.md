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
├── tasks/                        # Reproduction scripts, organised by task
│   ├── task1_classification/
│   │   ├── finetune_vlm.py       # QLoRA fine-tune Llama-3.2-11B (VLM)
│   │   ├── finetune_cnn.py       # Fine-tune CNN (EfficientNet / SwinV2 / Inception)
│   │   ├── predict_qwen.py       # Inference: QLoRA-finetuned Qwen2.5-VL
│   │   ├── predict_llama.py      # Inference: QLoRA-finetuned Llama-3.2-11B
│   │   ├── predict_cnn.py        # Inference: CNN classifier
│   │   └── predict_zeroshot.py   # Zero-shot inference (LLaVA / Phi / InstructBLIP)
│   ├── task2_extraction/
│   │   ├── finetune.py           # QLoRA fine-tune Qwen2.5-VL for data extraction
│   │   └── predict.py            # Extraction inference + submission zip
│   ├── task3_summarization/
│   │   ├── finetune.py           # QLoRA fine-tune Qwen2.5-VL for summarization
│   │   └── predict.py            # Summarization inference + submission zip
│   ├── task4_vqa/
│   │   ├── finetune.py           # QLoRA fine-tune Qwen2.5-VL for VQA
│   │   └── predict.py            # VQA inference + submission zip
│   └── pipeline.py               # End-to-end context-injection pipeline (Tasks 1→4)
├── src/                          # Core Python library
│   ├── prompts.py                # VLM prompt templates + selective label hints
│   ├── ensemble.py               # Weighted voting ensemble
│   ├── postprocess.py            # Label normalization
│   ├── evaluation.py             # Metrics (macro/weighted F1)
│   ├── cnn/                      # CNN classifier (EfficientNet, SwinV2, Inception)
│   ├── models/                   # Zero-shot VLM wrappers
│   └── vlm_finetune/             # QLoRA training + inference helpers
├── postprocessing/               # Post-processing for Task 1 output
│   ├── flag_panels.py            # Identify unknown/out-of-taxonomy predictions
│   └── merge_classifications.py  # Apply corrections to final JSON
├── data_prep/                    # Data preparation utilities
│   ├── prepare_external_data.py  # Map ACL-Fig / DocFigure to competition taxonomy
│   └── prepare_vqa_data.py       # Extract VQA pairs from competition JSONs
├── data/                         # Dataset metadata (CSVs only; images not included)
│   ├── train_panels.csv          # 2,421 training panels
│   ├── dev_panels.csv            # 363 dev panels
│   ├── train_aclfig.csv          # 3,962 ACL-Fig panels
│   ├── train_extraction.csv      # 12,667 extraction samples
│   ├── train_summarization.csv   # 1,573 summarization samples
│   └── train_vqa.csv             # 3,041 VQA pairs
├── paper/                        # ICDAR 2026 paper (LaTeX)
│   ├── sci_imageminer_2026.tex   # Main paper source
│   ├── local.bib                 # Bibliography
│   └── tibop-article.cls         # TIB Open Publishing class file
├── scripts/                      # Shell scripts for training and evaluation
│   ├── train_task1_vlm.sh        # Fine-tune Qwen2.5-VL (Task 1)
│   ├── train_task1_llama.sh      # Fine-tune Llama-3.2-11B (Task 1)
│   ├── train_task1_cnn.sh        # Fine-tune CNN models (Task 1)
│   ├── train_task2_extraction.sh # Fine-tune + infer Task 2
│   ├── build_ensemble.sh         # Build weighted voting ensemble
│   ├── eval_task1_ensemble.sh    # Evaluate ensemble on dev set
│   └── eval_task1_vlms.sh        # Evaluate VLMs on dev set
├── environment.yml               # Conda environment
└── ALD-E-ImageMiner/             # Competition data (cloned separately — see Setup)
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

All commands are run from the **repo root**.

**Fine-tune Qwen2.5-VL:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tasks/task1_classification/finetune_vlm.py \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --train-csv data/train_panels.csv \
    --output-dir outputs/qwen2vl_qlora \
    --epochs 5 --lora-r 32 --lora-alpha 64 --lr 1e-4
```

> To fine-tune Llama-3.2-11B instead, use the same script with `--model-id meta-llama/Llama-3.2-11B-Vision-Instruct`.

**Fine-tune CNN (EfficientNet / SwinV2 / Inception-ResNet-v2):**
```bash
python tasks/task1_classification/finetune_cnn.py --model efficientnet_b0
```

**Inference on dev/test set (Qwen QLoRA):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tasks/task1_classification/predict_qwen.py \
    --adapter-path outputs/qwen2vl_qlora/adapter \
    --test-root ALD-E-ImageMiner/icdar2026-competition-data/test \
    --output-dir outputs/qwen2vl_qlora/test
```

**Post-processing (flag unknown labels → merge corrections):**
```bash
python postprocessing/flag_panels.py \
    outputs/qwen2vl_qlora/test/prediction_data.json \
    --output-dir outputs/flagged

python postprocessing/merge_classifications.py \
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
# Task 2 — Data Extraction
python tasks/task2_extraction/finetune.py --output-dir outputs/extraction_qlora
python tasks/task2_extraction/predict.py \
    --adapter-path outputs/extraction_qlora/adapter \
    --mode test \
    --test-root ALD-E-ImageMiner/icdar2026-competition-data/test \
    --output-dir outputs/extraction_qlora/test

# Task 3 — Summarization
python tasks/task3_summarization/finetune.py --output-dir outputs/summarization_qlora
python tasks/task3_summarization/predict.py \
    --adapter-path outputs/summarization_qlora/adapter \
    --mode test --output-dir outputs/summarization_qlora/test

# Task 4 — VQA
python tasks/task4_vqa/finetune.py --output-dir outputs/vqa_qlora
python tasks/task4_vqa/predict.py \
    --adapter-path outputs/vqa_qlora/adapter \
    --mode test --output-dir outputs/vqa_qlora/test
```

### Cross-Task Context Injection Pipeline

Run all four tasks sequentially with context chained across tasks:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tasks/pipeline.py \
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
  author    = {VLMinators},
  booktitle = {Sci-ImageMiner 2026: Scientific Image Mining Challenge at ICDAR 2026},
  year      = {2026},
  publisher = {TIB Open Publishing}
}
```
