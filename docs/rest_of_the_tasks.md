# Plan: Sci-ImageMiner — Next Steps

## Context

The project is a submission to the ICDAR 2026 ALD/E-ImageMiner competition (4 tasks: Classification, Table Extraction, Summarization, VQA). Work so far covers **Task 1 (Classification)** only. Best CodaBench practice-phase score: **Acc=0.7925, F1=0.8059** (Submission 4: 6-model weighted ensemble). The competition has moved to its **evaluation phase**, which adds a `remaining-test-data` split merged with the original practice-phase test set. The GitHub repo has 3 new commits that are not yet pulled locally.

---

## Part 1 — Git Pull

**Goal:** Pull 3 remote commits into `ALD-E-ImageMiner/`:
- `c9406502` — Fixing incorrect sample_id issue (important — predictions may use wrong IDs)
- `6391f6f2` — Fixing incorrect sample_id issue
- `d15470a2` — Merging test data for evaluation phase (`remaining-test-data/` folder)

**Steps:**
```bash
cd /home/jovyan/tvinchur/sci_image_miner/ALD-E-ImageMiner
git pull origin main
```

**Impact:** After pull, check whether sample_id fix affects the prediction JSON format used in `src/classify.py`. Also confirm `remaining-test-data/` folder structure matches `practice-phase/`.

---

## Part 2 — Evaluation Phase Submission Zip

**Best ensemble (dev_v2_weighted):**
- 6 models: `llama_qlora_ep3` (F1=0.581), `qwen2vl_qlora_ep6` (F1=0.487), `swinv2_aclfig` (F1=0.426), `qwen2vl_zero_shot` (F1=0.348), `inception_resnet_v2_v2` (F1=0.353), `efficientnet_b0` (F1=0.283)
- Dev Macro F1: **0.6119** (best on dev set)
- Weights: proportional to dev macro F1 scores
- Existing practice-phase test predictions: `outputs/ensemble/test/prediction_data.json`

**The evaluation phase requires NEW predictions on `remaining-test-data/`.** Steps:

### Step 2a — Prepare remaining-test-data crops
```bash
python prepare_external_data.py  # or data_prep.py
```
Need to run `src/data_prep.py` (panel crop extraction) on the new remaining-test-data figures. Check existing `data/test_panels.csv` and add new rows, or create a separate `data/remaining_test_panels.csv`.

**Critical files:**
- `src/data_prep.py` — extracts panel crops from figure images
- `data/dev_panels.csv` / `data/train_panels.csv` — format reference

### Step 2b — Run all 6 model predictions on remaining-test-data

For each model, run predict script targeting remaining-test-data crops:
1. `run_cnn_predict.py` — efficientnet_b0, inception_resnet_v2, swinv2_aclfig
2. `run_llama_qlora_predict.py` — llama_qlora_ep3
3. `run_qwen_qlora_predict.py` — qwen2vl_qlora_ep6
4. `run_classify.py` — qwen2vl zero-shot

Save outputs under `outputs/*/remaining_test/prediction_data.json` (parallel to existing `dev/` and `test/` dirs).

### Step 2c — Merge predictions and generate zip

Use `src/ensemble.py` + `src/classify.py` to:
1. Load practice-phase test predictions for each model
2. Load remaining-test-data predictions for each model
3. Concatenate into merged prediction list
4. Apply weighted ensemble
5. Package as `prediction_data.json` → zip

**Output:** `outputs/submission_eval_phase_v2_weighted.zip`

**Key files:**
- `src/ensemble.py` — weighted ensemble logic
- `src/classify.py` — submission zip creation
- `build_6model_ensemble.sh` — script reference for weights/model order
- `outputs/ensemble/test/prediction_data.json` — existing practice-phase predictions

---

## Part 3 — Single Model Improvement Experiments

**Current best:**
| | Dev Macro F1 |
|---|---|
| `llama_qlora_ep3` | 0.581 |
| `dev_v2_weighted` ensemble | 0.612 |

**Two experiments to run (in parallel on separate GPUs):**

### Experiment A: Llama + ACL-Fig fine-tuning
- Add 1,541 ACL-Fig samples to training data alongside original 2,420 (total ~3,961)
- Create `data/train_aclfig_combined.csv` (orig train + ACL-Fig rows)
- Re-run `run_vlm_finetune.py` with Llama-3.2-11B-Vision-Instruct, same QLoRA config (r=16, alpha=32, lr=2e-4), 3 epochs
- Output dir: `outputs/llama_qlora_aclfig_ep3/`
- Motivation: CNNs got +6.4pts from ACL-Fig. VLMs may similarly benefit from more domain-relevant visual diversity.

### Experiment B: Llama higher LoRA rank (r=32)
- Same data as original (2,420 train samples)
- Change LoRA `r=32, alpha=64` in `run_vlm_finetune.py`
- 3 epochs (same as current best ep3)
- Output dir: `outputs/llama_qlora_r32_ep3/`
- Motivation: More trainable parameters for fine-grained 36-class discrimination

**Evaluation:** After each, run dev predictions and compare macro F1 vs 0.581 baseline. If either exceeds 0.612, it can replace/augment the ensemble.

---

## Part 4 — Plan for Remaining Tasks (Tasks 2–4)

Based on `plan_4_alltasks.jpg` and competition pages: same VLM pipeline as classification but different outputs.

### Architecture (shared VLM backbone)
- Base model: **Qwen2.5-VL-7B-Instruct** or fine-tuned **Llama-3.2-11B-Vision**
- Already have `src/models/qwen2vl.py` and `src/models/llama_vision.py`
- Already have `ALD-E-ImageMiner/Prompts.md` with Qwen prompts for all 4 tasks

### Task 2: Data Table Extraction
- **Input:** Panel crop image
- **Output:** Markdown table (columns + values)
- **Metrics:** RMS (Relative Mapping Similarity), TEDS (Tree Edit Distance Similarity)
- **Approach:** Qwen2.5-VL-7B zero-shot or fine-tuned with ground-truth `.data.txt` files from train set
- **Ground truth:** `filename.data.txt` already exists in `ALD-E-ImageMiner/` train figures
- **Script to write:** `run_table_extract.py` — analogous to `run_classify.py` but with table extraction prompt

### Task 3: Summarization
- **Input:** Panel crop + optional caption (`.caption.txt`)
- **Output:** 1–3 sentence scientific summary
- **Metrics:** ROUGE, BERTScore
- **Approach:** Qwen2.5-VL-7B with caption context (`.caption.txt` exists in data)
- **Ground truth:** `filename.summary.txt` in train set
- **Script to write:** `run_summarize.py`

### Task 4: VQA (4 subtasks)
- **Input:** Panel image + natural-language question
- **Output:** Yes/No | Factoid | List | Paragraph (depending on question type)
- **Metrics:** F1/accuracy (Yes/No, List), Exact Match + ROUGE (Factoid), ROUGE + BERTScore (Paragraph)
- **Approach:** Qwen2.5-VL-7B with question in prompt; parse answer into correct format
- **Script to write:** `run_vqa.py`

### Recommended execution order
1. Classification evaluation submission (Part 2) — highest priority
2. Task 3: Summarization — zero-shot first, then fine-tune on `.summary.txt` ground truth
3. Task 2: Table Extraction — zero-shot first, then fine-tune on `.data.txt` ground truth
4. Task 4: VQA — 4 subtasks, build answer-type router (Yes/No → classification head, Factoid/List/Paragraph → Qwen2.5-VL generation)

### Reuse
- `src/models/qwen2vl.py` — base for all new tasks
- `ALD-E-ImageMiner/Prompts.md` — prompts already written for Tasks 2–4 by organizers
- `src/evaluation.py` — add ROUGE/BERTScore/TEDS metrics alongside existing F1 code
- `src/classify.py` → create analogous `src/extract.py`, `src/summarize.py`, `src/vqa.py`

---

## Verification

### For Part 1:
- `git log --oneline -5` shows the 3 new commits merged
- Verify `remaining-test-data/` folder exists with same structure as `practice-phase/`

### For Part 2:
- Run ensemble eval script on dev: confirm dev Macro F1 ≈ 0.612 (sanity check)
- Check `prediction_data.json` sample count = practice-phase panels + remaining-test-data panels
- Validate zip format matches CodaBench requirements (same as practice phase)

### For Part 3:
- Compare Llama + ACL-Fig fine-tune dev F1 vs current best (0.581)
- Decide whether to replace ensemble or keep ensemble as primary

### For Parts 2–4 (future tasks):
- Use dev ground truth (`.data.txt`, `.summary.txt`) to score before submitting
- Run `BERTScore` and `rouge-score` pip packages for offline eval
