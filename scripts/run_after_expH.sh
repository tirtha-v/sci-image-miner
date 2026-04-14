#!/bin/bash
# Queue to run after Exp H finishes:
# 1. Dev eval on Exp H
# 2. Re-ensemble llama + expH + swinv2 + inception
# 3. Hybrid pipeline (Phases 2-7)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=/home/jovyan/tvinchur/uq_uma/conda_envs/llpr_og_paper/bin/python3
TEST_ROOT="ALD-E-ImageMiner/icdar2026-competition-data/test"

for HF_TOKEN_FILE in \
    "/home/jovyan/tvinchur/hf_token_for_cluster.txt" \
    "/home/jovyan/shared-scratch-tvinchur-pvc/tvinchur/hf_token_for_cluster.txt"; do
    if [ -f "$HF_TOKEN_FILE" ]; then
        export HUGGING_FACE_HUB_TOKEN=$(cat "$HF_TOKEN_FILE")
        break
    fi
done

echo "=== Step 1: Exp H dev eval ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON run_qwen_qlora_predict.py \
    --adapter-path outputs/qwen2vl_qlora_expH/adapter \
    --panels-csv data/dev_panels.csv \
    --output-dir outputs/qwen2vl_qlora_expH/dev \
    --device cuda:0

echo ""
echo "=== Step 1b: Exp H dev F1 ==="
$PYTHON -c "
import json, glob
from sklearn.metrics import f1_score
preds = json.load(open('outputs/qwen2vl_qlora_expH/dev/prediction_data.json'))
gt_map = {}
for jf in glob.glob('ALD-E-ImageMiner/icdar2026-competition-data/dev/**/images/*.json', recursive=True):
    d = json.load(open(jf))
    if isinstance(d, dict) and 'classification' in d:
        for pid, lbl in d['classification'].items():
            gt_map[(d['sample_id'], pid)] = lbl
y_true, y_pred = [], []
for p in preds:
    for pid, lbl in p['classification'].items():
        gt = gt_map.get((p['sample_id'], pid))
        if gt: y_true.append(gt); y_pred.append(lbl)
macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
print(f'  Exp H dev macro F1={macro:.4f}  weighted={weighted:.4f}')
# Write score file for ensemble script
with open('outputs/qwen2vl_qlora_expH/dev_macro_f1.txt', 'w') as f:
    f.write(str(round(macro, 4)))
"

echo ""
echo "=== Step 2: Exp H test inference ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON run_qwen_qlora_predict.py \
    --adapter-path outputs/qwen2vl_qlora_expH/adapter \
    --test-root "$TEST_ROOT" \
    --output-dir outputs/qwen2vl_qlora_expH/test \
    --device cuda:0

echo ""
echo "=== Step 3: Build best4 ensemble with Exp H ==="
EXPH_F1=$(cat outputs/qwen2vl_qlora_expH/dev_macro_f1.txt)
echo "  Using Exp H weight: $EXPH_F1"
mkdir -p outputs/submission_best4_expH
$PYTHON src/ensemble.py \
    --predictions \
        outputs/llama_vision/test/prediction_data.json \
        outputs/qwen2vl_qlora_expH/test/prediction_data.json \
        outputs/cnn/swinv2_aclfig/test/prediction_data.json \
        outputs/cnn/inception_resnet_v2/test_eval/prediction_data.json \
    --weights 0.58 "$EXPH_F1" 0.43 0.43 \
    --output outputs/submission_best4_expH/prediction_data.json
cd outputs/submission_best4_expH && zip submission_best4_expH.zip prediction_data.json && cd ../..
echo "  Zipped: outputs/submission_best4_expH/submission_best4_expH.zip"

echo ""
echo "=== Step 4: Hybrid pipeline (Phases 2-7) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON run_hybrid_pipeline.py \
    --classification-adapter outputs/qwen2vl_qlora_expG/adapter \
    --extraction-adapter outputs/qwen2vl_extraction_qlora/adapter \
    --summarization-adapter outputs/qwen2vl_summarization_qlora/adapter \
    --vqa-adapter outputs/qwen2vl_vqa_qlora/adapter \
    --test-root "$TEST_ROOT" \
    --output-dir outputs/hybrid \
    --force-phase 2

echo ""
echo "=== All done ==="
echo "  Submit to CodaBench:"
echo "  Task 1: outputs/submission_best4_expH/submission_best4_expH.zip"
echo "  Task 2: outputs/hybrid/task2_submission.zip"
echo "  Task 3: outputs/hybrid/task3_submission.zip"
echo "  Task 4: outputs/hybrid/task4_submission.zip"
