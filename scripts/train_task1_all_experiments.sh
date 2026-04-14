#!/bin/bash
# =============================================================================
# Fine-tuning experiments to push single-model F1 > 0.85
#
# Llama experiments (4-bit QLoRA, all 4 GPUs — bf16 materialization needs ~5.5GB/GPU):
#   A: Llama + ACL-Fig data (3961 samples)  → outputs/llama_qlora_aclfig_ep3/
#   B: Llama r=32                           → outputs/llama_qlora_r32_ep3/
#
# Qwen2.5-VL experiments (bfloat16, GPUs 0,1):
#   C: Qwen + ACL-Fig data (3961 samples)  → outputs/qwen2vl_qlora_aclfig_ep3/
#   D: Qwen r=32                           → outputs/qwen2vl_qlora_r32_ep3/
#
# All Qwen experiments run sequentially on GPUs 0,1 (14GB bfloat16 across 2x 11GB).
# Llama experiments run on all 4 GPUs (4-bit ~10GB across 4x 11GB).
#
# Baseline to beat:
#   Llama ep3: dev macro F1=0.581, weighted F1=0.737, est CodaBench ~0.80
#   Qwen ep6:  dev macro F1=0.487 (old run, only 2420 samples, single A6000)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"

# HUGGING_FACE_HUB_TOKEN must be set in your environment

mkdir -p logs

# ---------------------------------------------------------------------------
# Helper: evaluate adapter on dev and print macro/weighted F1
# ---------------------------------------------------------------------------
TEST_ROOT="ALD-E-ImageMiner/icdar2026-competition-data/test"

# ---------------------------------------------------------------------------
# Helper: run test inference and build submission zip
# ---------------------------------------------------------------------------
predict_test() {
    local ADAPTER_PATH="$1"
    local OUT_DIR="$2"
    local MODEL_TYPE="${3:-llama}"
    if [ -f "$OUT_DIR/prediction_data.json" ]; then
        echo "  SKIP test inference: $OUT_DIR/prediction_data.json already exists"
        return 0
    fi
    echo "  Test inference → $OUT_DIR ..."
    if [ "$MODEL_TYPE" = "qwen" ]; then
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON tasks/task1_classification/predict_qwen.py \
            --adapter-path "$ADAPTER_PATH" \
            --test-root "$TEST_ROOT" \
            --output-dir "$OUT_DIR" \
            --device cuda:0 2>&1 | tail -5
    else
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON tasks/task1_classification/predict_llama.py \
            --adapter-path "$ADAPTER_PATH" \
            --test-root "$TEST_ROOT" \
            --output-dir "$OUT_DIR" \
            --device cuda:0 2>&1 | tail -5
    fi
}

eval_on_dev() {
    local ADAPTER_PATH="$1"
    local OUT_DIR="$2"
    local MODEL_TYPE="${3:-llama}"
    if [ -f "$OUT_DIR/prediction_data.json" ]; then
        echo "  SKIP dev eval: $OUT_DIR/prediction_data.json already exists"
        # Still print F1 from existing results
        $PYTHON -c "
import json, glob
from sklearn.metrics import f1_score
preds = json.load(open('$OUT_DIR/prediction_data.json'))
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
print(f'  ★ Dev macro F1={f1_score(y_true,y_pred,average=\"macro\",zero_division=0):.4f}  weighted={f1_score(y_true,y_pred,average=\"weighted\",zero_division=0):.4f}')
" 2>/dev/null || true
        return 0
    fi
    echo "  Evaluating $ADAPTER_PATH ..."
    if [ "$MODEL_TYPE" = "qwen" ]; then
        CUDA_VISIBLE_DEVICES=0,1 $PYTHON tasks/task1_classification/predict_qwen.py \
            --adapter-path "$ADAPTER_PATH" \
            --panels-csv data/dev_panels.csv \
            --output-dir "$OUT_DIR" \
            --device cuda:0 2>&1 | tail -3
    else
        CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON tasks/task1_classification/predict_llama.py \
            --adapter-path "$ADAPTER_PATH" \
            --panels-csv data/dev_panels.csv \
            --output-dir "$OUT_DIR" \
            --device cuda:0 2>&1 | tail -3
    fi
    $PYTHON -c "
import json, glob
from sklearn.metrics import f1_score
preds = json.load(open('$OUT_DIR/prediction_data.json'))
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
print(f'  ★ Dev macro F1={f1_score(y_true,y_pred,average=\"macro\",zero_division=0):.4f}  weighted={f1_score(y_true,y_pred,average=\"weighted\",zero_division=0):.4f}')
" 2>/dev/null || echo "  (eval failed)"
}

# ---------------------------------------------------------------------------
# Experiment C: Qwen2.5-VL + ACL-Fig (3961 samples, all 4 GPUs)  ★ FIRST
# bfloat16 14GB spread across 4 GPUs; expandable_segments avoids fragmentation
# ---------------------------------------------------------------------------
echo "=== Exp C: Qwen2.5-VL + ACL-Fig (3961 samples) ★ ==="
if [ ! -f "outputs/qwen2vl_qlora_aclfig_ep3/adapter/adapter_config.json" ]; then
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON tasks/task1_classification/finetune_vlm.py \
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \
        --train-csv data/train_aclfig.csv \
        --output-dir outputs/qwen2vl_qlora_aclfig_ep3 \
        --epochs 3 --batch-size 1 --grad-accum 16 \
        --lr 2e-4 --lora-r 16 --lora-alpha 32 --device cuda:0 \
        2>&1 | tee logs/exp_c_qwen2vl_aclfig_ep3.log
else
    echo "  SKIP: adapter already exists"
fi
eval_on_dev "outputs/qwen2vl_qlora_aclfig_ep3/adapter" "outputs/qwen2vl_qlora_aclfig_ep3/dev" "qwen"
predict_test "outputs/qwen2vl_qlora_aclfig_ep3/adapter" "outputs/qwen2vl_qlora_aclfig_ep3/test" "qwen"

# ---------------------------------------------------------------------------
# Experiment D: Qwen2.5-VL r=32 (2420 samples, all 4 GPUs)
# ---------------------------------------------------------------------------
echo ""
echo "=== Exp D: Qwen2.5-VL r=32 (2420 samples) ★ ==="
if [ ! -f "outputs/qwen2vl_qlora_r32_ep3/adapter/adapter_config.json" ]; then
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON tasks/task1_classification/finetune_vlm.py \
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \
        --train-csv data/train_panels.csv \
        --output-dir outputs/qwen2vl_qlora_r32_ep3 \
        --epochs 3 --batch-size 1 --grad-accum 16 \
        --lr 2e-4 --lora-r 32 --lora-alpha 64 --device cuda:0 \
        2>&1 | tee logs/exp_d_qwen2vl_r32_ep3.log
else
    echo "  SKIP: adapter already exists"
fi
eval_on_dev "outputs/qwen2vl_qlora_r32_ep3/adapter" "outputs/qwen2vl_qlora_r32_ep3/dev" "qwen"
predict_test "outputs/qwen2vl_qlora_r32_ep3/adapter" "outputs/qwen2vl_qlora_r32_ep3/test" "qwen"

# Experiments A and B (Llama) — SKIPPED
# Llama 3.2-11B-Vision SDPA OOM on 2080 Ti even with 4 GPUs due to
# dual-representation (tile + global) producing 3200+ visual tokens.
echo ""
echo "=== Exp A/B: Llama — SKIPPED (persistent OOM on 2080 Ti) ==="

# ---------------------------------------------------------------------------
# Experiment E: Qwen2.5-VL r=32 + ACL-Fig (3961) + MLP modules + 5ep + LR=1e-4
# Best of everything: combined data, higher rank, MLP layers, more epochs
# ---------------------------------------------------------------------------
echo ""
echo "=== Exp E: Qwen2.5-VL + ACL-Fig + MLP LoRA + 5ep ==="
if [ ! -f "outputs/qwen2vl_qlora_expE/adapter/adapter_config.json" ]; then
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON tasks/task1_classification/finetune_vlm.py \
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \
        --train-csv data/train_aclfig.csv \
        --output-dir outputs/qwen2vl_qlora_expE \
        --epochs 5 --batch-size 1 --grad-accum 16 \
        --lr 1e-4 --lora-r 32 --lora-alpha 64 \
        --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --device cuda:0 \
        2>&1 | tee logs/exp_e_qwen2vl_mlp_ep5.log
else
    echo "  SKIP: adapter already exists"
fi
eval_on_dev "outputs/qwen2vl_qlora_expE/adapter" "outputs/qwen2vl_qlora_expE/dev" "qwen"
predict_test "outputs/qwen2vl_qlora_expE/adapter" "outputs/qwen2vl_qlora_expE/test" "qwen"

# ---------------------------------------------------------------------------
echo ""
echo "=== All experiments done. Dev results summary ==="
echo "  Baseline Llama ep3:  macro=0.581  weighted=0.737  (est CodaBench ~0.80)"
echo "  Target (single):     macro ~0.72  weighted ~0.78  (est CodaBench ~0.85)"
for DIR in llama_qlora_aclfig_ep3 llama_qlora_r32_ep3 qwen2vl_qlora_aclfig_ep3 qwen2vl_qlora_r32_ep3 qwen2vl_qlora_expE; do
    PRED="outputs/$DIR/dev/prediction_data.json"
    if [ -f "$PRED" ]; then
        $PYTHON -c "
import json, glob
from sklearn.metrics import f1_score
preds = json.load(open('$PRED'))
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
print(f'  $DIR: macro={f1_score(y_true,y_pred,average=\"macro\",zero_division=0):.4f}  weighted={f1_score(y_true,y_pred,average=\"weighted\",zero_division=0):.4f}')
" 2>/dev/null
    fi
done
