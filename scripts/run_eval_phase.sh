#!/bin/bash
# =============================================================================
# Evaluation Phase Submission Builder
# =============================================================================
# Usage:
#   Full 6-model (requires A6000 for VLMs):
#     bash run_eval_phase.sh
#
#   4-model only (already have predictions, runs immediately):
#     bash run_eval_phase.sh --skip-vlms
#
# Test root (merged practice-phase + remaining-test-data):
#   ALD-E-ImageMiner/icdar2026-competition-data/test/
#
# Best ensemble weights (from dev_v2_weighted, dev macro F1 = 0.6119):
#   llama_qlora_ep3:   0.581
#   qwen_qlora_ep6:    0.487
#   swinv2_aclfig:     0.426
#   inception_resnet:  0.353
#   qwen2vl_zero:      0.348
#   efficientnet_b0:   0.283
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/jovyan/tvinchur/uq_uma/conda_envs/llpr_og_paper/bin/python3}"
TEST_ROOT="ALD-E-ImageMiner/icdar2026-competition-data/test"
SKIP_VLMS="${1:-}"

mkdir -p logs outputs/ensemble/eval_phase

# ---------------------------------------------------------------------------
# Step 1: CNN predictions (all already done, but re-run if missing)
# ---------------------------------------------------------------------------
echo "=== Step 1: CNN Predictions ==="

for MODEL_KEY in swinv2_aclfig efficientnet_b0 inception_resnet_v2; do
    case $MODEL_KEY in
        swinv2_aclfig)
            MODEL_ARCH="swinv2_base"
            OUT_DIR="outputs/cnn/swinv2_aclfig/test_eval"
            ;;
        efficientnet_b0)
            MODEL_ARCH="efficientnet_b0"
            OUT_DIR="outputs/cnn/efficientnet_b0/test_eval"
            ;;
        inception_resnet_v2)
            MODEL_ARCH="inception_resnet_v2"
            OUT_DIR="outputs/cnn/inception_resnet_v2/test_eval"
            ;;
    esac
    CKPT="outputs/cnn/${MODEL_KEY}/best_model.pt"
    LMAP="outputs/cnn/${MODEL_KEY}/label_mapping.json"
    PRED="${OUT_DIR}/prediction_data.json"

    if [ -f "$PRED" ]; then
        COUNT=$(python3 -c "import json; d=json.load(open('$PRED')); print(len(d))")
        echo "  SKIP $MODEL_KEY: $PRED exists ($COUNT figs)"
    else
        echo "  RUN $MODEL_KEY ..."
        $PYTHON run_cnn_predict.py \
            --model "$MODEL_ARCH" \
            --checkpoint "$CKPT" \
            --label-mapping "$LMAP" \
            --test-root "$TEST_ROOT" \
            --output-dir "$OUT_DIR" \
            --device cuda:0
    fi
done

# ---------------------------------------------------------------------------
# Step 2: VLM Predictions (skip with --skip-vlms, needs A6000)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: VLM Predictions ==="

if [ "$SKIP_VLMS" = "--skip-vlms" ]; then
    echo "  Skipping VLMs (--skip-vlms flag set)"
else
    # --- Llama QLoRA ep3 (best single model, F1=0.581) ---
    LLAMA_PRED="outputs/llama_qlora_ft_ep3/test_eval/prediction_data.json"
    if [ -f "$LLAMA_PRED" ]; then
        COUNT=$(python3 -c "import json; d=json.load(open('$LLAMA_PRED')); print(len(d))")
        echo "  SKIP llama_qlora_ep3: $LLAMA_PRED exists ($COUNT figs)"
    else
        echo "  RUN llama_qlora_ep3 on GPU 0 ..."
        $PYTHON run_llama_qlora_predict.py \
            --adapter-path outputs/vlm_finetune/llama_qlora/checkpoint-152 \
            --test-root "$TEST_ROOT" \
            --output-dir outputs/llama_qlora_ft_ep3/test_eval \
            --device cuda:0 \
            2>&1 | tee logs/llama_qlora_ep3_eval_test.log
    fi

    # --- Qwen2VL zero-shot ---
    QWEN_PRED="outputs/qwen2vl_test_eval/prediction_data.json"
    if [ -f "$QWEN_PRED" ]; then
        COUNT=$(python3 -c "import json; d=json.load(open('$QWEN_PRED')); print(len(d))")
        echo "  SKIP qwen2vl_zero_shot: $QWEN_PRED exists ($COUNT figs)"
    else
        echo "  RUN qwen2vl zero-shot on GPU 1 ..."
        $PYTHON run_classify.py \
            --model qwen2vl \
            --test-root "$TEST_ROOT" \
            --output-dir outputs/qwen2vl_test_eval \
            --device cuda:1 \
            2>&1 | tee logs/qwen2vl_eval_test.log
    fi
fi

# qwen_qlora_ep6 already has full 580-fig predictions
QWEN_QLORA_PRED="outputs/qwen2vl_qlora_ep6_ft/test/prediction_data.json"
echo "  OK qwen_qlora_ep6: $QWEN_QLORA_PRED (pre-existing 580 figs)"

# ---------------------------------------------------------------------------
# Step 3: Build weighted ensemble
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Build Weighted Ensemble ==="

CNN_PREDS=(
    "outputs/cnn/swinv2_aclfig/test_eval/prediction_data.json"      # weight 0.426
    "outputs/cnn/inception_resnet_v2/test_eval/prediction_data.json" # weight 0.353
    "outputs/cnn/efficientnet_b0/test_eval/prediction_data.json"     # weight 0.283
)
CNN_WEIGHTS=(0.426 0.353 0.283)

VLM_PREDS=()
VLM_WEIGHTS=()
VLM_PREDS+=("$QWEN_QLORA_PRED")   # weight 0.487
VLM_WEIGHTS+=(0.487)

# Add VLM preds if available
LLAMA_PRED="outputs/llama_qlora_ft_ep3/test_eval/prediction_data.json"
if [ -f "$LLAMA_PRED" ]; then
    VLM_PREDS+=("$LLAMA_PRED")
    VLM_WEIGHTS+=(0.581)
    echo "  Including llama_qlora_ep3 (F1=0.581)"
else
    echo "  WARNING: llama_qlora_ep3 predictions missing — using 5-model ensemble"
fi

QWEN_ZERO_PRED="outputs/qwen2vl_test_eval/prediction_data.json"
if [ -f "$QWEN_ZERO_PRED" ]; then
    VLM_PREDS+=("$QWEN_ZERO_PRED")
    VLM_WEIGHTS+=(0.348)
    echo "  Including qwen2vl_zero_shot (F1=0.348)"
else
    echo "  WARNING: qwen2vl_zero_shot predictions missing — excluded from ensemble"
fi

# Combine all preds and weights
ALL_PREDS=("${VLM_PREDS[@]}" "${CNN_PREDS[@]}")
ALL_WEIGHTS=("${VLM_WEIGHTS[@]}" "${CNN_WEIGHTS[@]}")

N_MODELS=${#ALL_PREDS[@]}
echo "  Building ${N_MODELS}-model weighted ensemble ..."

$PYTHON -m src.ensemble \
    --predictions "${ALL_PREDS[@]}" \
    --weights "${ALL_WEIGHTS[@]}" \
    --output outputs/ensemble/eval_phase/prediction_data.json

COUNT=$(python3 -c "import json; d=json.load(open('outputs/ensemble/eval_phase/prediction_data.json')); print(len(d))")
echo "  Ensemble: $COUNT figures"

# ---------------------------------------------------------------------------
# Step 4: Generate submission zip
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Generate Submission Zip ==="

ZIP_PATH="outputs/submission_eval_phase_${N_MODELS}model_weighted.zip"
$PYTHON -c "
import sys; sys.path.insert(0, '.')
from src.classify import make_submission_zip
make_submission_zip('outputs/ensemble/eval_phase/prediction_data.json', '$ZIP_PATH')
print(f'Submission zip: $ZIP_PATH')
"

echo ""
echo "=== Done ==="
echo "Submit: $ZIP_PATH"
echo ""
echo "To run on A6000 for full 6-model, first run VLMs then re-run this script:"
echo "  GPU 0: python run_llama_qlora_predict.py --adapter-path outputs/vlm_finetune/llama_qlora/checkpoint-152 --test-root $TEST_ROOT --output-dir outputs/llama_qlora_ft_ep3/test_eval --device cuda:0"
echo "  GPU 1: python run_classify.py --model qwen2vl --test-root $TEST_ROOT --output-dir outputs/qwen2vl_test_eval --device cuda:1"
