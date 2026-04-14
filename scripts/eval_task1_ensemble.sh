#!/bin/bash
# Evaluate all available dev predictions and build ensemble.
#
# Usage:
#   bash run_eval_ensemble.sh

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
DEV_ROOT="ALD-E-ImageMiner/icdar2026-competition-data/dev"
GATE=0.25

mkdir -p logs

echo "=== Evaluating all dev predictions ==="
echo "Gate threshold: macro F1 >= $GATE"
echo ""

# Model name -> (dev pred path, test pred path)
# Paths reflect actual output directories (some use legacy names)
declare -A DEV_PREDS
declare -A TEST_PREDS

# Original strong models
DEV_PREDS["qwen2vl"]="outputs/qwen2vl_dev/prediction_data.json"
TEST_PREDS["qwen2vl"]="outputs/qwen2vl_test_practice/prediction_data.json"

# New VLMs from this run
DEV_PREDS["llava"]="outputs/llava/dev/prediction_data.json"
TEST_PREDS["llava"]="outputs/llava/test/prediction_data.json"

DEV_PREDS["llava_1_5"]="outputs/llava_1_5/dev/prediction_data.json"
TEST_PREDS["llava_1_5"]="outputs/llava_1_5/test/prediction_data.json"

DEV_PREDS["phi_3_5_vision"]="outputs/phi_3_5_vision/dev/prediction_data.json"
TEST_PREDS["phi_3_5_vision"]="outputs/phi_3_5_vision/test/prediction_data.json"

DEV_PREDS["llama_vision"]="outputs/llama_vision/dev/prediction_data.json"
TEST_PREDS["llama_vision"]="outputs/llama_vision/test/prediction_data.json"

# CNNs — prefer v2 (retrained), fall back to old if v2 not ready
if [ -f "outputs/cnn/efficientnet_b0_v2/dev/prediction_data.json" ]; then
    DEV_PREDS["efficientnet_b0"]="outputs/cnn/efficientnet_b0_v2/dev/prediction_data.json"
    TEST_PREDS["efficientnet_b0"]="outputs/cnn/efficientnet_b0_v2/test/prediction_data.json"
else
    DEV_PREDS["efficientnet_b0"]="outputs/cnn_efficientnet_b0_dev/prediction_data.json"
    TEST_PREDS["efficientnet_b0"]="outputs/cnn_efficientnet_b0_test/prediction_data.json"
fi

if [ -f "outputs/cnn/inception_resnet_v2_v2/dev/prediction_data.json" ]; then
    DEV_PREDS["inception_resnet_v2"]="outputs/cnn/inception_resnet_v2_v2/dev/prediction_data.json"
    TEST_PREDS["inception_resnet_v2"]="outputs/cnn/inception_resnet_v2_v2/test/prediction_data.json"
else
    DEV_PREDS["inception_resnet_v2"]="outputs/cnn_inception_resnet_v2_dev/prediction_data.json"
    TEST_PREDS["inception_resnet_v2"]="outputs/cnn_inception_resnet_v2_test/prediction_data.json"
fi

# QLoRA finetuned Llama epoch-3 (best checkpoint)
DEV_PREDS["llama_qlora_ft"]="outputs/llama_qlora_ft_ep3/dev/prediction_data.json"
TEST_PREDS["llama_qlora_ft"]="outputs/llama_qlora_ft_ep3/test/prediction_data.json"

# SwinV2-B (new backbone)
DEV_PREDS["swinv2_base"]="outputs/cnn/swinv2_base/dev/prediction_data.json"
TEST_PREDS["swinv2_base"]="outputs/cnn/swinv2_base/test/prediction_data.json"

# ConvNeXtV2-B (new backbone, retrain v2)
DEV_PREDS["convnextv2_base"]="outputs/cnn/convnextv2_base_v2/dev/prediction_data.json"
TEST_PREDS["convnextv2_base"]="outputs/cnn/convnextv2_base_v2/test/prediction_data.json"

# ViT-B/16 (augreg2_in21k_ft_in1k)
DEV_PREDS["vit_base"]="outputs/cnn/vit_base/dev/prediction_data.json"
TEST_PREDS["vit_base"]="outputs/cnn/vit_base/test/prediction_data.json"

# Qwen2.5-VL LoRA finetuned ep3
DEV_PREDS["qwen2vl_qlora"]="outputs/qwen2vl_qlora_ft/dev/prediction_data.json"
TEST_PREDS["qwen2vl_qlora"]="outputs/qwen2vl_qlora_ft/test/prediction_data.json"

# Qwen2.5-VL LoRA finetuned ep6 (better than ep3 by +0.020 macro F1)
DEV_PREDS["qwen2vl_qlora_ep6"]="outputs/qwen2vl_qlora_ep6_ft/dev/prediction_data.json"
TEST_PREDS["qwen2vl_qlora_ep6"]="outputs/qwen2vl_qlora_ep6_ft/test/prediction_data.json"

# SwinV2+ACLFig (retrained on ACL-Fig external data, +6.4pts over swinv2_base)
DEV_PREDS["swinv2_aclfig"]="outputs/cnn/swinv2_aclfig/dev/prediction_data.json"
TEST_PREDS["swinv2_aclfig"]="outputs/cnn/swinv2_aclfig/test/prediction_data.json"

# SwinV2+DocFig (retrained on DocFigure+ACLFig, 12,065 samples)
DEV_PREDS["swinv2_docfig"]="outputs/cnn/swinv2_docfig/dev/prediction_data.json"
TEST_PREDS["swinv2_docfig"]="outputs/cnn/swinv2_docfig/test/prediction_data.json"

# InceptionResNetV2+DocFig (retrained on DocFigure+ACLFig, 12,065 samples)
DEV_PREDS["inception_resnet_v2_docfig"]="outputs/cnn/inception_resnet_v2_docfig/dev/prediction_data.json"
TEST_PREDS["inception_resnet_v2_docfig"]="outputs/cnn/inception_resnet_v2_docfig/test/prediction_data.json"

PASSING_DEV_FILES=()
PASSING_TEST_FILES=()
PASSING_NAMES=()

echo "| Model | Macro F1 | Pass gate? |"
echo "|-------|----------|------------|"

for MODEL_NAME in qwen2vl llava llava_1_5 phi_3_5_vision llama_vision efficientnet_b0 inception_resnet_v2 llama_qlora_ft swinv2_base convnextv2_base vit_base qwen2vl_qlora qwen2vl_qlora_ep6 swinv2_aclfig swinv2_docfig inception_resnet_v2_docfig; do
    DEV_PRED="${DEV_PREDS[$MODEL_NAME]}"
    if [ ! -f "$DEV_PRED" ]; then
        echo "| $MODEL_NAME | (no predictions) | - |"
        continue
    fi

    METRICS_DIR="outputs/eval/${MODEL_NAME}"
    mkdir -p "$METRICS_DIR"

    $PYTHON -m src.evaluation \
        --predictions "$DEV_PRED" \
        --dev-root "$DEV_ROOT" \
        --output "${METRICS_DIR}/metrics.json" > /dev/null 2>&1 || { echo "| $MODEL_NAME | ERROR | - |"; continue; }

    MACRO_F1=$($PYTHON -c "
import json
m = json.load(open('${METRICS_DIR}/metrics.json'))
print(f\"{m.get('macro_f1', 0):.4f}\")
")
    PASSES=$($PYTHON -c "print('YES' if float('$MACRO_F1') >= $GATE else 'NO')")
    echo "| $MODEL_NAME | $MACRO_F1 | $PASSES |"

    if [ "$PASSES" = "YES" ]; then
        PASSING_DEV_FILES+=("$DEV_PRED")
        PASSING_NAMES+=("$MODEL_NAME")
        TEST_PRED="${TEST_PREDS[$MODEL_NAME]}"
        if [ -f "$TEST_PRED" ]; then
            PASSING_TEST_FILES+=("$TEST_PRED")
        else
            echo "  WARNING: test predictions missing for $MODEL_NAME ($TEST_PRED)"
        fi
    fi
done

echo ""
echo "Models passing gate (${#PASSING_NAMES[@]}): ${PASSING_NAMES[*]:-none}"
echo ""

if [ ${#PASSING_DEV_FILES[@]} -lt 2 ]; then
    echo "Not enough models passed the gate for ensemble. Exiting."
    exit 1
fi

# Dev ensemble
echo "=== Building dev ensemble ==="
mkdir -p outputs/ensemble/dev
$PYTHON -m src.ensemble \
    --predictions "${PASSING_DEV_FILES[@]}" \
    --output outputs/ensemble/dev/prediction_data.json \
    > logs/ensemble_build.log 2>&1

echo "=== Ensemble dev evaluation ==="
$PYTHON -m src.evaluation \
    --predictions outputs/ensemble/dev/prediction_data.json \
    --dev-root "$DEV_ROOT" \
    --output outputs/ensemble/dev/metrics.json

# Test ensemble
if [ ${#PASSING_TEST_FILES[@]} -ge 2 ]; then
    echo ""
    echo "=== Building test ensemble ==="
    mkdir -p outputs/ensemble/test
    $PYTHON -m src.ensemble \
        --predictions "${PASSING_TEST_FILES[@]}" \
        --output outputs/ensemble/test/prediction_data.json \
        >> logs/ensemble_build.log 2>&1

    # Build a descriptive zip name from passing model names
    MODELS_STR=$(IFS=_; echo "${PASSING_NAMES[*]}")
    ZIP_NAME="outputs/submission_${MODELS_STR}.zip"

    $PYTHON -c "
import sys; sys.path.insert(0, '.')
from src.classify import make_submission_zip
make_submission_zip('outputs/ensemble/test/prediction_data.json', '$ZIP_NAME')
print('Submission zip: $ZIP_NAME')
"
    echo ""
    echo "=== Final submission ready ==="
    echo "  $ZIP_NAME"
else
    echo "Insufficient test predictions (${#PASSING_TEST_FILES[@]}) — run test predictions first."
fi
