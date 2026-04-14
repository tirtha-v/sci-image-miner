#!/bin/bash
# Build 6-model weighted test ensemble once Qwen ep6 test predictions are ready.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/home/jovyan/shared-scratch-tvinchur-pvc/tvinchur/uq_uma/conda_envs/llpr_og_paper/bin/python}"

QWEN_EP6_TEST="outputs/qwen2vl_qlora_ep6_ft/test/prediction_data.json"

if [ ! -f "$QWEN_EP6_TEST" ]; then
    echo "Qwen ep6 test predictions not ready yet. Check logs/qwen_qlora_ep6_test.log"
    exit 1
fi

echo "=== Building 6-model weighted dev ensemble ==="
mkdir -p outputs/ensemble/dev_v2_weighted

$PYTHON -m src.ensemble \
    --predictions \
        outputs/qwen2vl_dev/prediction_data.json \
        outputs/cnn_efficientnet_b0_dev/prediction_data.json \
        outputs/cnn_inception_resnet_v2_dev/prediction_data.json \
        outputs/llama_qlora_ft_ep3/dev/prediction_data.json \
        outputs/cnn/swinv2_aclfig/dev/prediction_data.json \
        outputs/qwen2vl_qlora_ep6_ft/dev/prediction_data.json \
    --weights 0.348 0.283 0.353 0.581 0.426 0.487 \
    --output outputs/ensemble/dev_v2_weighted/prediction_data.json

$PYTHON -m src.evaluation \
    --predictions outputs/ensemble/dev_v2_weighted/prediction_data.json \
    --dev-root ALD-E-ImageMiner/icdar2026-competition-data/dev \
    --output outputs/ensemble/dev_v2_weighted/metrics.json

$PYTHON -c "import json; m = json.load(open('outputs/ensemble/dev_v2_weighted/metrics.json')); print(f'Dev: Macro F1={m[\"macro_f1\"]:.4f} Weighted F1={m[\"weighted_f1\"]:.4f} Acc={m[\"accuracy\"]:.4f}')"

echo ""
echo "=== Building 6-model weighted test ensemble ==="
mkdir -p outputs/ensemble/test_6model_weighted

$PYTHON -m src.ensemble \
    --predictions \
        outputs/qwen2vl_test_practice/prediction_data.json \
        outputs/cnn_efficientnet_b0_test/prediction_data.json \
        outputs/cnn_inception_resnet_v2_test/prediction_data.json \
        outputs/llama_qlora_ft_ep3/test/prediction_data.json \
        outputs/cnn/swinv2_aclfig/test/prediction_data.json \
        outputs/qwen2vl_qlora_ep6_ft/test/prediction_data.json \
    --weights 0.348 0.283 0.353 0.581 0.426 0.487 \
    --output outputs/ensemble/test_6model_weighted/prediction_data.json

$PYTHON -c "
import sys; sys.path.insert(0, '.')
from src.classify import make_submission_zip
zip_path = 'outputs/submission_weighted_qwen2vl_effnet_inceptionv2_llama_ep3_swinv2aclfig_qwen_qlora_ep6.zip'
make_submission_zip('outputs/ensemble/test_6model_weighted/prediction_data.json', zip_path)
print(f'Submission zip: {zip_path}')
"
