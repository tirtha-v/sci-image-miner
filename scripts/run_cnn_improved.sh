#!/bin/bash
# Retrain both CNNs with improved config (layerwise LR + label smoothing + domain augmentation).
# Run on GPU 0 and GPU 1 in parallel (after VLM eval frees those GPUs).
#
# Prerequisites:
#   conda activate llpr_og_paper
#   cd sci_image_miner
#   python src/data_prep.py   (if crops not yet generated)
#
# Usage:
#   bash run_cnn_improved.sh

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

# Re-exec under nohup if not already daemonized, so the job survives terminal disconnect.
if [ -z "${NOHUP_ACTIVE:-}" ]; then
    export NOHUP_ACTIVE=1
    nohup bash "$0" "$@" > logs/cnn_improved_master.log 2>&1 &
    echo "Started in background (PID $!)"
    echo "Master log: logs/cnn_improved_master.log"
    echo "Per-model logs: logs/efficientnet_b0_v2.log, logs/inception_resnet_v2_v2.log"
    exit 0
fi

PYTHON="${PYTHON:-python}"

echo "=== CNN retraining with improved config ==="
date

# GPU 0: EfficientNetB0
(
    echo "[GPU 0] EfficientNetB0: starting..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u run_cnn_train.py \
        --model efficientnet_b0 \
        --output-dir outputs/cnn/efficientnet_b0_v2 \
        --device cuda:0 \
        --epochs 30 \
        --frozen-epochs 5 \
        --batch-size 64 \
        --lr 1e-4 \
        --patience 7 \
        > logs/efficientnet_b0_v2.log 2>&1
    echo "[GPU 0] EfficientNetB0: done."
) &
PID0=$!

# GPU 1: InceptionResNetV2
(
    echo "[GPU 1] InceptionResNetV2: starting..."
    CUDA_VISIBLE_DEVICES=1 $PYTHON -u run_cnn_train.py \
        --model inception_resnet_v2 \
        --output-dir outputs/cnn/inception_resnet_v2_v2 \
        --device cuda:1 \
        --epochs 30 \
        --frozen-epochs 5 \
        --batch-size 32 \
        --lr 1e-4 \
        --patience 7 \
        > logs/inception_resnet_v2_v2.log 2>&1
    echo "[GPU 1] InceptionResNetV2: done."
) &
PID1=$!

echo "Waiting for GPU 0 (EfficientNetB0, PID=$PID0)..."
wait $PID0 && echo "GPU 0 done." || echo "GPU 0 FAILED (check logs/efficientnet_b0_v2.log)"

echo "Waiting for GPU 1 (InceptionResNetV2, PID=$PID1)..."
wait $PID1 && echo "GPU 1 done." || echo "GPU 1 FAILED (check logs/inception_resnet_v2_v2.log)"

echo ""
echo "=== CNN retraining complete ==="
date

# Run CNN predictions on dev and test
echo ""
echo "Running CNN predictions..."

DEV_ROOT="ALD-E-ImageMiner/icdar2026-competition-data/dev"
TEST_ROOT="ALD-E-ImageMiner/icdar2026-competition-data/test/practice-phase"

for MODEL in efficientnet_b0 inception_resnet_v2; do
    V2_DIR="outputs/cnn/${MODEL}_v2"

    echo "  $MODEL -> dev predictions..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u run_cnn_predict.py \
        --model "$MODEL" \
        --checkpoint "${V2_DIR}/best_model.pt" \
        --label-mapping "${V2_DIR}/label_mapping.json" \
        --test-root "$DEV_ROOT" \
        --output-dir "${V2_DIR}/dev" \
        --device cuda:0 \
        > "logs/${MODEL}_v2_dev_predict.log" 2>&1

    echo "  $MODEL -> test predictions..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u run_cnn_predict.py \
        --model "$MODEL" \
        --checkpoint "${V2_DIR}/best_model.pt" \
        --label-mapping "${V2_DIR}/label_mapping.json" \
        --test-root "$TEST_ROOT" \
        --output-dir "${V2_DIR}/test" \
        --device cuda:0 \
        > "logs/${MODEL}_v2_test_predict.log" 2>&1
done

echo "CNN predictions done. Run run_eval_ensemble.sh next."
