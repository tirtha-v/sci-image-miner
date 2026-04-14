#!/bin/bash
# Run 4 VLMs in parallel (one per GPU) on dev + practice-phase test.
# Each GPU runs dev then test sequentially; all 4 GPUs run concurrently.
#
# Prerequisites:
#   conda activate llpr_og_paper   (or use the full python path below)
#   cd sci_image_miner
#
# Usage:
#   bash run_vlm_parallel.sh

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

# Re-exec under nohup if not already daemonized, so the job survives terminal disconnect.
if [ -z "${NOHUP_ACTIVE:-}" ]; then
    export NOHUP_ACTIVE=1
    nohup bash "$0" "$@" > logs/vlm_parallel_master.log 2>&1 &
    echo "Started in background (PID $!)"
    echo "Master log: logs/vlm_parallel_master.log"
    echo "Per-model logs: logs/llava_dev.log, logs/phi_dev.log, etc."
    exit 0
fi

PYTHON="${PYTHON:-python}"
DEV_ROOT="ALD-E-ImageMiner/icdar2026-competition-data/dev"
TEST_ROOT="ALD-E-ImageMiner/icdar2026-competition-data/test/practice-phase"

echo "=== Parallel VLM evaluation ==="
echo "Dev:  $DEV_ROOT"
echo "Test: $TEST_ROOT"
date

# GPU 0: llava (llava-v1.6-mistral-7b-hf)
(
    echo "[GPU 0] llava: starting dev..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u run_classify.py \
        --model llava \
        --test-root "$DEV_ROOT" \
        --output-dir outputs/llava/dev \
        > logs/llava_dev.log 2>&1
    echo "[GPU 0] llava: dev done, starting test..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u run_classify.py \
        --model llava \
        --test-root "$TEST_ROOT" \
        --output-dir outputs/llava/test \
        > logs/llava_test.log 2>&1
    echo "[GPU 0] llava: done."
) &
PID0=$!

# GPU 1: phi_3_5_vision (Phi-3.5-vision-instruct)
(
    echo "[GPU 1] phi_3_5_vision: starting dev..."
    CUDA_VISIBLE_DEVICES=1 $PYTHON -u run_classify.py \
        --model phi_3_5_vision \
        --test-root "$DEV_ROOT" \
        --output-dir outputs/phi_3_5_vision/dev \
        > logs/phi_dev.log 2>&1
    echo "[GPU 1] phi_3_5_vision: dev done, starting test..."
    CUDA_VISIBLE_DEVICES=1 $PYTHON -u run_classify.py \
        --model phi_3_5_vision \
        --test-root "$TEST_ROOT" \
        --output-dir outputs/phi_3_5_vision/test \
        > logs/phi_test.log 2>&1
    echo "[GPU 1] phi_3_5_vision: done."
) &
PID1=$!

# GPU 2: llava_1_5 (llava-1.5-7b-hf)
(
    echo "[GPU 2] llava_1_5: starting dev..."
    CUDA_VISIBLE_DEVICES=2 $PYTHON -u run_classify.py \
        --model llava_1_5 \
        --test-root "$DEV_ROOT" \
        --output-dir outputs/llava_1_5/dev \
        > logs/llava_1_5_dev.log 2>&1
    echo "[GPU 2] llava_1_5: dev done, starting test..."
    CUDA_VISIBLE_DEVICES=2 $PYTHON -u run_classify.py \
        --model llava_1_5 \
        --test-root "$TEST_ROOT" \
        --output-dir outputs/llava_1_5/test \
        > logs/llava_1_5_test.log 2>&1
    echo "[GPU 2] llava_1_5: done."
) &
PID2=$!

# GPU 3: llama_vision (Llama-3.2-11B-Vision-Instruct)
(
    echo "[GPU 3] llama_vision: starting dev..."
    CUDA_VISIBLE_DEVICES=3 $PYTHON -u run_classify.py \
        --model llama_vision \
        --test-root "$DEV_ROOT" \
        --output-dir outputs/llama_vision/dev \
        > logs/llama_vision_dev.log 2>&1
    echo "[GPU 3] llama_vision: dev done, starting test..."
    CUDA_VISIBLE_DEVICES=3 $PYTHON -u run_classify.py \
        --model llama_vision \
        --test-root "$TEST_ROOT" \
        --output-dir outputs/llama_vision/test \
        > logs/llama_vision_test.log 2>&1
    echo "[GPU 3] llama_vision: done."
) &
PID3=$!

echo "Waiting for GPU 0 (llava, PID=$PID0)..."
wait $PID0 && echo "GPU 0 done." || echo "GPU 0 FAILED (check logs/llava_*.log)"

echo "Waiting for GPU 1 (phi_3_5_vision, PID=$PID1)..."
wait $PID1 && echo "GPU 1 done." || echo "GPU 1 FAILED (check logs/phi_*.log)"

echo "Waiting for GPU 2 (llava_1_5, PID=$PID2)..."
wait $PID2 && echo "GPU 2 done." || echo "GPU 2 FAILED (check logs/llava_1_5_*.log)"

echo "Waiting for GPU 3 (llama_vision, PID=$PID3)..."
wait $PID3 && echo "GPU 3 done." || echo "GPU 3 FAILED (check logs/llama_vision_*.log)"

echo ""
echo "=== All VLMs complete ==="
date
echo ""
echo "Run evaluation next:"
echo "  bash run_eval_ensemble.sh"
