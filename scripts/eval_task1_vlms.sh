#!/bin/bash
# =============================================================================
# Run VLM predictions on the test set (requires multi-GPU node)
#
# GPU 0: Llama QLoRA ep3  (11B 4-bit, ~6-7GB VRAM, ~2-3 hours)
# GPU 1: Qwen2VL zero-shot (7B bfloat16, ~14GB VRAM, ~1-2 hours)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
TEST_ROOT="ALD-E-ImageMiner/icdar2026-competition-data/test"

# Set HuggingFace token
HF_TOKEN_FILE="${HUGGING_FACE_HUB_TOKEN_FILE:-}"
if [ -f "$HF_TOKEN_FILE" ]; then
    export HUGGING_FACE_HUB_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

mkdir -p logs

echo "=== Launching VLM predictions in parallel ==="
echo "GPUs 0,1: Llama QLoRA ep3  (4-bit spread across 2x 2080 Ti)"
echo "GPUs 2,3: Qwen2VL zero-shot (bfloat16 spread across 2x 2080 Ti)"
echo ""

# Llama QLoRA ep3 (best single model, dev F1=0.581)
# Uses device_map="auto" internally — CUDA_VISIBLE_DEVICES pins to 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 $PYTHON tasks/task1_classification/predict_llama.py \
    --adapter-path outputs/vlm_finetune/llama_qlora/checkpoint-152 \
    --test-root "$TEST_ROOT" \
    --output-dir outputs/llama_qlora_ft_ep3/test_eval \
    --device cuda:0 \
    2>&1 | tee logs/llama_qlora_ep3_eval_test.log &
LLAMA_PID=$!

# Qwen2VL zero-shot (bfloat16, spread across 2 GPUs — bitsandbytes 4-bit
# has a known issue with Qwen attention weight shapes, so we use bfloat16)
CUDA_VISIBLE_DEVICES=2,3 $PYTHON tasks/task1_classification/predict_zeroshot.py \
    --model qwen2vl \
    --test-root "$TEST_ROOT" \
    --output-dir outputs/qwen2vl_test_eval \
    2>&1 | tee logs/qwen2vl_eval_test.log &
QWEN_PID=$!

echo "Llama PID: $LLAMA_PID | Qwen PID: $QWEN_PID"
echo "Tailing Llama log (Ctrl+C safe, jobs continue in background):"
tail -f logs/llama_qlora_ep3_eval_test.log &

wait $LLAMA_PID && echo "Llama done" || echo "Llama FAILED"
wait $QWEN_PID && echo "Qwen done" || echo "Qwen FAILED"

echo ""
echo "=== VLM predictions done. Now rebuild full 6-model submission: ==="
echo "  bash scripts/build_ensemble.sh"
