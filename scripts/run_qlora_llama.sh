#!/bin/bash
# QLoRA finetuning of Llama-3.2-11B-Vision-Instruct on chart classification.
# Run on GPU 2 or 3 after VLM evaluation frees them up.
#
# Prerequisites:
#   conda activate llpr_og_paper
#   pip install peft bitsandbytes   (if not already installed)
#   cd sci_image_miner
#
# Usage:
#   bash run_qlora_llama.sh [GPU_ID]
#   GPU_ID defaults to 2

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

GPU="${1:-2}"

# Re-exec under nohup if not already daemonized, so the job survives terminal disconnect.
if [ -z "${NOHUP_ACTIVE:-}" ]; then
    export NOHUP_ACTIVE=1
    nohup bash "$0" "$GPU" > logs/llama_qlora_master.log 2>&1 &
    echo "Started in background (PID $!)"
    echo "Master log: logs/llama_qlora_master.log"
    echo "Training log: logs/llama_qlora.log"
    exit 0
fi

PYTHON="${PYTHON:-python}"

echo "=== QLoRA finetuning: Llama-3.2-11B-Vision-Instruct on GPU $GPU ==="
date

CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u run_vlm_finetune.py \
    --model-id meta-llama/Llama-3.2-11B-Vision-Instruct \
    --output-dir outputs/vlm_finetune/llama_qlora \
    --device cuda:0 \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    > logs/llama_qlora.log 2>&1

echo "=== QLoRA training complete ==="
date

# Run predictions with finetuned model on dev (evaluation gate)
echo ""
echo "Running finetuned Llama predictions on dev..."
ADAPTER_DIR="outputs/vlm_finetune/llama_qlora/adapter"
DEV_ROOT="ALD-E-ImageMiner/icdar2026-competition-data/dev"
TEST_ROOT="ALD-E-ImageMiner/icdar2026-competition-data/test/practice-phase"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u -c "
import sys
sys.path.insert(0, '.')
from src.vlm_finetune.predict import FinetunedLLaVAClassifier
from src.classify import run_classification, make_submission_zip
from pathlib import Path

# Use finetuned Llama for inference
model = FinetunedLLaVAClassifier(adapter_path='$ADAPTER_DIR')
model.load_model()

# Dev predictions (for evaluation gate)
out_dev = Path('outputs/llama_qlora_ft/dev')
out_dev.mkdir(parents=True, exist_ok=True)
run_classification(model=model, test_root='$DEV_ROOT', output_path=str(out_dev / 'prediction_data.json'))

# Test predictions
out_test = Path('outputs/llama_qlora_ft/test')
out_test.mkdir(parents=True, exist_ok=True)
run_classification(model=model, test_root='$TEST_ROOT', output_path=str(out_test / 'prediction_data.json'))

model.unload_model()
print('Done.')
" > logs/llama_qlora_predict.log 2>&1

echo "Finetuned Llama predictions done."
echo ""
echo "Evaluate dev performance to check the gate (macro F1 vs zero-shot llama_vision):"
echo "  $PYTHON -m src.evaluation --predictions outputs/llama_qlora_ft/dev/prediction_data.json \\"
echo "          --dev-root $DEV_ROOT"
echo ""
echo "Then run run_eval_ensemble.sh to build the final ensemble."
