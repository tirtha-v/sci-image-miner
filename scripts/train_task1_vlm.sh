#!/bin/bash
# HUGGING_FACE_HUB_TOKEN must be set in your environment
PYTHON="${PYTHON:-python}"

CUDA_VISIBLE_DEVICES=2 $PYTHON -u tasks/task1_classification/finetune_vlm.py \
    --model-id meta-llama/Llama-3.2-11B-Vision-Instruct \
    --output-dir outputs/vlm_finetune/llama_qlora \
    --device cuda:0 \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32
