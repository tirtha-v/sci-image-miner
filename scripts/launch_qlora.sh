#!/bin/bash
export HF_TOKEN=$(cat /home/jovyan/shared-scratch-tvinchur-pvc/tvinchur/hf_token_for_cluster.txt | tr -d '[:space:]')
PYTHON=/home/jovyan/shared-scratch-tvinchur-pvc/tvinchur/uq_uma/conda_envs/llpr_og_paper/bin/python
cd /home/jovyan/shared-scratch-tvinchur-pvc/tvinchur/sci_image_miner

CUDA_VISIBLE_DEVICES=2 $PYTHON -u run_vlm_finetune.py \
    --model-id meta-llama/Llama-3.2-11B-Vision-Instruct \
    --output-dir outputs/vlm_finetune/llama_qlora \
    --device cuda:0 \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32
