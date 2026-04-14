#!/bin/bash
# Wait for dev inference to finish, then run test inference
cd /home/jovyan/tvinchur/sci_image_miner

echo "[queue] Waiting for dev inference (PID 2414) to finish..."
while kill -0 2414 2>/dev/null; do sleep 30; done
echo "[queue] Dev done. Starting test inference..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_qwen_extraction_predict.py \
    --adapter-path outputs/qwen2vl_extraction_qlora/adapter \
    --mode test \
    --test-root ALD-E-ImageMiner/icdar2026-competition-data/test \
    --output-dir outputs/qwen2vl_extraction_qlora/test \
    > logs/extraction_test.log 2>&1

echo "[queue] Test inference done."
