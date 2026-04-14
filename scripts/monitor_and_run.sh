#!/bin/bash
# Poll for Llama test inference (PID $1) to finish,
# rebuild 6-model ensemble, then run fine-tuning experiments.
set -euo pipefail
cd "$(dirname "$0")"

LLAMA_PID=${1:-}
PYTHON="${PYTHON:-/home/jovyan/tvinchur/uq_uma/conda_envs/llpr_og_paper/bin/python3}"

for HF_TOKEN_FILE in \
    "/home/jovyan/tvinchur/hf_token_for_cluster.txt" \
    "/home/jovyan/shared-scratch-tvinchur-pvc/tvinchur/hf_token_for_cluster.txt"; do
    if [ -f "$HF_TOKEN_FILE" ]; then
        export HUGGING_FACE_HUB_TOKEN=$(cat "$HF_TOKEN_FILE")
        break
    fi
done

# Poll until the Llama inference process exits (works across shell boundaries)
if [ -n "$LLAMA_PID" ]; then
    echo "[monitor] Polling for Llama test inference (PID $LLAMA_PID) ..."
    while kill -0 "$LLAMA_PID" 2>/dev/null; do
        sleep 30
    done
    echo "[monitor] Llama inference (PID $LLAMA_PID) finished."
fi

# Rebuild 6-model ensemble zip
if [ -f "outputs/llama_qlora_ft_ep3/test_eval/prediction_data.json" ] && \
   [ -f "outputs/qwen2vl_test_eval/prediction_data.json" ]; then
    echo "[monitor] Rebuilding 6-model ensemble zip ..."
    bash run_eval_phase.sh --skip-vlms 2>&1 | tee logs/ensemble_rebuild_monitor.log
    echo "[monitor] Ensemble rebuild done."
else
    echo "[monitor] WARNING: prediction files missing, skipping ensemble rebuild."
fi

# Run fine-tuning experiments C → D → A → B
echo "[monitor] Starting fine-tuning experiments ..."
PYTHON="$PYTHON" bash run_experiments.sh 2>&1 | tee logs/run_experiments_main.log
echo "[monitor] All experiments done."
