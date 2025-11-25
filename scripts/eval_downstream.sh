#!/usr/bin/env bash
# Evaluate AoE model on Downstream Transfer Tasks (SentEval).
set -euo pipefail

CKPT=${CKPT:-}
MODEL_NAME=${MODEL_NAME:-}

if [[ -z "${CKPT}" ]]; then
  echo "Usage: CKPT=<path> [MODEL_NAME=<name>] bash scripts/eval_downstream.sh"
  exit 1
fi

if [[ -z "${MODEL_NAME}" ]]; then
  MODEL_NAME=$(basename "${CKPT}")
fi

DATA_CACHE=${DATA_CACHE:-data}
RESULTS_DIR=${RESULTS_DIR:-output/transfer}

echo "Evaluating ${MODEL_NAME} on Downstream Tasks..."
echo "Checkpoint: ${CKPT}"
echo "Results: ${RESULTS_DIR}/${MODEL_NAME}"

python -m aoe.eval_downstream \
    --ckpt "${CKPT}" \
    --model_name "${MODEL_NAME}" \
    --data_cache "${DATA_CACHE}" \
    --results_dir "${RESULTS_DIR}" \
    --tasks all
