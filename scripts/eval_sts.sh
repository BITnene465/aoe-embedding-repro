#!/usr/bin/env bash
# Evaluate AoE model on STS tasks.
# Usage: bash scripts/eval_sts.sh <CHECKPOINT_DIR> [TASKS] [MODEL_NAME]
set -euo pipefail

CKPT=${CKPT:-}
TASKS=${TASKS:-all}
MODEL_NAME=${MODEL_NAME:-}

if [[ -z "${CKPT}" ]]; then
  echo "Usage: CKPT=<path> [TASKS=<tasks>] [MODEL_NAME=<name>] bash scripts/eval_sts.sh"
  exit 1
fi

DATA_CACHE=${DATA_CACHE:-data}
MODEL_CACHE=${MODEL_CACHE:-models}
MAX_LENGTH=${MAX_LENGTH:-128}
BATCH_SIZE=${BATCH_SIZE:-64}
EVAL_SPLITS=${EVAL_SPLITS:-test}
RESULTS_DIR=${RESULTS_DIR:-output/mteb}

echo "Evaluating STS tasks..."
echo "Checkpoint: ${CKPT}"
echo "Tasks: ${TASKS}"

# Build arguments array
ARGS=(
  --ckpt "${CKPT}"
  --tasks "${TASKS}"
  --data_cache "${DATA_CACHE}"
  --model_cache "${MODEL_CACHE}"
  --max_length "${MAX_LENGTH}"
  --batch_size "${BATCH_SIZE}"
  --results_dir "${RESULTS_DIR}"
  --eval_splits "${EVAL_SPLITS}"
)

if [[ -n "${MODEL_NAME}" ]]; then
  ARGS+=(--model_name "${MODEL_NAME}")
fi

python -m aoe.eval_sts "${ARGS[@]}"
