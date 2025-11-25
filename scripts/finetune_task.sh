#!/usr/bin/env bash
# Fine-tune AoE on a single STS task (e.g., STS-B).
set -euo pipefail

TASK_NAME=${TASK_NAME:-stsb}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}

# Validate checkpoint if provided
if [[ -n "${INIT_CHECKPOINT}" ]]; then
  if [[ ! -d "${INIT_CHECKPOINT}" ]]; then
    echo "[ERROR] INIT_CHECKPOINT directory '${INIT_CHECKPOINT}' not found." >&2
    exit 1
  fi
  echo "Initializing from checkpoint: ${INIT_CHECKPOINT}"
else
  echo "No INIT_CHECKPOINT provided. Training from backbone."
fi

# Default hyperparameters (can be overridden by env vars)
LEARNING_RATE=${LEARNING_RATE:-2e-5}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
DATA_CACHE=${DATA_CACHE:-data}
MODEL_CACHE=${MODEL_CACHE:-models}
MAX_LENGTH=${MAX_LENGTH:-128}
SEED=${SEED:-42}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
WARMUP_STEPS=${WARMUP_STEPS:-100}
STS_W_ANGLE=${STS_W_ANGLE:-0.02}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d_%H%M)}

# Construct run name
RUN_NAME="bert_${TASK_NAME}_aoe_${RUN_SUFFIX}"

LAUNCHER=${LAUNCHER:-python}

echo "Starting fine-tuning on task: ${TASK_NAME}"
echo "Checkpoint: ${INIT_CHECKPOINT}"
echo "Output: ${OUTPUT_ROOT}/${RUN_NAME}"

$LAUNCHER -m aoe.train \
  --dataset "${TASK_NAME}@train" \
  --train_split train \
  --eval_split validation \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --lr "${LEARNING_RATE}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --angle_tau 20 \
  --cl_scale 20 \
  --w_angle ${STS_W_ANGLE} \
  --w_cl 1.0 \
  --w_cosine 0.0 \
  --max_length "${MAX_LENGTH}" \
  --data_cache "${DATA_CACHE}" \
  --model_cache "${MODEL_CACHE}" \
  --seed "${SEED}" \
  --tensorboard_dir "${OUTPUT_ROOT}/${RUN_NAME}/tensorboard" \
  --metrics_path "${OUTPUT_ROOT}/${RUN_NAME}/metrics.jsonl" \
  --metrics_path "${OUTPUT_ROOT}/${RUN_NAME}/metrics.jsonl" \
  ${INIT_CHECKPOINT:+--init_checkpoint "${INIT_CHECKPOINT}"}
