#!/usr/bin/env bash
# Fine-tune AoE on mixed STS datasets with angle loss enabled.
set -euo pipefail

EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-256}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
DATA_CACHE=${DATA_CACHE:-data}
MODEL_CACHE=${MODEL_CACHE:-models}
MAX_LENGTH=${MAX_LENGTH:-128}
SEED=${SEED:-42}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
WARMUP_STEPS=${WARMUP_STEPS:-100}
STS_W_ANGLE=${STS_W_ANGLE:-0.02}
AOE_DATASETS=${AOE_DATASETS:-stsb@train,gis@train}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d_%H%M)}
RUN_NAME=${RUN_NAME:-bert_stsb_aoe_${RUN_SUFFIX}}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}

if [[ -z "${INIT_CHECKPOINT}" ]]; then
  echo "[ERROR] INIT_CHECKPOINT must point to a pretrained encoder directory (contains encoder.pt)." >&2
  exit 1
fi
if [[ ! -d "${INIT_CHECKPOINT}" ]]; then
  echo "[ERROR] INIT_CHECKPOINT directory '${INIT_CHECKPOINT}' not found." >&2
  exit 1
fi

python -m aoe.train \
  --dataset "${AOE_DATASETS}" \
  --train_split train \
  --eval_split validation \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --lr 2e-5 \
  --warmup_steps "${WARMUP_STEPS}" \
  --angle_tau 20 \
  --cl_scale 20 \
  --w_angle "${STS_W_ANGLE}" \
  --w_cl 1.0 \
  --max_length "${MAX_LENGTH}" \
  --data_cache "${DATA_CACHE}" \
  --model_cache "${MODEL_CACHE}" \
  --seed "${SEED}" \
  --tensorboard_dir "${OUTPUT_ROOT}/${RUN_NAME}/tensorboard" \
  --metrics_path "${OUTPUT_ROOT}/${RUN_NAME}/metrics.jsonl" \
  --init_checkpoint "${INIT_CHECKPOINT}"
