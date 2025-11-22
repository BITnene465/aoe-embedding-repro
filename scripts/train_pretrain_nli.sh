#!/usr/bin/env bash
# Pretrain AoE encoder on SNLI+MNLI with contrastive loss only.
set -euo pipefail

NLI_EPOCHS=${NLI_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-64}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
DATA_CACHE=${DATA_CACHE:-data}
MODEL_CACHE=${MODEL_CACHE:-models}
MAX_LENGTH=${MAX_LENGTH:-128}
SEED=${SEED:-42}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
WARMUP_STEPS=${WARMUP_STEPS:-600}
NLI_W_ANGLE=${NLI_W_ANGLE:-0.0}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d_%H%M)}
RUN_NAME=${RUN_NAME:-bert_nli_aoe_${RUN_SUFFIX}}

python -m aoe.train \
  --dataset nli \
  --train_split train \
  --eval_split none \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_ROOT}" \
  --epochs "${NLI_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --lr 2e-5 \
  --warmup_steps "${WARMUP_STEPS}" \
  --angle_tau 20 \
  --cl_scale 20 \
  --w_angle "${NLI_W_ANGLE}" \
  --w_cl 1.0 \
  --max_length "${MAX_LENGTH}" \
  --data_cache "${DATA_CACHE}" \
  --model_cache "${MODEL_CACHE}" \
  --seed "${SEED}" \
  --tensorboard_dir "${OUTPUT_ROOT}/${RUN_NAME}/tensorboard" \
  --metrics_path "${OUTPUT_ROOT}/${RUN_NAME}/metrics.jsonl"
