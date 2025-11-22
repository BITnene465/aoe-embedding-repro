#!/usr/bin/env bash
set -euo pipefail

CKPT=${CKPT:-output/bert_stsb_aoe/ckpt}
DATA_CACHE=${DATA_CACHE:-data}
MODEL_CACHE=${MODEL_CACHE:-models}
MAX_LENGTH=${MAX_LENGTH:-128}
STSB_SPLIT=${STSB_SPLIT:-validation}
DATASETS=${DATASETS:-stsb,gis,sickr}

python -m aoe.eval_sts \
  --ckpt "${CKPT}" \
  --datasets "${DATASETS}" \
  --stsb_split "${STSB_SPLIT}" \
  --data_cache "${DATA_CACHE}" \
  --model_cache "${MODEL_CACHE}" \
  --max_length "${MAX_LENGTH}"
