#!/usr/bin/env bash
set -euo pipefail

BACKBONE=${BACKBONE:-bert-base-uncased}
MAX_SAMPLES=${MAX_SAMPLES:-50000}
MAX_LENGTH=${MAX_LENGTH:-64}
DATA_CACHE=${DATA_CACHE:-data}
MODEL_CACHE=${MODEL_CACHE:-models}
PLOT_DIR=${PLOT_DIR:-output/plots}

python -m aoe.analysis \
  --mode cosine_saturation \
  --backbone "${BACKBONE}" \
  --max_samples "${MAX_SAMPLES}" \
  --max_length "${MAX_LENGTH}" \
  --data_cache "${DATA_CACHE}" \
  --model_cache "${MODEL_CACHE}" \
  --plot_dir "${PLOT_DIR}"
