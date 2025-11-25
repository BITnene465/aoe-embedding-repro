#!/usr/bin/env bash
set -euo pipefail

CKPT=${CKPT:-output/bert_stsb_aoe/ckpt}
DATA_CACHE=${DATA_CACHE:-data}
MODEL_CACHE=${MODEL_CACHE:-models}
MAX_LENGTH=${MAX_LENGTH:-128}
BATCH_SIZE=${BATCH_SIZE:-64}
TASKS=${TASKS:-STS12,STS13,STS14,STS15,STS16,STSBenchmark,SICK-R}
EVAL_SPLITS=${EVAL_SPLITS:-test}
RESULTS_DIR=${RESULTS_DIR:-output/mteb}

HF_CACHE_ROOT=${HF_CACHE_ROOT:-${DATA_CACHE}}
export HF_HOME=${HF_HOME:-${HF_CACHE_ROOT}/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/hf_datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_CACHE_ROOT}/hf_hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/hf_models}

python -m aoe.eval_sts \
  --ckpt "${CKPT}" \
  --tasks "${TASKS}" \
  --data_cache "${DATA_CACHE}" \
  --model_cache "${MODEL_CACHE}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --results_dir "${RESULTS_DIR}" \
  --eval_splits "${EVAL_SPLITS}"
