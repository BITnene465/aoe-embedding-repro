#!/usr/bin/env bash
# Usage: bash run_all_experiments.sh
# Optional env vars:
#   EPOCHS=3 BATCH_SIZE=64 OUTPUT_ROOT=output DATA_CACHE=data MODEL_CACHE=models bash run_all_experiments.sh
set -euo pipefail

EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-64}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
DATA_CACHE=${DATA_CACHE:-data}
MODEL_CACHE=${MODEL_CACHE:-models}
MAX_LENGTH=${MAX_LENGTH:-128}
SEED=${SEED:-42}

RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d_%H%M)}

if command -v tput >/dev/null 2>&1; then
  EMPH="$(tput bold 2>/dev/null)$(tput setaf 6 2>/dev/null)"
  RESET="$(tput sgr0 2>/dev/null)"
else
  EMPH=""
  RESET=""
fi

run_train() {
  local run_name=$1
  local angle_weight=$2
  echo "${EMPH}[TRAIN] dataset=stsb w_angle=${angle_weight} run=${run_name}${RESET}"
  python -m aoe.train \
    --dataset stsb \
    --train_split train \
    --eval_split validation \
    --run_name "${run_name}" \
    --output_dir "${OUTPUT_ROOT}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum_steps 8 \
    --lr 2e-5 \
    --warmup_steps 600 \
    --angle_tau 20 \
    --cl_scale 20 \
    --w_angle "${angle_weight}" \
    --w_cl 1.0 \
    --max_length "${MAX_LENGTH}" \
    --data_cache "${DATA_CACHE}" \
    --model_cache "${MODEL_CACHE}" \
    --seed "${SEED}" \
    --tensorboard_dir "${OUTPUT_ROOT}/${run_name}/tensorboard" \
    --metrics_path "${OUTPUT_ROOT}/${run_name}/metrics.jsonl"
}

run_eval() {
  local ckpt=$1
  local datasets=$2
  echo "${EMPH}[EVAL] ckpt=${ckpt} datasets=${datasets}${RESET}"
  python -m aoe.eval_sts \
    --ckpt "${ckpt}" \
    --datasets "${datasets}" \
    --data_cache "${DATA_CACHE}" \
    --model_cache "${MODEL_CACHE}" \
    --max_length "${MAX_LENGTH}"
}

# 1. Train contrastive baseline (w_angle=0)
BASELINE_RUN="bert_stsb_cl_${RUN_SUFFIX}"
run_train "${BASELINE_RUN}" 0.0

# 2. Train full AoE model (w_angle=0.02)
AOE_RUN="bert_stsb_aoe_${RUN_SUFFIX}"
run_train "${AOE_RUN}" 0.02

# 3. Evaluate the AoE checkpoint on STS datasets
run_eval "${OUTPUT_ROOT}/${AOE_RUN}/ckpt" stsb,gis,sickr

# 4. Run cosine saturation analysis (writes to output/plots by default)
echo "[ANALYSIS] cosine saturation"
python -m aoe.analysis --mode cosine_saturation --backbone bert-base-uncased --max_samples 50000 --output_dir "${OUTPUT_ROOT}/plots"

echo "All experiments finished. Logs and checkpoints live under ${OUTPUT_ROOT}."
