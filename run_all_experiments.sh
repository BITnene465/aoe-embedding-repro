#!/usr/bin/env bash
# Usage: bash scripts/run_all_experiments.sh
# Optional env vars:
#   EPOCHS=3 BATCH_SIZE=256 OUTPUT_ROOT=output DATA_CACHE=data MODEL_CACHE=models bash scripts/run_all_experiments.sh
set -euo pipefail

EPOCHS=${EPOCHS:-3}
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
  local task=$1
  local method=$2
  local run_name=$3
  echo "${EMPH}[TRAIN] task=${task} method=${method} run=${run_name}${RESET}"
  python -m aoe.train \
    --task "${task}" \
    --method "${method}" \
    --run_name "${run_name}" \
    --output_dir "${OUTPUT_ROOT}" \
    --epochs "${EPOCHS}" \
    --batch_size 128 \
    --grad_accum_steps 8 \
    --lr 1e-4 \
    --warmup_steps 600 \
    --max_length "${MAX_LENGTH}" \
    --data_cache "${DATA_CACHE}" \
    --model_cache "${MODEL_CACHE}" \
    --seed "${SEED}" \
    --eval_split validation \
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

# 1. Train baseline encoder on NLI
BASELINE_RUN="bert_nli_baseline_${RUN_SUFFIX}"
run_train nli baseline "${BASELINE_RUN}"

# 2. Train AoE encoder on NLI
AOE_RUN="bert_nli_aoe_${RUN_SUFFIX}"
run_train nli aoe "${AOE_RUN}"

# 3. Optional: train AoE on STS-B for domain finetune
STS_RUN="bert_stsb_aoe_${RUN_SUFFIX}"
run_train stsb aoe "${STS_RUN}"

# 4. Evaluate the best checkpoint on STS tasks
run_eval "${OUTPUT_ROOT}/${AOE_RUN}/ckpt" stsb,gis,sickr

# 5. Run cosine saturation analysis (writes to output/plots by default)
echo "[ANALYSIS] cosine saturation"
python -m aoe.analysis --mode cosine_saturation --backbone bert-base-uncased --max_samples 50000 --output_dir "${OUTPUT_ROOT}/plots"

echo "All experiments finished. Logs and checkpoints live under ${OUTPUT_ROOT}."
