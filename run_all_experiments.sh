#!/usr/bin/env bash
# Usage: bash run_all_experiments.sh
# Optional env vars:
#   EPOCHS=3 BATCH_SIZE=64 OUTPUT_ROOT=output DATA_CACHE=data MODEL_CACHE=models bash run_all_experiments.sh
set -euo pipefail

EPOCHS=${EPOCHS:-15}
NLI_EPOCHS=${NLI_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-64}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
DATA_CACHE=${DATA_CACHE:-data}
MODEL_CACHE=${MODEL_CACHE:-models}
MAX_LENGTH=${MAX_LENGTH:-128}
SEED=${SEED:-42}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
WARMUP_STEPS=${WARMUP_STEPS:-600}
BASELINE_W_ANGLE=${BASELINE_W_ANGLE:-0.0}
STS_W_ANGLE=${STS_W_ANGLE:-0.02}
NLI_W_ANGLE=${NLI_W_ANGLE:-0.0}

RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d_%H%M)}

if command -v tput >/dev/null 2>&1; then
  EMPH="$(tput bold 2>/dev/null)$(tput setaf 6 2>/dev/null)"
  RESET="$(tput sgr0 2>/dev/null)"
else
  EMPH=""
  RESET=""
fi

run_train() {
  local dataset=$1
  local run_name=$2
  local angle_weight=$3
  local train_split=$4
  local eval_split=$5
  local epochs=$6
  shift 6
  local extra_args=("$@")
  echo "${EMPH}[TRAIN] dataset=${dataset} w_angle=${angle_weight} run=${run_name}${RESET}"
  local cmd=(
    python -m aoe.train
    --dataset "${dataset}"
    --train_split "${train_split}"
    --eval_split "${eval_split}"
    --run_name "${run_name}"
    --output_dir "${OUTPUT_ROOT}"
    --epochs "${epochs}"
    --batch_size "${BATCH_SIZE}"
    --grad_accum_steps "${GRAD_ACCUM_STEPS}"
    --lr 2e-5
    --warmup_steps "${WARMUP_STEPS}"
    --angle_tau 20
    --cl_scale 20
    --w_angle "${angle_weight}"
    --w_cl 1.0
    --max_length "${MAX_LENGTH}"
    --data_cache "${DATA_CACHE}"
    --model_cache "${MODEL_CACHE}"
    --seed "${SEED}"
    --tensorboard_dir "${OUTPUT_ROOT}/${run_name}/tensorboard"
    --metrics_path "${OUTPUT_ROOT}/${run_name}/metrics.jsonl"
  )
  if ((${#extra_args[@]})); then
    cmd+=("${extra_args[@]}")
  fi
  "${cmd[@]}"
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
NLI_RUN="bert_nli_aoe_${RUN_SUFFIX}"
run_train nli "${NLI_RUN}" "${NLI_W_ANGLE}" train none "${NLI_EPOCHS}"

BASELINE_RUN="bert_stsb_cl_${RUN_SUFFIX}"
run_train stsb "${BASELINE_RUN}" "${BASELINE_W_ANGLE}" train validation "${EPOCHS}"

# 2. Train full AoE model (w_angle=0.02)
AOE_RUN="bert_stsb_aoe_${RUN_SUFFIX}"
run_train stsb "${AOE_RUN}" "${STS_W_ANGLE}" train validation "${EPOCHS}" --init_checkpoint "${OUTPUT_ROOT}/${NLI_RUN}/ckpt"

# 3. Evaluate the AoE checkpoint on STS datasets
run_eval "${OUTPUT_ROOT}/${AOE_RUN}/ckpt" stsb,gis,sickr

# 4. Run cosine saturation analysis (writes to output/plots by default)
echo "[ANALYSIS] cosine saturation"
python -m aoe.analysis --mode cosine_saturation --backbone bert-base-uncased --max_samples 50000 --plot_dir "${OUTPUT_ROOT}/plots"

echo "All experiments finished. Logs and checkpoints live under ${OUTPUT_ROOT}."
