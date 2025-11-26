#!/bin/bash

# Wrapper script for GIS evaluation
# Usage: 
#   CKPT=path/to/ckpt MODEL_NAME=my_model bash scripts/eval_gis.sh

set -e

: "${CKPT:?Need to set CKPT}"
: "${MODEL_NAME:=$(basename $(dirname $CKPT))}"
: "${OUTPUT_ROOT:=output}"
: "${DATA_CACHE:=data}"
: "${RESULTS_DIR:=${OUTPUT_ROOT}/mteb}"

echo "Evaluating GIS task..."
echo "Checkpoint: ${CKPT}"

python -m aoe.eval_gis \
    --ckpt "${CKPT}" \
    --model_name "${MODEL_NAME}" \
    --data_cache "${DATA_CACHE}" \
    --results_dir "${RESULTS_DIR}"
