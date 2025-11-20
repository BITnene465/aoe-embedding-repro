#!/usr/bin/env bash
set -euo pipefail

python -m aoe.train \
  --task nli \
  --method baseline \
  --backbone bert-base-uncased \
  --batch_size 128 \
  --epochs 1 \
  --lr 2e-5 \
  --output_dir output \
  --run_name bert_nli_baseline
