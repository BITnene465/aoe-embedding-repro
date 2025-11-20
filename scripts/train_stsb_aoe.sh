#!/usr/bin/env bash
set -euo pipefail

python -m aoe.train \
  --task stsb \
  --method aoe \
  --backbone bert-base-uncased \
  --batch_size 64 \
  --epochs 1 \
  --lr 2e-5 \
  --output_dir output \
  --run_name bert_stsb_aoe
