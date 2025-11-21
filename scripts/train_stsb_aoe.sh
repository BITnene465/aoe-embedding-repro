#!/usr/bin/env bash
set -euo pipefail

python -m aoe.train \
  --dataset stsb \
  --train_split train \
  --eval_split validation \
  --backbone bert-base-uncased \
  --batch_size 64 \
  --epochs 1 \
  --lr 2e-5 \
  --angle_tau 20 \
  --cl_scale 20 \
  --w_angle 0.02 \
  --w_cl 1.0 \
  --output_dir output \
  --run_name bert_stsb_aoe
