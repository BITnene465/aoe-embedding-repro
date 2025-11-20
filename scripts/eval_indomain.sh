#!/usr/bin/env bash
set -euo pipefail

python -m aoe.eval_sts \
  --ckpt output/bert_stsb_aoe/ckpt \
  --datasets stsb

python -m aoe.eval_sts \
  --ckpt output/bert_gis_aoe/ckpt \
  --datasets gis
