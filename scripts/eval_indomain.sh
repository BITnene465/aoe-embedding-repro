#!/usr/bin/env bash
set -euo pipefail

python -m aoe.eval_sts \
  --ckpt ckpt/bert_stsb_aoe \
  --datasets stsb

python -m aoe.eval_sts \
  --ckpt ckpt/bert_gis_aoe \
  --datasets gis
