#!/usr/bin/env bash
set -euo pipefail

python -m aoe.eval_sts \
  --ckpt output/ckpt/bert_nli_aoe \
  --datasets stsb,gis
