#!/usr/bin/env bash
set -euo pipefail

python -m aoe.eval_sts \
  --ckpt output/bert_nli_aoe/ckpt \
  --datasets stsb,gis
