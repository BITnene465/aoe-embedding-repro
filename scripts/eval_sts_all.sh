#!/usr/bin/env bash
set -euo pipefail

python -m aoe.eval_sts \
  --ckpt ckpt/bert_nli_aoe \
  --datasets stsb,gis
