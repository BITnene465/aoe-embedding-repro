#!/usr/bin/env bash
set -euo pipefail

python -m aoe.analysis \
  --mode cosine_saturation \
  --backbone bert-base-uncased \
  --max_samples 50000 \
  --checkpoint output/bert_nli_aoe/ckpt \
  --plot_dir output/bert_nli_aoe/plot
