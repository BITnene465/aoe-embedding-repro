#!/usr/bin/env bash
set -euo pipefail

python -m aoe.analysis \
  --mode cosine_saturation \
  --backbone bert-base-uncased \
  --max_samples 50000 \
  --plot_dir output/plot
