# aoe-embeddings-repro

Reproduces the AoE (Angle-optimized Embeddings) method from the ACL 2024 paper using a trainable BERT-base encoder to produce sentence embeddings tuned for semantic textual similarity.

## Main Libraries

- torch
- transformers
- datasets
- numpy
- scipy
- tqdm
- matplotlib (optional)
- tensorboard (optional, for experiment tracking)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train an encoder

The training CLI now materializes a dedicated run directory at `<output_dir>/<run_name>` and snapshots every hyperparameter into `train_config.json` for easy provenance tracking.

```bash
python -m aoe.train \
	--task nli \
	--method aoe \
	--run_name bert_nli_aoe \
	--output_dir output \
	--epochs 3 \
	--batch_size 256 \
	--eval_split validation
```

This command creates `output/bert_nli_aoe/ckpt` (weights + config), `metrics.jsonl` (per-epoch loss snapshots), and `tensorboard/` event files unless you pass `--metrics_path none` or `--tensorboard_dir none`.

### Evaluate a checkpoint

```bash
python -m aoe.eval_sts \
	--ckpt output/bert_nli_aoe/ckpt \
	--datasets stsb,gis \
	--data_cache data
```

Use `--datasets sts_all` to evaluate STS-B, GIS, and SICK-R in one shot. The evaluation command automatically reloads the model config stored inside the checkpoint directory.

### Helper scripts

Common experiment recipes live under `scripts/`. For example:

```bash
bash scripts/train_nli_aoe.sh
bash scripts/train_nli_baseline.sh
bash scripts/eval_sts_all.sh
```

Feel free to copy these scripts and adapt hyperparameters or `run_name` values for your own runs.

```python
from aoe.model import SentenceEncoder

sentences = [
	"A man is playing guitar.",
	"Musicians perform live on stage.",
]

# Baseline (real-valued) embeddings
encoder = SentenceEncoder(pooling="mean")
embeddings = encoder.encode(sentences)

# AoE-style complex embeddings (real + imaginary parts)
complex_encoder = SentenceEncoder(complex_mode=True)
embeddings_re, embeddings_im = complex_encoder.encode(sentences)
```

## Implementation Details

The training objective follows a simplified AoE-inspired design: we measure angle differences between complex sentence embeddings so that high-similarity pairs keep smaller angles, and combine that signal with a supervised contrastive objective. The final loss is a weighted sum of the angle-ranking term and the contrastive loss, preserving the key intuition of AoE without reproducing every detail from the original paper.

## Cache & Output Layout

- Hugging Face datasets default to `data/` (override with `--data_cache`).
- Hugging Face model weights default to `models/` (override with `--model_cache`).
- Each training run writes to `output/<run_name>/` by default. Inside you will find `ckpt/` (weights + encoder config), `tensorboard/`, `metrics.jsonl`, and the serialized `train_config.json` that mirrors the CLI arguments.
- Pass `--metrics_path none` or `--tensorboard_dir none` to disable the corresponding artifacts or point them elsewhere.
- Analysis artifacts (plots) still default to `output/plot/` (override with `--plot_dir`).
- Run `tensorboard --logdir output --port 6006` to browse every run directory.

## Datasets

- SNLI (`snli`) + MultiNLI (`multi_nli`) for supervised NLI pretraining.
- STS-B (`glue`, config `stsb`) as a standard English STS benchmark.
- GitHub Issue Similarity (`WhereIsAI/github-issue-similarity`) for in-domain STS-style evaluation.

## Analysis

Run `python -m aoe.analysis --mode cosine_saturation --backbone bert-base-uncased --max_samples 50000` to inspect the cosine similarity distribution of sampled NLI pairs and reproduce the high-cosine saturation phenomenon described in the AoE paper. The script logs saturation percentages and saves a histogram (`cosine_hist.png`).

## Planned Experiments

1. Standard STS evaluation (STS-B, SICK-R, etc.)
2. In-domain STS (STS-B and GIS)
3. Cosine saturation analysis on NLI pairs
