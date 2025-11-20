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

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

> The full training and evaluation code will be added soon; commands below are placeholders.

```bash
# Example 1: Train an AoE model on NLI data
python -m aoe.train --task nli --method aoe --output_dir ckpt/bert_nli_aoe

# Example 2: Evaluate a checkpoint on STS-B
python -m aoe.eval_sts --ckpt ckpt/bert_nli_aoe --datasets stsb
```

```bash
# Train a baseline (contrastive-only) encoder on SNLI + MultiNLI
python -m aoe.train --task nli --method baseline --output_dir ckpt/bert_nli_baseline

# Train an AoE encoder with angle + contrastive objectives
python -m aoe.train --task nli --method aoe --output_dir ckpt/bert_nli_aoe

# Evaluate an AoE checkpoint on STS-B and GIS
python -m aoe.eval_sts --ckpt ckpt/bert_nli_aoe --datasets stsb,gis
```

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
