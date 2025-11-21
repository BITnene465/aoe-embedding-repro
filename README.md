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

### Conda workflow

```bash
conda create -n aoe python=3.10 -y
conda activate aoe
pip install --upgrade pip
pip install -r requirements.txt
```

### Virtualenv alternative

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train an encoder

The training CLI follows the exact AnglE setup: every batch contains zigzag sentence pairs from an STS dataset, complex embeddings are produced in `SentenceEncoder`, and the AoE loss combines the CoSENT ranking angle with an InfoNCE contrastive term.

```bash
python -m aoe.train \
	--dataset stsb \
	--train_split train \
	--eval_split validation \
	--run_name bert_stsb_aoe \
	--output_dir output \
	--epochs 3 \
	--batch_size 64 \
	--grad_accum_steps 8 \
	--angle_tau 20 \
	--cl_scale 20 \
	--w_angle 0.02 \
	--w_cl 1.0
```

To pretrain on SNLI+MNLI with the full AnglE loss active, just target the `nli` dataset (the loader automatically merges both corpora and maps labels to continuous scores):

```bash
python -m aoe.train \
	--dataset nli \
	--train_split train \
	--eval_split none \
	--run_name bert_nli_aoe \
	--output_dir output \
	--epochs 1 \
	--batch_size 128 \
	--angle_tau 20 \
	--cl_scale 20 \
	--w_angle 0.02 \
	--w_cl 1.0
```

Every run writes `output/<run_name>/ckpt/encoder.pt` (full `SentenceEncoder` object), `metrics.jsonl` (per-batch/epoch loss snapshots), and `tensorboard/` event files unless you pass `--metrics_path none` or `--tensorboard_dir none`.

### Evaluate a checkpoint

```bash
python -m aoe.eval_sts \
	--ckpt output/bert_nli_aoe/ckpt \
	--datasets stsb,gis \
	--data_cache data
```

Use `--datasets sts_all` to evaluate STS-B, GIS, and SICK-R in one shot. The evaluation command automatically reloads the model config stored inside the checkpoint directory.

### Helper scripts

`run_all_experiments.sh` now mirrors the two-stage AoE workflow: (1) AoE pretraining on SNLI+MNLI with the angle loss enabled, (2) a contrastive STS-B baseline, (3) STS-B AoE fine-tuning initialized from the NLI checkpoint, (4) STS evaluation (STS-B/GIS/SICK-R), and (5) cosine saturation analysis. Environment variables such as `NLI_EPOCHS`, `NLI_W_ANGLE`, `STS_W_ANGLE`, or `GRAD_ACCUM_STEPS` let you tweak each stage without editing the script.

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

- **Loss**: The new `aoe.loss` module matches the official AnglE engineering. `angle_loss` implements the complex division / pooling pipeline from the paper and performs CoSENT-style pair-of-pairs ranking with `tau=20`. `supervised_contrastive_loss` mirrors Eq.(4) (InfoNCE with aligned anchors/positives). `aoe_total_loss` combines the two with weights `w_angle=0.02`, `w_cl=1.0`.
- **Data pipeline**: `aoe.train_utils` now builds zigzag batches straight from STS-style datasets (currently STS-B, GIS, SICK-R). Each batch looks like `[s0_1, s0_2, s1_1, s1_2, ...]`, and the matching scores are injected into `y_true` so the loss can build the order matrix.
- **Encoder**: `SentenceEncoder` always runs in `complex_mode=True` for AoE training. The real and imaginary parts are concatenated after encoding so the loss sees `[2 * dim]` features per text as required by AnglE.
- **Logging/checkpointing**: Every run records the full `TrainConfig`, metrics JSONL, and TensorBoard summaries. Checkpoints store the entire encoder object for direct `torch.load` usage.

## Cache & Output Layout

- Hugging Face datasets default to `data/` (override with `--data_cache`).
- Hugging Face model weights default to `models/` (override with `--model_cache`).
- Each training run writes to `output/<run_name>/` by default. Inside you will find `ckpt/` (weights + encoder config), `tensorboard/`, `metrics.jsonl`, and the serialized `train_config.json` that mirrors the CLI arguments.
- Pass `--metrics_path none` or `--tensorboard_dir none` to disable the corresponding artifacts or point them elsewhere.
- Analysis artifacts (plots) still default to `output/plot/` (override with `--plot_dir`).
- Run `tensorboard --logdir output --port 6006` to browse every run directory.

## Datasets

- **Training**: AoE is trained on STS-style sentence pairs with gold similarity scores. The default workflow uses STS-B train for optimization and STS-B validation for monitoring.
- **Evaluation**: `aoe.eval_sts` evaluates checkpoints on STS-B test, GIS, and SICK-R, reporting Spearman correlations just like the paper. Additional classic STS sets (STS12-16) can be wired up via `aoe.data.load_angle_pairs` if needed.

## Analysis

Run `python -m aoe.analysis --mode cosine_saturation --backbone bert-base-uncased --max_samples 50000` to inspect the cosine similarity distribution of sampled NLI pairs and reproduce the high-cosine saturation phenomenon described in the AoE paper. The script logs saturation percentages and saves a histogram (`cosine_hist.png`).

## Planned Experiments

1. Standard STS evaluation (STS-B, SICK-R, etc.)
2. In-domain STS (STS-B and GIS)
3. Cosine saturation analysis on NLI pairs

## Citation


```bibtex
@inproceedings{li2024aoe,
	title     = {AoE: Angle-Optimized Embeddings for Semantic Textual Similarity},
	author    = {Li, Xianming and Li, Jing},
	booktitle = {Proceedings of the 62nd Annual Meeting of the ACL},
	pages     = {1825--1839},
	year      = {2024},
	doi       = {10.18653/v1/2024.acl-long.101}
}
```
