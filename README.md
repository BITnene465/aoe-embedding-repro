# aoe-embeddings-repro

Reproduces the AoE (Angle-optimized Embeddings) method from the ACL 2024 paper using a trainable BERT-base encoder to produce sentence embeddings tuned for semantic textual similarity.

## Main Libraries

- torch
- transformers
- datasets
- numpy
- scipy
- tqdm
- accelerate
- matplotlib
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

## Data & Model Preparation

Before training, download the necessary datasets and models.

```bash
# Download datasets (NLI, STS-B, SICK-R, GIS, MTEB STS)
python3 scripts/download/download_data.py --output_dir data --datasets all

# Download backbone model
python3 scripts/download/download_model.py --model_name bert-base-uncased --output_dir models
```

## Quick Start

For a detailed step-by-step guide, see **[experiment/reproduction_guide.md](experiment/reproduction_guide.md)**.

### Stage 1: NLI Pretraining

```bash
# Single GPU
bash scripts/train_pretrain_nli.sh

# Multi-GPU (Accelerate)
accelerate launch scripts/train_pretrain_nli.sh
```

Official configuration:
- Binary classification: **excludes neutral samples** (entailment vs contradiction only)
- `batch_size=32`, `grad_accum=16` (effective batch: 512)
- `epochs=1`, `warmup_steps=100`, `lr=2e-5`
- `w_angle=1.0`, `w_cl=30.0` (angle + strong contrastive)

### Stage 2: STS Mixed Fine-tuning

```bash
export INIT_CHECKPOINT=output/bert_nli_aoe_<timestamp>/ckpt

# Single GPU
bash scripts/finetune_aoe_mixed.sh

# Multi-GPU (Accelerate)
accelerate launch scripts/finetune_aoe_mixed.sh
```

Fine-tune on STS-B + GIS + SICK-R:
- `batch_size=32`, `grad_accum=16`, `epochs=5`
- `w_angle=0.02`, `w_cl=1.0` (reduced angle weight)
- Default datasets: `stsb@train,gis@train,sickr@validation`

### Evaluation

```bash
CKPT=output/bert_stsb_aoe_<timestamp>/ckpt bash scripts/eval_sts_all.sh
```

Reports Spearman correlations on **7 official STS datasets** (STS12-16, STS-B, SICK-R) following the AnglE paper's evaluation protocol.

## Advanced Usage

### Custom Training

```bash
python -m aoe.train \
	--dataset stsb@train,gis@train \
	--train_split train \
	--eval_split validation \
	--run_name my_experiment \
	--output_dir output \
	--epochs 5 \
	--batch_size 32 \
	--grad_accum_steps 16 \
	--lr 2e-5 \
	--warmup_steps 100 \
	--angle_tau 20 \
	--cl_scale 20 \
	--w_angle 0.02 \
	--w_cl 1.0
```

### Environment Variables

**NLI Training** (`train_pretrain_nli.sh`):
- `NLI_EPOCHS` (default: 1)
- `BATCH_SIZE` (default: 32)
- `GRAD_ACCUM_STEPS` (default: 16)
- `NLI_W_ANGLE` (default: 1.0)
- `WARMUP_STEPS` (default: 100)
- `LEARNING_RATE` (default: 5e-5)

**STS Fine-tuning** (`finetune_aoe_mixed.sh`):
- `EPOCHS` (default: 5)
- `STS_W_ANGLE` (default: 0.02)
- `AOE_DATASETS` (default: `stsb@train,gis@train,sickr@validation`)
- `INIT_CHECKPOINT` (required: path to NLI checkpoint)
- `LEARNING_RATE` (default: 2e-5)

Example:
```bash
NLI_EPOCHS=2 NLI_W_ANGLE=0.5 bash scripts/train_pretrain_nli.sh
EPOCHS=10 STS_W_ANGLE=0.01 INIT_CHECKPOINT=output/bert_nli_aoe_exp3/ckpt bash scripts/finetune_aoe_mixed.sh
```

### Python API

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

### Output Artifacts

Each run writes to `output/<run_name>/`:
- `ckpt/encoder.pt` - Full `SentenceEncoder` object
- `metrics.jsonl` - Per-batch/epoch loss snapshots
- `tensorboard/` - TensorBoard event files
- `train_config.json` - Complete training configuration

Browse training curves:
```bash
tensorboard --logdir output --port 6006
```

## Implementation Details

### Loss Functions

Fully aligned with official AnglE implementation:

- **`angle_loss`**: Complex division → pooling → abs → CoSENT pair-ranking (tau=20)
- **`supervised_contrastive_loss`**: InfoNCE with aligned anchor/positive pairs (scale=20)
- **Hyperparameters**:
  - **NLI stage**: `w_angle=1.0`, `w_cl=30.0` (strong contrastive supervision)
  - **STS stage**: `w_angle=0.02`, `w_cl=1.0` (angle loss becomes dominant)

### Data Pipeline

**NLI (SNLI + MultiNLI)**:
- Binary classification (excludes neutral samples, matching official repo)
- `entailment` → score 1.0
- `contradiction` → score 0.0
- `neutral` → **excluded** (official configuration)
- ~550k training pairs after filtering

**STS Datasets**:
- Zigzag batch construction: `[s0_1, s0_2, s1_1, s1_2, ...]`
- Gold similarity scores injected for ranking loss
- Dataset mixing: `stsb@train,gis@train,sickr@validation`

### Encoder Architecture

`SentenceEncoder` operates in `complex_mode=True` during training:
- Splits hidden dimension: `[real_part, imaginary_part]`
- Outputs concatenated features `[2 * dim]` for angle loss
- Pooling: configurable (`cls` or `mean`)

### Training Configuration

Official AnglE defaults:
- Batch size: **32** (single GPU)
- Gradient accumulation: **16** (effective batch: 512)
- Learning rate: **2e-5** (both stages)
- Warmup steps: **100**
- NLI epochs: **1**
- STS epochs: **5**

## Datasets

### NLI (SNLI + MultiNLI)
- Used for Stage 1 pretraining
- Binary classification: `entailment` (1.0) vs `contradiction` (0.0)
- **Neutral samples excluded** (official configuration)
- ~550k training pairs after filtering

### STS Evaluation (Official AnglE Benchmark)
Official evaluation uses 7 STS datasets following SentEval protocol:
- **STS12** (5 sub-tasks): MSRpar, MSRvid, SMTeuroparl, OnWN, SMTnews
- **STS13** (3 sub-tasks): FNWN, headlines, OnWN
- **STS14** (6 sub-tasks): deft-forum, deft-news, headlines, images, OnWN, tweet-news
- **STS15** (5 sub-tasks): answers-forums, answers-students, belief, headlines, images
- **STS16** (5 sub-tasks): answer-answer, headlines, plagiarism, postediting, question-question
- **STS-B** (STSBenchmark): 5.7k train, 1.5k dev, 1.4k test pairs (scores 0-5)
- **SICK-R** (SICKRelatedness): 4.5k train, 0.5k dev, 4.9k test pairs (scores 1-5)

### STS Training Datasets
- **STS-B** (GLUE): 5.7k train, 1.5k validation pairs (scores 0-5, normalized to 0-1)
- **GIS** (GitHub Issue Similarity): ~28k pairs with binary similarity labels
- **SICK-R**: Relatedness scores from SICK validation split

### Dataset Mixing

Combine multiple sources via comma-separated specs:
```bash
--dataset stsb@train,gis@train,sickr@validation
```

The `@split` suffix overrides the global `--train_split` for specific datasets.

## Cache & Output Layout

- **Datasets**: `data/` (override with `--data_cache`)
- **Models**: `models/` (override with `--model_cache`)
- **Outputs**: `output/<run_name>/`
  - `ckpt/` - Weights + encoder config
  - `tensorboard/` - Training curves
  - `metrics.jsonl` - Per-batch metrics
  - `train_config.json` - CLI arguments

## Experiments

Based on official AnglE configuration:

### exp1 (Baseline)
NLI pretraining only → evaluate on STS-B
- Expected: Poor performance (~0.50 Spearman) due to cosine saturation

### exp2 (Previous Attempt)
Mixed configuration with incorrect hyperparameters
- Issues identified: wrong loss weights, included neutral samples, incorrect batch size

### exp3 (Official Config)
Two-stage training with corrected parameters:
- Stage 1: NLI with `w_angle=1.0, w_cl=30.0`, binary classification
- Stage 2: STS mix with `w_angle=0.02, w_cl=1.0`
- **Target**: Match official AnglE performance across all 7 STS benchmarks
  - Expected average: ~85% Spearman (STS12-16 + STS-B + SICK-R)
  - Individual targets: STS-B ~88%, SICK-R ~81%

### Analysis
Cosine saturation phenomenon on NLI pairs:
```bash
python -m aoe.analysis --mode cosine_saturation --backbone bert-base-uncased --max_samples 50000
```
- Measure high-cosine percentage (expected: >90% above 0.8)
- Visualize distribution shift after angle optimization

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
