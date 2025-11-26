# AoE Reproduction Guide

This document outlines the steps to reproduce the experiments from the paper "AoE: Angle-optimized Embeddings for Semantic Textual Similarity".

## 1. Environment Setup

Ensure you have the required dependencies installed.

```bash
pip install -r requirements.txt
```

## 2. Data & Model Preparation

Before training, you must download the datasets and the backbone model. This ensures all experiments run in an offline-compatible environment.

### Download Datasets
Downloads NLI (SNLI, MultiNLI), STS-B, SICK-R, GIS, and MTEB STS tasks (STS12-16).

```bash
# Download all datasets (including MTEB eval tasks)
python scripts/download/download_data.py --output_dir data --datasets all

# Or download specific datasets
python scripts/download/download_data.py --output_dir data --datasets nli stsb mteb_sts
```

### Download Backbone Model
Downloads `bert-base-uncased` (or other specified backbones).

```bash
python scripts/download/download_model.py --model_name bert-base-uncased --output_dir models
```

## 3. Pre-training on NLI

The core of AoE is training on NLI data using a combination of Contrastive Loss and Angle Loss.

**Script:** `scripts/train_pretrain_nli.sh`

**Key Hyperparameters (aligned with paper):**
- **Backbone:** BERT-base-uncased
- **Batch Size:** 512 (simulated via gradient accumulation if needed)
- **Learning Rate:** 5e-5
- **Pooling:** CLS
- **Projection Head:** MLP (Linear-Tanh-Linear) during training; None during inference
- **Angle Tau:** 20.0
- **Contrastive Scale:** 20.0
- **Angle Weight (`w_angle`):** 1.0 (for NLI)
- **Contrastive Weight (`w_cl`):** 30.0 (for NLI)

**Command:**
```bash
# Run NLI pre-training

# Single GPU
BATCH_SIZE=64 \
GRAD_ACCUM_STEPS=8 \
LEARNING_RATE=4e-5 \
NLI_EPOCHS=2 \
RUN_SUFFIX=2ep \
bash ./scripts/train_pretrain_nli.sh

# 4x RTX 3090 Configuration
# Effective batch size = 64 * 4 (GPUs) * 2 (Accum) = 512
BATCH_SIZE=64 \
GRAD_ACCUM_STEPS=2 \
LEARNING_RATE=2e-5 \
NLI_EPOCHS=2 \
RUN_SUFFIX=2ep \
LAUNCHER="accelerate launch" \
bash ./scripts/train_pretrain_nli.sh
```

**Output:**
Checkpoints will be saved to `output/bert_nli_aoe_2ep/ckpt`.

## 4. Standard Experiment (Transfer Learning)
**Goal**: Evaluate the NLI-pretrained model on standard STS tasks (Zero-shot transfer). This tests the backbone's generalization capability.
**Tasks**: STS12, STS13, STS14, STS15, STS16, STS-B, SICK-R. (Excluding GIS)

**Command**:
```bash
# Evaluate on all standard STS tasks
CKPT=output/bert_nli_aoe_2ep/ckpt \
TASKS=all \
MODEL_NAME=bert_nli_standard_2ep \
bash scripts/eval_sts.sh
```

## 5. In-domain Experiments (Supervised)
**Goal**: Fine-tune and evaluate on specific domains.

### 5.1 STS-B In-domain
**Training**: Fine-tune `bert-base-uncased` directly on STS-B Train (Single-stage).
**Evaluation**: Evaluate on STS-B Test.

**Train**:
```bash
# Ensure we start from BERT, not an NLI checkpoint
# 4x RTX 3090 Configuration
# Effective batch size = 64 * 4 (GPUs) * 1 (Accum) = 256
TASK_NAME=stsb \
LAUNCHER="accelerate launch" \
EPOCHS=30 \
BATCH_SIZE=64 \
MAX_LENGTH=128 \
RUN_SUFFIX=30ep \
bash scripts/finetune_task.sh
```

**Evaluate**:
```bash
CKPT=output/bert_stsb_aoe_30ep/ckpt \
TASKS=STSBenchmark \
MODEL_NAME=bert_stsb_indomain_30ep \
bash scripts/eval_sts.sh
```

### 5.2 GIS In-domain
**Training**: Fine-tune `bert-base-uncased` directly on GIS Train (Single-stage).
**Evaluation**: Evaluate on GIS Test.

**Train**:
```bash
# Ensure we start from BERT, not an NLI checkpoint
# 4x RTX 3090 Configuration
# Effective batch size = 32 * 4 (GPUs) * 2 (Accum) = 256    
TASK_NAME=gis \
LAUNCHER="accelerate launch" \
EPOCHS=20 \
BATCH_SIZE=32 \
GRAD_ACCUM_STEPS=2 \
MAX_LENGTH=512 \
RUN_SUFFIX=20ep \
bash scripts/finetune_task.sh
```

**Evaluate**:
you have to set MAX_LENGTH=512
```bash
CKPT=output/bert_gis_aoe_20ep/ckpt \
MAX_LENGTH=512 \
MODEL_NAME=bert_gis_indomain_20ep \
bash scripts/eval_gis.sh
```

## 6. Downstream Experiments
**Goal**: Fine-tune the NLI-pretrained model on various downstream tasks (e.g., retrieval, clustering) as described in the paper.

### Transfer Tasks
Evaluate the embedding quality on standard classification tasks by training a classifier on top of frozen embeddings.
**Tasks**: MR, CR, SUBJ, MPQA, SST2, TREC, MRPC.

**Command**:
```bash
# Evaluate NLI-pretrained model
CKPT=output/bert_nli_aoe_2ep/ckpt \
MODEL_NAME=bert_nli_2ep_downstream \
bash scripts/eval_downstream.sh
```

**Expected Results**:
Compare with Table 3 in the paper. Metrics are usually Accuracy.

## 7. Analysis (Cosine Saturation)

To verify the mitigation of the representation collapse (cosine saturation) problem:

```bash
python -m aoe.analysis \
    --mode cosine_saturation \
    --model_cache models \
    --data_cache data
```
This will generate a histogram of cosine similarities for random NLI pairs. AoE should produce a more spread-out distribution compared to raw BERT.

```bash
python -m aoe.analysis \
    --mode cosine_saturation \
    --backbone output/bert_nli_aoe_2ep/ckpt \
    --max_samples 50000 \
    --plot_dir output/plot_nli_aoe_2ep
```


