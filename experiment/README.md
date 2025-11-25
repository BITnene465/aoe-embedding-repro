# AoE Reproduction Experiments

This directory contains the planned experiments to reproduce the Angle-optimized Embeddings (AoE) results.

## Prerequisites

Ensure the environment is set up and dependencies are installed:
```bash
pip install -r requirements.txt
```

## Experiment 1: NLI Pretraining (Contrastive Learning)

**Goal**: Train the encoder on NLI data (SNLI + MNLI) using the fixed `in_batch_negative_loss`. This teaches the model basic semantic similarity and handles contradictions correctly.

**Script**: `scripts/train_pretrain_nli.sh`

**Parameters**:
- `NLI_W_ANGLE`: 0.02 (Aligned with official)
- `w_cl`: 1.0 (Aligned with official)
- `BATCH_SIZE`: 256 (Adjust based on GPU memory)

**Execution**:
```bash
# Run NLI pretraining
# Output will be in output/bert_nli_aoe_<timestamp>
bash scripts/train_pretrain_nli.sh
```

**Verification**:
- Check `metrics.jsonl` in the output directory.
- `train_contrast` should decrease over time.

## Experiment 2: STS Finetuning (AoE Loss)

**Goal**: Finetune the NLI-pretrained model on STS datasets (STS-B, GIS, SICK-R) using the full AoE loss (Angle + Contrastive).

**Script**: `scripts/finetune_aoe_mixed.sh`

**Prerequisite**: Requires the checkpoint from Experiment 1.

**Execution**:
```bash
# Set the path to the NLI pretrained checkpoint
export INIT_CHECKPOINT="output/bert_nli_aoe_YOUR_TIMESTAMP/ckpt"

# Run STS finetuning
bash scripts/finetune_aoe_mixed.sh
```

**Parameters**:
- `STS_W_ANGLE`: 0.02
- `w_cl`: 1.0

## Experiment 3: Evaluation

**Goal**: Evaluate the trained models on STS benchmarks (MTEB).

**Script**: `scripts/eval_sts_all.sh`

**Execution**:
```bash
# Evaluate the final STS model
export CKPT="output/bert_stsb_aoe_YOUR_TIMESTAMP/ckpt"
bash scripts/eval_sts_all.sh
```

## Directory Structure
- `experiment/`: This documentation.
- `output/`: Training artifacts (checkpoints, logs).
- `data/`: Cached datasets.
