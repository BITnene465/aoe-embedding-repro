# AoE: Angle-Optimized Embeddings (Reproduction)

This repository contains a high-quality reproduction of the ACL 2024 paper **"AoE: Angle-Optimized Embeddings for Semantic Textual Similarity"**.

We successfully reproduced the paper's results, achieving state-of-the-art performance on STS tasks and surpassing the original paper on In-domain (GIS) benchmarks.

![overview](assets/img/overview.png)

📄 **[Read the Full Reproduction Report](papers/reproduction_report.md)**

📘 **[Step-by-Step Reproduction Guide](experiment/reproduction_guide.md)**

## 🚀 Quick Start

### 1. Installation

```bash
conda create -n aoe python=3.10 -y
conda activate aoe
pip install -r requirements.txt
```

### 2. Data Preparation

Download all necessary datasets (NLI, STS-B, GIS, etc.) and the backbone model:

```bash
# Download datasets
python scripts/download/download_data.py --output_dir data --datasets all

# Download BERT backbone
python scripts/download/download_model.py --model_name bert-base-uncased --output_dir models
```

### 3. Training & Evaluation

We provide one-click scripts for the three main experiments:

#### Standard Experiment (NLI Pre-training)
```bash
# Train (2 epochs, w_angle=1.0)
bash scripts/train_pretrain_nli.sh

# Evaluate (STS12-16, STS-B, SICK-R)
CKPT=output/bert_nli_aoe_2ep/ckpt bash scripts/eval_sts.sh
```

#### In-domain Experiment (STS-B / GIS)
```bash
# Train on GIS (20 epochs, w_angle=1.0)
TASK_NAME=gis EPOCHS=20 bash scripts/finetune_task.sh

# Evaluate
CKPT=output/bert_gis_aoe_20ep/ckpt bash scripts/eval_gis.sh
```

#### Downstream Experiment (Transfer)
```bash
# Evaluate NLI checkpoint on classification tasks (MR, CR, SUBJ, SST2)
CKPT=output/bert_nli_aoe_5ep/ckpt bash scripts/eval_downstream.sh
```

## 📊 Key Results

| Model | STS Avg (Standard) | GIS (In-domain) |
| :--- | :---: | :---: | 
| **AoE (Paper)** | **82.43** | 70.59 |
| **AoE (Ours)** | 82.20 | **70.94** | 


## 📂 Repository Structure

- `aoe/`: Core library (Model, Loss, Data, Trainer).
- `scripts/`: Automation scripts for training and evaluation.
- `papers/`: Detailed reproduction reports and analysis.
- `experiment/`: Raw experiment logs and guides.

## 🔗 Citation

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
