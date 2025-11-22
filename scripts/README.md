## Scripts

Helper scripts for the AoE reproduction workflow:

- `train_pretrain_nli.sh`: Stage 1 SNLI+MNLI contrastive pretraining (produces `output/<run>/ckpt`).
- `finetune_aoe_mixed.sh`: Stage 2 AoE fine-tuning on the mixed STS dataset specified via `AOE_DATASETS` (requires `INIT_CHECKPOINT`).
- `train_stsb_aoe.sh`: single-run AoE training on STS-B (train split) with validation monitoring.
- `eval_sts_all.sh`: evaluate a checkpoint on STS-B + GIS using Spearman correlation.
- `eval_indomain.sh`: convenience wrapper for evaluating checkpoints on their source dataset only.
- `run_cosine_saturation.sh`: launch the cosine saturation analysis over sampled NLI pairs (unchanged from the paper).
- `run_all_experiments.sh`: sequentially runs Stage 1 + Stage 2 + evaluation/analysis (set `SKIP_NLI=true` with `INIT_CHECKPOINT` to reuse an existing pretrain).


## training pipeline

Example commands for the current `exp2` run (adjust env vars as needed):

1. **NLI pretrain (Stage 1)**

	```bash
	RUN_SUFFIX=exp2 bash scripts/train_pretrain_nli.sh
	```

2. **AoE fine-tune (Stage 2)**

	```bash
	INIT_CHECKPOINT=output/bert_nli_aoe_exp2/ckpt \
	RUN_SUFFIX=exp2 \
	bash scripts/finetune_aoe_mixed.sh
	```

3. **Evaluate on STS-B/GIS/SICK-R**

	```bash
	CKPT=output/bert_stsb_aoe_exp2/ckpt \
	STSB_SPLIT=validation \
	bash scripts/eval_sts_all.sh
	```

4. **Cosine saturation analysis**

	```bash
	PLOT_DIR=output/plots_exp2 bash scripts/run_cosine_saturation.sh
	```
