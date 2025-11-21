# Scripts

Helper scripts for the AoE reproduction workflow:

- `train_stsb_aoe.sh`: single-run AoE training on STS-B (train split) with validation monitoring.
- `eval_sts_all.sh`: evaluate a checkpoint on STS-B + GIS using Spearman correlation.
- `eval_indomain.sh`: convenience wrapper for evaluating checkpoints on their source dataset only.
- `run_cosine_saturation.sh`: launch the cosine saturation analysis over sampled NLI pairs (unchanged from the paper).
