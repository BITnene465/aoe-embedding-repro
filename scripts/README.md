# Scripts

Helper scripts for common experiments:

- `train_nli_baseline.sh`: train the contrastive-only baseline on SNLI + MultiNLI.
- `train_nli_aoe.sh`: train the AoE model on SNLI + MultiNLI.
- `train_stsb_aoe.sh`: run AoE fine-tuning on STS-B.
- `train_gis_aoe.sh`: run AoE fine-tuning on the GitHub Issue Similarity dataset.
- `eval_sts_all.sh`: evaluate the AoE NLI checkpoint on STS-B and GIS.
- `eval_indomain.sh`: evaluate in-domain checkpoints (STS-B model on STS-B, GIS model on GIS).
- `run_cosine_saturation.sh`: launch cosine saturation analysis over sampled NLI pairs.
