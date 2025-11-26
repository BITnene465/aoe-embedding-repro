#!/usr/bin/env bash
# Run batch experiments for AoE reproduction.
# Total Experiments: 6
# 1. NLI Pre-training: 1, 2 epochs (2 runs)
# 2. STS-B Fine-tuning (In-domain): 5, 10 epochs (2 runs)
# 3. GIS Fine-tuning (In-domain): 5, 10 epochs (2 runs)

set -euo pipefail

# ANSI Color Codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Global Configuration for 4x GPUs
export BATCH_SIZE=64
export GRAD_ACCUM_STEPS=2
export LAUNCHER=${LAUNCHER:-"accelerate launch"}

echo -e "${BOLD}${BLUE}========================================================${NC}"
echo -e "${BOLD}${BLUE}Starting Batch Experiments (Total: 6)${NC}"
echo -e "${BOLD}${BLUE}Configuration: 4x GPUs, Batch=${BATCH_SIZE}, Accum=${GRAD_ACCUM_STEPS}${NC}"
echo -e "${BOLD}${BLUE}========================================================${NC}"

# 1. NLI Pre-training
echo ""
echo -e "${BOLD}${YELLOW}>>> [1/3] Starting NLI Pre-training Experiments (2 runs)${NC}"
for epochs in 1 2; do
    echo -e "${GREEN}--------------------------------------------------------${NC}"
    echo -e "${GREEN}Running NLI Pre-training for ${epochs} epochs...${NC}"
    # Suffix only contains epochs, e.g., "1ep"
    # Resulting run name: bert_nli_aoe_1ep
    NLI_EPOCHS=${epochs} RUN_SUFFIX="${epochs}ep" bash scripts/train_pretrain_nli.sh
done

# 2. STS-B Fine-tuning (In-domain, from BERT)
echo ""
echo -e "${BOLD}${YELLOW}>>> [2/3] Starting STS-B Fine-tuning Experiments (2 runs)${NC}"
for epochs in 5 10; do
    echo -e "${GREEN}--------------------------------------------------------${NC}"
    echo -e "${GREEN}Running STS-B Fine-tuning for ${epochs} epochs...${NC}"
    # Resulting run name: bert_stsb_aoe_5ep
    TASK_NAME=stsb LEARNING_RATE=2e-5 EPOCHS=${epochs} RUN_SUFFIX="${epochs}ep" bash scripts/finetune_task.sh
done

# 3. GIS Fine-tuning (In-domain, from BERT)
echo ""
echo -e "${BOLD}${YELLOW}>>> [3/3] Starting GIS Fine-tuning Experiments (2 runs)${NC}"
for epochs in 5 10; do
    echo -e "${GREEN}--------------------------------------------------------${NC}"
    echo -e "${GREEN}Running GIS Fine-tuning for ${epochs} epochs...${NC}"
    # Resulting run name: bert_gis_aoe_5ep
    TASK_NAME=gis LEARNING_RATE=2e-5 EPOCHS=${epochs} RUN_SUFFIX="${epochs}ep" bash scripts/finetune_task.sh
done

echo ""
echo -e "${BOLD}${BLUE}========================================================${NC}"
echo -e "${BOLD}${BLUE}All 6 experiments completed successfully.${NC}"
echo -e "${BOLD}${BLUE}========================================================${NC}"
