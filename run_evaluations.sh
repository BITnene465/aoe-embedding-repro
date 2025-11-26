#!/usr/bin/env bash
# Run batch evaluations for AoE reproduction.
# Total Evaluations: 12
# 1. NLI Models -> Standard STS Transfer (3 runs)
# 2. NLI Models -> Downstream Transfer (3 runs)
# 3. STS-B Models -> STS-B In-domain (3 runs)
# 4. GIS Models -> GIS In-domain (3 runs)

set -euo pipefail

# ANSI Color Codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

OUTPUT_ROOT=${OUTPUT_ROOT:-output}

echo -e "${BOLD}${BLUE}========================================================${NC}"
echo -e "${BOLD}${BLUE}Starting Batch Evaluations (Total: 12)${NC}"
echo -e "${BOLD}${BLUE}========================================================${NC}"

# 1. NLI Models -> Standard STS Evaluation (Zero-shot)
# echo ""
# echo -e "${BOLD}${YELLOW}>>> [1/4] Evaluating NLI Models on Standard STS Tasks (3 runs)${NC}"
# for epochs in 1 2 5; do
#     RUN_NAME="bert_nli_aoe_${epochs}ep"
#     CKPT="${OUTPUT_ROOT}/${RUN_NAME}/ckpt"
    
#     if [[ ! -d "${CKPT}" ]]; then
#         echo -e "${YELLOW}[WARNING] Checkpoint not found: ${CKPT}. Skipping...${NC}"
#         continue
#     fi

#     echo -e "${GREEN}--------------------------------------------------------${NC}"
#     echo -e "${GREEN}Evaluating ${RUN_NAME} on ALL STS tasks...${NC}"
    
#     CKPT="${CKPT}" TASKS="all" MODEL_NAME="${RUN_NAME}_sts" bash scripts/eval_sts.sh
# done

# # 2. NLI Models -> Downstream Evaluation (Transfer)
echo ""
echo -e "${BOLD}${YELLOW}>>> [2/4] Evaluating NLI Models on Downstream Tasks (3 runs)${NC}"
for epochs in 1 2 5; do
    RUN_NAME="bert_nli_aoe_${epochs}ep"
    CKPT="${OUTPUT_ROOT}/${RUN_NAME}/ckpt"
    
    if [[ ! -d "${CKPT}" ]]; then
        echo -e "${YELLOW}[WARNING] Checkpoint not found: ${CKPT}. Skipping...${NC}"
        continue
    fi

    echo -e "${GREEN}--------------------------------------------------------${NC}"
    echo -e "${GREEN}Evaluating ${RUN_NAME} on Downstream tasks...${NC}"
    
    CKPT="${CKPT}" MODEL_NAME="${RUN_NAME}_downstream" bash scripts/eval_downstream.sh
done

# 3. STS-B Models -> STS-B Evaluation (In-domain)
# echo ""
# echo -e "${BOLD}${YELLOW}>>> [3/4] Evaluating STS-B Models on STS-B Task (3 runs)${NC}"
# for epochs in 5 10 20; do
#     RUN_NAME="bert_stsb_aoe_${epochs}ep"
#     CKPT="${OUTPUT_ROOT}/${RUN_NAME}/ckpt"

#     if [[ ! -d "${CKPT}" ]]; then
#         echo -e "${YELLOW}[WARNING] Checkpoint not found: ${CKPT}. Skipping...${NC}"
#         continue
#     fi

#     echo -e "${GREEN}--------------------------------------------------------${NC}"
#     echo -e "${GREEN}Evaluating ${RUN_NAME} on STSBenchmark...${NC}"
    
#     CKPT="${CKPT}" TASKS="STSBenchmark" MODEL_NAME="${RUN_NAME}_stsb" bash scripts/eval_sts.sh
# done

# 4. GIS Models -> GIS Evaluation (In-domain)
# echo ""
# echo -e "${BOLD}${YELLOW}>>> [4/4] Evaluating GIS Models on GIS Task (3 runs)${NC}"
# for epochs in 5 10 20; do
#     RUN_NAME="bert_gis_aoe_${epochs}ep"
#     CKPT="${OUTPUT_ROOT}/${RUN_NAME}/ckpt"

#     if [[ ! -d "${CKPT}" ]]; then
#         echo -e "${YELLOW}[WARNING] Checkpoint not found: ${CKPT}. Skipping...${NC}"
#         continue
#     fi

#     echo -e "${GREEN}--------------------------------------------------------${NC}"
#     echo -e "${GREEN}Evaluating ${RUN_NAME} on GIS...${NC}"
    
#     CKPT="${CKPT}" MODEL_NAME="${RUN_NAME}_gis" bash scripts/eval_gis.sh
# done

# echo ""
# echo -e "${BOLD}${BLUE}========================================================${NC}"
# echo -e "${BOLD}${BLUE}All evaluations completed.${NC}"
# echo -e "${BOLD}${BLUE}========================================================${NC}"
