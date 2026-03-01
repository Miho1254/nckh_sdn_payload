#!/bin/bash
# Script Huan Luyen AI tu du lieu mang (TFT-DQN) - Optimized UI

# Mau sac ANSI
BLUE='\033[1;34m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
BOLD='\033[1m'
NC='\033[0m'

clear
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}        NCKH SDN - TRAINING MODULE (TFT-DQN)                     ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""

echo -e "${BOLD}Buoc 1:${NC} ${CYAN}Dang xu ly du lieu tu flow_stats.csv...${NC}"
docker exec -it nckh-sdn-mininet python3 /work/ai_model/data_processor.py

echo -e "\n${BOLD}Buoc 2:${NC} ${CYAN}Tien hanh huan luyen mo hinh (PyTorch)...${NC}"
docker exec -it nckh-sdn-mininet python3 /work/ai_model/train.py

echo -e "\n${GREEN}=================================================================${NC}"
echo -e "${GREEN} HOAN TAT! Mo hinh + Bieu do da duoc luu.                        ${NC}"
echo -e "${GREEN}=================================================================${NC}"
echo -e "${CYAN}Bieu do Training:  ai_model/processed_data/charts/${NC}"
echo -e "${CYAN}Model checkpoint:  ai_model/checkpoints/tft_dqn_master.pth${NC}"
echo ""
