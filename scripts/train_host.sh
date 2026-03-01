#!/bin/bash
# Script Huan Luyen AI truc tiep tren moi truong Host (De dung GPU NVIDIA)

# Mau sac ANSI
BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
BOLD='\033[1m'
NC='\033[0m'

clear
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}        NCKH SDN - HOST GPU TRAINING MODE                        ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Kiem tra va kich hoat moi truong ao (Venv)
if [ -d "venv" ]; then
    echo -e "${CYAN}Dang kich hoat moi truong ao (venv)...${NC}"
    source venv/bin/activate 2>/dev/null || source venv/bin/activate.fish 2>/dev/null || true
else
    echo -e "${YELLOW}Canh bao: Khong tim thay thu muc 'venv'.${NC}"
    echo -e "   Goi y: Chay 'python3 -m venv venv && source venv/bin/activate && pip install -r ai_model/requirements.txt' truoc."
fi

# Kiem tra Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Loi: Khong tim thay python3 tren may Host.${NC}"
    exit 1
fi

echo -e "${BOLD}Buoc 1:${NC} ${CYAN}Xu ly du lieu (Dung Python Host)...${NC}"
python3 ai_model/data_processor.py

echo -e "\n${BOLD}Buoc 2:${NC} ${CYAN}Tien hanh huan luyen (Se tu dong dung CUDA neu co)...${NC}"
python3 ai_model/train.py

echo -e "\n${GREEN}=================================================================${NC}"
echo -e "${GREEN} HOAN TAT! Bo nao AI da duoc cung co.                            ${NC}"
echo -e "${GREEN}=================================================================${NC}"
echo -e "${CYAN}Bieu do Training:  ai_model/processed_data/charts/${NC}"
echo -e "${CYAN}Model checkpoint:  ai_model/checkpoints/tft_dqn_master.pth${NC}"
echo -e "${CYAN}Raw metrics:       ai_model/processed_data/training_metrics.json${NC}"
echo -e "\n${YELLOW}Goi y: Bay gio chay evaluate_sdn.sh va chon AI mode de so sanh.${NC}\n"
