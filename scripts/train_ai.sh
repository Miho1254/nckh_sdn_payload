#!/bin/bash
# ================================================================
# NCKH SDN - TRAINING MODULE (TFT-DQN)
# Merge data + Process + Train trong 1 lệnh duy nhất
# Cách dùng: bash scripts/train_ai.sh
# ================================================================

BLUE='\033[1;34m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}        NCKH SDN - TRAINING MODULE (TFT-DQN)                     ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Bước 0: Merge data từ COLLECT results (nếu có)
COLLECT_DIRS=$(ls -d stats/results/COLLECT_*/flow_stats.csv 2>/dev/null)
if [ -n "$COLLECT_DIRS" ]; then
    echo -e "${BOLD}Buoc 0:${NC} ${YELLOW}Merge data tu 4 kich ban COLLECT...${NC}"
    
    docker exec nckh-sdn-mininet bash -c '
        cd /work
        FIRST=true
        for s in flash_crowd predictable_ramping targeted_congestion gradual_shift; do
            F="stats/results/COLLECT_${s}/flow_stats.csv"
            P="stats/results/COLLECT_${s}/port_stats.csv"
            if [ -f "$F" ]; then
                if [ "$FIRST" = true ]; then
                    head -1 "$F" > stats/flow_stats.csv
                    [ -f "$P" ] && head -1 "$P" > stats/port_stats.csv
                    FIRST=false
                fi
                tail -n +2 "$F" >> stats/flow_stats.csv
                echo "" >> stats/flow_stats.csv
                if [ -f "$P" ]; then
                    tail -n +2 "$P" >> stats/port_stats.csv
                    echo "" >> stats/port_stats.csv
                fi
                echo "  + Merged: $s"
            fi
        done
        echo "  Flow stats: $(wc -l < stats/flow_stats.csv) rows"
        echo "  Port stats: $(wc -l < stats/port_stats.csv) rows"
    '
    echo ""
else
    echo -e "${CYAN}Khong tim thay COLLECT data, dung file hien tai.${NC}"
fi

# Bước 1: Xử lý dữ liệu
echo -e "${BOLD}Buoc 1:${NC} ${CYAN}Xu ly du lieu (data_processor.py)...${NC}"
docker exec nckh-sdn-mininet bash -c "PYTHONUNBUFFERED=1 python3 /work/ai_model/data_processor.py"
if [ $? -ne 0 ]; then
    echo -e "${RED}[LOI] data_processor.py that bai!${NC}"
    exit 1
fi

# Bước 2: Train model
echo -e "\n${BOLD}Buoc 2:${NC} ${CYAN}Huan luyen mo hinh (train.py)...${NC}"
docker exec -t nckh-sdn-mininet bash -c "PYTHONUNBUFFERED=1 python3 /work/ai_model/train.py"
if [ $? -ne 0 ]; then
    echo -e "${RED}[LOI] train.py that bai!${NC}"
    exit 1
fi

echo -e "\n${GREEN}=================================================================${NC}"
echo -e "${GREEN} HOAN TAT! Mo hinh + Bieu do da duoc luu.                        ${NC}"
echo -e "${GREEN}=================================================================${NC}"
echo -e "${CYAN}Bieu do Training:  ai_model/processed_data/charts/${NC}"
echo -e "${CYAN}Model checkpoint:  ai_model/checkpoints/tft_dqn_master.pth${NC}"
echo ""
