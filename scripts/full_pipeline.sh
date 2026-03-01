#!/bin/bash
# ================================================================
# full_pipeline.sh - Chay toan bo quy trinh: Thu thap -> Train -> Bieu do
# Su dung: ./scripts/full_pipeline.sh [algo]
#   algo: COLLECT (mac dinh) | RR | WRR | AI
# ================================================================

BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
BOLD='\033[1m'
NC='\033[0m'

# Thuat toan co the truyen vao, mac dinh la COLLECT
ALGO="${1:-COLLECT}"

SCENARIOS=("flash_crowd" "predictable_ramping" "targeted_congestion" "gradual_shift")
SCENARIO_FILES=("flash_crowd.yml" "predictable_ramping.yml" "targeted_congestion.yml" "gradual_shift.yml")

clear
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}    NCKH SDN - FULL PIPELINE: THU THAP 4 KICH BAN + TRAIN AI      ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "Che do: ${CYAN}${ALGO}${NC} | So kich ban: ${YELLOW}${#SCENARIOS[@]}${NC}"
echo -e "${YELLOW}Thoi gian uoc tinh: ~${#SCENARIOS[@]} x 15 phut = ~$((${#SCENARIOS[@]} * 15)) phut${NC}"
echo ""
echo -e "${RED}CANH BAO: Qua trinh nay se mat khoang 1 gio, dung tat script giua chung!${NC}"
echo ""
echo -ne "${YELLOW}Nhan Enter de bat dau, Ctrl+C de huy: ${NC}"
read -r _confirm

# ----------------------------------------------------------------
# BUOC 0: Xoa CSV cu, gop lai tu dau sach
# ----------------------------------------------------------------
echo -e "\n${CYAN}=== BUOC 0: Chuan bi moi truong sach ===${NC}"
docker exec nckh-sdn-mininet bash -c "rm -f /work/stats/flow_stats.csv /work/stats/port_stats.csv"
sudo chown -R "$USER:$USER" stats/ 2>/dev/null || true
mkdir -p stats/results
echo -e "${GREEN}Sach!${NC}"

# ----------------------------------------------------------------
# BUOC 1: Lap qua 4 kich ban
# ----------------------------------------------------------------
TOTAL=${#SCENARIOS[@]}
SUCCESS_COUNT=0

for i in "${!SCENARIOS[@]}"; do
    SCENE="${SCENARIOS[$i]}"
    SCENE_FILE="${SCENARIO_FILES[$i]}"
    STEP=$((i + 1))
    RESULT_DIR="stats/results/${ALGO}_${SCENE}"

    echo ""
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${BOLD}[${STEP}/${TOTAL}] Kich ban: ${CYAN}${SCENE_FILE}${NC} | Thuat toan: ${GREEN}${ALGO}${NC}"
    echo -e "${BLUE}=================================================================${NC}"

    # Xoa CSV cu trong container truoc moi kich ban
    docker exec nckh-sdn-mininet bash -c "rm -f /work/stats/flow_stats.csv /work/stats/port_stats.csv"

    # Don dep Mininet + Ryu
    docker exec nckh-sdn-mininet mn -c > /dev/null 2>&1
    docker exec nckh-sdn-mininet pkill -9 -f ryu-manager > /dev/null 2>&1 || true
    
    echo -e "${CYAN}Khoi dong Ryu Controller...${NC}"
    docker exec -d nckh-sdn-mininet bash -c "ryu-manager /work/controller_stats.py --log-file /tmp/ryu.log"
    sleep 3

    echo -e "${CYAN}Chay Mininet cho kich ban nay (tu dong doi Artillery xong moi thoat)...${NC}"
    docker exec -it nckh-sdn-mininet bash -c "cd /work && SCENARIO=${SCENE_FILE} LB_ALGO=${ALGO} python3 run_lms_mininet.py"

    # Sao luu ket qua
    mkdir -p "$RESULT_DIR"
    cp stats/flow_stats.csv "$RESULT_DIR/" 2>/dev/null
    cp stats/port_stats.csv "$RESULT_DIR/" 2>/dev/null
    echo "{\"algo\": \"${ALGO}\", \"scenario\": \"${SCENE_FILE}\", \"timestamp\": \"$(date -Iseconds)\"}" > "$RESULT_DIR/metadata.json"

    # Gop CSV vao file merged
    if [ -f "$RESULT_DIR/flow_stats.csv" ]; then
        LINE_COUNT=$(wc -l < "$RESULT_DIR/flow_stats.csv")
        echo -e "${GREEN}Da sao luu ket qua (${LINE_COUNT} dong) vao ${RESULT_DIR}/${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}Canh bao: Khong tim thay flow_stats.csv cho kich ban nay!${NC}"
    fi
done

# ----------------------------------------------------------------
# BUOC 2: Gop tat ca CSV cho Mode COLLECT
# ----------------------------------------------------------------
if [ "$ALGO" == "COLLECT" ]; then
    echo ""
    echo -e "${BLUE}=== BUOC 2: GOP TAT CA DU LIEU (${SUCCESS_COUNT} kich ban) ===${NC}"
    
    MERGED_CSV="stats/flow_stats_merged.csv"
    > "$MERGED_CSV"  # Xoa file cu neu co
    
    FIRST=true
    for scene in "${SCENARIOS[@]}"; do
        SRC="stats/results/${ALGO}_${scene}/flow_stats.csv"
        if [ -f "$SRC" ]; then
            if [ "$FIRST" = true ]; then
                cat "$SRC" >> "$MERGED_CSV"
                FIRST=false
            else
                # Bo qua header (dong dau) khi append
                tail -n +2 "$SRC" >> "$MERGED_CSV"
            fi
            echo -e "  ${GREEN}Da gop: ${SRC}${NC}"
        fi
    done
    
    TOTAL_LINES=$(wc -l < "$MERGED_CSV")
    echo -e "${GREEN}File merged: ${MERGED_CSV} (${TOTAL_LINES} dong)${NC}"
    
    # Copy merged file vao stats/ de data_processor.py doc
    cp "$MERGED_CSV" "stats/flow_stats.csv"
    docker cp "stats/flow_stats.csv" nckh-sdn-mininet:/work/stats/flow_stats.csv 2>/dev/null || true

    # ----------------------------------------------------------------
    # BUOC 3: Train AI
    # ----------------------------------------------------------------
    echo ""
    echo -e "${BLUE}=== BUOC 3: TIEN HANH TRAIN AI (TFT-DQN) ===${NC}"
    
    # Kiem tra xem dung venv hay docker
    if [ -d "venv" ]; then
        echo -e "${CYAN}Dung venv (Host GPU Mode)...${NC}"
        source venv/bin/activate 2>/dev/null || source venv/bin/activate.fish 2>/dev/null || true
        python3 ai_model/data_processor.py && python3 ai_model/train.py
    else
        echo -e "${CYAN}Dung Docker (CPU Mode)...${NC}"
        docker exec -it nckh-sdn-mininet bash -c "cd /work && python3 ai_model/data_processor.py && python3 ai_model/train.py"
    fi

else
    # Voi RR/WRR/AI: Sau khi co du data, tu dong tao bieu do so sanh
    echo ""
    echo -e "${BLUE}=== BUOC 2: TU DONG TAO BIEU DO SO SANH ===${NC}"
    
    for scene in "${SCENARIOS[@]}"; do
        ALGO_COUNT=$(ls -d stats/results/*_"${scene}" 2>/dev/null | wc -l)
        if [ "$ALGO_COUNT" -ge 2 ]; then
            echo -e "${CYAN}Dang ve bieu do so sanh cho kich ban: ${scene}...${NC}"
            source venv/bin/activate 2>/dev/null || true
            python3 ai_model/generate_comparison_charts.py --scenario "$scene" 2>&1
        fi
    done
fi

# ----------------------------------------------------------------
# HOAN TAT
# ----------------------------------------------------------------
echo ""
echo -e "${GREEN}=================================================================${NC}"
echo -e "${GREEN}  FULL PIPELINE HOAN TAT!                                        ${NC}"
echo -e "${GREEN}=================================================================${NC}"

if [ "$ALGO" == "COLLECT" ]; then
    echo -e "${CYAN}Bieu do Training:    ai_model/processed_data/charts/${NC}"
    echo -e "${CYAN}Model checkpoint:    ai_model/checkpoints/tft_dqn_master.pth${NC}"
    echo -e "${CYAN}Du lieu gop:         stats/flow_stats_merged.csv${NC}"
    echo ""
    echo -e "${YELLOW}Buoc tiep theo: Chay ./scripts/full_pipeline.sh RR${NC}"
    echo -e "${YELLOW}             sau do  ./scripts/full_pipeline.sh WRR${NC}"
    echo -e "${YELLOW}             sau do  ./scripts/full_pipeline.sh AI${NC}"
    echo -e "${YELLOW}De co bieu do so sanh AI vs RR vs WRR!${NC}"
else
    echo -e "${CYAN}Ket qua tung kich ban: stats/results/${NC}"
    echo -e "${CYAN}Bieu do so sanh:       stats/results/charts/${NC}"
fi
echo ""
