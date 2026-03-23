#!/bin/bash
# Script Đánh giá khả năng chịu tải của hệ thống (Optimized UI + Auto-Save Results)

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
echo -e "${BLUE}        NCKH SDN - HE THONG DANH GIA TAI (LMS)                   ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""

echo -e "${BOLD}Danh sach Kich ban (Scenarios):${NC}"
echo -e "  1) ${CYAN}golden_hour.yml${NC}       (Gio cao diem 1000+ users)"
echo -e "  2) ${CYAN}video_conference.yml${NC}  (Tai video on dinh dai)"
echo -e "  3) ${CYAN}hardware_degradation.yml${NC} (Server suy giam tu tu)"
echo -e "  4) ${CYAN}low_rate_dos.yml${NC}     (Tan cong toc do thap)"
echo -e "  5) ${CYAN}burst_traffic.yml${NC}    (NEW - Traffic dot ngot tang 10x)"
echo -e "  6) ${CYAN}server_failure.yml${NC}   (NEW - Server mat capacity dot ngot)"
echo ""
if [ -n "$2" ]; then
    input_scene=$2
    echo -e "${YELLOW}Auto-selected Scenario: ${CYAN}$input_scene${NC}"
else
    echo -ne "${YELLOW}Nhap so (1-6) hoac ten file YAML (Mac dinh: 1): ${NC}"
    read input_scene
fi

case $input_scene in
    1| "golden_hour") SCENARIO="golden_hour.yml" ;;
    2| "video_conference") SCENARIO="video_conference.yml" ;;
    3| "hardware_degradation") SCENARIO="hardware_degradation.yml" ;;
    4| "low_rate_dos"| "") SCENARIO="low_rate_dos.yml" ;;
    5| "burst_traffic") SCENARIO="burst_traffic.yml" ;;
    6| "server_failure") SCENARIO="server_failure.yml" ;;
    *) SCENARIO="$input_scene" ;;
esac

# Lay ten kich ban khong co .yml
SCENE_NAME=$(echo "$SCENARIO" | sed 's/.yml//')

echo -e "\n${BLUE}-----------------------------------------------------------------${NC}"
echo -e "${BOLD}Chon Thuat toan Load Balancing:${NC}"
echo -e "  1) ${GREEN}RR${NC}       (Round Robin - Xoay vong don gian)"
echo -e "  2) ${GREEN}WRR${NC}      (Weighted Round Robin - Trong so 1:2:3)"
echo -e "  3) ${CYAN}AI${NC}       (TFT-CQL Actor-Critic Inference Mode)"
echo -e "  4) ${YELLOW}COLLECT${NC}  (Thu thap du lieu cho Training)"
echo ""
if [ -n "$1" ]; then
    algo_choice=$1
    echo -e "${YELLOW}Auto-selected Algorithm: ${GREEN}$algo_choice${NC}"
else
    echo -ne "${YELLOW}Nhap lua chon (1-4): ${NC}"
    read algo_choice
fi

case $algo_choice in
    1| "RR") ALGO="RR" ;;
    2| "WRR") ALGO="WRR" ;;
    3| "AI") ALGO="AI" ;;
    4| "COLLECT") ALGO="COLLECT" ;;
    *) ALGO="RR" ;;
esac

# CANH BAO THOI GIAN
if [[ "$ALGO" == "COLLECT" || "$ALGO" == "AI" ]]; then
    echo -e "\n${RED}QUAN TRONG:${NC} Che do ${BOLD}$ALGO${NC} can it nhat ${YELLOW}10-15 phut${NC} de thu thap du du lieu."
    echo -e "   Vui long khong tat script som de dam bao chat luong Dataset/Ket qua."
fi

echo -e "\n${BLUE}-----------------------------------------------------------------${NC}"
echo -e "Khoi dong: ${CYAN}$SCENARIO${NC} | ${GREEN}$ALGO${NC}"
echo -e "${BLUE}-----------------------------------------------------------------${NC}\n"

# ================================================================
# CHUAN BI THU MUC KET QUA RIENG
# ================================================================
RESULT_DIR="stats/results/${ALGO}_${SCENE_NAME}"
echo -e "${CYAN}Don dep & Tao thu muc ket qua: ${RESULT_DIR}${NC}"
docker exec nckh-sdn-mininet rm -rf "/work/${RESULT_DIR}"
mkdir -p "$RESULT_DIR"

# Xoa CSV cu de bat dau sach (Host & Container)
docker exec nckh-sdn-mininet bash -c "rm -f /work/stats/flow_stats.csv /work/stats/port_stats.csv /work/stats/inference_log.csv"

echo -e "${CYAN}Don dep Mininet...${NC}"
docker exec nckh-sdn-mininet mn -c > /dev/null 2>&1

echo -e "${CYAN}Kiem tra OVS...${NC}"
docker exec nckh-sdn-mininet ovs-vswitchd --pidfile --detach --log-file 2>/dev/null || true

echo -e "${CYAN}Khoi dong Ryu Controller...${NC}"
docker exec nckh-sdn-mininet pkill -9 -f ryu-manager > /dev/null 2>&1 || true
sleep 1  # Đảm bảo process đã chết hoàn toàn
docker exec -d nckh-sdn-mininet bash -c "export LB_ALGO=$ALGO && ryu-manager /work/controller_stats.py --log-file /tmp/ryu.log"

# Cho Ryu khoi tao và kiểm tra log
sleep 5
echo -e "${YELLOW}Kiểm tra controller strategy...${NC}"
docker exec nckh-sdn-mininet bash -c "timeout 5 tail -f /tmp/ryu.log | grep -E 'RIU Load Balancer|Strategy' | head -1 || echo 'Không thể đọc log'"

echo -e "${CYAN}Thiet lap mang...${NC}"
echo -e "${GREEN}=================================================================${NC}"
# Chay Mininet
docker exec nckh-sdn-mininet bash -c "cd /work && SCENARIO=$SCENARIO LB_ALGO=$ALGO python3 run_lms_mininet.py"

# ================================================================
# SAU KHI MININET THOAT -> SAO LUU KET QUA
# ================================================================
echo -e "\n${BLUE}-----------------------------------------------------------------${NC}"
echo -e "${CYAN}Dang sao luu ket qua vao ${RESULT_DIR}/ ...${NC}"

# Copy CSV ket qua (Tu file Controller ghi ra)
cp stats/flow_stats.csv "$RESULT_DIR/" 2>/dev/null && echo -e "  ${GREEN}flow_stats.csv${NC}" || echo -e "  ${RED}flow_stats.csv (khong tim thay)${NC}"
cp stats/port_stats.csv "$RESULT_DIR/" 2>/dev/null && echo -ne ""
cp stats/inference_log.csv "$RESULT_DIR/" 2>/dev/null && echo -ne ""

# Ghi metadata
echo "{\"algo\": \"$ALGO\", \"scenario\": \"$SCENARIO\", \"timestamp\": \"$(date -Iseconds)\"}" > "$RESULT_DIR/metadata.json"
echo -e "  ${GREEN}metadata.json${NC}"

echo -e "\n${CYAN}Tich hop phan tich so lieu Traffic truc tiep...${NC}"
if [ -d "venv" ]; then
    source venv/bin/activate.fish 2>/dev/null || source venv/bin/activate 2>/dev/null || true
fi
python3 scripts/analyze_stats.py --scenario "$SCENE_NAME"

echo -e "\n${GREEN}=================================================================${NC}"
echo -e "${GREEN} KET QUA DA LUU TAI: ${RESULT_DIR}/                               ${NC}"
echo -e "${GREEN}=================================================================${NC}"

# Tu dong ve bieu do so sanh neu co du 2 thuat toan tro len
ALGO_COUNT=$(ls -d stats/results/*_${SCENE_NAME} 2>/dev/null | wc -l)
if [ "$ALGO_COUNT" -ge 2 ]; then
    echo -e "\n${CYAN}Tim thay ${ALGO_COUNT} bo ket qua cho kich ban '${SCENE_NAME}'. Dang xuat bieu do so sanh...${NC}"
    
    # Kiem tra virtual env
    if [ -d "venv" ]; then
        source venv/bin/activate.fish 2>/dev/null || source venv/bin/activate 2>/dev/null || true
    fi
    
    python3 ai_model/generate_comparison_charts.py --scenario "$SCENE_NAME" 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Bieu do so sanh da xuat tai: stats/results/charts/${NC}"
    else
        echo -e "${YELLOW}Loi khi xuat bieu do. Ban co the chay thu cong:${NC}"
        echo -e "  python3 ai_model/generate_comparison_charts.py --scenario $SCENE_NAME"
    fi
else
    echo -e "\n${YELLOW}Goi y: Chay them cac thuat toan khac (RR, WRR, AI) de xuat bieu do so sanh tu dong.${NC}"
fi
echo ""
