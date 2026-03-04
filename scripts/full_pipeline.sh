#!/bin/bash
# ================================================================
# NCKH SDN - MASTER PIPELINE (AUTOMATED)
# Chạy toàn bộ quy trình thí nghiệm không dừng (Non-stop)
# ================================================================

# Màu sắc ANSI
BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
BOLD='\033[1m'
NC='\033[0m'

clear
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}        NCKH SDN - MASTER PIPELINE ORCHESTRATOR                  ${NC}"
echo -e "${BLUE}=================================================================${NC}"

if [ -z "$1" ]; then
    echo -e "${YELLOW}Cách dùng:${NC} ./scripts/full_pipeline.sh [ALGO]"
    echo -e "  [ALGO]: COLLECT | RR | WRR | AI"
    echo -e "\n${BOLD}Ví dụ:${NC}"
    echo -e "  Phase 1: ${CYAN}./scripts/full_pipeline.sh COLLECT${NC} (Chạy 4 kịch bản -> Train AI)"
    echo -e "  Phase 2: ${CYAN}./scripts/full_pipeline.sh RR${NC}      (Để so sánh)"
    exit 1
fi

ALGO=$1
SKIP_COLLECT=false
if [ "$2" == "--skip-collect" ]; then
    SKIP_COLLECT=true
fi
SCENARIOS=("flash_crowd" "predictable_ramping" "targeted_congestion" "gradual_shift")

echo -e "\n${BOLD}Bắt đầu quy trình thí nghiệm cho thuật toán: ${GREEN}$ALGO${NC}"
if [ "$SKIP_COLLECT" = true ]; then
    echo -e "${YELLOW}>>> SKIP MODE: Bỏ qua thu thập dữ liệu, nhảy thẳng tới Training <<<${NC}"
else
    echo -e "Các kịch bản sẽ chạy: ${CYAN}${SCENARIOS[*]}${NC}"
fi
echo -e "${BLUE}-----------------------------------------------------------------${NC}\n"

# Đếm ngược 3s
for i in {3..1}; do echo -ne "${YELLOW}$i... ${NC}"; sleep 1; done
echo -e "${GREEN}BẮT ĐẦU!${NC}\n"

if [ "$SKIP_COLLECT" = false ]; then
    # DỌN DẸP DỮ LIỆU CỦA THUẬT TOÁN NÀY (Algo Cleanup)
    echo -e "${YELLOW}Xóa dữ liệu cũ của $ALGO...${NC}"
    docker exec nckh-sdn-mininet bash -c "rm -rf /work/stats/results/${ALGO}_*"

    for scene in "${SCENARIOS[@]}"; do
        echo -e "${BLUE} >>> ĐANG CHẠY KỊCH BẢN: ${BOLD}$scene${NC} (${ALGO}) <<<${NC}"
        
        # Gọi evaluate_sdn.sh với tham số (Non-interactive)
        bash scripts/evaluate_sdn.sh "$ALGO" "$scene"
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}[LỖI] Kịch bản $scene thất bại. Dừng pipeline.${NC}"
            exit 1
        fi
        echo -e "${GREEN}[DONE] Hoàn tất $scene.${NC}\n"
        sleep 2
    done
else
    echo -e "${CYAN}Bỏ qua thu thập dữ liệu (--skip-collect)${NC}\n"
fi

# ================================================================
# XỬ LÝ ĐẶC BIỆT CHO MODE COLLECT (PHASE 1)
# ================================================================
if [ "$ALGO" == "COLLECT" ]; then
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${BOLD} PHASE 1: TỔNG HỢP DỮ LIỆU & HUẤN LUYỆN AI                      ${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    
    MERGED_CSV="stats/flow_stats_merged.csv"
    echo -e "${CYAN}Đang gộp dữ liệu từ 4 kịch bản vào $MERGED_CSV...${NC}"
    
    # Gộp CSV (giữ header dòng đầu, bỏ header các file sau)
    FIRST_FILE=true
    for scene in "${SCENARIOS[@]}"; do
        SRC="stats/results/COLLECT_${scene}/flow_stats.csv"
        if [ -f "$SRC" ]; then
            if [ "$FIRST_FILE" = true ]; then
                cat "$SRC" > "$MERGED_CSV"
                FIRST_FILE=false
            else
                tail -n +2 "$SRC" >> "$MERGED_CSV"
            fi
            echo -e "  + Đã gộp flow_stats: $scene"
        fi
    done
    
    # Gộp port_stats.csv (cho 5-feature pipeline)
    MERGED_PORT="stats/port_stats_merged.csv"
    FIRST_PORT=true
    for scene in "${SCENARIOS[@]}"; do
        SRC="stats/results/COLLECT_${scene}/port_stats.csv"
        if [ -f "$SRC" ]; then
            if [ "$FIRST_PORT" = true ]; then
                cat "$SRC" > "$MERGED_PORT"
                FIRST_PORT=false
            else
                tail -n +2 "$SRC" >> "$MERGED_PORT"
            fi
            echo -e "  + Đã gộp port_stats: $scene"
        fi
    done
    
    # Copy file gộp vào vị trí train AI mong đợi
    cp "$MERGED_CSV" "stats/flow_stats.csv"
    [ -f "$MERGED_PORT" ] && cp "$MERGED_PORT" "stats/port_stats.csv"
    
    echo -e "\n${YELLOW}Bắt đầu quá trình huấn luyện AI (scripts/train_ai.sh)...${NC}"
    bash scripts/train_ai.sh
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}[THÀNH CÔNG] AI đã được huấn luyện xong.${NC}"
        echo -e "${BOLD}Bây giờ bạn có thể chạy Phase 2:${NC}"
        echo -e "  ./scripts/full_pipeline.sh RR"
        echo -e "  ./scripts/full_pipeline.sh WRR"
        echo -e "  ./scripts/full_pipeline.sh AI"
    else
        echo -e "${RED}[LỖI] Huấn luyện AI thất bại.${NC}"
        exit 1
    fi
fi

# ================================================================
# XỬ LÝ CHO PHASE 2 (SO SÁNH)
# ================================================================
if [[ "$ALGO" == "AI" || "$ALGO" == "RR" || "$ALGO" == "WRR" ]]; then
    echo -e "${GREEN}=================================================================${NC}"
    echo -e "${BOLD} KẾT THÚC CHU KỲ CHẠY ${ALGO}                                   ${NC}"
    echo -e "${GREEN}=================================================================${NC}"
    echo -e "Kết quả đã được lưu tại: ${CYAN}stats/results/${ALGO}_*/${NC}"
    echo -e "Báo cáo traffic tự động đã hiển thị ở trên."
    
    # Nếu chạy xong AI mode, gợi ý xem chart tổng hợp
    if [ "$ALGO" == "AI" ]; then
        echo -e "\n${CYAN}Tất cả biểu đồ so sánh khoa học đã có tại: ${BOLD}stats/results/charts/${NC}"
    fi
fi

echo -e "\n${BLUE}Quy trình hoàn tất!${NC}\n"
