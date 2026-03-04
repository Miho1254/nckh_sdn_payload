#!/bin/bash
# ================================================================
# NCKH SDN - NON-STOP EXPERIMENT (ULTIMATE RUN)
# Chạy toàn bộ từ Thu thập -> Train -> So sánh 3 thuật toán
# Phù hợp khi cần treo máy để lấy kết quả cuối cùng.
#
# FIX ĐÃ ÁP DỤNG:
# - controller_stats.py: idle_timeout=0 cho flow NAT (priority=100)
#   (trước đây idle_timeout=30 khiến flow bị xóa trước khi stats collection)
# - analyze_stats.py: Thêm trường duration_nsec vào rows dictionary
# ================================================================

# Màu sắc ANSI
BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m'

SKIP_FLAG=""
if [ "$1" == "--skip-collect" ]; then
    SKIP_FLAG="--skip-collect"
    echo -e "${YELLOW}>>> SKIP MODE: Bỏ qua thu thập data, nhảy thẳng tới Training <<<${NC}"
fi

clear
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}        NCKH SDN - ULTIMATE NON-STOP EXPERIMENT                  ${NC}"
echo -e "${BLUE}=================================================================${NC}"
if [ -z "$SKIP_FLAG" ]; then
    echo -e "${YELLOW}Dự kiến thời gian chạy: ~4 giờ (4 giai đoạn x 1 giờ)${NC}"
else
    echo -e "${YELLOW}Dự kiến thời gian chạy: ~3 giờ (skip data collection)${NC}"
fi
echo -e "${RED}Vui lòng đảm bảo máy không vào chế độ Sleep/Hibernate.${NC}"
echo ""

# DỌN DẸP TOÀN CỤC TRƯỚC KHI CHẠY (Global Cleanup)
if [ -z "$SKIP_FLAG" ]; then
    echo -e "${YELLOW}Đang dọn dẹp các kết quả và biểu đồ cũ...${NC}"
    docker exec nckh-sdn-mininet bash -c "rm -rf /work/stats/results/*"
    docker exec nckh-sdn-mininet bash -c "rm -rf /work/ai_model/processed_data/charts/*"
    docker exec nckh-sdn-mininet bash -c "rm -f /work/stats/flow_stats.csv /work/stats/port_stats.csv /work/stats/flow_stats_merged.csv"
    rm -f stats/results/final_report.txt
    mkdir -p stats/results/charts
    mkdir -p ai_model/processed_data/charts
else
    echo -e "${YELLOW}Bỏ qua dọn dẹp (giữ lại data cũ để retrain)${NC}"
    # Chỉ xóa model cũ và charts
    docker exec nckh-sdn-mininet bash -c "rm -rf /work/ai_model/processed_data/charts/*"
    docker exec nckh-sdn-mininet bash -c "rm -f /work/ai_model/checkpoints/tft_dqn_master.pth"
    rm -f stats/results/final_report.txt
    mkdir -p stats/results/charts
    mkdir -p ai_model/processed_data/charts
fi

# Đếm ngược khởi động
for i in {5..1}; do echo -ne "${YELLOW}Bắt đầu sau $i s... ${NC}"; sleep 1; done
echo -e "${GREEN}GO!${NC}\n"

# GIAI ĐOẠN 1: COLLECT DATA & TRAIN AI
echo -e "${BLUE}>>> GIAI ĐOẠN 1: THU THẬP DỮ LIỆU & HUẤN LUYỆN AI <<<${NC}"
if [ "$SKIP_FLAG" == "--skip-collect" ]; then
    echo -e "${YELLOW}Skip collect → Merge data cũ + Retrain trực tiếp trong Docker...${NC}"
    docker exec nckh-sdn-mininet bash -c '
        cd /work
        echo "=== MERGE COLLECT DATA ==="
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
                [ -f "$P" ] && tail -n +2 "$P" >> stats/port_stats.csv
                echo "  + Merged: $s"
            fi
        done
        echo "  Flow stats: $(wc -l < stats/flow_stats.csv) rows"
        [ -f stats/port_stats.csv ] && echo "  Port stats: $(wc -l < stats/port_stats.csv) rows"
        echo ""
        echo "=== DATA PROCESSING ==="
        PYTHONUNBUFFERED=1 python3 ai_model/data_processor.py
        echo ""
        echo "=== TRAINING ==="
        PYTHONUNBUFFERED=1 python3 ai_model/train.py
    '
    if [ $? -ne 0 ]; then echo -e "${RED}Thất bại ở Giai đoạn 1 (Retrain)${NC}"; exit 1; fi
else
    ./scripts/full_pipeline.sh COLLECT
    if [ $? -ne 0 ]; then echo -e "${RED}Thất bại ở Giai đoạn 1${NC}"; exit 1; fi
fi

# GIAI ĐOẠN 2: CHẠY ROUND ROBIN (BASELINE)
echo -e "${BLUE}>>> GIAI ĐOẠN 2: CHẠY ROUND ROBIN <<<${NC}"
./scripts/full_pipeline.sh RR
if [ $? -ne 0 ]; then echo -e "${RED}Thất bại ở Giai đoạn 2${NC}"; exit 1; fi

# GIAI ĐOẠN 3: CHẠY WEIGHTED ROUND ROBIN
echo -e "${BLUE}>>> GIAI ĐOẠN 3: CHẠY WEIGHTED ROUND ROBIN <<<${NC}"
./scripts/full_pipeline.sh WRR
if [ $? -ne 0 ]; then echo -e "${RED}Thất bại ở Giai đoạn 3${NC}"; exit 1; fi

# GIAI ĐOẠN 4: CHẠY AI INFERENCE & XUẤT BIỂU ĐỒ SO SÁNH
echo -e "${BLUE}>>> GIAI ĐOẠN 4: CHẠY AI INFERENCE & SO SÁNH CUỐI CÙNG <<<${NC}"
./scripts/full_pipeline.sh AI
if [ $? -ne 0 ]; then echo -e "${RED}Thất bại ở Giai đoạn 4${NC}"; exit 1; fi

echo -e "${YELLOW}Đang tổng hợp báo cáo sau cùng cho toàn bộ kịch bản...${NC}"
python3 scripts/analyze_stats.py

echo -e "\n${GREEN}=================================================================${NC}"
echo -e "${GREEN} 🔥 TẤT CẢ THÍ NGHIỆM ĐÃ HOÀN TẤT NON-STOP! 🔥                    ${NC}"
echo -e "${GREEN}=================================================================${NC}"
echo -e "${CYAN}Kết quả biểu đồ so sánh: stats/results/charts/${NC}"
echo -e "${CYAN}Kết quả training:       ai_model/processed_data/charts/${NC}"
echo -e "${YELLOW}Bạn có thể xem báo cáo khoa học ngay bây giờ.${NC}\n"
