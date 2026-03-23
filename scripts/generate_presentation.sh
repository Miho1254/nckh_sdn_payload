#!/bin/bash
# =================================================================
# NCKH SDN - GENERATE ALL PRESENTATION ASSETS
# Script tổng hợp để tạo ra toàn bộ hình ảnh phục vụ thuyết trình/báo cáo.
# =================================================================

BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}       HỆ THỐNG TẠO ASSET THUYẾT TRÌNH TỰ ĐỘNG                   ${NC}"
echo -e "${BLUE}=================================================================${NC}"

# Chuyển về thư mục gốc của project
cd "$(dirname "$0")/.."

# 1. Vẽ Dashboard huấn luyện chuyên nghiệp (Professional Style)
echo -e "\n${YELLOW}[1/2] Đang tạo Dashboard huấn luyện AI chuyên nghiệp...${NC}"
python3 ai_model/replot_training.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Hoàn tất: ai_model/processed_data/charts_professional/${NC}"
else
    echo -e "${RED}✗ Lỗi khi chạy replot_training.py${NC}"
fi

# 2. Vẽ biểu đồ Thực nghiệm cuối (IEEE Edition)
echo -e "\n${YELLOW}[2/2] Đang kết xuất biểu đồ benchmark (Chuẩn IEEE)...${NC}"
python3 scripts/plot_ieee_benchmark.py
python3 scripts/plot_actual_dist.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Hoàn tất: stats/results/charts_presentation/${NC}"
else
    echo -e "${RED}✗ Lỗi khi chạy plot_ieee_benchmark.py${NC}"
fi

echo -e "\n${BLUE}=================================================================${NC}"
echo -e "${GREEN} 🔥 TẤT CẢ BIỂU ĐỒ ĐÃ SẴN SÀNG CHO BUỔI THUYẾT TRÌNH! 🔥          ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "Check folder: ${YELLOW}stats/results/charts_presentation/${NC}"
echo -e "Check folder: ${YELLOW}ai_model/processed_data/charts_professional/${NC}\n"
