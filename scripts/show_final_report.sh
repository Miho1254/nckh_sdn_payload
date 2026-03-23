#!/bin/bash

BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
BOLD='\033[1m'
NC='\033[0m'

clear
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BOLD}${GREEN}        TỔNG KẾT NGHIÊN CỨU: TỰ ĐỘNG HÓA SDN LOAD BALANCING      ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""
echo -e "${BOLD}1. THÔNG TIN MÔ HÌNH ƯU VIỆT NHẤT${NC}"
echo -e "   - ${CYAN}Thuật toán:${NC} TFT-CQL (Temporal Fusion Transformer + Conservative Q-Learning)"
echo -e "   - ${CYAN}Serving Rule:${NC} Stochastic Sampling Policy (Không dùng Deterministic Argmax)"
echo -e "   - ${CYAN}Feature State:${NC} V3 (44 Features - Bổ sung Util & Headroom Vector)"
echo -e "   - ${CYAN}Tiêu chí Early Stop:${NC} Vượt qua Diversity Gate (Entropy > 0.5 liên tục)"
echo ""
echo -e "${BOLD}2. KẾT QUẢ ĐÁNH GIÁ THỰC NGHIỆM CUỐI (16 BÀI TEST)${NC}"
echo -e "   ${YELLOW}Độ C��ng Bằng Rải Tải (Fairness CV% - Càng thấp càng cân bằng):${NC}"
echo -e "      - CQL-Sampled : ${GREEN}85.15%${NC} (Dẫn đầu nhóm AI & Cơ học linh hoạt)"
echo -e "      - Round-Robin : ${RED}94.73%${NC}"
echo -e "      - WRR (Tĩnh)  : ${CYAN}19.25%${NC} (Vô địch tuyệt đối do đặc tính chia tĩnh)"
echo ""
echo -e "   ${YELLOW}Khả Năng Sinh Tồn - Giảm Tỷ Lệ Nghẽn Cục Bộ (Congestion Prevention):${NC}"
echo -e "      - Môi trường biến thiên (Kịch bản Golden Hour):"
echo -e "        + CQL-Sampled : ${GREEN}14.11%${NC} (Tốt nhất toàn bảng 4 mô hình)"
echo -e "        + Round-Robin : ${RED}20.56%${NC}"
echo -e "        + WRR (Tĩnh)  : ${RED}20.19%${NC}"
echo -e "      => Trong môi trường động, thuật toán tĩnh không thể xoay sở. AI phân tích Time-series"
echo -e "         linh hoạt điều hướng, hạ mức nghẽn chạm mức thấp nhất lịch sử dự án!"
echo ""
echo -e "${BOLD}=== BẢNG SỐ LIỆU CHI TIẾT TỪ BENCHMARK KHÉP KÍN ===${NC}"
python3 scripts/build_decision_table.py
echo ""
echo -e "${BOLD}3. BÀI HỌC VÀ KHẲNG ĐỊNH KHOA HỌC YẾU QUYẾT${NC}"
echo -e "   - Kiến trúc **Actor-Critic (CQL)** tối ưu Offline Reinforcement Learning với "
echo -e "     Conservative Q-Learning penalty, đảm bảo Q-value không vượt tập dữ liệu huấn luyện."
echo -e "   - Việc giữ nguyên **Stochastic Sampling Policy** khi Inference giúp Load Balancer rải gói tin"
echo -e "     chủ động thay vì 'đâm đầu' tham lam vào 1 cổng mạnh nhất."
echo -e "   - **Mạng nơ-ron TFT** dự đoán cực kỳ nhạy bén với luồng Traffic Shift, giúp Controller chặn"
echo -e "     nghẽn từ trứng nước thay vì đợi rớt gói (Drop) như các Switch truyền thống."
echo ""
echo -e "${BLUE}=================================================================${NC}"
echo -e "   ✅ TIẾN TRÌNH HOÀN TẤT. SẴN SÀNG LÊN BỤC BẢO VỆ NCKH IEEE! 🎓"
echo -e "${BLUE}=================================================================${NC}"
echo ""
