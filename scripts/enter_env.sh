#!/bin/bash
# Script hỗ trợ mở môi trường Interactive Development (Vào bash của Docker)

# Màu sắc ANSI
BLUE='\033[1;34m'
CYAN='\033[1;36m'
NC='\033[0m'

clear
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}        🔌 ĐANG KẾT NỐI VÀO KHÔNG GIAN ẢO MININET...             ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${CYAN}Gợi ý: Gõ 'exit' để quay lại máy Host.${NC}\n"

docker exec -it nckh-sdn-mininet /bin/bash
