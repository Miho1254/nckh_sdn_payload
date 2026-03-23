#!/bin/bash
# ================================================================
# NCKH SDN - NON-STOP EXPERIMENT (FOR INSIDE CONTAINER)
# Chạy trực tiếp bên trong Docker container
# ================================================================
# V14 Config:
#   - ENTROPY_COEFF: 0.5
#   - KL_COEFF: 0.01
#   - Training: 100 epochs, 3-phase
#   - 9000 samples (3000 per action)
# ================================================================

BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
NC='\033[0m'

SYNTHETIC_FLAG="--synthetic"
SKIP_FLAG=""

for arg in "$@"; do
    case "$arg" in
        --skip-collect) SKIP_FLAG="--skip-collect" ;;
    esac
done

clear
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}   NCKH SDN - V14 THE ULTIMATE EQUILIBRIUM (INSIDE CONTAINER)    ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${CYAN}V14 Config: ENTROPY=0.5, KL=0.01, epochs=100${NC}"
echo -e "${YELLOW}Dự kiến thời gian: ~3-4 giờ${NC}"
echo ""

# Dọn dẹp
echo -e "${YELLOW}Đang dọn dẹp kết quả cũ...${NC}"
rm -rf /work/stats/results/*
rm -rf /work/ai_model/processed_data/charts/*
rm -f /work/stats/flow_stats.csv /work/stats/port_stats.csv
mkdir -p /work/stats/results/charts /work/ai_model/processed_data/charts /work/ai_model/training_logs

echo -e "${GREEN}Bắt đầu sau 5s...${NC}"
sleep 5

# ================================================================
# GIAI ĐOẠN 0: SYNTHETIC DATA
# ================================================================
echo -e "${CYAN}>>> GIAI ĐOẠN 0: TẠO SYNTHETIC DATA <<<${NC}"

echo -e "${YELLOW}Tạo synthetic data V14: 9000 samples (3000 x 3 actions)${NC}"

cd /work
python3 scripts/generate_synthetic_data_v3.py \
    --samples 3000 \
    --output ai_model/processed_data

if [ $? -ne 0 ]; then
    echo -e "${RED}Thất bại khi tạo synthetic data${NC}"
    exit 1
fi

# Copy vào vị trí chính
cp ai_model/processed_data/X_v3.npy ai_model/processed_data/X_v3_synthetic.npy 2>/dev/null || true
cp ai_model/processed_data/y_v3.npy ai_model/processed_data/y_v3_synthetic.npy 2>/dev/null || true

echo -e "${GREEN}Synthetic data: 9000 samples hoàn tất${NC}"

# ================================================================
# GIAI ĐOẠN 2: TRAINING TFT-CQL ACTOR-CRITIC
# ================================================================
echo -e "${CYAN}>>> GIAI ĐOẠN 2: TRAINING TFT-CQL <<<${NC}"

echo "=== TRAINING V14 (3-phase, 100 epochs) ==="
PYTHONUNBUFFERED=1 python3 ai_model/train_actor_critic.py \
    --phase all \
    --epochs 100 \
    --batch_size 64 \
    --hidden_size 64

if [ $? -ne 0 ]; then
    echo -e "${RED}Cảnh báo: Training thất bại${NC}"
else
    echo -e "${GREEN}Training hoàn tất${NC}"
fi

# ================================================================
# GIAI ĐOẠN 7: BENCHMARK VỚI METRIC MỚI
# ================================================================
echo -e "${CYAN}>>> GIAI ĐOẠN 7: BENCHMARK VỚI METRIC MỚI <<<${NC}"

python3 scripts/quick_benchmark_new_scenarios.py --scenario all --compare BOTH

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Benchmark hoàn tất${NC}"
else
    echo -e "${RED}Cảnh báo: Benchmark thất bại${NC}"
fi

# ================================================================
# REPORT
# ================================================================
echo -e "${YELLOW}Đang tổng hợp báo cáo...${NC}"
python3 scripts/analyze_stats.py

echo -e "\n${GREEN}=================================================================${NC}"
echo -e "${GREEN} TẤT CẢ THÍ NGHIỆM HOÀN TẤT!                               ${NC}"
echo -e "${GREEN}=================================================================${NC}"
echo -e "${CYAN}Biểu đồ: stats/results/charts/${NC}"
echo -e "${CYAN}Training logs: ai_model/training_logs/${NC}"
echo -e "${CYAN}Benchmark: stats/benchmark_final/${NC}"
