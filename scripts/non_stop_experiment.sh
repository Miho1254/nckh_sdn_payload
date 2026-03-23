#!/bin/bash
# ================================================================
# NCKH SDN - NON-STOP EXPERIMENT (CQL ONLY)
# V14 "THE ULTIMATE EQUILIBRIUM" - FINAL VERSION
# Chạy toàn bộ từ Thu thập -> Train CQL -> So sánh
# Chỉ hỗ trợ TFT-CQL Actor-Critic (đã loại bỏ DQN)
# ================================================================
# V14 Config:
#   - ENTROPY_COEFF: 0.5
#   - KL_COEFF: 0.01 (giảm bảo thủ)
#   - overload_penalty: 5.0 / 20.0 (đã scale)
#   - wastage_penalty: 0.015
#   - saving_bonus: h5=1.0, h7=0.3, h8=0.0
#   - Training: 100 epochs, 3-phase
# ================================================================
# Options:
#   --skip-collect    : Bỏ qua thu thập data, dùng data cũ
#   --synthetic       : Sử dụng synthetic data thay vì collect
#   --synthetic-only  : Chỉ tạo synthetic data, không train/benchmark
# ================================================================

# Màu sắc ANSI
BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
NC='\033[0m'

SKIP_FLAG=""
SYNTHETIC_FLAG=""
SYNTHETIC_ONLY=""

for arg in "$@"; do
    case "$arg" in
        --skip-collect) SKIP_FLAG="--skip-collect" ;;
        --synthetic) SYNTHETIC_FLAG="--synthetic" ;;
        --synthetic-only) SYNTHETIC_ONLY="--synthetic-only" ;;
    esac
done

if [ -n "$SYNTHETIC_ONLY" ]; then
    echo -e "${YELLOW}>>> SYNTHETIC ONLY MODE: Chỉ tạo synthetic data <<<${NC}"
fi

if [ -n "$SYNTHETIC_FLAG" ]; then
    echo -e "${YELLOW}>>> SYNTHETIC MODE: Sử dụng synthetic data <<<${NC}"
fi

if [ -n "$SKIP_FLAG" ]; then
    echo -e "${YELLOW}>>> SKIP MODE: Bỏ qua thu thập data <<<${NC}"
fi

clear
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}   NCKH SDN - V14 THE ULTIMATE EQUILIBRIUM                       ${NC}"
echo -e "${BLUE}   (TFT-CQL Actor-Critic Pipeline ONLY)                          ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${CYAN}V14 Config: ENTROPY=0.5, KL=0.01, epochs=100${NC}"

if [ -z "$SKIP_FLAG" ]; then
    echo -e "${YELLOW}Dự kiến thời gian: ~4-5 giờ (9000 samples + CQL + benchmark)${NC}"
else
    echo -e "${YELLOW}Dự kiến thời gian: ~3 giờ (skip data collection)${NC}"
fi
echo -e "${RED}Vui lòng đảm bảo máy không vào chế độ Sleep/Hibernate.${NC}"
echo ""

# ── DỌN DẸP ──
if [ -z "$SKIP_FLAG" ]; then
    echo -e "${YELLOW}Đang dọn dẹp kết quả và biểu đồ cũ...${NC}"
    docker exec nckh-sdn-mininet bash -c "rm -rf /work/stats/results/*"
    docker exec nckh-sdn-mininet bash -c "rm -rf /work/ai_model/processed_data/charts/*"
    docker exec nckh-sdn-mininet bash -c "rm -f /work/stats/flow_stats.csv /work/stats/port_stats.csv"
    rm -f stats/results/final_report.txt
    mkdir -p stats/results/charts ai_model/processed_data/charts ai_model/training_logs
else
    echo -e "${YELLOW}Bỏ qua dọn dẹp (giữ lại data cũ)${NC}"
    docker exec nckh-sdn-mininet bash -c "rm -rf /work/ai_model/processed_data/charts/*"
    rm -f stats/results/final_report.txt
    mkdir -p stats/results/charts ai_model/processed_data/charts ai_model/training_logs
fi

for i in {5..1}; do echo -ne "${YELLOW}Bắt đầu sau $i s... ${NC}"; sleep 1; done
echo -e "${GREEN}GO!${NC}\n"

# ================================================================
# GIAI ĐOẠN 0: SYNTHETIC DATA (nếu có flag --synthetic hoặc --synthetic-only)
# ================================================================
if [ -n "$SYNTHETIC_FLAG" ] || [ -n "$SYNTHETIC_ONLY" ]; then
    echo -e "${CYAN}>>> GIAI ĐOẠN 0: TẠO SYNTHETIC DATA <<<${NC}"
    
    echo -e "${YELLOW}Tạo synthetic data V14 (đủ lớn cho TFT-CQL):${NC}"
    echo "  - Samples per action: 3000 (tối thiểu NCKH)"
    echo "  - Total: 9000 samples (3000 x 3 actions)"
    echo ""
    echo -e "${RED}Lưu ý: Nếu GPU không nhận (CPU mode), kiểm tra:${NC}"
    echo "  docker run ... --gpus all"
    echo "  nvidia-smi (trong container)"
    
    # V14: Use v3 generator - 3000 samples per action = 9000 total
    python3 scripts/generate_synthetic_data_v3.py \
        --samples 3000 \
        --output ai_model/processed_data
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Thất bại khi tạo synthetic data${NC}"
        exit 1
    fi
    
    # Copy synthetic data to main data files
    cp ai_model/processed_data/X_v3_synthetic.npy ai_model/processed_data/X_v3.npy
    cp ai_model/processed_data/y_v3_synthetic.npy ai_model/processed_data/y_v3.npy
    cp ai_model/processed_data/scenarios_v3.npy ai_model/processed_data/scenarios_v3_main.npy 2>/dev/null || true
    cp ai_model/processed_data/feature_metadata.json ai_model/processed_data/feature_metadata.json 2>/dev/null || true
    
    echo -e "${GREEN}Synthetic data đã được tạo và copy vào X_v3.npy, y_v3.npy${NC}"
    
    # Nếu chỉ tạo synthetic data thì thoát
    if [ -n "$SYNTHETIC_ONLY" ]; then
        echo -e "\n${GREEN}=================================================================${NC}"
        echo -e "${GREEN} SYNTHETIC DATA CREATION COMPLETED!                                ${NC}"
        echo -e "${GREEN}=================================================================${NC}"
        echo -e "${CYAN}Data files:${NC}"
        echo -e "${CYAN}  - ai_model/processed_data/X_v3_synthetic.npy${NC}"
        echo -e "${CYAN}  - ai_model/processed_data/y_v3_synthetic.npy${NC}"
        echo -e "${CYAN}  - ai_model/processed_data/scenarios_v3.npy${NC}"
        echo -e "${CYAN}  - ai_model/processed_data/feature_metadata.json${NC}"
        echo -e "${CYAN}  - ai_model/processed_data/X_v3.npy (copy)${NC}"
        echo -e "${CYAN}  - ai_model/processed_data/y_v3.npy (copy)${NC}"
        exit 0
    fi
fi

# ================================================================
# GIAI ĐOẠN 1: COLLECT DATA
# ================================================================
echo -e "${BLUE}>>> GIAI ĐOẠN 1: THU THẬP DỮ LIỆU <<<${NC}"

if [ -n "$SYNTHETIC_FLAG" ]; then
    echo -e "${YELLOW}Skip collect → Sử dụng synthetic data đã tạo${NC}"
elif [ "$SKIP_FLAG" == "--skip-collect" ]; then
    echo -e "${YELLOW}Skip collect → Merge data cũ...${NC}"
    docker exec nckh-sdn-mininet bash -c '
        cd /work
        echo "=== MERGE COLLECT DATA ==="
        python3 scripts/merge_data.py
        echo "  Flow stats: $(wc -l < stats/flow_stats.csv) rows"
        [ -f stats/port_stats.csv ] && echo "  Port stats: $(wc -l < stats/port_stats.csv) rows"
    '
    if [ $? -ne 0 ]; then echo -e "${RED}Thất bại ở Giai đoạn 1${NC}"; exit 1; fi
else
    ./scripts/full_pipeline.sh COLLECT
    if [ $? -ne 0 ]; then echo -e "${RED}Thất bại ở Giai đoạn 1${NC}"; exit 1; fi
fi

# ================================================================
# GIAI ĐOẠN 2: TFT-CQL ACTOR-CRITIC TRAINING (V3, 44 features)
# ================================================================
echo -e "${CYAN}>>> GIAI ĐOẠN 2: TRAINING TFT-CQL ACTOR-CRITIC <<<${NC}"

if [ -n "$SYNTHETIC_FLAG" ]; then
    # Với synthetic data, copy vào Docker container và train trực tiếp
    echo -e "${YELLOW}Sử dụng synthetic data - Skip data processing${NC}"
    
    # Copy synthetic data vào Docker container
    docker cp ai_model/processed_data/X_v3.npy nckh-sdn-mininet:/work/ai_model/processed_data/
    docker cp ai_model/processed_data/y_v3.npy nckh-sdn-mininet:/work/ai_model/processed_data/
    docker cp ai_model/processed_data/scenarios_v3.npy nckh-sdn-mininet:/work/ai_model/processed_data/ 2>/dev/null || true
    docker cp ai_model/processed_data/feature_metadata.json nckh-sdn-mininet:/work/ai_model/processed_data/ 2>/dev/null || true
    
    echo -e "${GREEN}Synthetic data copied to Docker container${NC}"
    
    # Training trực tiếp
    docker exec nckh-sdn-mininet bash -c '
        cd /work
        echo "=== TRAINING V14 (3-phase, 100 epochs) ==="
        PYTHONUNBUFFERED=1 python3 ai_model/train_actor_critic.py \
            --phase all \
            --epochs 100 \
            --batch_size 64 \
            --hidden_size 64
    '
else
    # Với real data, cần process từ flow_stats.csv
    docker exec nckh-sdn-mininet bash -c '
        cd /work

        echo "=== DATA PROCESSING V3 (44 features) ==="
        PYTHONUNBUFFERED=1 python3 << EOF
import sys
sys.path.insert(0, "/work")
from ai_model.data_processor import (
    load_flow_features, load_port_server_loads,
    aggregate_and_visualize, create_time_series_windows_v3
)

FLOW_CSV = "/work/stats/flow_stats.csv"
PORT_CSV = "/work/stats/port_stats.csv"

df_raw  = load_flow_features(FLOW_CSV)
server_loads = load_port_server_loads(PORT_CSV)
df_agg  = aggregate_and_visualize(df_raw)
X_v3, y_v3, meta = create_time_series_windows_v3(df_agg, server_loads, sequence_length=5)
features_count = meta.get("num_features")
print(f"V3 dataset: X={X_v3.shape}, features={features_count}")
EOF

        echo ""
        echo "=== TRAINING V14 (3-phase, 100 epochs) ==="
        PYTHONUNBUFFERED=1 python3 ai_model/train_actor_critic.py \
            --phase all \
            --epochs 100 \
            --batch_size 64 \
            --hidden_size 64
    '
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Cảnh báo: TFT-CQL training thất bại.${NC}"
else
    echo -e "${GREEN}TFT-CQL training hoàn tất.${NC}"
fi

# ================================================================
# GIAI ĐOẠN 3-4: BENCHMARK RR / WRR / AI
# ================================================================
echo -e "${BLUE}>>> GIAI ĐOẠN 3: ROUND ROBIN <<<${NC}"
./scripts/full_pipeline.sh RR
if [ $? -ne 0 ]; then echo -e "${RED}Thất bại ở Giai đoạn 3${NC}"; exit 1; fi

echo -e "${BLUE}>>> GIAI ĐOẠN 4: WEIGHTED ROUND ROBIN <<<${NC}"
./scripts/full_pipeline.sh WRR
if [ $? -ne 0 ]; then echo -e "${RED}Thất bại ở Giai đoạn 4${NC}"; exit 1; fi

echo -e "${BLUE}>>> GIAI ĐOẠN 5: AI INFERENCE & SO SÁNH <<<${NC}"
./scripts/full_pipeline.sh AI
if [ $? -ne 0 ]; then echo -e "${RED}Thất bại ở Giai đoạn 5${NC}"; exit 1; fi

# ================================================================
# GIAI ĐOẠN 6: EVALUATION TFT-CQL vs BASELINES
# ================================================================
echo -e "${CYAN}>>> GIAI ĐOẠN 6: ĐÁNH GIÁ TFT-CQL vs RR/WRR <<<${NC}"
docker exec nckh-sdn-mininet bash -c '
    cd /work
    if [ -f "ai_model/checkpoints/tft_ac_final.pth" ]; then
        CKPT="ai_model/checkpoints/tft_ac_final.pth"
    elif [ -f "ai_model/checkpoints/tft_ac_best.pth" ]; then
        CKPT="ai_model/checkpoints/tft_ac_best.pth"
    else
        echo "Không tìm thấy TFT-CQL checkpoint — bỏ qua."
        exit 0
    fi

    echo "=== EVALUATION TFT-CQL: $CKPT ==="
    PYTHONUNBUFFERED=1 python3 ai_model/evaluator.py --checkpoint "$CKPT"
    echo "Kết quả: ai_model/evaluation_results.json"
'
[ $? -eq 0 ] && echo -e "${GREEN}TFT-CQL evaluation hoàn tất.${NC}"

# ================================================================
# GIAI ĐOẠN 7: BENCHMARK VỚI METRIC MỚI (Synthetic Data)
# ================================================================
echo -e "${CYAN}>>> GIAI ĐOẠN 7: BENCHMARK VỚI METRIC MỚI <<<${NC}"
echo -e "${YELLOW}Chạy benchmark với synthetic data và metric mới...${NC}"

# Chạy benchmark với synthetic data
python3 scripts/quick_benchmark_new_scenarios.py --scenario all --compare BOTH

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Benchmark với metric mới hoàn tất.${NC}"
    echo -e "${CYAN}Kết quả benchmark:${NC}"
    echo "  - stats/benchmark_final/golden_hour_vs_wrr.json"
    echo "  - stats/benchmark_final/video_conference_vs_wrr.json"
    echo "  - stats/benchmark_final/hardware_degradation_vs_wrr.json"
    echo "  - stats/benchmark_final/low_rate_dos_vs_wrr.json"
else
    echo -e "${RED}Cảnh báo: Benchmark với metric mới thất bại.${NC}"
fi

# ================================================================
# GIAI ĐOẠN 8: STATISTICAL SIGNIFICANCE TESTING (IEEE)
# ================================================================
echo -e "${CYAN}>>> GIAI ĐOẠN 8: STATISTICAL SIGNIFICANCE TESTING (IEEE) <<<${NC}"
echo -e "${YELLOW}Running paired t-tests with N=3 iterations...${NC}"

python3 scripts/statistical_significance_test.py --n_runs 3 --output stats/ieee_stats.json

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Statistical significance testing completed.${NC}"
    echo "Results: stats/ieee_stats.json"
    python3 -c "
import json
with open('stats/ieee_stats.json') as f:
    data = json.load(f)
print(f\"AI wins {data['summary']['ai_wins']}/{data['summary']['total_metrics']} metrics with p < 0.05\")
" 2>/dev/null || echo "  (Summary not available)"
else
    echo -e "${RED}Cảnh báo: Statistical significance testing failed.${NC}"
fi

# ================================================================
# GIAI ĐOẠN 9: GENERATE KILLER CHARTS
# ================================================================
echo -e "${CYAN}>>> GIAI ĐOẠN 9: GENERATE KILLER CHARTS <<<${NC}"
python3 scripts/generate_killer_charts.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}4 Killer Charts generated successfully.${NC}"
else
    echo -e "${YELLOW}Chart generation skipped or failed.${NC}"
fi

# ── REPORT ──
echo -e "${YELLOW}Đang tổng hợp báo cáo...${NC}"
python3 scripts/analyze_stats.py

echo -e "\n${GREEN}=================================================================${NC}"
echo -e "${GREEN} TẤT CẢ THÍ NGHIỆM ĐÃ HOÀN TẤT NON-STOP!                        ${NC}"
echo -e "${GREEN}=================================================================${NC}"
echo -e "${CYAN}Biểu đồ so sánh:         stats/results/charts/${NC}"
echo -e "${CYAN}Training TFT-CQL:        ai_model/training_logs/${NC}"
echo -e "${CYAN}Evaluation CQL:          ai_model/evaluation_results.json${NC}"
echo -e "${CYAN}Benchmark mới:           stats/benchmark_final/${NC}"
echo -e "${YELLOW}Bạn có thể xem báo cáo khoa học ngay bây giờ.${NC}\n"
