#!/bin/bash
# Chạy đánh giá PPO và WRR trên môi trường Mininet/Ryu thực tế

cd /work

# Cấu hình
SCENARIO=${1:-"golden_hour.yml"}
OUTPUT_DIR="benchmark_real"

echo "=========================================================="
echo "🚀 REAL MININET BENCHMARK: PPO vs WRR ($SCENARIO)"
echo "=========================================================="

mkdir -p $OUTPUT_DIR/wrr
mkdir -p $OUTPUT_DIR/ppo

# Cleanup
rm -f stats/*.csv
rm -f /tmp/h*.log
mn -c > /dev/null 2>&1

# ====== 1. Định chuẩn WRR ======
echo -e "\n[1/2] Đang chạy WRR Policy..."
export LB_ALGO=WRR
# Chạy Ryu controller ngầm
ryu-manager controller_stats.py > /tmp/ryu_wrr.log 2>&1 &
RYU_PID=$!
sleep 5 # Đợi Ryu khởi động

echo "Bắt đầu Mininet và ép traffic WRR..."
export SCENARIO=$SCENARIO
python3 run_lms_mininet.py

echo "Mininet đã kết thúc. Đang lưu metrics WRR..."
kill -9 $RYU_PID
cp stats/flow_stats.csv $OUTPUT_DIR/wrr/
cp stats/port_stats.csv $OUTPUT_DIR/wrr/
cp stats/inference_log.csv $OUTPUT_DIR/wrr/ 2>/dev/null
cp /tmp/h*_stress.log $OUTPUT_DIR/wrr/ 2>/dev/null

sleep 2
# Cleanup
rm -f stats/*.csv
rm -f /tmp/h*.log
mn -c > /dev/null 2>&1

# ====== 2. Thử nghiệm PPO ======
echo -e "\n[2/2] Đang chạy AI Policy (PPO)..."
export LB_ALGO=AI
# Chạy Ryu controller ngầm
ryu-manager controller_stats.py > /tmp/ryu_ppo.log 2>&1 &
RYU_PID=$!
sleep 5 # Đợi Ryu khởi động

echo "Bắt đầu Mininet và ép traffic PPO..."
export SCENARIO=$SCENARIO
python3 run_lms_mininet.py

echo "Mininet đã kết thúc. Đang lưu metrics PPO..."
kill -9 $RYU_PID
cp stats/flow_stats.csv $OUTPUT_DIR/ppo/
cp stats/port_stats.csv $OUTPUT_DIR/ppo/
cp stats/inference_log.csv $OUTPUT_DIR/ppo/ 2>/dev/null
cp /tmp/h*_stress.log $OUTPUT_DIR/ppo/ 2>/dev/null

echo -e "\n🎉 Benchmark trên REAL MININET đã hoàn tất!"
echo "Đang phân tích kết quả..."
python3 scripts/analyze_real_mininet_results.py $OUTPUT_DIR
