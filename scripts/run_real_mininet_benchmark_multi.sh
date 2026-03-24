#!/bin/bash

# scripts/run_real_mininet_benchmark_multi.sh
# Usage: ./run_real_mininet_benchmark_multi.sh <num_runs>

NUM_RUNS=${1:-5}
OUTPUT_ROOT="benchmark_results_multi"
RYU_APP="controller_stats.py"

mkdir -p $OUTPUT_ROOT

for i in $(seq 1 $NUM_RUNS); do
    RUN_DIR="$OUTPUT_ROOT/run_$i"
    mkdir -p "$RUN_DIR/wrr"
    mkdir -p "$RUN_DIR/ppo"

    echo "=========================================================="
    echo "🚀 BẮT ĐẦU RUN $i / $NUM_RUNS"
    echo "=========================================================="

    # --- 1. WRR Run ---
    echo "[Run $i] Đang chạy WRR Policy..."
    export AI_POLICY_ENABLED=false
    
    # Clean up previous state
    rm -rf stats/
    mn -c > /dev/null 2>&1
    pkill -9 ryu-manager > /dev/null 2>&1
    
    # Start Ryu
    ryu-manager $RYU_APP > /tmp/ryu_wrr.log 2>&1 &
    RYU_PID=$!
    sleep 5
    
    # Start Mininet with Traffic
    python3 run_lms_mininet.py
    
    # Save Metrics
    kill -9 $RYU_PID
    cp stats/flow_stats.csv "$RUN_DIR/wrr/" 2>/dev/null
    cp stats/port_stats.csv "$RUN_DIR/wrr/" 2>/dev/null
    mv /tmp/h*_stress.log "$RUN_DIR/wrr/" 2>/dev/null
    
    echo "[Run $i] WRR hoàn tất. Chờ 10s cooldown CPU..."
    sleep 10

    # --- 2. PPO Run ---
    echo "[Run $i] Đang chạy PPO Policy..."
    export AI_POLICY_ENABLED=true
    
    # Clean up
    mn -c > /dev/null 2>&1
    pkill -9 ryu-manager > /dev/null 2>&1
    
    # Start Ryu
    ryu-manager $RYU_APP > /tmp/ryu_ppo.log 2>&1 &
    RYU_PID=$!
    sleep 5
    
    # Start Mininet with Traffic
    python3 run_lms_mininet.py
    
    # Save Metrics
    kill -9 $RYU_PID
    cp stats/flow_stats.csv "$RUN_DIR/ppo/" 2>/dev/null
    cp stats/port_stats.csv "$RUN_DIR/ppo/" 2>/dev/null
    cp stats/inference_log.csv "$RUN_DIR/ppo/" 2>/dev/null
    mv /tmp/h*_stress.log "$RUN_DIR/ppo/" 2>/dev/null

    echo "[Run $i] PPO hoàn tất. Chờ 10s cooldown CPU..."
    sleep 10
done

echo "✅ Đã hoàn thành $NUM_RUNS lượt benchmark!"
echo "Dữ liệu lưu tại thư mục: $OUTPUT_ROOT"
