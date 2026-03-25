#!/bin/bash
# =====================================================================
# NCKH SDN BENCHMARK - ALL SCENARIOS
# Chạy benchmark cho tất cả 6 scenarios: golden_hour, video_conference, burst_traffic, low_rate_dos, server_failure, hardware_degradation
# =====================================================================

echo "============================================================"
echo "🚀 NCKH SDN BENCHMARK - ALL SCENARIOS"
echo "============================================================"

# Check if inside Docker
if [ ! -f "/.dockerenv" ]; then
    echo "❌ LỖI: Phải chạy TRONG Docker container!"
    exit 1
fi

# Clean up
echo "[1/6] Cleaning up..."
pkill -9 ryu-manager 2>/dev/null || true
mn -c 2>/dev/null || true
sleep 3

cd /work
OUTPUT_ROOT="benchmark_results_all"
rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

# Verify model
echo "[2/6] Verifying PPO V3 model..."
python3 -c "from stable_baselines3 import PPO; m = PPO.load('ai_model/models/ppo_v3_real.zip'); print(f'Model: {m.observation_space}')"

# Verify controller
echo "[3/6] Verifying controller..."
grep -c "load_h5, load_h7, load_h8" controller_stats.py > /dev/null && echo "✓ 20 features confirmed"

# List of scenarios to run
SCENARIOS=("golden_hour" "video_conference" "burst_traffic" "low_rate_dos" "server_failure" "hardware_degradation")

echo ""
echo "============================================================"
echo "[4/6] Running benchmarks for all scenarios"
echo "============================================================"

# Run each scenario
for SCENARIO in "${SCENARIOS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running scenario: $SCENARIO"
    echo "============================================================"
    
    RUN=1
    RUN_DIR="$OUTPUT_ROOT/${SCENARIO}/run_$RUN"
    mkdir -p "$RUN_DIR/ppo" "$RUN_DIR/wrr"
    
    # ── WRR ──
    echo ""
    echo "[WRR] Starting..."
    rm -rf stats/ && mkdir -p stats/
    export LB_ALGO="RR"
    export SCENARIO="${SCENARIO}.yml"
    ryu-manager controller_stats.py > "$RUN_DIR/wrr/ryu_wrr.log" 2>&1 &
    RYU_PID=$!
    echo "[WRR] Ryu PID: $RYU_PID"
    sleep 15
    
    # Run Mininet
    echo "[WRR] Running Mininet (will take ~3-5 minutes)..."
    python3 run_lms_mininet.py
    
    # Wait a bit more for stats to be written
    sleep 5
    
    # Collect
    echo "[WRR] Collecting stats..."
    cp stats/flow_stats.csv "$RUN_DIR/wrr/" 2>/dev/null && echo "  ✓ flow_stats.csv copied" || echo "  ✗ No flow_stats.csv"
    cp stats/port_stats.csv "$RUN_DIR/wrr/" 2>/dev/null && echo "  ✓ port_stats.csv copied" || echo "  ✗ No port_stats.csv"
    mv /tmp/h*_stress.log "$RUN_DIR/wrr/" 2>/dev/null || true
    
    # Kill Ryu
    kill $RYU_PID 2>/dev/null || true
    sleep 3
    
    WRR_PACKETS=$(cat $RUN_DIR/wrr/flow_stats.csv 2>/dev/null | tail -n +2 | cut -d',' -f10 | paste -sd+ | bc 2>/dev/null || echo 'N/A')
    echo "[WRR] Done! Packets: $WRR_PACKETS"
    
    sleep 20
    
    # ── PPO ──
    echo ""
    echo "[PPO] Starting..."
    mn -c > /dev/null 2>&1 || true
    rm -rf stats/ && mkdir -p stats/
    export LB_ALGO="AI"
    export SCENARIO="${SCENARIO}.yml"
    ryu-manager controller_stats.py > "$RUN_DIR/ppo/ryu_ppo.log" 2>&1 &
    RYU_PID=$!
    echo "[PPO] Ryu PID: $RYU_PID"
    sleep 15
    
    # Run Mininet
    echo "[PPO] Running Mininet (will take ~3-5 minutes)..."
    python3 run_lms_mininet.py
    
    # Wait a bit more for stats to be written
    sleep 5
    
    # Collect
    echo "[PPO] Collecting stats..."
    cp stats/flow_stats.csv "$RUN_DIR/ppo/" 2>/dev/null && echo "  ✓ flow_stats.csv copied" || echo "  ✗ No flow_stats.csv"
    cp stats/port_stats.csv "$RUN_DIR/ppo/" 2>/dev/null && echo "  ✓ port_stats.csv copied" || echo "  ✗ No port_stats.csv"
    cp stats/inference_log.csv "$RUN_DIR/ppo/" 2>/dev/null && echo "  ✓ inference_log.csv copied" || echo "  ✗ No inference_log.csv"
    mv /tmp/h*_stress.log "$RUN_DIR/ppo/" 2>/dev/null || true
    
    echo "[PPO] Done!"
    
    # ── Results ──
    echo ""
    echo "============================================================"
    echo "Results for $SCENARIO"
    echo "============================================================"
    
    python3 -c "
import pandas as pd
import os

wrr_path = '$OUTPUT_ROOT/${SCENARIO}/run_1/wrr/flow_stats.csv'
ppo_path = '$OUTPUT_ROOT/${SCENARIO}/run_1/ppo/flow_stats.csv'

try:
    wrr_df = pd.read_csv(wrr_path, low_memory=False)
    wrr_packets = wrr_df['packet_count'].sum()
    print(f'WRR packets: {wrr_packets:,}')
except Exception as e:
    print(f'WRR error: {e}')

try:
    ppo_df = pd.read_csv(ppo_path, low_memory=False)
    ppo_packets = ppo_df['packet_count'].sum()
    print(f'PPO packets: {ppo_packets:,}')
except Exception as e:
    print(f'PPO error: {e}')

try:
    diff = ((ppo_packets - wrr_packets) / wrr_packets) * 100
    winner = 'PPO' if ppo_packets > wrr_packets else 'WRR'
    print(f'Winner: {winner} ({diff:+.1f}%)')
except:
    pass

# Check inference log
inf_path = '$OUTPUT_ROOT/${SCENARIO}/run_1/ppo/inference_log.csv'
if os.path.exists(inf_path):
    with open(inf_path, 'r') as f:
        lines = f.readlines()
        print(f'Inference records: {len(lines)-1} rows')
else:
    print('No inference log found')
"
    
    sleep 5
done

echo ""
echo "============================================================"
echo "[5/6] Summary of all scenarios"
echo "============================================================"

# Create summary CSV
python3 -c "
import pandas as pd
import os

scenarios = ['golden_hour', 'video_conference', 'burst_traffic', 'low_rate_dos', 'server_failure', 'hardware_degradation']
results = []

for scenario in scenarios:
    wrr_path = f'benchmark_results_all/{scenario}/run_1/wrr/flow_stats.csv'
    ppo_path = f'benchmark_results_all/{scenario}/run_1/ppo/flow_stats.csv'
    
    wrr_packets = None
    ppo_packets = None
    
    try:
        wrr_df = pd.read_csv(wrr_path, low_memory=False)
        wrr_packets = wrr_df['packet_count'].sum()
    except:
        pass
    
    try:
        ppo_df = pd.read_csv(ppo_path, low_memory=False)
        ppo_packets = ppo_df['packet_count'].sum()
    except:
        pass
    
    if wrr_packets and ppo_packets:
        diff = ((ppo_packets - wrr_packets) / wrr_packets) * 100
        winner = 'PPO' if ppo_packets > wrr_packets else 'WRR'
    else:
        diff = None
        winner = 'N/A'
    
    results.append({
        'scenario': scenario,
        'wrr_packets': wrr_packets,
        'ppo_packets': ppo_packets,
        'diff_pct': diff,
        'winner': winner
    })

df = pd.DataFrame(results)
df.to_csv('benchmark_results_all/summary.csv', index=False)
print(df.to_string(index=False))
"

echo ""
echo "✅ All benchmarks done! Results in: $OUTPUT_ROOT"
echo "============================================================"
