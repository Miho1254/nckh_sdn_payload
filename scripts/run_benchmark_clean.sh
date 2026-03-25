#!/bin/bash
# =====================================================================
# NCKH SDN BENCHMARK - CHẠY CHUẨN TRONG DOCKER
# =====================================================================

# Don't exit on error - we want to continue even if some commands fail
# set -e

echo "============================================================"
echo "🚀 NCKH SDN BENCHMARK - PPO vs WRR"
echo "============================================================"
echo ""

# Check if inside Docker
if [ ! -f "/.dockerenv" ]; then
    echo "❌ LỖI: Script này phải chạy TRONG Docker container!"
    echo ""
    echo "👉 Từ máy HOST, chạy:"
    echo "   docker exec -it nckh-sdn-mininet bash"
    echo "   cd /work && bash scripts/run_benchmark_clean.sh"
    exit 1
fi

# Kill any existing benchmark processes
echo "[1/6] Cleaning up existing processes..."
sudo pkill -9 ryu-manager 2>/dev/null || true
sudo pkill -9 mininet 2>/dev/null || true
sudo mn -c 2>/dev/null || true
sleep 3

# Setup
WORK_DIR="/work"
cd "$WORK_DIR"
OUTPUT_ROOT="benchmark_results_v4_fixed"
SCENARIOS=("golden_hour" "video_conference" "hardware_degradation" "low_rate_dos")
NUM_RUNS=5
COOLDOWN_SEC=20

mkdir -p "$OUTPUT_ROOT"

echo "[2/6] Verifying PPO V3 model..."
ls -la ai_model/models/ppo_v3_real.zip
python3 -c "from stable_baselines3 import PPO; m = PPO.load('ai_model/models/ppo_v3_real.zip'); print(f'  Model: {m.observation_space}')"

echo ""
echo "[3/6] Verifying controller_stats.py has 20 features..."
grep -c "load_h5, load_h7, load_h8" controller_stats.py > /dev/null && echo "  ✓ Feature vector OK (20 features)"

echo ""
echo "============================================================"
echo "[4/6] Starting Benchmark: 4 Scenarios × 5 Runs × 2 Algorithms"
echo "============================================================"

for SCENARIO in "${SCENARIOS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📂 SCENARIO: $SCENARIO"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━��━"

    for RUN in $(seq 1 $NUM_RUNS); do
        RUN_DIR="$OUTPUT_ROOT/${SCENARIO}/run_$RUN"
        mkdir -p "$RUN_DIR/ppo" "$RUN_DIR/wrr"

        echo ""
        echo "  [Run $RUN/$NUM_RUNS]"
        echo "  ─────────────────────────────────────────"

        # Clean slate
        sudo mn -c > /dev/null 2>&1
        sudo pkill -9 ryu-manager > /dev/null 2>&1 || true
        rm -rf stats/ && mkdir -p stats/

        # ── WRR BENCHMARK ──
        echo "    [1/2] Running WRR..."
        export LB_ALGO="WRR"
        sudo ryu-manager controller_stats.py > "$RUN_DIR/wrr/ryu_wrr.log" 2>&1 &
        RYU_PID=$!
        sleep 12
        
        export SCENARIO_NAME="$SCENARIO"
        sudo python3 run_lms_mininet.py > /dev/null 2>&1 &
        
        sleep 90  # Wait for benchmark to complete
        sudo kill -9 $RYU_PID > /dev/null 2>&1 || true
        
        cp stats/flow_stats.csv "$RUN_DIR/wrr/" 2>/dev/null || true
        cp stats/port_stats.csv "$RUN_DIR/wrr/" 2>/dev/null || true
        
        echo "      ✓ WRR done. Cooldown ${COOLDOWN_SEC}s..."
        sleep $COOLDOWN_SEC

        # ── PPO BENCHMARK ──
        echo "    [2/2] Running PPO..."
        sudo mn -c > /dev/null 2>&1
        sudo pkill -9 ryu-manager > /dev/null 2>&1 || true
        rm -rf stats/ && mkdir -p stats/
        
        export LB_ALGO="AI"
        sudo ryu-manager controller_stats.py > "$RUN_DIR/ppo/ryu_ppo.log" 2>&1 &
        RYU_PID=$!
        sleep 12
        
        sudo python3 run_lms_mininet.py > /dev/null 2>&1 &
        
        sleep 90  # Wait for benchmark to complete
        sudo kill -9 $RYU_PID > /dev/null 2>&1 || true
        
        cp stats/flow_stats.csv "$RUN_DIR/ppo/" 2>/dev/null || true
        cp stats/port_stats.csv "$RUN_DIR/ppo/" 2>/dev/null || true
        cp stats/inference_log.csv "$RUN_DIR/ppo/" 2>/dev/null || true
        
        echo "      ✓ PPO done. Cooldown ${COOLDOWN_SEC}s..."
        sleep $COOLDOWN_SEC
    done
done

echo ""
echo "============================================================"
echo "[5/6] Benchmark Complete!"
echo "============================================================"

echo ""
echo "[6/6] Analyzing results..."
python3 -c "
import pandas as pd
import numpy as np
from scipy import stats

results = []
scenarios = ['golden_hour', 'video_conference', 'hardware_degradation', 'low_rate_dos']

for scenario in scenarios:
    for run in range(1, 6):
        for algo in ['ppo', 'wrr']:
            path = f'$OUTPUT_ROOT/{scenario}/run_{run}/{algo}/flow_stats.csv'
            try:
                df = pd.read_csv(path, low_memory=False)
                packets = df['packet_count'].sum()
                results.append({'scenario': scenario, 'run': run, 'algo': algo, 'packets': packets})
            except: pass

df = pd.DataFrame(results)

print()
print('=' * 60)
print('BENCHMARK RESULTS - PPO V3 vs WRR')
print('=' * 60)

ppo_wins = 0
for scenario in scenarios:
    sc = df[df['scenario'] == scenario]
    ppo = sc[sc['algo'] == 'ppo']['packets'].values
    wrr = sc[sc['algo'] == 'wrr']['packets'].values
    
    if len(ppo) == 5 and len(wrr) == 5:
        ppo_mean = np.mean(ppo)
        wrr_mean = np.mean(wrr)
        diff_pct = ((ppo_mean - wrr_mean) / wrr_mean) * 100
        t_stat, p_val = stats.ttest_rel(ppo, wrr)
        
        winner = 'PPO' if ppo_mean > wrr_mean else 'WRR'
        if winner == 'PPO':
            ppo_wins += 1
        
        sig = '(SIGNIFICANT)' if p_val < 0.05 else ''
        print(f'{scenario}:')
        print(f'  PPO: {ppo_mean:>12,.0f} ± {np.std(ppo):>10,.0f}')
        print(f'  WRR: {wrr_mean:>12,.0f} ± {np.std(wrr):>10,.0f}')
        print(f'  Winner: {winner:>4} ({diff_pct:>+6.1f}%)  p={p_val:.4f} {sig}')
        print()

print('=' * 60)
print(f'Overall: PPO won {ppo_wins}/4 scenarios')
ppo_all = df[df['algo'] == 'ppo']['packets']
wrr_all = df[df['algo'] == 'wrr']['packets']
print(f'Overall PPO: {ppo_all.mean():,.0f} ± {ppo_all.std():,.0f}')
print(f'Overall WRR: {wrr_all.mean():,.0f} ± {wrr_all.std():,.0f}')
print(f'Difference: {((ppo_all.mean() - wrr_all.mean()) / wrr_all.mean()) * 100:+.1f}%')
"

echo ""
echo "✅ Benchmark complete! Results saved in: $OUTPUT_ROOT"
echo "============================================================"
