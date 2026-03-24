#!/bin/bash


# =====================================================================
#  NCKH SDN BENCHMARK ORCHESTRATOR (N=5)
#  Automate: 4 Scenarios x 5 Runs x 2 Algorithms (WRR & PPO)
# =====================================================================

# ── CONFIGURATION ─────────────────────────────────────────────────────
SCENARIOS=("golden_hour.yml" "video_conference.yml" "hardware_degradation.yml" "low_rate_dos.yml")
NUM_RUNS=5
OUTPUT_ROOT="benchmark_results_final_n5"
RYU_APP="controller_stats.py"
PYTHON_RUNNER="run_lms_mininet.py"
COOLDOWN_SEC=15

# ── DOCKER CHECK ──────────────────────────────────────────────────────
# Script này BẮT BUỘC phải chạy bên trong Container 'nckh-sdn-mininet'.
if [ ! -f "/.dockerenv" ] && [ "$DOCKER_CONTEXT" != "1" ]; then
    echo "❌ LỖI: Bạn đang chạy script trên máy HOST!"
    echo "Thư viện Mininet chỉ có sẵn bên trong Docker container."
    echo ""
    echo "👉 Hãy chạy lệnh sau từ máy ngoài:"
    echo "   docker exec -it nckh-sdn-mininet /work/scripts/run_all_scenarios_n5.sh"
    echo ""
    exit 1
fi

# ── ENVIRONMENT DETECTION ─────────────────────────────────────────────
# Mininet thường nằm ở system python, không nằm trong venv.
# Chúng ta sẽ tìm python có mininet.
PYTHON_CMD="python3"
if ! $PYTHON_CMD -c "import mininet" >/dev/null 2>&1; then
    if /usr/bin/python3 -c "import mininet" >/dev/null 2>&1; then
        PYTHON_CMD="/usr/bin/python3"
    fi
fi

# Mininet YÊU CẦU sudo!
SUDO="sudo"
if [ "$EUID" -eq 0 ]; then
    SUDO=""
fi

# ── INIT ─────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_ROOT"
echo "=========================================================="
echo "🚀 BẮT ĐẦU CHIẾN DỊCH BENCHMARK TỔNG LỰC (N=$NUM_RUNS)"
echo "🚀 Python: $PYTHON_CMD | Mode: $SUDO"
echo "🚀 Kịch bản: ${SCENARIOS[*]}"
echo "=========================================================="

# Check dependencies
if [ ! -f "$RYU_APP" ]; then echo "❌ Lỗi: Không tìm thấy $RYU_APP tại $(pwd)"; exit 1; fi
if [ ! -f "$PYTHON_RUNNER" ]; then echo "❌ Lỗi: Không tìm thấy $PYTHON_RUNNER tại $(pwd)"; exit 1; fi

SCENARIO_DIR="lms/evaluation"

for SCENARIO in "${SCENARIOS[@]}"; do
    SCENARIO_NAME=$(echo "$SCENARIO" | cut -f 1 -d '.')
    SCENARIO_PATH="$SCENARIO_DIR/$SCENARIO"
    
    # Check if scenario exists
    if [ ! -f "$SCENARIO_PATH" ]; then
        echo "⚠️ Cảnh báo: Không tìm thấy file kịch bản $SCENARIO_PATH. Bỏ qua."
        continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📂 KỊCH BẢN: $SCENARIO_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for i in $(seq 1 $NUM_RUNS); do
        RUN_DIR="$OUTPUT_ROOT/$SCENARIO_NAME/run_$i"
        mkdir -p "$RUN_DIR/wrr"
        mkdir -p "$RUN_DIR/ppo"

        echo "[Run $i/$NUM_RUNS] Processing $SCENARIO_NAME..."

        # ── 1. WRR BENCHMARK ──
        echo "   [+] Running WRR..."
        export LB_ALGO="RR"        # Dùng Round Robin (hoặc WRR tùy config)
        export AI_POLICY_ENABLED=false
        export SCENARIO="$SCENARIO"
        
        # Clean
        $SUDO mn -c > /dev/null 2>&1
        $SUDO pkill -9 ryu-manager > /dev/null 2>&1
        rm -rf stats/ && mkdir -p stats/
        
        # Start Ryu
        $SUDO ryu-manager "$RYU_APP" > "$RUN_DIR/wrr/ryu_wrr.log" 2>&1 &
        RYU_PID=$!
        sleep 10
        
        # Run Traffic
        $SUDO $PYTHON_CMD "$PYTHON_RUNNER"
        
        # Collect
        $SUDO kill -9 $RYU_PID > /dev/null 2>&1
        cp stats/flow_stats.csv "$RUN_DIR/wrr/" 2>/dev/null
        cp stats/port_stats.csv "$RUN_DIR/wrr/" 2>/dev/null
        mv /tmp/h*_stress.log "$RUN_DIR/wrr/" 2>/dev/null
        
        echo "   [+] WRR Done. Cooldown ${COOLDOWN_SEC}s..."
        sleep $COOLDOWN_SEC

        # ── 2. PPO BENCHMARK ──
        echo "   [+] Running PPO..."
        export LB_ALGO="AI"
        export AI_SERVING_RULE="sampled"
        export AI_POLICY_ENABLED=true
        export SCENARIO="$SCENARIO"

        # Clean
        $SUDO mn -c > /dev/null 2>&1
        $SUDO pkill -9 ryu-manager > /dev/null 2>&1
        rm -rf stats/ && mkdir -p stats/

        # Start Ryu
        $SUDO ryu-manager "$RYU_APP" > "$RUN_DIR/ppo/ryu_ppo.log" 2>&1 &
        RYU_PID=$!
        sleep 10

        # Run Traffic
        $SUDO $PYTHON_CMD "$PYTHON_RUNNER"

        # Collect
        $SUDO kill -9 $RYU_PID > /dev/null 2>&1
        cp stats/flow_stats.csv "$RUN_DIR/ppo/" 2>/dev/null
        cp stats/port_stats.csv "$RUN_DIR/ppo/" 2>/dev/null
        cp stats/inference_log.csv "$RUN_DIR/ppo/" 2>/dev/null
        mv /tmp/h*_stress.log "$RUN_DIR/ppo/" 2>/dev/null


        echo "   [+] PPO Done. Cooldown ${COOLDOWN_SEC}s..."
        sleep $COOLDOWN_SEC
    done
done


echo "=========================================================="
echo "✅ HOÀN TẤT TOÀN BỘ CHIẾN DỊCH!"
echo "📍 Dữ liệu tại: $OUTPUT_ROOT"
echo "=========================================================="
