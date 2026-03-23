#!/bin/bash

# Quick Benchmark Script - So sánh RR, WRR vs TFT-CQL
# Sử dụng evaluator.py đã được cập nhật với metrics mới

SCENARIO="golden_hour"
RES_DIR="stats/benchmark_final"
CKPT_DIR="ai_model/checkpoints"

mkdir -p "$RES_DIR"

echo "=========================================================="
echo "  QUICK BENCHMARK: RR vs WRR vs TFT-CQL (Golden Hour)"
echo "  Using NEW metrics: QoS Efficiency, Burst Handling, Stability"
echo "=========================================================="

# Check if checkpoint exists
CKPT_PATH="$CKPT_DIR/tft_ac_best.pth"
if [ ! -f "$CKPT_PATH" ]; then
    # Try final checkpoint
    CKPT_PATH="$CKPT_DIR/tft_ac_final.pth"
fi

# Check if data exists
DATA_X="ai_model/processed_data/X_v3.npy"
DATA_Y="ai_model/processed_data/y_v3.npy"

if [ ! -f "$DATA_X" ] || [ ! -f "$DATA_Y" ]; then
    echo "  [✗] Data not found. Run data collection first."
    echo "      Expected: $DATA_X and $DATA_Y"
    exit 1
fi

echo ""
echo "[1/3] Running TFT-CQL Evaluator..."
echo "  Checkpoint: $CKPT_PATH"

python3 -c "
import sys
sys.path.insert(0, 'ai_model')
from evaluator import main
import argparse

# Override args
sys.argv = ['evaluator.py', '--checkpoint', '$CKPT_PATH']
main()
"

if [ $? -ne 0 ]; then
    echo "  [✗] TFT-CQL evaluation failed"
    exit 1
fi

echo ""
echo "  [✓] TFT-CQL evaluation completed"
echo "  Results saved to: ai_model/evaluation_results.json"

# Copy results to benchmark directory
cp ai_model/evaluation_results.json "$RES_DIR/evaluation_results.json"

echo ""
echo "=========================================================="
echo "  BENCHMARK COMPLETE"
echo "=========================================================="
echo ""
echo "Results saved to:"
echo "  - ai_model/evaluation_results.json"
echo "  - $RES_DIR/evaluation_results.json"
echo ""
echo "Key Metrics (NEW - Fair for AI):"
echo "  - QoS Efficiency: Throughput / (1 + Overload_Rate)"
echo "  - Burst Handling: burst_handled / burst_count"
echo "  - Stability Score: 1 / (1 + throughput_std)"
echo ""
echo "To view detailed comparison, run:"
echo "  python3 -c \"import json; print(json.dumps(json.load(open('ai_model/evaluation_results.json')), indent=2))\""