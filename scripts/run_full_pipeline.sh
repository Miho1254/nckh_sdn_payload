#!/bin/bash
# Full Pipeline: Generate Data -> Train Model -> Benchmark
# Usage: ./run_full_pipeline.sh

set -e

echo "=============================================="
echo "  FULL PIPELINE: DATA -> TRAIN -> BENCHMARK"
echo "=============================================="

cd "$(dirname "$0")/.."

# Step 1: Generate Synthetic Data
echo ""
echo "[Step 1/4] Generating Synthetic Data..."
echo "----------------------------------------------"
python scripts/generate_synthetic_data.py \
    --normal 200 \
    --overload 100 \
    --burst 100 \
    --degradation 100 \
    --dos 100 \
    --output ai_model/processed_data

# Step 2: Copy synthetic data to main data files
echo ""
echo "[Step 2/4] Copying Synthetic Data to Main Files..."
echo "----------------------------------------------"
cp ai_model/processed_data/X_v3_synthetic.npy ai_model/processed_data/X_v3.npy
cp ai_model/processed_data/y_v3_synthetic.npy ai_model/processed_data/y_v3.npy
echo "Done! Data files updated."

# Step 3: Train Model (optional - comment out if not needed)
echo ""
echo "[Step 3/4] Training Model..."
echo "----------------------------------------------"
echo "Note: Training may take a while. Press Ctrl+C to skip."
echo "To skip training, comment out this section in the script."
# python ai_model/train_actor_critic.py --epochs 100

# Step 4: Run Benchmark
echo ""
echo "[Step 4/4] Running Benchmark..."
echo "----------------------------------------------"
python scripts/quick_benchmark_new_scenarios.py --scenario all --compare BOTH

echo ""
echo "=============================================="
echo "  PIPELINE COMPLETED!"
echo "=============================================="
echo ""
echo "Results saved to: stats/benchmark_final/"
echo ""
echo "To view results:"
echo "  cat stats/benchmark_final/golden_hour_vs_wrr.json"
echo "  cat stats/benchmark_final/video_conference_vs_wrr.json"
echo "  cat stats/benchmark_final/hardware_degradation_vs_wrr.json"
echo "  cat stats/benchmark_final/low_rate_dos_vs_wrr.json"