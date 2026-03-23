#!/bin/bash
# ═════════════════════════════════════════════════════════════════════
# V11 BALANCED EXPLORATION TRAINING SCRIPT
# ═════════════════════════════════════════════════════════════════════
# Thay đổi từ V10:
#   - V11: Tăng wastage_penalty (0.00025 -> 0.005, x20)
#   - V11: Tăng entropy_coeff (0.5 -> 2.0, x4)
#   - Mục tiêu: AI không quá thận trọng (h8 100%), thử nghiệm nhiều actions
# ═════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")/.."
cd ai_model

echo "════════════════════════════════════════════════════════════════════"
echo "  V11 BALANCED EXPLORATION"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  Config:"
echo "    - ENTROPY_COEFF: 2.0 (V11: tăng exploration)"
echo "    - KL_COEFF: 0.05"
echo "    - CQL_ALPHA: 1.0"
echo "    - Early Stopping: DISABLED"
echo "    - Epochs: 100 (full run)"
echo "    - StandardScaler: ENABLED"
echo ""
echo "  Reward Scale (V10):"
echo "    - overload_penalty (util > 0.85): 5.0"
echo "    - overload_penalty (util > 0.95): 20.0"
echo "    - wastage_penalty: 0.005 (V11: tăng x20)"
echo ""
echo "  Mục tiêu: AI thử nghiệm h5, h7, h8 theo điều kiện mạng"
echo "    - Normal traffic -> h5/h7 (tiết kiệm)"
echo "    - High traffic -> h7/h8 (cân bằng)"
echo "    - DOS -> h8 (mạnh nhất)"
echo ""
echo "  Press Ctrl+C to abort"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python train_actor_critic.py --phase all --epochs 100 --batch_size 64

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE - Check training_logs/ for results"
echo "════════════════════════════════════════════════════════════════════"