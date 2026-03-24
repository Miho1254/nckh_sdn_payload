#!/bin/bash
# ═════════════════════════════════════════════════════════════════════
# V8 GOLDILOCKS TRAINING SCRIPT
# ═════════════════════════════════════════════════════════════════════
# Thay đổi từ V7:
#   - ENTROPY_COEFF: 0.05 → 0.5 (Vùng Goldilocks - tiêu chuẩn SAC/CQL)
#   - Tắt Early Stopping (ép chạy đủ 100 epochs để AI nếm hình phạt)
# ═════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")/.."
cd ai_model

echo "════════════════════════════════════════════════════════════════════"
echo "  V8 GOLDILOCKS TRAINING - ENTROPY COEFF = 0.5"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  Config:"
echo "    - ENTROPY_COEFF: 0.5 (Vùng Goldilocks)"
echo "    - KL_COEFF: 0.05"
echo "    - CQL_ALPHA: 1.0"
echo "    - Early Stopping: DISABLED"
echo "    - Epochs: 100 (full run)"
echo ""
echo "  Press Ctrl+C to abort"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python train_actor_critic.py --phase all --epochs 100 --batch_size 64

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE - Check training_logs/ for results"
echo "════════════════════════════════════════════════════════════════════"