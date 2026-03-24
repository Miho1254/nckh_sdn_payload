#!/bin/bash
# ═════════════════════════════════════════════════════════════════════
# V9 NORMALIZED TRAINING SCRIPT
# ═════════════════════════════════════════════════════════════════════
# Thay đổi từ V8:
#   - Thêm StandardScaler normalization (V9: MỞ MẮT Neural Network)
#   - Tất cả 44 features được ép về mean=0, std=1
#   - AI có thể phân biệt normal vs dos scenarios
# ═════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")/.."
cd ai_model

echo "════════════════════════════════════════════════════════════════════"
echo "  V9 NORMALIZED TRAINING - STANDARDSCALER"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  Config:"
echo "    - ENTROPY_COEFF: 0.5 (Vùng Goldilocks)"
echo "    - KL_COEFF: 0.05"
echo "    - CQL_ALPHA: 1.0"
echo "    - Early Stopping: DISABLED"
echo "    - Epochs: 100 (full run)"
echo "    - StandardScaler: ENABLED (V9 - MỞ MẮT AI)"
echo ""
echo "  Press Ctrl+C to abort"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python train_actor_critic.py --phase all --epochs 100 --batch_size 64

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE - Check training_logs/ for results"
echo "════════════════════════════════════════════════════════════════════"