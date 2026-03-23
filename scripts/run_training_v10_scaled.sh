#!/bin/bash
# ═════════════════════════════════════════════════════════════════════
# V10 SCALED REWARD TRAINING SCRIPT
# ═════════════════════════════════════════════════════════════════════
# Thay đổi từ V9:
#   - V10: Scale down reward để tránh Gradient Explosion
#   - Normalized input (mean=0, std=1) + scaled reward (nhỏ hơn)
#   - Giữ nguyên CÁN CÂN: rớt mạng đáng sợ hơn lãng phí
# ═════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")/.."
cd ai_model

echo "════════════════════════════════════════════════════════════════════"
echo "  V10 SCALED REWARD - TRÁNH GRADIENT EXPLOSION"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  Config:"
echo "    - ENTROPY_COEFF: 0.5 (Vùng Goldilocks)"
echo "    - KL_COEFF: 0.05"
echo "    - CQL_ALPHA: 1.0"
echo "    - Early Stopping: DISABLED"
echo "    - Epochs: 100 (full run)"
echo "    - StandardScaler: ENABLED"
echo "    - Reward SCALED DOWN: để phù hợp với normalized input"
echo ""
echo "  Reward Scale (V10 - chia 1000 để tránh gradient explosion):"
echo "    - overload_penalty (util > 0.85): 5.0 (từ 5000.0)"
echo "    - overload_penalty (util > 0.95): 20.0 (từ 20000.0)"
echo "    - wastage_penalty: 0.00025 (từ 0.25)"
echo "    - saving_bonus: 0.001 (từ 1.0)"
echo "    - throughput_reward: 0.01 (từ 10.0)"
echo ""
echo "  Press Ctrl+C to abort"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python train_actor_critic.py --phase all --epochs 100 --batch_size 64

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE - Check training_logs/ for results"
echo "════════════════════════════════════════════════════════════════════"