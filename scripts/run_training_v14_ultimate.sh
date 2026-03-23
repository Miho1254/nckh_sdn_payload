#!/bin/bash
# ═════════════════════════════════════════════════════════════════════
# V14 "THE ULTIMATE EQUILIBRIUM" TRAINING SCRIPT
# ═════════════════════════════════════════════════════════════════════
# Thay đổi từ V13:
#   - V14: saving_bonus h5 (0.5 -> 1.0) - đủ lớn để dám đối mặt rủi ro
#   - V14: wastage_penalty (0.01 -> 0.015) - cân bằng với bonus
#   - V14: KL_COEFF (0.05 -> 0.01) - giảm bảo thủ
# ═════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")/.."
cd ai_model

echo "════════════════════════════════════════════════════════════════════"
echo "  V14 THE ULTIMATE EQUILIBRIUM"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  Config V14:"
echo "    - ENTROPY_COEFF: 0.5"
echo "    - KL_COEFF: 0.01 (V14: giảm từ 0.05)"
echo "    - overload_penalty: 5.0 / 20.0"
echo "    - wastage_penalty: 0.015 (V14: tăng từ 0.01)"
echo "    - saving_bonus h5: 1.0 (V14: tăng từ 0.5)"
echo "    - saving_bonus h7: 0.3 (V14: tăng từ 0.2)"
echo ""
echo "  Cán cân kinh tế V14:"
echo "    - h8 lãng phí: ~1.35 điểm (đau)"
echo "    - h7 lãng phí: ~0.63 điểm (nhẹ)"
echo "    - h5 tiết kiệm: +1.0 điểm (thưởng)"
echo "    - h5 rớt mạng: -5 đến -20 điểm (hủy diệt)"
echo ""
echo "  AI sẽ tự động:"
echo "    - Normal traffic -> h5 (lãi 1.0, h7 lỗ 0.63)"
echo "    - High traffic -> h7 (an toàn hơn h5)"
echo "    - DOS -> h8 (duy trì throughput)"
echo ""
echo "  Press Ctrl+C to abort"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python train_actor_critic.py --phase all --epochs 100 --batch_size 64

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE - Check training_logs/ for results"
echo "════════════════════════════════════════════════════════════════════"