#!/bin/bash
# ═════════════════════════════════════════════════════════════════════
# V13 "CÂN BẰNG TINH TẾ" TRAINING SCRIPT
# ═════════════════════════════════════════════════════════════════════
# Thay đổi từ V12:
#   - V13: Giảm entropy (2.0 -> 0.5) - ổn định hơn
#   - V13: Giảm saving_bonus (2.0 -> 0.5) - không phải mồi nhử
#   - V13: Giảm wastage (0.02 -> 0.01) - h8 có thể tồn tại
# ═════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")/.."
cd ai_model

echo "════════════════════════════════════════════════════════════════════"
echo "  V13 CÂN BẰNG TINH TẾ"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  Config V13:"
echo "    - ENTROPY_COEFF: 0.5 (V13: giảm từ 2.0)"
echo "    - overload_penalty: 5.0 / 20.0"
echo "    - wastage_penalty: 0.01 (V13: giảm từ 0.02)"
echo "    - saving_bonus h5: 0.5 (V13: giảm từ 2.0)"
echo "    - saving_bonus h7: 0.2 (V13: giảm từ 1.0)"
echo ""
echo "  Cán cân kinh tế V13:"
echo "    - h8 lãng phí: ~0.9 điểm (hơi đau)"
echo "    - h5 tiết kiệm: 0.5 điểm (an ủi)"
echo "    - h5 rớt mạng: 5-20 điểm (hủy diệt)"
echo ""
echo "  AI sẽ tự động:"
echo "    - Normal traffic -> ưu tiên h5 (vì h8 mất 0.9)"
echo "    - Overload/DOS -> chọn h8 (vì rủi ro h5 quá lớn)"
echo ""
echo "  Press Ctrl+C to abort"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python train_actor_critic.py --phase all --epochs 100 --batch_size 64

echo ""
echo "══════��═════════════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE - Check training_logs/ for results"
echo "════════════════════════════════════════════════════════════════════"