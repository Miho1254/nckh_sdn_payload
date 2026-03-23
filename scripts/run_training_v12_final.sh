#!/bin/bash
# ═════════════════════════════════════════════════════════════════════
# V12 "PHONG TƯỚC CHO H5" TRAINING SCRIPT
# ═════════════════════════════════════════════════════════════════════
# Thay đổi từ V11:
#   - V12: Tăng wastage_penalty (0.005 -> 0.02, x4) - Ép h8 nhả tài nguyên
#   - V12: Tăng saving_bonus h5 (0.001 -> 2.0) - Thưởng hậu cho h5 thành công
#   - Mục tiêu: Cân bằng h5, h7, h8 theo điều kiện mạng
# ═════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")/.."
cd ai_model

echo "════════════════════════════════════════════════════════════════════"
echo "  V12 PHONG TƯỚC CHO H5"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  Reward V12:"
echo "    - overload_penalty (util > 0.85): 5.0"
echo "    - overload_penalty (util > 0.95): 20.0"
echo "    - wastage_penalty: 0.02 (V12: tăng x4)"
echo "    - saving_bonus h5: 2.0 (V12: tăng mạnh!)"
echo "    - saving_bonus h7: 1.0 (V12: tăng mạnh!)"
echo ""
echo "  Cán cân kinh tế V12:"
echo "    - h5 thành công: +2.0 bonus"
echo "    - h8 lãng phí (92MB dư): -1.84 penalty"
echo "    - AI sẽ chọn h5 khi traffic nhẹ"
echo ""
echo "  Press Ctrl+C to abort"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python train_actor_critic.py --phase all --epochs 100 --batch_size 64

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE - Check training_logs/ for results"
echo "════════════════════════════════════════════════════════════════════"