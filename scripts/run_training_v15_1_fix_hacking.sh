#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# V15.1: FIX OVERESTIMATION BIAS (REWARD HACKING h5)
# ═══════════════════════════════════════════════════════════════
# Thay đổi so với V15:
#   - CQL_ALPHA: 0.1 -> 0.5 (Tăng tính bảo thủ để tránh ảo giác)
#   - saving_bonus h5: max 3.0 -> max 1.0 (Giảm tính hấp dẫn quá mức)
#   - saving_bonus h7: max 1.5 -> max 0.5
#   - saving_bonus h8: 0.0 -> 0.2 (khi util < 0.8) (Cấp lương cứng)

echo "═══════════════════════════════════════════════════════"
echo "  V15.1: FIX OVERESTIMATION BIAS - Training"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Changes from V15:"
echo "    - CQL_ALPHA: 0.1 -> 0.5"
echo "    - saving_bonus h5: 1.0 * (1 - u)"
echo "    - saving_bonus h7: 0.5 * (1 - u)"
echo "    - saving_bonus h8: +0.2 (if u < 0.8)"
echo "    - Entropy decay: 2.0 -> 0.5 (giữ nguyên V15)"
echo ""

cd "$(dirname "$0")/.."

# Run test first
echo "[*] Running reward V15.1 tests..."
python3 /tmp/test_reward_v15_1.py
if [ $? -ne 0 ]; then
    echo "[!] WARNING: Some reward tests failed. Proceeding anyway..."
fi

echo ""
echo "[*] Starting V15.1 training..."
python3 ai_model/train_actor_critic.py \
    --phase all \
    --epochs 100 \
    --batch_size 64 \
    --hidden_size 64

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  V15.1 Training Complete!"
echo "═══════════════════════════════════════════════════════"
echo "  Check action distribution - EXPECTED:"
echo "    Distribution nên cân bằng hơn (VD: h5 ~30%, h7 ~30%, h8 ~40%)"
echo "  FAIL IF: h5 = 100% HOẶC h8 = 100%"
echo ""
