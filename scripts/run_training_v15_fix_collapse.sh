#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# V15: FIX POLICY COLLAPSE
# ═══════════════════════════════════════════════════════════════
# Thay đổi so với V14:
#   - CQL_ALPHA: 1.0 -> 0.1 (giảm bảo thủ CQL)
#   - ENTROPY_COEFF: 0.5 -> 2.0 (initial, decay về 0.5 sau 80% epochs)
#   - overload_penalty: step function max=100 -> smooth quadratic coeff=50.0
#   - saving_bonus: fixed 1.0 -> headroom-scaled max 3.0
#   - wastage_penalty: 0.015 -> 0.01
#   - packet_loss threshold: 0.01 -> 0.02

echo "═══════════════════════════════════════════════════════"
echo "  V15: FIX POLICY COLLAPSE - Training"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Changes from V14:"
echo "    - CQL_ALPHA: 1.0 -> 0.1"
echo "    - ENTROPY_COEFF: 2.0 (warmup) -> 0.5 (decay)"
echo "    - overload_penalty: 50.0 * (u - 0.8)^2 (smooth)"
echo "    - saving_bonus h5: 3.0 * (1 - u) (headroom-scaled)"
echo "    - saving_bonus h7: 1.5 * (1 - u)"
echo "    - Crossover penalty > bonus at u ≈ 0.89"
echo ""

cd "$(dirname "$0")/.."

# Run test first
echo "[*] Running reward V15 tests..."
python3 /tmp/test_reward_v15.py
if [ $? -ne 0 ]; then
    echo "[!] WARNING: Some reward tests failed. Proceeding anyway..."
fi

echo ""
echo "[*] Starting V15 training..."
python3 ai_model/train_actor_critic.py \
    --phase all \
    --epochs 100 \
    --batch_size 64 \
    --hidden_size 64

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  V15 Training Complete!"
echo "═══════════════════════════════════════════════════════"
echo "  Check action distribution - EXPECTED:"
echo "    h8 < 80%, h5 > 5%, h7 > 10%"
echo "  FAIL IF: h8 = 100% (policy collapse not fixed)"
echo ""
