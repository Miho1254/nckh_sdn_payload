#!/bin/bash
# Script training V2 - TFT-DQN với Simulated Load Tracking
# Các thay đổi chính so với V1:
# 1. Reward Function V2: Dùng simulated load thay vì tải tĩnh
# 2. Balance penalty: phạt mũ (std^2 * 8.0)
# 3. Epsilon decay: 0.985 per epoch (không per step)
# 4. Gamma: 0.95 (focus ngắn hạn SDN)
# 5. LR: 3e-4 (tăng từ 1e-5)
# 6. Bỏ reward scaling /100

cd /home/miho/Downloads/nckh/nckh_sdn/ai_model

echo "============================================================"
echo "       NCKH SDN - TRAINING V2 (TFT-DQN + Simulated Load)"
echo "============================================================"
echo "Episodes: 200"
echo "Balance penalty: exponential (std^2 * 8.0)"
echo "Epsilon decay: 0.985 per epoch"
echo "Gamma: 0.95 | LR: 3e-4"
echo "============================================================"

python3 train.py
