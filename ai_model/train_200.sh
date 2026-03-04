#!/bin/bash
# Script training cải tiến - TFT-DQN với 200 epochs
# Các thay đổi:
# 1. Tăng số epochs: 100 -> 200
# 2. Tăng hệ số phạt balance: 3.0
# 3. Tăng epsilon decay: 0.9999

cd /home/miho/Downloads/nckh/nckh_sdn/ai_model

echo "============================================================"
echo "       NCKH SDN - TRAINING MODULE (TFT-DQN) - 200 EPOCHS"
echo "============================================================"
echo "Episodes: 200 (tăng từ 100)"
echo "Balance penalty: 3.0"
echo "Epsilon decay: 0.9999"
echo "============================================================"

python3 train.py
