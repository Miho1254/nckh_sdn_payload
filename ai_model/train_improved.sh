#!/bin/bash
# Script training cải tiến - TFT-DQN với 100 epochs
# Các thay đổi:
# 1. Tăng số epochs: 50 -> 100
# 2. Tăng hệ số phạt balance: 1.5 -> 3.0
# 3. Tăng epsilon decay: 0.9997 -> 0.9999

cd /home/miho/Downloads/nckh/nckh_sdn/ai_model

echo "============================================================"
echo "       NCKH SDN - TRAINING MODULE (TFT-DQN) - CẢI TIẾN"
echo "============================================================"
echo "Episodes: 100 (tăng từ 50)"
echo "Balance penalty: 3.0 (tăng từ 1.5)"
echo "Epsilon decay: 0.9999 (tăng từ 0.9997)"
echo "============================================================"

python3 train.py
