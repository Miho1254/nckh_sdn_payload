#!/bin/bash
# Train TFT-CQL V3 - Bỏ Phase 0, train trực tiếp từ Phase 2
# Mục tiêu: Tránh "học vẹt" từ prior distribution

set -e

echo "=============================================="
echo "  TFT-CQL Training V3 - No Phase 0"
echo "=============================================="

cd /home/miho/Downloads/nckh/nckh_sdn

# Sử dụng augmented data
export DATA_PATH="ai_model/processed_data"
export X_FILE="X_v3_aug.npy"
export Y_FILE="y_v3_aug.npy"

echo ""
echo "[Config] Using augmented data:"
echo "  X: ${X_FILE}"
echo "  y: ${Y_FILE}"
echo ""

# Kiểm tra GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU] Checking GPU..."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    export CUDA_VISIBLE_DEVICES=0
fi

echo ""
echo "[Training] Starting..."
echo "  - Bỏ Phase 0 (supervised pretraining)"
echo "  - Train trực tiếp Phase 2 (Offline RL)"
echo "  - Target Entropy: 0.9"
echo "  - Critic LR = Actor LR = 5e-5"
echo "  - CQL Alpha: 1.0"
echo ""

python3 ai_model/train_actor_critic.py \
    --phase 2 \
    --epochs 60 \
    --batch_size 64 \
    --skip_phase0 \
    --use_augmented_data

echo ""
echo "[Done] Training completed!"