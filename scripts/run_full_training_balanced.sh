#!/bin/bash
# Run FULL TFT-CQL Training Pipeline with BALANCED dataset
# Includes ALL phases: Phase 0, 1, 2, 3

cd "$(dirname "$0")/.."

echo "========================================"
echo "  TFT-CQL FULL TRAINING - BALANCED DATA"
echo "========================================"
echo ""
echo "Dataset: X_v3_aug.npy, y_v3_aug.npy"
echo "Samples: 7200 (2400 per action - BALANCED)"
echo "Distribution: [33.33%, 33.33%, 33.33%]"
echo ""

# Check data
python3 -c "
import numpy as np
X = np.load('ai_model/processed_data/X_v3_aug.npy')
y = np.load('ai_model/processed_data/y_v3_aug.npy')
print(f'Data loaded: X={X.shape}, y={y.shape}')
unique, counts = np.unique(y, return_counts=True)
print(f'Action distribution:')
for u, c in zip(unique, counts):
    print(f'  Action {int(u)}: {c} ({c/len(y)*100:.1f}%)')
"

echo ""
echo "========================================"
echo "  RUNNING ALL PHASES"
echo "========================================"
echo ""
echo "Phase 0: Capacity Ratio Supervised Pretraining"
echo "Phase 1: Encoder Pretraining (Forecast)"
echo "Phase 2: Offline CQL Actor-Critic Training"
echo "Phase 3: Constraint Tuning"
echo ""

# Run full training with --augment flag
python3 ai_model/train_actor_critic.py \
    --phase all \
    --epochs 100 \
    --batch_size 64 \
    --hidden_size 64 \
    --augment

echo ""
echo "========================================"
echo "  TRAINING COMPLETED"
echo "========================================"
echo ""
echo "Checkpoints saved in: ai_model/checkpoints/"
echo "  - encoder_best.pth (Phase 0/1)"
echo "  - tft_ac_best.pth (Best model)"
echo "  - tft_ac_final.pth (Final model)"
echo ""
echo "Logs saved in: ai_model/training_logs/"