#!/bin/bash
# Run TFT-CQL V3 Training with BALANCED dataset
# Dataset: 7200 samples, 33.33% each action

cd "$(dirname "$0")/.."

echo "========================================"
echo "  TFT-CQL V3 Training - BALANCED DATA"
echo "========================================"
echo ""
echo "Dataset: X_v3_aug.npy, y_v3_aug.npy"
echo "Samples: 7200 (2400 per action)"
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
echo "Starting training..."
echo ""

# Run training
python3 scripts/train_v3_no_phase0.py --epochs 100 --batch_size 64

echo ""
echo "Training completed!"