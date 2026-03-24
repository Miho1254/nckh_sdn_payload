#!/usr/bin/env python3
"""
Data Augmentation cho TFT-CQL Training
- Gaussian Noise: Thêm nhiễu nhỏ vào features
- Time-shifting: Dịch chuyển chuỗi thời gian
- SMOTE-like oversampling: Cân bằng các action classes
- Scenario-specific augmentation: Tạo thêm samples cho các scenario hiếm

Mục tiêu: Nhân bản 600 samples → 6000+ samples
"""

import numpy as np
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def add_gaussian_noise(X, y, noise_std=0.01, n_copies=5):
    """
    Thêm Gaussian noise vào features để tạo samples mới.
    
    Args:
        X: [N, seq_len, features] - Input sequences
        y: [N] - Labels
        noise_std: Độ lệch chuẩn của noise (nhỏ để giữ nguyên distribution)
        n_copies: Số bản sao với noise khác nhau
    
    Returns:
        Augmented X, y
    """
    X_aug = [X]
    y_aug = [y]
    
    for i in range(n_copies):
        noise = np.random.normal(0, noise_std, X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)
    
    return np.concatenate(X_aug), np.concatenate(y_aug)


def time_shift(X, y, shift_range=2):
    """
    Dịch chuyển chuỗi thời gian để tạo samples mới.
    
    Args:
        X: [N, seq_len, features]
        y: [N]
        shift_range: Số bước dịch chuyển tối đa
    
    Returns:
        Time-shifted X, y
    """
    X_shifted = []
    y_shifted = []
    
    for shift in range(-shift_range, shift_range + 1):
        if shift == 0:
            continue
        X_shift = np.roll(X, shift, axis=1)
        X_shifted.append(X_shift)
        y_shifted.append(y)
    
    return np.concatenate([X] + X_shifted), np.concatenate([y] + y_shifted)


def smote_oversample(X, y, target_samples_per_class=500):
    """
    Cân bằng các action classes bằng SMOTE-like oversampling.
    
    Args:
        X: [N, seq_len, features]
        y: [N] - Labels (action indices)
        target_samples_per_class: Số samples mục tiêu cho mỗi class
    
    Returns:
        Balanced X, y
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"  Original distribution: {dict(zip(unique_classes, counts))}")
    
    X_balanced = [X]
    y_balanced = [y]
    
    for cls in unique_classes:
        cls_mask = y == cls
        cls_count = np.sum(cls_mask)
        
        if cls_count < target_samples_per_class:
            # Oversample minority class
            n_needed = target_samples_per_class - cls_count
            cls_indices = np.where(cls_mask)[0]
            
            # Random oversampling with slight noise
            oversample_indices = np.random.choice(cls_indices, n_needed, replace=True)
            X_oversample = X[oversample_indices].copy()
            
            # Add small noise to make samples unique
            noise = np.random.normal(0, 0.005, X_oversample.shape)
            X_oversample += noise
            
            X_balanced.append(X_oversample)
            y_balanced.append(np.full(n_needed, cls))
    
    X_result = np.concatenate(X_balanced)
    y_result = np.concatenate(y_balanced)
    
    unique_classes, counts = np.unique(y_result, return_counts=True)
    print(f"  Balanced distribution: {dict(zip(unique_classes, counts))}")
    
    return X_result, y_result


def scenario_specific_augmentation(X, y, scenarios, n_augment=3):
    """
    Tạo thêm samples cho các scenario hiếm (DoS, Degradation).
    
    Args:
        X: [N, seq_len, features]
        y: [N]
        scenarios: [N] - Scenario labels
        n_augment: Số bản sao cho mỗi scenario hiếm
    
    Returns:
        Augmented X, y
    """
    X_aug = [X]
    y_aug = [y]
    
    # Augment rare scenarios more
    rare_scenarios = ['dos', 'degradation']
    
    for scenario in rare_scenarios:
        scenario_mask = scenarios == scenario
        if np.sum(scenario_mask) > 0:
            for _ in range(n_augment):
                X_scenario = X[scenario_mask].copy()
                noise = np.random.normal(0, 0.02, X_scenario.shape)
                X_scenario += noise
                X_aug.append(X_scenario)
                y_aug.append(y[scenario_mask])
    
    return np.concatenate(X_aug), np.concatenate(y_aug)


def augment_dataset(X, y, scenarios=None, target_samples=6000):
    """
    Pipeline augmentation hoàn chỉnh.
    
    Args:
        X: [N, seq_len, features]
        y: [N]
        scenarios: [N] - Optional scenario labels
        target_samples: Số samples mục tiêu
    
    Returns:
        Augmented X, y
    """
    print(f"[Augmentation] Starting with {len(X)} samples")
    
    # Step 1: Gaussian noise augmentation
    print("\n[1/4] Adding Gaussian noise...")
    X_noisy, y_noisy = add_gaussian_noise(X, y, noise_std=0.01, n_copies=3)
    print(f"  After noise: {len(X_noisy)} samples")
    
    # Step 2: Time shifting
    print("\n[2/4] Time shifting...")
    X_shifted, y_shifted = time_shift(X_noisy, y_noisy, shift_range=1)
    print(f"  After time-shift: {len(X_shifted)} samples")
    
    # Step 3: SMOTE-like oversampling
    print("\n[3/4] SMOTE oversampling...")
    target_per_class = target_samples // 3  # 3 actions
    X_balanced, y_balanced = smote_oversample(X_shifted, y_shifted, target_per_class)
    print(f"  After SMOTE: {len(X_balanced)} samples")
    
    # Step 4: Scenario-specific augmentation
    if scenarios is not None:
        print("\n[4/4] Scenario-specific augmentation...")
        X_final, y_final = scenario_specific_augmentation(X_balanced, y_balanced, scenarios)
        print(f"  After scenario aug: {len(X_final)} samples")
    else:
        X_final, y_final = X_balanced, y_balanced
    
    # Shuffle
    indices = np.random.permutation(len(X_final))
    X_final = X_final[indices]
    y_final = y_final[indices]
    
    print(f"\n[Augmentation] Final: {len(X_final)} samples")
    
    # Print action distribution
    unique, counts = np.unique(y_final, return_counts=True)
    print(f"[Augmentation] Action distribution:")
    for u, c in zip(unique, counts):
        print(f"  Action {int(u)}: {c} samples ({c/len(y_final)*100:.1f}%)")
    
    return X_final, y_final


def load_and_augment_data():
    """Load existing data và augment."""
    data_path = 'ai_model/processed_data'
    
    # Load original data
    X_path = os.path.join(data_path, 'X_v3.npy')
    y_path = os.path.join(data_path, 'y_v3.npy')
    scenarios_path = os.path.join(data_path, 'scenarios_v3.npy')
    
    if not os.path.exists(X_path):
        print(f"Error: {X_path} not found")
        return None, None
    
    X = np.load(X_path)
    y = np.load(y_path)
    scenarios = np.load(scenarios_path) if os.path.exists(scenarios_path) else None
    
    print(f"Loaded: X={X.shape}, y={y.shape}")
    
    # Augment
    X_aug, y_aug = augment_dataset(X, y, scenarios, target_samples=6000)
    
    return X_aug, y_aug


def save_augmented_data(X, y, output_path='ai_model/processed_data'):
    """Save augmented data."""
    os.makedirs(output_path, exist_ok=True)
    
    # Save with new names
    np.save(os.path.join(output_path, 'X_v3_aug.npy'), X)
    np.save(os.path.join(output_path, 'y_v3_aug.npy'), y)
    
    print(f"\nSaved augmented data to {output_path}")
    print(f"  X_v3_aug.npy: {X.shape}")
    print(f"  y_v3_aug.npy: {y.shape}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment training data')
    parser.add_argument('--target', type=int, default=6000, help='Target number of samples')
    parser.add_argument('--output', type=str, default='ai_model/processed_data', help='Output directory')
    
    args = parser.parse_args()
    
    # Load and augment
    X_aug, y_aug = load_and_augment_data()
    
    if X_aug is not None:
        save_augmented_data(X_aug, y_aug, args.output)
        
        # Verify
        print("\n[Verification]")
        print(f"  Total samples: {len(X_aug)}")
        print(f"  Sequence length: {X_aug.shape[1]}")
        print(f"  Features: {X_aug.shape[2]}")
        unique, counts = np.unique(y_aug, return_counts=True)
        print(f"  Action distribution: {dict(zip(unique, counts))}")