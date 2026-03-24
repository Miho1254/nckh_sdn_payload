#!/usr/bin/env python3
"""
Data Augmentation V2 - Sửa lỗi scenario mismatch
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_gaussian_noise(X, y, noise_std=0.01, n_copies=3):
    """Thêm Gaussian noise vào features."""
    X_aug = [X]
    y_aug = [y]
    
    for i in range(n_copies):
        noise = np.random.normal(0, noise_std, X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)
    
    return np.concatenate(X_aug), np.concatenate(y_aug)


def time_shift(X, y, shift_range=1):
    """Dịch chuyển chuỗi thời gian."""
    X_shifted = [X]
    y_shifted = [y]
    
    for shift in range(-shift_range, shift_range + 1):
        if shift == 0:
            continue
        X_shift = np.roll(X, shift, axis=1)
        X_shifted.append(X_shift)
        y_shifted.append(y)
    
    return np.concatenate(X_shifted), np.concatenate(y_shifted)


def smote_oversample(X, y, target_per_class=2000):
    """Cân bằng các action classes."""
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"  Original distribution: {dict(zip(unique_classes, counts))}")
    
    X_balanced = [X]
    y_balanced = [y]
    
    for cls in unique_classes:
        cls_mask = y == cls
        cls_count = np.sum(cls_mask)
        
        if cls_count < target_per_class:
            n_needed = target_per_class - cls_count
            cls_indices = np.where(cls_mask)[0]
            
            oversample_indices = np.random.choice(cls_indices, n_needed, replace=True)
            X_oversample = X[oversample_indices].copy()
            
            noise = np.random.normal(0, 0.005, X_oversample.shape)
            X_oversample += noise
            
            X_balanced.append(X_oversample)
            y_balanced.append(np.full(n_needed, cls))
    
    X_result = np.concatenate(X_balanced)
    y_result = np.concatenate(y_balanced)
    
    unique_classes, counts = np.unique(y_result, return_counts=True)
    print(f"  Balanced distribution: {dict(zip(unique_classes, counts))}")
    
    return X_result, y_result


def augment_dataset(X, y, target_samples=6000):
    """Pipeline augmentation hoàn chỉnh."""
    print(f"[Augmentation] Starting with {len(X)} samples")
    
    # Step 1: Gaussian noise
    print("\n[1/3] Adding Gaussian noise...")
    X_noisy, y_noisy = add_gaussian_noise(X, y, noise_std=0.01, n_copies=3)
    print(f"  After noise: {len(X_noisy)} samples")
    
    # Step 2: Time shifting
    print("\n[2/3] Time shifting...")
    X_shifted, y_shifted = time_shift(X_noisy, y_noisy, shift_range=1)
    print(f"  After time-shift: {len(X_shifted)} samples")
    
    # Step 3: SMOTE
    print("\n[3/3] SMOTE oversampling...")
    target_per_class = target_samples // 3
    X_balanced, y_balanced = smote_oversample(X_shifted, y_shifted, target_per_class)
    print(f"  After SMOTE: {len(X_balanced)} samples")
    
    # Shuffle
    indices = np.random.permutation(len(X_balanced))
    X_final = X_balanced[indices]
    y_final = y_balanced[indices]
    
    print(f"\n[Augmentation] Final: {len(X_final)} samples")
    
    unique, counts = np.unique(y_final, return_counts=True)
    print(f"[Augmentation] Action distribution:")
    for u, c in zip(unique, counts):
        print(f"  Action {int(u)}: {c} samples ({c/len(y_final)*100:.1f}%)")
    
    return X_final, y_final


def main():
    data_path = 'ai_model/processed_data'
    
    X = np.load(os.path.join(data_path, 'X_v3.npy'))
    y = np.load(os.path.join(data_path, 'y_v3.npy'))
    
    print(f"Loaded: X={X.shape}, y={y.shape}")
    
    # Augment
    X_aug, y_aug = augment_dataset(X, y, target_samples=6000)
    
    # Save
    np.save(os.path.join(data_path, 'X_v3_aug.npy'), X_aug)
    np.save(os.path.join(data_path, 'y_v3_aug.npy'), y_aug)
    
    print(f"\nSaved: X_v3_aug.npy {X_aug.shape}, y_v3_aug.npy {y_aug.shape}")


if __name__ == '__main__':
    main()
