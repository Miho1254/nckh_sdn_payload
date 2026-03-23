#!/usr/bin/env python3
"""
Train TFT-CQL V3 - No Phase 0
Bỏ supervised pretraining để tránh "học vẹt"
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ai_model'))

from cql_agent import CQLAgent
from sdn_env_v2 import SDN_Offline_Env_V2
from config import (
    SEQUENCE_LENGTH, NUM_ACTIONS, CAPACITY_RATIOS,
    CQL_ALPHA, ENTROPY_COEFF, FORECAST_LOSS_WEIGHT, CONSTRAINT_WEIGHTS
)

DATA_DIR = 'ai_model/processed_data'
CKPT_DIR = 'ai_model/checkpoints'
LOG_DIR = 'ai_model/training_logs'


def load_augmented_data():
    """Load augmented data."""
    X_path = os.path.join(DATA_DIR, 'X_v3_aug.npy')
    y_path = os.path.join(DATA_DIR, 'y_v3_aug.npy')
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    print(f"[*] Loaded augmented data: X={X.shape}, y={y.shape}")
    
    # Action distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"[*] Action distribution:")
    for u, c in zip(unique, counts):
        print(f"    Action {int(u)}: {c} ({c/len(y)*100:.1f}%)")
    
    metadata = {'version': 'v3_aug', 'num_features': X.shape[2]}
    return X, y, metadata


def train_phase2_only(epochs=60, batch_size=64):
    """Train chỉ Phase 2 (Offline RL) - bỏ Phase 0."""
    
    print("\n" + "="*60)
    print("  TFT-CQL V3 Training - NO PHASE 0")
    print("="*60)
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Critic LR: 5e-5 (same as Actor)")
    print(f"  Target Entropy Ratio: 0.9")
    print(f"  CQL Alpha: 1.0")
    print("="*60 + "\n")
    
    # Load data
    X, y, metadata = load_augmented_data()
    num_features = X.shape[2]
    
    # Init agent with new hyperparameters
    agent = CQLAgent(
        input_size=num_features,
        seq_len=SEQUENCE_LENGTH,
        hidden_size=64,
        num_actions=NUM_ACTIONS,
        actor_lr=5e-5,       # Cân bằng với Critic
        critic_lr=5e-5,      # Giảm từ 3e-4 xuống 5e-5
        cql_alpha=1.0,       # Giảm từ 10.0 xuống 1.0
        entropy_coeff=0.5,   # Giảm từ 5.0 xuống 0.5
        kl_coeff=0.1,
        target_entropy_ratio=0.9,  # Tăng từ 0.4 lên 0.9
        forecast_loss_weight=FORECAST_LOSS_WEIGHT,
        constraint_weights=CONSTRAINT_WEIGHTS,
        capacity_prior=list(CAPACITY_RATIOS[:NUM_ACTIONS]),
    )
    
    # Init env
    env = SDN_Offline_Env_V2(
        os.path.join(DATA_DIR, 'X_v3_aug.npy'),
        os.path.join(DATA_DIR, 'y_v3_aug.npy'),
        mode='train', metadata=metadata
    )
    
    # Training loop
    all_metrics = []
    best_score = -float('inf')
    
    print("\n[*] Starting Phase 2: Offline RL Training...")
    print("[*] Skipping Phase 0 (supervised pretraining)")
    print("[*] Skipping Phase 1 (encoder pretraining)")
    print("")
    
    for epoch in range(epochs):
        epoch_rewards = []
        epoch_losses = []
        action_counts = np.zeros(NUM_ACTIONS)
        
        # Shuffle data
        indices = np.random.permutation(len(X) - 1)
        
        for i in range(0, len(X) - batch_size - 1, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            states = X[batch_idx]
            next_states = X[batch_idx + 1]
            actions = y[batch_idx]
            
            # Get rewards from environment
            rewards = []
            dones = []
            info_batch = []
            
            for j, idx in enumerate(batch_idx):
                env.current_step = idx
                action = int(actions[j])
                _, reward, done, info = env.step(action)
                rewards.append(reward)
                dones.append(done)
                info_batch.append(info)
                epoch_rewards.append(reward)
                action_counts[action] += 1
            
            # Convert to tensors
            rewards_t = np.array(rewards, dtype=np.float32)
            dones_t = np.array(dones, dtype=np.float32)
            
            # Train step - returns dict
            loss_dict = agent.train_step(states, next_states, rewards_t, actions, dones_t, info_batch)
            epoch_losses.append(loss_dict['total_critic_loss'])
        
        # Metrics
        avg_reward = np.mean(epoch_rewards)
        avg_loss = np.mean(epoch_losses)
        action_dist = action_counts / np.sum(action_counts)
        
        # Entropy
        eps = 1e-8
        entropy = -np.sum(action_dist * np.log(action_dist + eps))
        max_entropy = np.log(NUM_ACTIONS)
        entropy_ratio = entropy / max_entropy
        
        metrics = {
            'epoch': epoch,
            'avg_reward': avg_reward,
            'avg_loss': avg_loss,
            'entropy_ratio': entropy_ratio,
            'action_dist': action_dist.tolist()
        }
        all_metrics.append(metrics)
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: reward={avg_reward:.4f}, loss={avg_loss:.4f}, "
                  f"entropy={entropy_ratio:.2f}, actions=[{action_dist[0]:.2f}, {action_dist[1]:.2f}, {action_dist[2]:.2f}]")
        
        # Save best
        if avg_reward > best_score:
            best_score = avg_reward
            agent.save_checkpoint(os.path.join(CKPT_DIR, 'tft_ac_best.pth'))
    
    # Save final
    agent.save_checkpoint(os.path.join(CKPT_DIR, 'tft_ac_final.pth'))
    
    print(f"\n[*] Training completed!")
    print(f"[*] Best reward: {best_score:.4f}")
    print(f"[*] Final action distribution: [{action_dist[0]:.2f}, {action_dist[1]:.2f}, {action_dist[2]:.2f}]")
    
    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    train_phase2_only(epochs=args.epochs, batch_size=args.batch_size)
