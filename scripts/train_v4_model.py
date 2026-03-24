#!/usr/bin/env python3
"""
Quick Train TFT-AC V4 - Capacity-Weighted Model
===============================================

Train model with fixed V4 dataset (capacity-weighted distribution).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, '/work')
from ai_model.tft_ac_net import TFT_ActorCritic_Model


def train_v4_model(
    n_epochs=50,
    batch_size=64,
    lr=5e-5,
    cql_alpha=1.0,
    entropy_coeff=0.5,
    kl_coeff=0.01,
    save_dir='ai_model/checkpoints'
):
    """Train TFT-AC model with V4 capacity-weighted data."""
    
    print("=" * 60)
    print("TRAINING TFT-AC V4 - CAPACITY-WEIGHTED MODEL")
    print("=" * 60)
    
    # Load V4 data
    X = np.load('ai_model/processed_data/X_v4.npy')
    y = np.load('ai_model/processed_data/y_v4.npy')
    
    print(f"Loaded V4 dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Action distribution: {np.bincount(y)}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize model
    model = TFT_ActorCritic_Model(
        input_size=44,
        seq_len=5,
        hidden_size=64,
        num_actions=3
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    
    # Capacity prior (for KL penalty)
    capacity_prior = torch.tensor([0.0625, 0.3125, 0.625], dtype=torch.float32).to(device)
    
    # Training loop
    n_samples = len(X)
    n_batches = n_samples // batch_size
    
    best_score = -float('inf')
    history = []
    
    print(f"\nTraining for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        model.train()
        
        # Shuffle
        perm = torch.randperm(n_samples)
        X_shuffled = X_tensor[perm]
        y_shuffled = y_tensor[perm]
        
        epoch_loss = 0
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        epoch_entropy = 0
        correct = 0
        action_counts = torch.zeros(3, device=device)
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]
            
            # Forward pass - model returns 5 outputs
            # Output 0: actor_logits [batch, 3]
            # Output 1: critic_value [batch]
            # Output 2: safety_value [batch]
            # Output 3: forecast [batch, 44]
            # Output 4: safety_probs [batch, 3]
            outputs = model(batch_X)
            actor_logits = outputs[0]  # [batch, 3]
            critic_value = outputs[1]  # [batch]
            safety_probs = outputs[4]  # [batch, 3]
            
            # Use safety_probs as policy (sigmoid output)
            policy = safety_probs
            
            # Actor loss: negative log probability of correct action
            log_probs = torch.log(policy + 1e-8)
            actor_loss = F.nll_loss(log_probs, batch_y)
            
            # Critic loss: MSE between critic value and rewards
            # Use action labels as proxy for rewards (higher capacity = higher reward)
            rewards = batch_y.float() / 2.0  # Normalize: 0->0, 1->0.5, 2->1
            critic_loss = F.mse_loss(critic_value.squeeze(), rewards)
            
            # CQL penalty: penalize Q-values for non-taken actions
            cql_loss = cql_alpha * (actor_logits.logsumexp(dim=1).mean() - actor_logits.mean())
            
            # Entropy bonus
            entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=1).mean()
            entropy_loss = -entropy_coeff * entropy
            
            # KL divergence from capacity prior
            kl_div = F.kl_div(
                torch.log(policy + 1e-8),
                capacity_prior.expand(policy.shape[0], -1),
                reduction='batchmean'
            )
            kl_loss = kl_coeff * kl_div
            
            # Total loss
            loss = actor_loss + critic_loss + cql_loss + entropy_loss + kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_actor_loss += actor_loss.item()
            epoch_critic_loss += critic_loss.item()
            epoch_entropy += entropy.item()
            
            # Accuracy
            pred = policy.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            
            # Action distribution
            action_counts += pred.bincount(minlength=3).float()
        
        # Epoch metrics
        avg_loss = epoch_loss / n_batches
        avg_actor_loss = epoch_actor_loss / n_batches
        avg_critic_loss = epoch_critic_loss / n_batches
        avg_entropy = epoch_entropy / n_batches
        accuracy = correct / n_samples
        
        # Action distribution
        action_dist = action_counts / action_counts.sum()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor[:1000])
            policy = outputs[4]  # safety_probs
            eval_dist = policy.mean(dim=0).cpu().numpy()
        
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy,
            'accuracy': accuracy,
            'action_dist': action_dist.cpu().numpy().tolist(),
            'eval_dist': eval_dist.tolist()
        })
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f} actor={avg_actor_loss:.4f} "
                  f"critic={avg_critic_loss:.4f} entropy={avg_entropy:.4f} acc={accuracy:.2%}")
            print(f"  Action dist: [{action_dist[0]:.1%}, {action_dist[1]:.1%}, {action_dist[2]:.1%}]")
            print(f"  Eval dist:   [{eval_dist[0]:.1%}, {eval_dist[1]:.1%}, {eval_dist[2]:.1%}]")
        
        # Save best model
        if accuracy > best_score:
            best_score = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'accuracy': accuracy,
                'action_dist': action_dist.cpu().numpy().tolist()
            }, os.path.join(save_dir, 'tft_ac_v4_best.pth'))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': n_epochs,
        'accuracy': accuracy,
        'action_dist': action_dist.cpu().numpy().tolist()
    }, os.path.join(save_dir, 'tft_ac_v4_final.pth'))
    
    # Save history
    with open(os.path.join(save_dir, 'training_history_v4.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best accuracy: {best_score:.2%}")
    print(f"Final action distribution: [{action_dist[0]:.1%}, {action_dist[1]:.1%}, {action_dist[2]:.1%}]")
    print(f"Saved to: {save_dir}")
    print(f"{'='*60}")
    
    return model, history


if __name__ == '__main__':
    train_v4_model(
        n_epochs=50,
        batch_size=64,
        lr=5e-5,
        cql_alpha=1.0,
        entropy_coeff=0.5,
        kl_coeff=0.01
    )