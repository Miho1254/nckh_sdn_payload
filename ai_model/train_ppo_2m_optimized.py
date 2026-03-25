#!/usr/bin/env python3
"""
PPO Training Script - Optimized with SubprocVecEnv and CPU-only

Optimizations:
- SubprocVecEnv: 4 parallel environments for faster sampling
- CPU-only: PPO is inefficient on GPU for simple environments
- Larger batch size: Better CPU utilization

Chạy: python ai_model/train_ppo_2m_optimized.py
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
import torch
from multiprocessing import cpu_count

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdn_sim_env import SDNLoadBalancerEnv, GoldenHourEnv, VideoConferenceEnv, HardwareDegradationEnv, LowRateDosEnv, make_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


class DetailedRewardLogger(BaseCallback):
    """Callback để log chi tiết reward và action distribution."""
    
    def __init__(self, verbose=1, log_freq=5000):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = np.zeros(3)
        self.total_steps = 0
        self.log_freq = log_freq
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self):
        self.total_steps += 1
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Track action distribution
        actions = self.locals['actions'][0]
        if isinstance(actions, np.ndarray):
            actions_clipped = np.clip(actions, 0, None)
            if actions_clipped.sum() > 0:
                actions_normalized = actions_clipped / actions_clipped.sum()
            else:
                actions_normalized = np.array([0.33, 0.33, 0.34])
            self.action_counts += actions_normalized
        else:
            self.action_counts[int(actions)] += 1
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        if self.total_steps % self.log_freq == 0 and self.verbose > 0:
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                total = self.action_counts.sum()
                if total > 0:
                    action_dist = self.action_counts / total
                else:
                    action_dist = np.array([0.33, 0.33, 0.34])
                print(f"  Step {self.total_steps}: avg_reward={avg_reward:.2f} | actions: h5={action_dist[0]*100:.1f}%, h7={action_dist[1]*100:.1f}%, h8={action_dist[2]*100:.1f}%")
        
        return True


def make_parallel_envs(env_id, num_envs):
    """Create multiple parallel environments."""
    def make_env_fn():
        return make_env(env_id)
    
    return SubprocVecEnv([make_env_fn for _ in range(num_envs)])


def train_ppo_2m_optimized():
    """Train PPO với 2M samples - Optimized version."""
    
    print("="*70)
    print("  PPO TRAINING - 2M SAMPLES OPTIMIZED")
    print("="*70)
    print("\n[*] Optimizations:")
    print("    - SubprocVecEnv: 4 parallel environments")
    print("    - CPU-only: PPO is inefficient on GPU for simple envs")
    print("    - Larger batch size: Better CPU utilization")
    print()
    
    # Create directories
    os.makedirs("ai_model/models", exist_ok=True)
    os.makedirs("ai_model/logs", exist_ok=True)
    
    # Training configuration
    total_timesteps = 2_000_000
    num_envs = 4  # Number of parallel environments
    
    # Scenarios to train on
    scenarios = [
        ('SDN-v0', 0.3),
        ('GoldenHour-v0', 0.25),
        ('VideoConference-v0', 0.2),
        ('HardwareDegradation-v0', 0.15),
        ('LowRateDoS-v0', 0.1),
    ]
    
    print("[*] Training scenarios:")
    for scenario, weight in scenarios:
        print(f"    {scenario}: {weight*100:.0f}%")
    print()
    
    # Create parallel environments
    print(f"[*] Creating {num_envs} parallel environments...")
    env = make_parallel_envs('SDN-v0', num_envs)
    
    # PPO Hyperparameters - Optimized for CPU
    print("[*] Initializing PPO model (CPU-only)...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,        # Higher LR for faster learning
        n_steps=2048,              # Steps per env per update
        batch_size=64,             # Batch size
        n_epochs=10,               # PPO epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,             # Higher entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),  # Smaller network for CPU
        ),
        device='cpu',              # Force CPU
        verbose=1,
    )
    
    print(f"    Policy network: [64, 64]")
    print(f"    Learning rate: {model.learning_rate}")
    print(f"    N steps: {model.n_steps}")
    print(f"    Batch size: {model.batch_size}")
    print(f"    Device: CPU")
    print(f"    Num envs: {num_envs}")
    print()
    
    # Callbacks
    reward_logger = DetailedRewardLogger(verbose=1, log_freq=5000)
    
    # Training
    print("[*] Starting training...")
    print(f"[*] Total timesteps: {total_timesteps:,}")
    print(f"[*] Effective batch size: {model.n_steps * num_envs}")
    print(f"[*] Estimated time: {total_timesteps/(num_envs * 1000):.0f}-{total_timesteps/(num_envs * 500):.0f} minutes")
    print()
    
    start_time = time.time()
    
    # Multi-scenario training
    timesteps_per_scenario = [int(total_timesteps * weight) for _, weight in scenarios]
    
    for i, (scenario, weight) in enumerate(scenarios):
        scenario_timesteps = timesteps_per_scenario[i]
        print(f"\n{'='*60}")
        print(f"  TRAINING ON: {scenario} ({scenario_timesteps:,} steps)")
        print(f"{'='*60}")
        
        # Close current env
        env.close()
        
        # Create new parallel envs for this scenario
        env = make_parallel_envs(scenario, num_envs)
        model.set_env(env)
        
        # Train
        model.learn(
            total_timesteps=scenario_timesteps,
            callback=[reward_logger],
            reset_num_timesteps=False,
        )
        
        print(f"\n[*] Completed {scenario}")
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"[*] Total training time: {training_time/60:.1f} minutes")
    print(f"[*] Total timesteps: {reward_logger.total_steps:,}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"ai_model/models/ppo_2m_optimized_{timestamp}.zip"
    model.save(save_path)
    print(f"[✓] Model saved: {save_path}")
    
    # Final evaluation
    print("\n[*] Final evaluation on all scenarios:")
    print("-" * 60)
    
    for scenario, _ in scenarios:
        eval_env = make_env(scenario)
        eval_rewards = []
        action_counts = np.zeros(3)
        
        for _ in range(20):
            obs, _ = eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_counts += np.clip(action, 0, None)
                obs, reward, done, trunc, info = eval_env.step(action)
                total_reward += reward
            eval_rewards.append(total_reward)
        
        action_dist = action_counts / action_counts.sum()
        print(f"\n{scenario}:")
        print(f"  Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"  Action dist: h5={action_dist[0]*100:.1f}%, h7={action_dist[1]*100:.1f}%, h8={action_dist[2]*100:.1f}%")
    
    return model


if __name__ == "__main__":
    train_ppo_2m_optimized()
    print("\n[*] Done!")