#!/usr/bin/env python3
"""
Train PPO with Stable Baselines3 using V4 Capacity-Weighted Data
================================================================

This script trains a PPO model using SB3 with the fixed V4 dataset.
"""

import os
import sys
import numpy as np
import torch
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, '/work')

# SB3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv


class SDNEnvGymnasium(gym.Env):
    """Gymnasium wrapper for SDN Load Balancer environment."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        super().__init__()
        
        # Action space: 3 discrete actions (h5, h7, h8)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 11 features
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.float32
        )
        
        # Server capacities
        self.capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.max_capacity = float(np.max(self.capacities))
        
        # State variables
        self.state = None
        self.current_step = 0
        self.max_steps = 200
        
        # Traffic parameters
        self.traffic_intensity = 0.3
        self.burst_probability = 0.15
        self.in_burst = False
        self.burst_duration = 0
        self.burst_intensity = 0.0
        
        # Episode stats
        self.episode_stats = {
            'total_throughput': 0.0,
            'max_latency': 0.0,
            'overload_count': 0,
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random initial state
        self.state = np.random.rand(11).astype(np.float32) * 0.3
        self.state[6] = 0.0
        self.state[7] = 0.3
        self.state[8] = self.capacities[0] / 150.0
        self.state[9] = self.capacities[1] / 150.0
        self.state[10] = self.capacities[2] / 150.0
        
        self.current_step = 0
        self.traffic_intensity = np.random.uniform(0.2, 0.4)
        self.in_burst = False
        self.burst_duration = 0
        
        self.episode_stats = {
            'total_throughput': 0.0,
            'max_latency': 0.0,
            'overload_count': 0,
        }
        
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1
        
        # Convert discrete action to weight vector
        weights = np.zeros(3, dtype=np.float32)
        weights[action] = 1.0
        
        # Update traffic
        if np.random.random() < self.burst_probability and not self.in_burst:
            self.in_burst = True
            self.burst_duration = np.random.randint(5, 15)
            self.burst_intensity = np.random.uniform(0.5, 0.8)
        
        if self.in_burst:
            self.traffic_intensity = min(0.9, self.traffic_intensity + self.burst_intensity)
            self.burst_duration -= 1
            if self.burst_duration <= 0:
                self.in_burst = False
        else:
            self.traffic_intensity = max(0.1, self.traffic_intensity + np.random.uniform(-0.05, 0.05))
        
        # Calculate load on each server
        load_h5 = self.traffic_intensity * weights[0] / self.capacities[0] * self.max_capacity
        load_h7 = self.traffic_intensity * weights[1] / self.capacities[1] * self.max_capacity
        load_h8 = self.traffic_intensity * weights[2] / self.capacities[2] * self.max_capacity
        
        load_h5 = np.clip(load_h5, 0, 1)
        load_h7 = np.clip(load_h7, 0, 1)
        load_h8 = np.clip(load_h8, 0, 1)
        
        # Calculate latency (M/M/1 queue)
        base_lat = 10.0
        lat_h5 = base_lat / (1 - load_h5 + 1e-8) if load_h5 < 0.99 else 1000.0
        lat_h7 = base_lat / (1 - load_h7 + 1e-8) if load_h7 < 0.99 else 1000.0
        lat_h8 = base_lat / (1 - load_h8 + 1e-8) if load_h8 < 0.99 else 1000.0
        
        avg_latency = (lat_h5 + lat_h7 + lat_h8) / 3
        
        # Calculate throughput
        throughput = self.traffic_intensity * (1 - 0.5 * (load_h5 + load_h7 + load_h8) / 3)
        
        # Calculate reward
        # Reward = throughput - latency penalty - overload penalty
        latency_penalty = -0.1 * avg_latency
        overload_penalty = -10.0 * (max(0, load_h5 - 0.9) + max(0, load_h7 - 0.9) + max(0, load_h8 - 0.9))
        
        # Capacity-weighted bonus (encourage matching capacity distribution)
        capacity_ratios = self.capacities / np.sum(self.capacities)
        if action == 2:  # h8 (strongest)
            capacity_bonus = 1.0  # Encourage using strongest server
        elif action == 1:  # h7 (medium)
            capacity_bonus = 0.5
        else:  # h5 (weakest)
            capacity_bonus = -0.5  # Penalize using weakest server
        
        reward = throughput + latency_penalty + overload_penalty + capacity_bonus
        
        # Update state
        self.state = np.array([
            load_h5, load_h7, load_h8,
            lat_h5 / 100.0, lat_h7 / 100.0, lat_h8 / 100.0,
            self.traffic_intensity,
            avg_latency / 100.0,
            self.capacities[0] / 150.0,
            self.capacities[1] / 150.0,
            self.capacities[2] / 150.0
        ], dtype=np.float32)
        
        # Episode stats
        self.episode_stats['total_throughput'] += throughput
        self.episode_stats['max_latency'] = max(self.episode_stats['max_latency'], avg_latency)
        if load_h5 > 0.9 or load_h7 > 0.9 or load_h8 > 0.9:
            self.episode_stats['overload_count'] += 1
        
        # Done condition
        done = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'throughput': throughput,
            'latency': avg_latency,
            'overload': load_h5 > 0.9 or load_h7 > 0.9 or load_h8 > 0.9,
            'episode_stats': self.episode_stats
        }
        
        return self.state, float(reward), done, truncated, info


def train_ppo_sb3_v4(
    total_timesteps=500_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    save_dir='ai_model/checkpoints'
):
    """Train PPO with SB3 using V4 data."""
    
    print("=" * 60)
    print("TRAINING PPO WITH SB3 - V4 CAPACITY-WEIGHTED DATA")
    print("=" * 60)
    
    # Load V4 data for reference
    X = np.load('ai_model/processed_data/X_v4.npy')
    y = np.load('ai_model/processed_data/y_v4.npy')
    
    print(f"Loaded V4 dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Action distribution: {np.bincount(y)}")
    
    # Create environment
    env = SDNEnvGymnasium()
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        device='cpu'  # Use CPU for MLP policy
    )
    
    print(f"\nModel architecture:")
    print(model.policy)
    
    # Action distribution callback
    class ActionDistCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.action_counts = [0, 0, 0]
            
        def _on_step(self):
            actions = self.locals.get('actions', None)
            if actions is not None:
                for action in actions.flatten():
                    if action < 3:
                        self.action_counts[int(action)] += 1
            return True
        
        def get_distribution(self):
            total = sum(self.action_counts)
            if total > 0:
                return [c / total for c in self.action_counts]
            return [0.33, 0.33, 0.34]
    
    action_callback = ActionDistCallback()
    
    # Train
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[action_callback],
        progress_bar=False  # Disable progress bar (requires tqdm/rich)
    )
    
    elapsed = datetime.now() - start_time
    print(f"\nTraining completed in {elapsed}")
    
    # Get action distribution
    action_dist = action_callback.get_distribution()
    print(f"\nAction distribution during training:")
    print(f"  Action 0 (h5 - weakest):  {action_dist[0]:.1%}")
    print(f"  Action 1 (h7 - medium):   {action_dist[1]:.1%}")
    print(f"  Action 2 (h8 - strongest): {action_dist[2]:.1%}")
    
    # Save final model
    final_path = f"{save_dir}/ppo_v4_sb3_final.zip"
    model.save(final_path)
    print(f"\nSaved final model to: {final_path}")
    
    # Test model
    print("\n" + "=" * 60)
    print("TESTING PPO V4 MODEL")
    print("=" * 60)
    
    obs, info = env.reset()
    total_reward = 0
    action_counts = [0, 0, 0]
    
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_counts[action] += 1
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done or truncated:
            break
    
    print(f"Test episode reward: {total_reward:.2f}")
    print(f"Action distribution: [{action_counts[0]/sum(action_counts):.1%}, {action_counts[1]/sum(action_counts):.1%}, {action_counts[2]/sum(action_counts):.1%}]")
    
    return model


if __name__ == '__main__':
    train_ppo_sb3_v4(
        total_timesteps=500_000,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01
    )