#!/usr/bin/env python3
"""
Train PPO with Realistic Environment - NO BIAS
================================================

Environment changes to remove bias:
1. Linear latency model (not M/M/1 exponential)
2. Packet loss simulation
3. Network delay variation
4. Load balancing constraint in reward
5. Randomized capacities
"""

import os
import sys
import numpy as np
import torch
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from scipy import stats
import json

sys.path.insert(0, '/work')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class SDNEnvRealistic(gym.Env):
    """
    Realistic SDN Environment - NO BIAS
    
    Key changes:
    1. Linear latency model (not M/M/1)
    2. Packet loss simulation
    3. Network delay variation
    4. Load balancing constraint
    5. Randomized capacities
    """
    
    def __init__(self, scenario='normal'):
        super().__init__()
        
        self.scenario = scenario
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(14,), dtype=np.float32
        )
        
        # Randomized capacities (±20% each episode)
        self.base_capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.capacities = self.base_capacities.copy()
        self.max_capacity = 100.0
        self.max_steps = 200
        self.current_step = 0
        self.state = None
        
        # Traffic parameters
        self.traffic_intensity = 0.3
        self.burst_probability = 0.15
        self.in_burst = False
        self.burst_duration = 0
        
        # Network dynamics
        self.base_latency = 10.0  # ms
        self.packet_loss_rate = 0.01  # 1% base packet loss
        self.network_delay_variation = 5.0  # ±5ms random delay
        
        # Load balancing tracking
        self.load_history = [[], [], []]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize capacities ±20%
        self.capacities = self.base_capacities * np.random.uniform(0.8, 1.2, size=3)
        
        # Randomize network conditions
        self.packet_loss_rate = np.random.uniform(0.005, 0.02)
        self.network_delay_variation = np.random.uniform(2.0, 8.0)
        
        # Random initial state (14 features)
        self.state = np.zeros(14, dtype=np.float32)
        self.state[:3] = np.random.rand(3) * 0.3  # loads
        self.state[6] = self.traffic_intensity
        self.state[8:11] = self.capacities / 150.0
        
        self.current_step = 0
        self.traffic_intensity = np.random.uniform(0.2, 0.5)  # Higher traffic
        self.in_burst = False
        self.burst_duration = 0
        self.load_history = [[], [], []]
        
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1
        
        # Convert discrete action to weight vector
        if isinstance(action, np.ndarray):
            action = int(action[0]) if action.ndim > 0 else int(action)
        
        # One-hot action
        weights = np.zeros(3, dtype=np.float32)
        weights[action] = 1.0
        
        # Update traffic
        if np.random.random() < self.burst_probability and not self.in_burst:
            self.in_burst = True
            self.burst_duration = np.random.randint(5, 15)
        
        if self.in_burst:
            self.traffic_intensity = min(0.95, self.traffic_intensity + np.random.uniform(0.3, 0.5))
            self.burst_duration -= 1
            if self.burst_duration <= 0:
                self.in_burst = False
        else:
            self.traffic_intensity = max(0.1, self.traffic_intensity + np.random.uniform(-0.05, 0.05))
        
        # Calculate load
        load_h5 = self.traffic_intensity * weights[0] / self.capacities[0] * self.max_capacity
        load_h7 = self.traffic_intensity * weights[1] / self.capacities[1] * self.max_capacity
        load_h8 = self.traffic_intensity * weights[2] / self.capacities[2] * self.max_capacity
        
        load_h5 = np.clip(load_h5, 0, 1)
        load_h7 = np.clip(load_h7, 0, 1)
        load_h8 = np.clip(load_h8, 0, 1)
        
        # LINEAR latency model (not M/M/1 exponential)
        # latency = base + load * factor + random variation
        lat_h5 = self.base_latency * (1 + load_h5) + np.random.uniform(0, self.network_delay_variation)
        lat_h7 = self.base_latency * (1 + load_h7) + np.random.uniform(0, self.network_delay_variation)
        lat_h8 = self.base_latency * (1 + load_h8) + np.random.uniform(0, self.network_delay_variation)
        
        # Packet loss
        packet_loss_h5 = self.packet_loss_rate * (1 + load_h5)
        packet_loss_h7 = self.packet_loss_rate * (1 + load_h7)
        packet_loss_h8 = self.packet_loss_rate * (1 + load_h8)
        
        avg_latency = (lat_h5 + lat_h7 + lat_h8) / 3
        avg_packet_loss = (packet_loss_h5 + packet_loss_h7 + packet_loss_h8) / 3
        
        # Throughput (affected by packet loss)
        throughput = self.traffic_intensity * (1 - avg_packet_loss) * (1 - 0.5 * (load_h5 + load_h7 + load_h8) / 3)
        
        # Track load history for load balancing constraint
        self.load_history[0].append(load_h5)
        self.load_history[1].append(load_h7)
        self.load_history[2].append(load_h8)
        
        # Calculate load balance (std of loads)
        if len(self.load_history[0]) > 10:
            load_std = np.std([np.mean(self.load_history[0][-10:]),
                              np.mean(self.load_history[1][-10:]),
                              np.mean(self.load_history[2][-10:])])
        else:
            load_std = 0
        
        # UNBIASED REWARD
        latency_penalty = -0.1 * avg_latency
        packet_loss_penalty = -10.0 * avg_packet_loss
        overload_penalty = -5.0 * (max(0, load_h5 - 0.9) + max(0, load_h7 - 0.9) + max(0, load_h8 - 0.9))
        
        # Load balancing bonus (encourage distributed load)
        load_balance_bonus = -2.0 * load_std
        
        reward = throughput + latency_penalty + packet_loss_penalty + overload_penalty + load_balance_bonus
        
        # Update state (14 features)
        self.state = np.array([
            load_h5, load_h7, load_h8,
            lat_h5 / 100.0, lat_h7 / 100.0, lat_h8 / 100.0,
            self.traffic_intensity,
            avg_latency / 100.0,
            self.capacities[0] / 150.0,
            self.capacities[1] / 150.0,
            self.capacities[2] / 150.0,
            packet_loss_h5,
            packet_loss_h7,
            packet_loss_h8
        ], dtype=np.float32)
        
        done = self.current_step >= self.max_steps
        
        info = {
            'throughput': throughput,
            'latency': avg_latency,
            'packet_loss': avg_packet_loss,
            'load_balance': load_std,
            'load_h5': load_h5,
            'load_h7': load_h7,
            'load_h8': load_h8,
        }
        
        return self.state, float(reward), done, False, info


def train_ppo_realistic(total_timesteps=500_000, save_dir='ai_model/checkpoints'):
    """Train PPO with realistic environment."""
    
    print("=" * 60)
    print("TRAINING PPO - REALISTIC ENVIRONMENT (NO BIAS)")
    print("=" * 60)
    print("Key changes:")
    print("1. Linear latency model (not M/M/1)")
    print("2. Packet loss simulation")
    print("3. Network delay variation")
    print("4. Load balancing constraint")
    print("5. Randomized capacities")
    print("=" * 60)
    
    env = SDNEnvRealistic()
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device='cpu'
    )
    
    print(f"\nModel architecture:")
    print(model.policy)
    
    # Action distribution callback
    class ActionDistCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.action_counts = [0, 0, 0]
            self.episode_rewards = []
            self.current_episode_reward = 0
            
        def _on_step(self):
            actions = self.locals.get('actions', None)
            rewards = self.locals.get('rewards', None)
            
            if actions is not None:
                for action in actions.flatten():
                    if action < 3:
                        self.action_counts[int(action)] += 1
            
            if rewards is not None:
                for reward in rewards.flatten():
                    self.current_episode_reward += reward
            
            dones = self.locals.get('dones', None)
            if dones is not None:
                for done in dones.flatten():
                    if done:
                        self.episode_rewards.append(self.current_episode_reward)
                        self.current_episode_reward = 0
            
            return True
        
        def get_distribution(self):
            total = sum(self.action_counts)
            if total > 0:
                return [c / total for c in self.action_counts]
            return [0.33, 0.33, 0.34]
    
    action_callback = ActionDistCallback()
    
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[action_callback],
        progress_bar=False
    )
    
    elapsed = datetime.now() - start_time
    print(f"\nTraining completed in {elapsed}")
    
    action_dist = action_callback.get_distribution()
    print(f"\nAction distribution during training:")
    print(f"  Action 0 (h5):  {action_dist[0]:.1%}")
    print(f"  Action 1 (h7):  {action_dist[1]:.1%}")
    print(f"  Action 2 (h8):  {action_dist[2]:.1%}")
    
    # Save model
    final_path = f"{save_dir}/ppo_realistic_final.zip"
    model.save(final_path)
    print(f"\nSaved final model to: {final_path}")
    
    return model, action_dist


def benchmark_realistic(model_path, n_episodes=100):
    """Benchmark on realistic environment."""
    
    print("\n" + "=" * 60)
    print("BENCHMARK - REALISTIC ENVIRONMENT")
    print("=" * 60)
    
    model = PPO.load(model_path)
    env = SDNEnvRealistic()
    
    # PPO benchmark
    ppo_rewards = []
    ppo_actions = [0, 0, 0]
    ppo_load_balance = []
    
    print("\nRunning PPO benchmark...")
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action[0]) if action.ndim > 0 else int(action)
            
            ppo_actions[action] += 1
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if 'load_balance' in info:
                ppo_load_balance.append(info['load_balance'])
        
        ppo_rewards.append(total_reward)
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}: reward={total_reward:.2f}")
    
    # WRR benchmark
    wrr_rewards = []
    wrr_load_balance = []
    weights = np.array([1.0, 5.0, 10.0])
    weights = weights / weights.sum()
    
    print("\nRunning WRR benchmark...")
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            r = np.random.random()
            if r < weights[0]:
                action = 0
            elif r < weights[0] + weights[1]:
                action = 1
            else:
                action = 2
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if 'load_balance' in info:
                wrr_load_balance.append(info['load_balance'])
        
        wrr_rewards.append(total_reward)
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}: reward={total_reward:.2f}")
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(ppo_rewards, wrr_rewards)
    
    # Cohen's d
    pooled_std = np.sqrt((np.std(ppo_rewards)**2 + np.std(wrr_rewards)**2) / 2)
    cohens_d = (np.mean(ppo_rewards) - np.mean(wrr_rewards)) / pooled_std if pooled_std > 0 else 0
    
    # 95% CI
    ppo_ci = stats.t.interval(0.95, len(ppo_rewards)-1, loc=np.mean(ppo_rewards), scale=stats.sem(ppo_rewards))
    wrr_ci = stats.t.interval(0.95, len(wrr_rewards)-1, loc=np.mean(wrr_rewards), scale=stats.sem(wrr_rewards))
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nPPO:")
    print(f"  Mean Reward: {np.mean(ppo_rewards):.2f} ± {np.std(ppo_rewards):.2f}")
    print(f"  95% CI: [{ppo_ci[0]:.2f}, {ppo_ci[1]:.2f}]")
    print(f"  Action Distribution: [{ppo_actions[0]/sum(ppo_actions):.1%}, {ppo_actions[1]/sum(ppo_actions):.1%}, {ppo_actions[2]/sum(ppo_actions):.1%}]")
    print(f"  Load Balance (std): {np.mean(ppo_load_balance):.4f}")
    
    print(f"\nWRR:")
    print(f"  Mean Reward: {np.mean(wrr_rewards):.2f} ± {np.std(wrr_rewards):.2f}")
    print(f"  95% CI: [{wrr_ci[0]:.2f}, {wrr_ci[1]:.2f}]")
    print(f"  Load Balance (std): {np.mean(wrr_load_balance):.4f}")
    
    print(f"\nStatistical Tests:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")
    
    winner = "PPO" if np.mean(ppo_rewards) > np.mean(wrr_rewards) else "WRR"
    improvement = ((np.mean(ppo_rewards) - np.mean(wrr_rewards)) / abs(np.mean(wrr_rewards)) * 100) if np.mean(wrr_rewards) != 0 else 0
    print(f"\nWinner: {winner} ({improvement:+.1f}%)")
    
    results = {
        'PPO': {
            'mean': float(np.mean(ppo_rewards)),
            'std': float(np.std(ppo_rewards)),
            'ci_95': [float(ppo_ci[0]), float(ppo_ci[1])],
            'action_dist': [c / sum(ppo_actions) for c in ppo_actions],
            'load_balance': float(np.mean(ppo_load_balance))
        },
        'WRR': {
            'mean': float(np.mean(wrr_rewards)),
            'std': float(np.std(wrr_rewards)),
            'ci_95': [float(wrr_ci[0]), float(wrr_ci[1])],
            'load_balance': float(np.mean(wrr_load_balance))
        },
        'statistical': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': p_value < 0.05
        }
    }
    
    return results


if __name__ == '__main__':
    # Train
    model, action_dist = train_ppo_realistic(total_timesteps=500_000)
    
    # Benchmark
    results = benchmark_realistic('ai_model/checkpoints/ppo_realistic_final.zip', n_episodes=100)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'ai_model/benchmark_realistic_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: ai_model/benchmark_realistic_{timestamp}.json")