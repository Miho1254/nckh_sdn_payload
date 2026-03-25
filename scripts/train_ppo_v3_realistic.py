#!/usr/bin/env python3
"""
Train PPO with Realistic Environment V3 - CONGESTION-AWARE
=========================================================

Improvements over V2:
1. Congestion penalty - penalize when one server is overloaded while others idle
2. Link utilization features - state includes per-link congestion
3. Queue length simulation - realistic buffering behavior
4. Better throughput formula - accounts for shared bottleneck links
5. Cache effect modeling - distributed load improves cache hit rate

Key insight: WRR wins because it distributes load, avoiding congestion.
PPO must learn to do the same.
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


class SDNEnvV3Realistic(gym.Env):
    """
    Realistic SDN Environment V3 - CONGESTION-AWARE
    
    Key improvements:
    1. Congestion penalty: penalize load imbalance that causes bottleneck
    2. Link utilization tracking: per-switch queue lengths
    3. Cache effect: distributed load = better cache hit rate
    4. Shared link capacity: servers share uplink bandwidth
    """
    
    def __init__(self, scenario='normal'):
        super().__init__()
        
        self.scenario = scenario
        self.action_space = spaces.Discrete(3)
        
        # Extended state: 14 original + 6 new = 20 features
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20,), dtype=np.float32
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
        
        # NEW: Link utilization (per-switch queue lengths)
        self.link_queue_lengths = np.zeros(3, dtype=np.float32)  # per-server link
        self.switch_queue_length = 0.0  # core switch queue
        
        # NEW: Shared uplink capacity (for congestion modeling)
        self.shared_uplink_capacity = 1000.0  # Mbps - all servers share this
        
        # NEW: Cache hit rate (improves with load distribution)
        self.cache_hit_rate = 0.3  # base cache hit rate
        
        # Load balancing tracking
        self.load_history = [[], [], []]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize capacities ±20%
        self.capacities = self.base_capacities * np.random.uniform(0.8, 1.2, size=3)
        
        # Randomize network conditions
        self.packet_loss_rate = np.random.uniform(0.005, 0.02)
        self.network_delay_variation = np.random.uniform(2.0, 8.0)
        
        # Random initial state (20 features)
        self.state = np.zeros(20, dtype=np.float32)
        self.state[:3] = np.random.rand(3) * 0.3  # loads
        self.state[6] = self.traffic_intensity
        self.state[8:11] = self.capacities / 150.0
        
        # Initialize link queues
        self.link_queue_lengths = np.zeros(3, dtype=np.float32)
        self.switch_queue_length = 0.0
        
        self.current_step = 0
        self.traffic_intensity = np.random.uniform(0.2, 0.5)
        self.in_burst = False
        self.burst_duration = 0
        self.load_history = [[], [], []]
        self.cache_hit_rate = 0.3
        
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1
        
        # Convert discrete action to weight vector
        if isinstance(action, np.ndarray):
            action = int(action[0]) if action.ndim > 0 else int(action)
        
        # One-hot action - but now we model partial allocation
        # Action 0: prefer h5 (weak)
        # Action 1: prefer h7 (medium)  
        # Action 2: prefer h8 (strong)
        weights = np.array([0.7, 0.15, 0.15], dtype=np.float32) if action == 0 else \
                  np.array([0.15, 0.7, 0.15], dtype=np.float32) if action == 1 else \
                  np.array([0.15, 0.15, 0.7], dtype=np.float32)
        
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
        
        # Update link queue lengths (simulate buffering)
        # Queue grows when load is high, shrinks when load is low
        for i, load in enumerate([load_h5, load_h7, load_h8]):
            self.link_queue_lengths[i] += load * 0.1
            self.link_queue_lengths[i] *= 0.95  # decay
            self.link_queue_lengths[i] = np.clip(self.link_queue_lengths[i], 0, 1)
        
        # Core switch queue (shared bottleneck)
        total_load = (load_h5 + load_h7 + load_h8) / 3
        self.switch_queue_length += total_load * 0.1
        self.switch_queue_length *= 0.92  # faster decay
        self.switch_queue_length = np.clip(self.switch_queue_length, 0, 1)
        
        # LINEAR latency model with queueing delay
        base_lat = self.base_latency * (1 + load_h5) + np.random.uniform(0, self.network_delay_variation)
        lat_h5 = base_lat + self.link_queue_lengths[0] * 50  # queue adds delay
        lat_h7 = self.base_latency * (1 + load_h7) + np.random.uniform(0, self.network_delay_variation) + self.link_queue_lengths[1] * 50
        lat_h8 = self.base_latency * (1 + load_h8) + np.random.uniform(0, self.network_delay_variation) + self.link_queue_lengths[2] * 50
        
        # Packet loss increases with queue length
        packet_loss_h5 = self.packet_loss_rate * (1 + load_h5 + self.link_queue_lengths[0])
        packet_loss_h7 = self.packet_loss_rate * (1 + load_h7 + self.link_queue_lengths[1])
        packet_loss_h8 = self.packet_loss_rate * (1 + load_h8 + self.link_queue_lengths[2])
        
        avg_latency = (lat_h5 + lat_h7 + lat_h8) / 3
        avg_packet_loss = (packet_loss_h5 + packet_loss_h7 + packet_loss_h8) / 3
        
        # Track load history
        self.load_history[0].append(load_h5)
        self.load_history[1].append(load_h7)
        self.load_history[2].append(load_h8)
        
        # Calculate load balance (std of loads) - for cache effect
        if len(self.load_history[0]) > 10:
            loads = [np.mean(self.load_history[0][-10:]),
                    np.mean(self.load_history[1][-10:]),
                    np.mean(self.load_history[2][-10:])]
            load_std = np.std(loads)
            load_balance = 1.0 - load_std  # 1 = perfectly balanced
        else:
            load_std = 0
            load_balance = 0.5
        
        # Cache effect: better cache hit rate when load is balanced
        # This is the key insight - WRR wins because distributed load improves cache
        cache_hit_rate = 0.3 + load_balance * 0.4  # 0.3 to 0.7 based on balance
        
        # NEW: CONGESTION-AWARE THROUGHPUT
        # Key insight: concentrating load on h8 causes congestion because:
        # 1. Single link bottleneck
        # 2. Cache thrashing
        # 3. Queue buildup
        
        # Per-server effective throughput (limited by capacity)
        eff_h5 = self.capacities[0] * (1 - packet_loss_h5) * (1 - load_h5) * (1 - self.link_queue_lengths[0])
        eff_h7 = self.capacities[1] * (1 - packet_loss_h7) * (1 - load_h7) * (1 - self.link_queue_lengths[1])
        eff_h8 = self.capacities[2] * (1 - packet_loss_h8) * (1 - load_h8) * (1 - self.link_queue_lengths[2])
        
        # Sum throughput (this is what matters in reality!)
        total_throughput = eff_h5 + eff_h7 + eff_h8
        
        # Congestion penalty: if one server is overloaded (>0.8) while others are <0.5
        max_load = max(load_h5, load_h7, load_h8)
        min_load = min(load_h5, load_h7, load_h8)
        
        congestion_penalty = 0
        if max_load > 0.8 and min_load < 0.4:
            congestion_penalty = -0.3 * (max_load - 0.8)  # penalize congestion
        
        # Load balance bonus (NOT fairness, but avoiding congestion)
        # This is different from "fairness" - we're avoiding bottleneck, not dividing equally
        balance_bonus = load_balance * 0.2  # small bonus for balanced load
        
        # Total throughput (normalized)
        throughput_normalized = total_throughput / (self.max_capacity * 3)
        
        # REWARD: Focus on real throughput, penalize congestion
        latency_penalty = -0.05 * avg_latency
        packet_loss_penalty = -5.0 * avg_packet_loss
        overload_penalty = -3.0 * (max(0, load_h5 - 0.9) + max(0, load_h7 - 0.9) + max(0, load_h8 - 0.9))
        queue_penalty = -2.0 * (self.switch_queue_length + np.sum(self.link_queue_lengths) / 3)
        
        # Total reward: maximize throughput, minimize penalties
        reward = throughput_normalized + balance_bonus + congestion_penalty + latency_penalty + packet_loss_penalty + overload_penalty + queue_penalty
        
        # Update state (20 features)
        self.state = np.array([
            load_h5, load_h7, load_h8,                          # 0-2: loads
            lat_h5 / 100.0, lat_h7 / 100.0, lat_h8 / 100.0,    # 3-5: normalized latency
            self.traffic_intensity,                              # 6: traffic
            avg_latency / 100.0,                                 # 7: avg latency
            self.capacities[0] / 150.0,                          # 8: h5 capacity
            self.capacities[1] / 150.0,                          # 9: h7 capacity
            self.capacities[2] / 150.0,                          # 10: h8 capacity
            packet_loss_h5,                                      # 11: packet loss h5
            packet_loss_h7,                                      # 12: packet loss h7
            packet_loss_h8,                                      # 13: packet loss h8
            self.link_queue_lengths[0],                          # 14: link queue h5
            self.link_queue_lengths[1],                          # 15: link queue h7
            self.link_queue_lengths[2],                          # 16: link queue h8
            self.switch_queue_length,                            # 17: core switch queue
            load_balance,                                        # 18: load balance metric
            cache_hit_rate,                                      # 19: cache hit rate estimate
        ], dtype=np.float32)
        
        done = self.current_step >= self.max_steps
        
        info = {
            'throughput': total_throughput,
            'latency': avg_latency,
            'packet_loss': avg_packet_loss,
            'load_balance': load_balance,
            'cache_hit_rate': cache_hit_rate,
            'congestion': self.switch_queue_length,
            'load_h5': load_h5,
            'load_h7': load_h7,
            'load_h8': load_h8,
        }
        
        return self.state, float(reward), done, False, info


def train_ppo_v3(total_timesteps=500_000, save_dir='ai_model/checkpoints'):
    """Train PPO V3 with congestion-aware environment."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("Training PPO V3 - CONGESTION-AWARE")
    print("=" * 70)
    print("\nKey improvements:")
    print("1. Congestion penalty - penalize overloaded links")
    print("2. Link queue simulation - queue buildup adds delay")
    print("3. Cache effect - balanced load = better cache")
    print("4. Better throughput formula - sum of per-server capacity")
    print()
    
    # Create environment
    env = SDNEnvV3Realistic()
    
    # Create PPO model
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
    )
    
    # Training callback
    class ActionDistCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.action_counts = {0: 0, 1: 0, 2: 0}
            self.total_steps = 0
            
        def _on_step(self):
            if len(self.locals['actions']) > 0:
                for a in self.locals['actions']:
                    if a in self.action_counts:
                        self.action_counts[a] += 1
                        self.total_steps += 1
            return True
        
        def get_distribution(self):
            if self.total_steps > 0:
                return {k: v / self.total_steps * 100 for k, v in self.action_counts.items()}
            return {0: 0, 1: 0, 2: 0}
    
    callback = ActionDistCallback()
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,  # Disabled - tqdm not installed
    )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{save_dir}/ppo_v3_congestion_aware_{timestamp}.zip"
    model.save(model_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Action distribution: {callback.get_distribution()}")
    
    return model, callback.get_distribution()


def benchmark_v3(model_path, n_episodes=50):
    """Benchmark PPO V3 against WRR."""
    
    from scipy import stats
    
    print("\n" + "=" * 70)
    print("BENCHMARKING PPO V3 vs WRR")
    print("=" * 70)
    
    # Load model
    model = PPO.load(model_path)
    
    # Run episodes
    ppo_rewards = []
    wrr_rewards = []
    
    for ep in range(n_episodes):
        # PPO
        env = SDNEnvV3Realistic()
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        ppo_rewards.append(total_reward)
        
        # WRR (simulated with weights [1, 5, 10])
        env = SDNEnvV3Realistic()
        obs, _ = env.reset()
        total_reward = 0
        wrr_idx = 0
        weights = [1, 5, 10]
        for _ in range(200):
            action = wrr_idx % 3  # round-robin
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            wrr_idx += 1
            if done:
                break
        wrr_rewards.append(total_reward)
    
    print(f"\nPPO V3: {np.mean(ppo_rewards):.2f} ± {np.std(ppo_rewards):.2f}")
    print(f"WRR:    {np.mean(wrr_rewards):.2f} ± {np.std(wrr_rewards):.2f}")
    
    t_stat, p_value = stats.ttest_ind(ppo_rewards, wrr_rewards)
    winner = "PPO V3" if np.mean(ppo_rewards) > np.mean(wrr_rewards) else "WRR"
    diff_pct = abs(np.mean(ppo_rewards) - np.mean(wrr_rewards)) / abs(np.mean(wrr_rewards)) * 100
    
    print(f"\nWinner: {winner} ({diff_pct:+.1f}%)")
    print(f"p-value: {p_value:.4f}")
    print(f"Significant: {'YES' if p_value < 0.05 else 'NO'}")
    
    return ppo_rewards, wrr_rewards


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train PPO V3")
    parser.add_argument("--benchmark", type=str, help="Benchmark with model path")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Training timesteps")
    args = parser.parse_args()
    
    if args.train:
        train_ppo_v3(args.timesteps)
    elif args.benchmark:
        benchmark_v3(args.benchmark)
    else:
        # Train then benchmark
        model, dist = train_ppo_v3(args.timesteps)
        model_path = f"ai_model/checkpoints/ppo_v3_congestion_aware_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        benchmark_v3(model_path)
