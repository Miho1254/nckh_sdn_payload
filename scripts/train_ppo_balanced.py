#!/usr/bin/env python3
"""
Train PPO với Balanced Environment - Tránh Reward Hacking
==========================================================

Problem: Model bị reward hacking - chỉ chọn action 0 (h5) vì environment bị exploit.

Solution:
1. Normalize throughput theo action (không phải capacity đơn thuần)
2. Thêm penalty nếu không sử dụng tất cả servers
3. Đảm bảo mỗi action có cơ hội được chọn
"""

import os
import sys
import numpy as np
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, '/work')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class SDNEnvBalanced(gym.Env):
    """
    Balanced Environment - Tránh reward hacking.
    
    Key changes:
    1. Throughput = f(loads) không phải f(action)
    2. Action chỉ quyết định phân bổ, không phải kết quả trực tiếp
    3. Load balancing bonus thực sự
    """
    
    def __init__(self):
        super().__init__()
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20,), dtype=np.float32
        )
        
        # Server capacities (h5=10, h7=50, h8=100)
        self.base_capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.capacities = self.base_capacities.copy()
        self.max_capacity = 100.0
        
        self.max_steps = 200
        self.current_step = 0
        self.state = None
        
        # Traffic
        self.traffic_intensity = 0.3
        self.burst_probability = 0.15
        self.in_burst = False
        self.burst_duration = 0
        
        # Network
        self.base_latency = 10.0
        self.packet_loss_rate = 0.01
        self.network_delay_variation = 5.0
        
        # Queues
        self.link_queue_lengths = np.zeros(3, dtype=np.float32)
        self.switch_queue_length = 0.0
        
        # Tracking
        self.load_history = [[], [], []]
        self.action_counts = [0, 0, 0]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize capacities ±20%
        self.capacities = self.base_capacities * np.random.uniform(0.8, 1.2, size=3)
        
        # Randomize network
        self.packet_loss_rate = np.random.uniform(0.005, 0.02)
        self.network_delay_variation = np.random.uniform(2.0, 8.0)
        
        # Initial state
        self.state = np.zeros(20, dtype=np.float32)
        self.state[:3] = np.random.rand(3) * 0.3
        self.state[6] = self.traffic_intensity
        self.state[7] = 0.1
        self.state[8:11] = self.capacities / 150.0
        self.state[11:14] = 0.01
        self.state[14:18] = 0.0
        self.state[18] = 1.0
        self.state[19] = 0.3
        
        self.current_step = 0
        self.in_burst = False
        self.action_counts = [0, 0, 0]
        
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1
        
        # Track action usage
        self.action_counts[action] += 1
        
        # Get current loads
        loads = self.state[:3].copy()
        
        # Action = chọn server để tăng load
        loads[action] = min(1.0, loads[action] + 0.25)
        
        # Traffic variation
        if np.random.rand() < self.burst_probability:
            self.traffic_intensity = min(1.0, self.traffic_intensity + 0.2)
            self.in_burst = True
            self.burst_duration = 5
        elif self.in_burst:
            self.burst_duration -= 1
            if self.burst_duration <= 0:
                self.in_burst = False
                self.traffic_intensity = max(0.1, self.traffic_intensity - 0.3)
        
        # Natural load decay
        loads = loads * 0.85 + np.random.uniform(0.05, 0.15, size=3)
        
        load_h5, load_h7, load_h8 = loads[0], loads[1], loads[2]
        
        # Update queues
        self.link_queue_lengths = np.minimum(1.0, self.link_queue_lengths + np.array([
            max(0, load_h5 - 0.7),
            max(0, load_h7 - 0.7),
            max(0, load_h8 - 0.7)
        ]))
        self.link_queue_lengths *= 0.9
        self.switch_queue_length = min(1.0, self.switch_queue_length + max(0, np.mean(loads) - 0.5))
        self.switch_queue_length *= 0.85
        
        # Latency
        lat_h5 = self.base_latency * (1 + load_h5) + np.random.uniform(0, self.network_delay_variation) + self.link_queue_lengths[0] * 50
        lat_h7 = self.base_latency * (1 + load_h7) + np.random.uniform(0, self.network_delay_variation) + self.link_queue_lengths[1] * 50
        lat_h8 = self.base_latency * (1 + load_h8) + np.random.uniform(0, self.network_delay_variation) + self.link_queue_lengths[2] * 50
        
        # Packet loss
        packet_loss_h5 = self.packet_loss_rate * (1 + load_h5 + self.link_queue_lengths[0])
        packet_loss_h7 = self.packet_loss_rate * (1 + load_h7 + self.link_queue_lengths[1])
        packet_loss_h8 = self.packet_loss_rate * (1 + load_h8 + self.link_queue_lengths[2])
        
        avg_latency = (lat_h5 + lat_h7 + lat_h8) / 3
        avg_packet_loss = (packet_loss_h5 + packet_loss_h7 + packet_loss_h8) / 3
        
        # ===== REWARD =====
        
        # Throughput: phụ thuộc vào CAPACITY thực sự, không phải action
        # h8 (capacity=100) phải cho throughput cao hơn h5 (capacity=10)
        # Nếu h8 load=0.5 và h5 load=0.5 thì h8 đóng góp nhiều hơn
        eff_h5 = self.capacities[0] * (1 - packet_loss_h5) * (1 - load_h5)
        eff_h7 = self.capacities[1] * (1 - packet_loss_h7) * (1 - load_h7)
        eff_h8 = self.capacities[2] * (1 - packet_loss_h8) * (1 - load_h8)
        
        total_throughput = eff_h5 + eff_h7 + eff_h8
        
        # Normalize by max possible throughput
        max_possible_throughput = np.sum(self.capacities)  # 160
        throughput_score = total_throughput / max_possible_throughput
        
        # Load balance check
        load_std = np.std(loads)
        load_balance = max(0, 1.0 - load_std * 3)
        
        # ===== PENALTIES =====
        
        # Latency penalty
        latency_penalty = -0.05 * avg_latency
        
        # Packet loss penalty
        packet_loss_penalty = -5.0 * avg_packet_loss
        
        # Overload penalty
        overload_penalty = -3.0 * (
            max(0, load_h5 - 0.9) +
            max(0, load_h7 - 0.9) +
            max(0, load_h8 - 0.9)
        )
        
        # Queue penalty
        queue_penalty = -2.0 * (self.switch_queue_length + np.sum(self.link_queue_lengths) / 3)
        
        # Balance bonus - THỰC SỰ cân bằng
        balance_bonus = 0.2 * load_balance
        
        # Action diversity bonus - KHÔNG ĐƯỢC BỎ QUA SERVER MẠNH
        action_usage = np.array(self.action_counts) / max(1, sum(self.action_counts))
        # Bonus nếu sử dụng tất cả servers
        diversity_bonus = 0.1 * (1.0 - np.max(action_usage))  # Penalize if one action dominates
        
        # Capacity utilization bonus - DÙNG H8 NHIỀU HƠN
        # h8 có capacity gấp 10 lần h5, nên được ưu tiên hơn
        capacity_weight = self.capacities / np.sum(self.capacities)  # [0.0625, 0.3125, 0.625]
        expected_action_usage = capacity_weight
        
        # Reward if actual usage matches capacity distribution
        usage_match_bonus = 0.2 * (1.0 - np.sum(np.abs(action_usage - expected_action_usage)) / 2)
        
        # Congestion penalty
        max_load = max(load_h5, load_h7, load_h8)
        min_load = min(load_h5, load_h7, load_h8)
        congestion_penalty = 0
        if max_load > 0.8 and min_load < 0.4:
            congestion_penalty = -0.3 * (max_load - 0.8)
        
        # Total reward
        reward = (
            1.5 * throughput_score +
            balance_bonus +
            diversity_bonus +
            usage_match_bonus +
            congestion_penalty +
            latency_penalty +
            packet_loss_penalty +
            overload_penalty +
            queue_penalty
        )
        
        # Update state
        cache_hit_rate = 0.3 + 0.4 * (1 - load_std)
        
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
            packet_loss_h8,
            self.link_queue_lengths[0],
            self.link_queue_lengths[1],
            self.link_queue_lengths[2],
            self.switch_queue_length,
            load_balance,
            cache_hit_rate,
        ], dtype=np.float32)
        
        done = self.current_step >= self.max_steps
        
        info = {
            'throughput': total_throughput,
            'latency': avg_latency,
            'packet_loss': avg_packet_loss,
            'load_balance': load_balance,
            'action_counts': self.action_counts.copy(),
            'reward': reward
        }
        
        return self.state, float(reward), done, False, info


def train_balanced(total_timesteps=500_000, save_path=None):
    """Train PPO với balanced environment."""
    
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"ai_model/checkpoints/ppo_balanced_{timestamp}.zip"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print("=" * 70)
    print("Training PPO - BALANCED ENVIRONMENT")
    print("=" * 70)
    print("\nKey changes to prevent reward hacking:")
    print("1. Throughput phụ thuộc capacity thực, không phải action")
    print("2. Action diversity bonus - khuyến khích dùng tất cả servers")
    print("3. Usage match bonus - h8 (mạnh) được ưu tiên hơn h5 (yếu)")
    print()
    
    env = SDNEnvBalanced()
    
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
    
    class ActionDistCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.action_counts = {0: 0, 1: 0, 2: 0}
            self.total_steps = 0
            self.episode_actions = []
            self.current_episode_actions = []
            
        def _on_step(self):
            if len(self.locals.get('actions', [])) > 0:
                for a in self.locals['actions']:
                    if a in self.action_counts:
                        self.action_counts[a] += 1
                        self.total_steps += 1
                        self.current_episode_actions.append(a)
            return True
        
        def _on_rollout_end(self):
            if self.current_episode_actions:
                self.episode_actions.append(self.current_episode_actions.copy())
                self.current_episode_actions = []
            return True
        
        def get_distribution(self):
            if self.total_steps > 0:
                return {k: v / self.total_steps * 100 for k, v in self.action_counts.items()}
            return {0: 0, 1: 0, 2: 0}
        
        def get_episode_stats(self):
            if not self.episode_actions:
                return {}
            last_ep = self.episode_actions[-1]
            counts = {0: 0, 1: 0, 2: 0}
            for a in last_ep:
                counts[a] += 1
            return {k: v / len(last_ep) * 100 for k, v in counts.items()}
    
    callback = ActionDistCallback()
    
    print("Training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )
    
    model.save(save_path)
    
    print(f"\nModel saved to: {save_path}")
    print(f"Final action distribution: {callback.get_distribution()}")
    print(f"Last episode action dist: {callback.get_episode_stats()}")
    
    return model, callback


def benchmark_balanced(model_path, n_episodes=20):
    """Benchmark model với WRR."""
    from stable_baselines3.common.evaluation import evaluate_policy
    
    print("\n" + "=" * 60)
    print("BENCHMARKING")
    print("=" * 60)
    
    env = SDNEnvBalanced()
    
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # PPO evaluation
    ppo_throughputs, ppo_latencies, ppo_actions = [], [], []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_tp, ep_lat = 0, 0
        actions = []
        
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
            obs, reward, done, _, info = env.step(action)
            ep_tp += info.get('throughput', 0)
            ep_lat += info.get('latency', 0)
        
        ppo_throughputs.append(ep_tp / 200)
        ppo_latencies.append(ep_lat / 200)
        
        # Action distribution for this episode
        total = len(actions)
        action_dist = {i: actions.count(i) / total * 100 for i in range(3)}
        
        if ep == n_episodes - 1:
            print(f"PPO Action Distribution: {action_dist}")
    
    # WRR evaluation
    wrr_throughputs, wrr_latencies = [], []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_tp, ep_lat = 0, 0
        action = 0
        
        for _ in range(200):
            obs, reward, done, _, info = env.step(action % 3)
            action += 1
            ep_tp += info.get('throughput', 0)
            ep_lat += info.get('latency', 0)
        
        wrr_throughputs.append(ep_tp / 200)
        wrr_latencies.append(ep_lat / 200)
    
    print(f"\n--- Results ---")
    print(f"PPO Throughput: {np.mean(ppo_throughputs):.2f} ± {np.std(ppo_throughputs):.2f}")
    print(f"WRR Throughput: {np.mean(wrr_throughputs):.2f} ± {np.std(wrr_throughputs):.2f}")
    print(f"PPO Latency: {np.mean(ppo_latencies):.2f} ms")
    print(f"WRR Latency: {np.mean(wrr_latencies):.2f} ms")
    
    return {
        'ppo_throughput': np.mean(ppo_throughputs),
        'wrr_throughput': np.mean(wrr_throughputs),
        'ppo_latency': np.mean(ppo_latencies),
        'wrr_latency': np.mean(wrr_latencies)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--model', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.benchmark and args.model:
        benchmark_balanced(args.model)
    else:
        save_path = f"ai_model/checkpoints/ppo_balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        model, callback = train_balanced(args.timesteps, save_path)
        
        print("\nBenchmarking...")
        benchmark_balanced(save_path)
