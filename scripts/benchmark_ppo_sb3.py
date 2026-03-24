#!/usr/bin/env python3
"""
Benchmark PPO SB3 vs WRR
========================
"""

import os
import sys
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, '/work')

from stable_baselines3 import PPO


class SDNEnvGymnasium:
    """Simple SDN environment for benchmarking."""
    
    def __init__(self):
        # Server capacities
        self.capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.max_capacity = float(np.max(self.capacities))
        self.max_steps = 200
        self.current_step = 0
        self.state = None
        self.traffic_intensity = 0.3
        self.burst_probability = 0.15
        self.in_burst = False
        self.burst_duration = 0
        self.burst_intensity = 0.0
        
    def reset(self, seed=None):
        self.state = np.random.rand(11).astype(np.float32) * 0.3
        self.state[6] = 0.0
        self.state[7] = 0.3
        self.state[8] = self.capacities[0] / 150.0
        self.state[9] = self.capacities[1] / 150.0
        self.state[10] = self.capacities[2] / 150.0
        self.current_step = 0
        self.traffic_intensity = np.random.uniform(0.2, 0.4)
        self.in_burst = False
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1
        
        # Convert discrete action to weight vector
        if isinstance(action, np.ndarray):
            action = int(action[0]) if action.ndim > 0 else int(action)
        
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
        
        # Calculate load
        load_h5 = self.traffic_intensity * weights[0] / self.capacities[0] * self.max_capacity
        load_h7 = self.traffic_intensity * weights[1] / self.capacities[1] * self.max_capacity
        load_h8 = self.traffic_intensity * weights[2] / self.capacities[2] * self.max_capacity
        
        load_h5 = np.clip(load_h5, 0, 1)
        load_h7 = np.clip(load_h7, 0, 1)
        load_h8 = np.clip(load_h8, 0, 1)
        
        # Latency
        base_lat = 10.0
        lat_h5 = base_lat / (1 - load_h5 + 1e-8) if load_h5 < 0.99 else 1000.0
        lat_h7 = base_lat / (1 - load_h7 + 1e-8) if load_h7 < 0.99 else 1000.0
        lat_h8 = base_lat / (1 - load_h8 + 1e-8) if load_h8 < 0.99 else 1000.0
        
        avg_latency = (lat_h5 + lat_h7 + lat_h8) / 3
        
        # Throughput
        throughput = self.traffic_intensity * (1 - 0.5 * (load_h5 + load_h7 + load_h8) / 3)
        
        # Reward
        latency_penalty = -0.1 * avg_latency
        overload_penalty = -10.0 * (max(0, load_h5 - 0.9) + max(0, load_h7 - 0.9) + max(0, load_h8 - 0.9))
        
        if action == 2:
            capacity_bonus = 1.0
        elif action == 1:
            capacity_bonus = 0.5
        else:
            capacity_bonus = -0.5
        
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
        
        done = self.current_step >= self.max_steps
        
        return self.state, float(reward), done, False, {'throughput': throughput, 'latency': avg_latency}


def benchmark_ppo(model_path, n_episodes=20):
    """Benchmark PPO model."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking PPO: {model_path}")
    print(f"{'='*60}\n")
    
    # Load model
    model = PPO.load(model_path)
    
    env = SDNEnvGymnasium()
    
    episode_rewards = []
    action_counts = [0, 0, 0]
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action[0]) if action.ndim > 0 else int(action)
            
            action_counts[action] += 1
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        episode_rewards.append(total_reward)
        print(f"  Episode {ep+1}: reward={total_reward:.2f}")
    
    results = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'action_distribution': [c / sum(action_counts) for c in action_counts],
        'n_episodes': n_episodes
    }
    
    print(f"\nPPO Results:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Action Distribution: [{results['action_distribution'][0]:.1%}, {results['action_distribution'][1]:.1%}, {results['action_distribution'][2]:.1%}]")
    
    return results


def benchmark_wrr(n_episodes=20):
    """Benchmark WRR policy."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking WRR")
    print(f"{'='*60}\n")
    
    env = SDNEnvGymnasium()
    
    # WRR weights
    weights = np.array([1.0, 5.0, 10.0], dtype=np.float32)
    weights = weights / weights.sum()
    
    episode_rewards = []
    action_counts = [0, 0, 0]
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # WRR action selection
            r = np.random.random()
            if r < weights[0]:
                action = 0
            elif r < weights[0] + weights[1]:
                action = 1
            else:
                action = 2
            
            action_counts[action] += 1
            
            # Convert to weight vector for environment
            weights_vec = np.zeros(3, dtype=np.float32)
            weights_vec[action] = 1.0
            
            # Step with weight vector
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        episode_rewards.append(total_reward)
        print(f"  Episode {ep+1}: reward={total_reward:.2f}")
    
    results = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'action_distribution': [c / sum(action_counts) for c in action_counts],
        'n_episodes': n_episodes
    }
    
    print(f"\nWRR Results:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Action Distribution: [{results['action_distribution'][0]:.1%}, {results['action_distribution'][1]:.1%}, {results['action_distribution'][2]:.1%}]")
    
    return results


def main():
    print("=" * 60)
    print("PPO SB3 vs WRR BENCHMARK")
    print("=" * 60)
    
    # Benchmark PPO
    ppo_results = benchmark_ppo('ai_model/checkpoints/ppo_v4_sb3_final.zip', n_episodes=20)
    
    # Benchmark WRR
    wrr_results = benchmark_wrr(n_episodes=20)
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    print(f"\nPPO SB3:")
    print(f"  Mean Reward: {ppo_results['mean_reward']:.2f} ± {ppo_results['std_reward']:.2f}")
    print(f"  Action Dist: [{ppo_results['action_distribution'][0]:.1%}, {ppo_results['action_distribution'][1]:.1%}, {ppo_results['action_distribution'][2]:.1%}]")
    
    print(f"\nWRR:")
    print(f"  Mean Reward: {wrr_results['mean_reward']:.2f} ± {wrr_results['std_reward']:.2f}")
    print(f"  Action Dist: [{wrr_results['action_distribution'][0]:.1%}, {wrr_results['action_distribution'][1]:.1%}, {wrr_results['action_distribution'][2]:.1%}]")
    
    winner = "PPO" if ppo_results['mean_reward'] > wrr_results['mean_reward'] else "WRR"
    improvement = ((ppo_results['mean_reward'] - wrr_results['mean_reward']) / abs(wrr_results['mean_reward']) * 100) if wrr_results['mean_reward'] != 0 else 0
    
    print(f"\nWinner: {winner} ({improvement:+.1f}%)")
    
    # Save results
    results = {
        'PPO_SB3': ppo_results,
        'WRR': wrr_results,
        'winner': winner,
        'improvement_pct': improvement
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'ai_model/benchmark_ppo_sb3_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: ai_model/benchmark_ppo_sb3_{timestamp}.json")


if __name__ == '__main__':
    main()