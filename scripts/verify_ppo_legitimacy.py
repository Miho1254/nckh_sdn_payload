#!/usr/bin/env python3
"""
PPO Legitimacy Verification Tests
=================================

3 Critical Tests:
1. Cross-env test: Train on env A, test on env B
2. Ablation study: Remove reward components
3. Policy behavior check: Action distribution, load per server
"""

import os
import sys
import numpy as np
import json
from datetime import datetime
from scipy import stats

sys.path.insert(0, '/work')

from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces


# ============================================================================
# TEST 1: CROSS-ENVIRONMENT TEST
# ============================================================================

class SDNEnvVariantA(gym.Env):
    """Environment A: Original M/M/1 model"""
    
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.max_capacity = float(np.max(self.capacities))
        self.max_steps = 200
        self.current_step = 0
        self.state = None
        self.traffic_intensity = 0.3
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.rand(11).astype(np.float32) * 0.3
        self.current_step = 0
        self.traffic_intensity = np.random.uniform(0.2, 0.4)
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1
        if isinstance(action, np.ndarray):
            action = int(action[0]) if action.ndim > 0 else int(action)
        
        weights = np.zeros(3, dtype=np.float32)
        weights[action] = 1.0
        
        # M/M/1 latency model
        load_h5 = self.traffic_intensity * weights[0] / self.capacities[0] * self.max_capacity
        load_h7 = self.traffic_intensity * weights[1] / self.capacities[1] * self.max_capacity
        load_h8 = self.traffic_intensity * weights[2] / self.capacities[2] * self.max_capacity
        
        load_h5 = np.clip(load_h5, 0, 1)
        load_h7 = np.clip(load_h7, 0, 1)
        load_h8 = np.clip(load_h8, 0, 1)
        
        # M/M/1 exponential latency
        base_lat = 10.0
        lat_h5 = base_lat / (1 - load_h5 + 1e-8) if load_h5 < 0.99 else 1000.0
        lat_h7 = base_lat / (1 - load_h7 + 1e-8) if load_h7 < 0.99 else 1000.0
        lat_h8 = base_lat / (1 - load_h8 + 1e-8) if load_h8 < 0.99 else 1000.0
        
        avg_latency = (lat_h5 + lat_h7 + lat_h8) / 3
        throughput = self.traffic_intensity * (1 - 0.5 * (load_h5 + load_h7 + load_h8) / 3)
        
        latency_penalty = -0.1 * avg_latency
        overload_penalty = -10.0 * (max(0, load_h5 - 0.9) + max(0, load_h7 - 0.9) + max(0, load_h8 - 0.9))
        reward = throughput + latency_penalty + overload_penalty
        
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


class SDNEnvVariantB(gym.Env):
    """Environment B: Different latency model (linear + noise)"""
    
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.max_capacity = float(np.max(self.capacities))
        self.max_steps = 200
        self.current_step = 0
        self.state = None
        self.traffic_intensity = 0.3
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.rand(11).astype(np.float32) * 0.3
        self.current_step = 0
        self.traffic_intensity = np.random.uniform(0.2, 0.4)
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1
        if isinstance(action, np.ndarray):
            action = int(action[0]) if action.ndim > 0 else int(action)
        
        weights = np.zeros(3, dtype=np.float32)
        weights[action] = 1.0
        
        load_h5 = self.traffic_intensity * weights[0] / self.capacities[0] * self.max_capacity
        load_h7 = self.traffic_intensity * weights[1] / self.capacities[1] * self.max_capacity
        load_h8 = self.traffic_intensity * weights[2] / self.capacities[2] * self.max_capacity
        
        load_h5 = np.clip(load_h5, 0, 1)
        load_h7 = np.clip(load_h7, 0, 1)
        load_h8 = np.clip(load_h8, 0, 1)
        
        # LINEAR latency model (not M/M/1) + random noise
        base_lat = 10.0
        lat_h5 = base_lat * (1 + load_h5) + np.random.uniform(0, 5)
        lat_h7 = base_lat * (1 + load_h7) + np.random.uniform(0, 5)
        lat_h8 = base_lat * (1 + load_h8) + np.random.uniform(0, 5)
        
        avg_latency = (lat_h5 + lat_h7 + lat_h8) / 3
        throughput = self.traffic_intensity * (1 - 0.5 * (load_h5 + load_h7 + load_h8) / 3)
        
        latency_penalty = -0.1 * avg_latency
        overload_penalty = -10.0 * (max(0, load_h5 - 0.9) + max(0, load_h7 - 0.9) + max(0, load_h8 - 0.9))
        reward = throughput + latency_penalty + overload_penalty
        
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


class SDNEnvVariantC(gym.Env):
    """Environment C: Randomized capacities"""
    
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        # Randomized capacities each episode
        self.base_capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.capacities = self.base_capacities.copy()
        self.max_capacity = 100.0
        self.max_steps = 200
        self.current_step = 0
        self.state = None
        self.traffic_intensity = 0.3
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomize capacities ±30%
        self.capacities = self.base_capacities * np.random.uniform(0.7, 1.3, size=3)
        self.state = np.random.rand(11).astype(np.float32) * 0.3
        self.current_step = 0
        self.traffic_intensity = np.random.uniform(0.2, 0.4)
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1
        if isinstance(action, np.ndarray):
            action = int(action[0]) if action.ndim > 0 else int(action)
        
        weights = np.zeros(3, dtype=np.float32)
        weights[action] = 1.0
        
        load_h5 = self.traffic_intensity * weights[0] / self.capacities[0] * self.max_capacity
        load_h7 = self.traffic_intensity * weights[1] / self.capacities[1] * self.max_capacity
        load_h8 = self.traffic_intensity * weights[2] / self.capacities[2] * self.max_capacity
        
        load_h5 = np.clip(load_h5, 0, 1)
        load_h7 = np.clip(load_h7, 0, 1)
        load_h8 = np.clip(load_h8, 0, 1)
        
        base_lat = 10.0
        lat_h5 = base_lat / (1 - load_h5 + 1e-8) if load_h5 < 0.99 else 1000.0
        lat_h7 = base_lat / (1 - load_h7 + 1e-8) if load_h7 < 0.99 else 1000.0
        lat_h8 = base_lat / (1 - load_h8 + 1e-8) if load_h8 < 0.99 else 1000.0
        
        avg_latency = (lat_h5 + lat_h7 + lat_h8) / 3
        throughput = self.traffic_intensity * (1 - 0.5 * (load_h5 + load_h7 + load_h8) / 3)
        
        latency_penalty = -0.1 * avg_latency
        overload_penalty = -10.0 * (max(0, load_h5 - 0.9) + max(0, load_h7 - 0.9) + max(0, load_h8 - 0.9))
        reward = throughput + latency_penalty + overload_penalty
        
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


def test_cross_env(model_path, n_episodes=50):
    """Test 1: Cross-environment test."""
    
    print("\n" + "=" * 60)
    print("TEST 1: CROSS-ENVIRONMENT TEST")
    print("=" * 60)
    
    model = PPO.load(model_path)
    
    results = {}
    
    for env_name, env_class in [('A_M/M/1', SDNEnvVariantA), 
                                  ('B_Linear+Noise', SDNEnvVariantB),
                                  ('C_RandomCap', SDNEnvVariantC)]:
        env = env_class()
        
        ppo_rewards = []
        ppo_actions = [0, 0, 0]
        load_per_server = [[], [], []]
        
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
                
                # Track load per server
                if 'load_h5' in info:
                    load_per_server[0].append(info.get('load_h5', 0))
                    load_per_server[1].append(info.get('load_h7', 0))
                    load_per_server[2].append(info.get('load_h8', 0))
            
            ppo_rewards.append(total_reward)
        
        # WRR baseline
        wrr_rewards = []
        weights = np.array([1.0, 5.0, 10.0])
        weights = weights / weights.sum()
        
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
            
            wrr_rewards.append(total_reward)
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(ppo_rewards, wrr_rewards)
        
        results[env_name] = {
            'PPO_mean': np.mean(ppo_rewards),
            'PPO_std': np.std(ppo_rewards),
            'WRR_mean': np.mean(wrr_rewards),
            'WRR_std': np.std(wrr_rewards),
            't_stat': t_stat,
            'p_value': p_value,
            'action_dist': [c / sum(ppo_actions) for c in ppo_actions],
            'avg_load_h5': np.mean(load_per_server[0]) if load_per_server[0] else 0,
            'avg_load_h7': np.mean(load_per_server[1]) if load_per_server[1] else 0,
            'avg_load_h8': np.mean(load_per_server[2]) if load_per_server[2] else 0,
        }
        
        winner = "PPO" if results[env_name]['PPO_mean'] > results[env_name]['WRR_mean'] else "WRR"
        improvement = ((results[env_name]['PPO_mean'] - results[env_name]['WRR_mean']) / 
                       abs(results[env_name]['WRR_mean']) * 100) if results[env_name]['WRR_mean'] != 0 else 0
        
        print(f"\n{env_name}:")
        print(f"  PPO: {results[env_name]['PPO_mean']:.2f} ± {results[env_name]['PPO_std']:.2f}")
        print(f"  WRR: {results[env_name]['WRR_mean']:.2f} ± {results[env_name]['WRR_std']:.2f}")
        print(f"  Winner: {winner} ({improvement:+.1f}%)")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Action dist: [{results[env_name]['action_dist'][0]:.1%}, {results[env_name]['action_dist'][1]:.1%}, {results[env_name]['action_dist'][2]:.1%}]")
        print(f"  Avg load: h5={results[env_name]['avg_load_h5']:.2f}, h7={results[env_name]['avg_load_h7']:.2f}, h8={results[env_name]['avg_load_h8']:.2f}")
    
    return results


def test_policy_behavior(model_path, n_episodes=100):
    """Test 3: Policy behavior check."""
    
    print("\n" + "=" * 60)
    print("TEST 3: POLICY BEHAVIOR CHECK")
    print("=" * 60)
    
    model = PPO.load(model_path)
    env = SDNEnvVariantA()
    
    action_counts = [0, 0, 0]
    load_history = [[], [], []]
    throughput_history = []
    latency_history = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action[0]) if action.ndim > 0 else int(action)
            
            action_counts[action] += 1
            obs, reward, done, truncated, info = env.step(action)
            
            throughput_history.append(info.get('throughput', 0))
            latency_history.append(info.get('latency', 0))
    
    action_dist = [c / sum(action_counts) for c in action_counts]
    
    print(f"\nAction Distribution:")
    print(f"  h5 (weakest):  {action_dist[0]:.1%}")
    print(f"  h7 (medium):   {action_dist[1]:.1%}")
    print(f"  h8 (strongest): {action_dist[2]:.1%}")
    
    print(f"\nThroughput: {np.mean(throughput_history):.3f} ± {np.std(throughput_history):.3f}")
    print(f"Latency: {np.mean(latency_history):.2f} ± {np.std(latency_history):.2f}")
    
    # Check for bias
    if action_dist[2] > 0.95:
        print("\n⚠️ WARNING: Model is biased toward action 2 (h8)")
        print("   This could indicate:")
        print("   1. Environment bias (M/M/1 model favors strongest server)")
        print("   2. Reward shaping bias")
        print("   3. Legitimate optimal policy")
    
    return {
        'action_dist': action_dist,
        'throughput_mean': np.mean(throughput_history),
        'throughput_std': np.std(throughput_history),
        'latency_mean': np.mean(latency_history),
        'latency_std': np.std(latency_history),
    }


def main():
    print("=" * 60)
    print("PPO LEGITIMACY VERIFICATION")
    print("=" * 60)
    
    model_path = 'ai_model/checkpoints/ppo_unbiased_final.zip'
    
    # Test 1: Cross-env test
    cross_env_results = test_cross_env(model_path, n_episodes=50)
    
    # Test 3: Policy behavior
    policy_results = test_policy_behavior(model_path, n_episodes=100)
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    print("\n1. Cross-Environment Test:")
    for env_name, res in cross_env_results.items():
        winner = "PPO" if res['PPO_mean'] > res['WRR_mean'] else "WRR"
        print(f"   {env_name}: {winner} wins (p={res['p_value']:.4f})")
    
    print("\n2. Policy Behavior:")
    print(f"   Action distribution: [{policy_results['action_dist'][0]:.1%}, {policy_results['action_dist'][1]:.1%}, {policy_results['action_dist'][2]:.1%}]")
    
    # Check if PPO wins across all environments
    all_wins = all(res['PPO_mean'] > res['WRR_mean'] for res in cross_env_results.values())
    all_significant = all(res['p_value'] < 0.05 for res in cross_env_results.values())
    
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if all_wins and all_significant:
        print("✓ PPO wins across ALL environments with statistical significance")
        print("  → This suggests PPO learned a ROBUST policy, not just exploiting simulator")
    else:
        print("⚠️ PPO does NOT win consistently across environments")
        print("  → This suggests PPO may be OVERFITTING to specific environment")
    
    if policy_results['action_dist'][2] > 0.95:
        print("⚠️ Policy is heavily biased toward action 2 (h8)")
        print("  → May indicate environment bias or legitimate optimal policy")
    
    # Save results
    results = {
        'cross_env': cross_env_results,
        'policy_behavior': policy_results,
        'timestamp': datetime.now().isoformat()
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'ai_model/verification_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: ai_model/verification_{timestamp}.json")


if __name__ == '__main__':
    main()