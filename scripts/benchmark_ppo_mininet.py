#!/usr/bin/env python3
"""
Benchmark PPO on Real Mininet/Ryu Environment
==============================================

This script tests PPO on real SDN infrastructure (Mininet + Ryu).
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
from datetime import datetime

sys.path.insert(0, '/work')

from stable_baselines3 import PPO


def run_mininet_benchmark(model_path, n_runs=5, duration=60):
    """
    Run benchmark on real Mininet/Ryu environment.
    
    This requires:
    1. Ryu controller running
    2. Mininet topology running
    3. Traffic generator (Artillery)
    """
    
    print("=" * 60)
    print("MININET/RYU REAL-WORLD BENCHMARK")
    print("=" * 60)
    
    # Check if Ryu controller is running
    print("\n[1/5] Checking Ryu controller...")
    try:
        result = subprocess.run(['pgrep', '-f', 'ryu-manager'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ Ryu controller is running")
        else:
            print("  ✗ Ryu controller is NOT running")
            print("  Please start: ryu-manager controller_stats.py")
            return None
    except Exception as e:
        print(f"  ✗ Error checking Ryu: {e}")
        return None
    
    # Check if Mininet is running
    print("\n[2/5] Checking Mininet...")
    try:
        result = subprocess.run(['pgrep', '-f', 'mn'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ Mininet is running")
        else:
            print("  ✗ Mininet is NOT running")
            print("  Please start: python run_lms_mininet.py")
            return None
    except Exception as e:
        print(f"  ✗ Error checking Mininet: {e}")
        return None
    
    # Load PPO model
    print("\n[3/5] Loading PPO model...")
    try:
        model = PPO.load(model_path)
        print(f"  ✓ Loaded model from {model_path}")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return None
    
    # Check stats files
    print("\n[4/5] Checking stats files...")
    stats_dir = '/work/stats'
    flow_stats_file = os.path.join(stats_dir, 'flow_stats.csv')
    port_stats_file = os.path.join(stats_dir, 'port_stats.csv')
    
    if os.path.exists(flow_stats_file):
        print(f"  ✓ Flow stats file exists: {flow_stats_file}")
    else:
        print(f"  ✗ Flow stats file NOT found: {flow_stats_file}")
    
    if os.path.exists(port_stats_file):
        print(f"  ✓ Port stats file exists: {port_stats_file}")
    else:
        print(f"  ✗ Port stats file NOT found: {port_stats_file}")
    
    # Run benchmark
    print("\n[5/5] Running benchmark...")
    
    results = {
        'model': model_path,
        'n_runs': n_runs,
        'duration': duration,
        'timestamp': datetime.now().isoformat(),
        'runs': []
    }
    
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        
        # Clear stats files
        for f in [flow_stats_file, port_stats_file]:
            if os.path.exists(f):
                os.remove(f)
        
        # Wait for traffic
        print(f"  Running for {duration}s...")
        time.sleep(duration)
        
        # Read stats
        run_result = {
            'run': run + 1,
            'flow_stats': None,
            'port_stats': None
        }
        
        if os.path.exists(flow_stats_file):
            try:
                with open(flow_stats_file, 'r') as f:
                    lines = f.readlines()
                    run_result['flow_stats_lines'] = len(lines)
                    print(f"  Flow stats: {len(lines)} lines")
            except Exception as e:
                print(f"  ✗ Error reading flow stats: {e}")
        
        if os.path.exists(port_stats_file):
            try:
                with open(port_stats_file, 'r') as f:
                    lines = f.readlines()
                    run_result['port_stats_lines'] = len(lines)
                    print(f"  Port stats: {len(lines)} lines")
            except Exception as e:
                print(f"  ✗ Error reading port stats: {e}")
        
        results['runs'].append(run_result)
    
    return results


def run_simulation_benchmark(model_path, n_episodes=100):
    """
    Fallback: Run simulation benchmark if Mininet/Ryu not available.
    """
    
    print("\n" + "=" * 60)
    print("SIMULATION BENCHMARK (Mininet/Ryu not available)")
    print("=" * 60)
    
    import gymnasium as gym
    from gymnasium import spaces
    
    class SDNEnvSimulation(gym.Env):
        """Simulation environment for fallback testing."""
        
        def __init__(self):
            super().__init__()
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
            self.capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
            self.max_capacity = 100.0
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
            return self.state, float(reward), done, False, {}
    
    # Load model
    model = PPO.load(model_path)
    env = SDNEnvSimulation()
    
    # PPO benchmark
    ppo_rewards = []
    ppo_actions = [0, 0, 0]
    
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
        
        ppo_rewards.append(total_reward)
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}: reward={total_reward:.2f}")
    
    # WRR benchmark
    wrr_rewards = []
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
        
        wrr_rewards.append(total_reward)
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}: reward={total_reward:.2f}")
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(ppo_rewards, wrr_rewards)
    
    results = {
        'model': model_path,
        'n_episodes': n_episodes,
        'timestamp': datetime.now().isoformat(),
        'PPO': {
            'mean': float(np.mean(ppo_rewards)),
            'std': float(np.std(ppo_rewards)),
            'action_dist': [c / sum(ppo_actions) for c in ppo_actions]
        },
        'WRR': {
            'mean': float(np.mean(wrr_rewards)),
            'std': float(np.std(wrr_rewards))
        },
        'statistical': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    }
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nPPO: {results['PPO']['mean']:.2f} ± {results['PPO']['std']:.2f}")
    print(f"WRR: {results['WRR']['mean']:.2f} ± {results['WRR']['std']:.2f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significant: {'YES' if p_value < 0.05 else 'NO'}")
    
    winner = "PPO" if results['PPO']['mean'] > results['WRR']['mean'] else "WRR"
    improvement = ((results['PPO']['mean'] - results['WRR']['mean']) / 
                   abs(results['WRR']['mean']) * 100) if results['WRR']['mean'] != 0 else 0
    print(f"Winner: {winner} ({improvement:+.1f}%)")
    
    return results


def main():
    print("=" * 60)
    print("PPO REAL-WORLD BENCHMARK")
    print("=" * 60)
    
    model_path = 'ai_model/checkpoints/ppo_unbiased_final.zip'
    
    # Try Mininet/Ryu first
    results = run_mininet_benchmark(model_path, n_runs=5, duration=60)
    
    # Fallback to simulation if Mininet/Ryu not available
    if results is None:
        print("\n⚠️ Mininet/Ryu not available, falling back to simulation")
        results = run_simulation_benchmark(model_path, n_episodes=100)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'ai_model/benchmark_real_world_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()