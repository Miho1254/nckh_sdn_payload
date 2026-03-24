#!/usr/bin/env python3
"""
PPO vs WRR Benchmark

So sánh hiệu năng của PPO-based Load Balancer với WRR truyền thống.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdn_sim_env import SDNLoadBalancerEnv, make_env
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')


class WRRBalancer:
    """Weighted Round Robin - baseline truyền thống."""
    
    def __init__(self):
        # Capacity ratios: h5=10, h7=50, h8=100 → total=160
        self.ratios = np.array([10.0, 50.0, 100.0])
        self.ratios = self.ratios / np.sum(self.ratios)  # [0.0625, 0.3125, 0.625]
        self.counter = 0
        
    def predict(self, obs):
        """Return WRR weights."""
        # WRR cố định, không quan tâm obs
        return self.ratios.copy()
    
    def get_weights(self, stats):
        """Return fixed WRR weights."""
        return self.ratios.copy()


class RandomBalancer:
    """Random baseline - thử ngẫu nhiên."""
    
    def predict(self, obs):
        weights = np.random.rand(3)
        return weights / np.sum(weights)
    
    def get_weights(self, stats):
        weights = np.random.rand(3)
        return weights / np.sum(weights)


def run_episode(env, balancer, max_steps=200):
    """Chạy một episode với balancer."""
    obs, _ = env.reset()
    total_reward = 0
    total_latency = 0
    crash_count = 0
    steps = 0
    
    for step in range(max_steps):
        # Get action/weights from balancer
        if hasattr(balancer, 'predict'):
            weights = balancer.predict(obs)
        else:
            weights = balancer.get_weights({})
        
        # Convert weights to action (đánh giá theo traffic distribution)
        # Map weights sang action space của env
        # Ưu tiên server có weight cao nhất
        action_idx = np.argmax(weights)
        
        # Create action as probability distribution
        action = weights.astype(np.float32)
        
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        total_latency += info.get('latency', 0)
        if info.get('crash', False):
            crash_count += 1
        steps += 1
        
        if done:
            break
    
    return {
        'total_reward': total_reward,
        'avg_reward': total_reward / max(1, steps),
        'avg_latency': total_latency / max(1, steps),
        'crash_count': crash_count,
        'steps': steps,
        'weights_used': weights.tolist(),
    }


def run_benchmark(n_episodes=10):
    """Run benchmark comparison."""
    
    print("="*70)
    print("  PPO vs WRR vs RANDOM BENCHMARK")
    print("="*70)
    
    results = {
        'ppo': [],
        'wrr': [],
        'random': [],
    }
    
    # Load PPO model
    print("\n[*] Loading PPO model...")
    ppo_balancer = PPOLoadBalancer('ai_model/ppo_sdn_load_balancer.zip')
    wrr_balancer = WRRBalancer()
    random_balancer = RandomBalancer()
    
    # Run episodes
    for episode in range(n_episodes):
        print(f"\n[*] Episode {episode + 1}/{n_episodes}...")
        
        # PPO
        env = make_env('SDN-v0')
        ppo_result = run_episode(env, ppo_balancer)
        results['ppo'].append(ppo_result)
        
        # WRR  
        env = make_env('SDN-v0')
        wrr_result = run_episode(env, wrr_balancer)
        results['wrr'].append(wrr_result)
        
        # Random
        env = make_env('SDN-v0')
        random_result = run_episode(env, random_balancer)
        results['random'].append(random_result)
        
        print(f"    PPO:    reward={ppo_result['avg_reward']:.2f}, "
              f"latency={ppo_result['avg_latency']:.1f}ms, "
              f"crashes={ppo_result['crash_count']}")
        print(f"    WRR:    reward={wrr_result['avg_reward']:.2f}, "
              f"latency={wrr_result['avg_latency']:.1f}ms, "
              f"crashes={wrr_result['crash_count']}")
        print(f"    Random: reward={random_result['avg_reward']:.2f}, "
              f"latency={random_result['avg_latency']:.1f}ms, "
              f"crashes={random_result['crash_count']}")
    
    # Summary
    print("\n" + "="*70)
    print("  BENCHMARK RESULTS")
    print("="*70)
    
    for name, data in results.items():
        rewards = [d['avg_reward'] for d in data]
        latencies = [d['avg_latency'] for d in data]
        crashes = [d['crash_count'] for d in data]
        
        print(f"\n{name.upper()}:")
        print(f"  Avg Reward:    {np.mean(rewards):7.2f} ± {np.std(rewards):.2f}")
        print(f"  Avg Latency:  {np.mean(latencies):7.1f} ± {np.std(latencies):.1f} ms")
        print(f"  Total Crashes:{np.sum(crashes):7d}")
    
    # Winner
    print("\n" + "="*70)
    print("  WINNER ANALYSIS")
    print("="*70)
    
    ppo_reward = np.mean([d['avg_reward'] for d in results['ppo']])
    wrr_reward = np.mean([d['avg_reward'] for d in results['wrr']])
    random_reward = np.mean([d['avg_reward'] for d in results['random']])
    
    ppo_latency = np.mean([d['avg_latency'] for d in results['ppo']])
    wrr_latency = np.mean([d['avg_latency'] for d in results['wrr']])
    random_latency = np.mean([d['avg_latency'] for d in results['random']])
    
    ppo_crashes = sum(d['crash_count'] for d in results['ppo'])
    wrr_crashes = sum(d['crash_count'] for d in results['wrr'])
    random_crashes = sum(d['crash_count'] for d in results['random'])
    
    print(f"\n{'Metric':<20} {'PPO':>12} {'WRR':>12} {'Random':>12} {'Winner':>10}")
    print("-" * 70)
    print(f"{'Avg Reward':<20} {ppo_reward:>12.2f} {wrr_reward:>12.2f} {random_reward:>12.2f} "
          f"{'PPO' if ppo_reward > wrr_reward else 'WRR':>10}")
    print(f"{'Avg Latency (ms)':<20} {ppo_latency:>12.1f} {wrr_latency:>12.1f} {random_latency:>12.1f} "
          f"{'PPO' if ppo_latency < wrr_latency else 'WRR':>10}")
    print(f"{'Total Crashes':<20} {ppo_crashes:>12d} {wrr_crashes:>12d} {random_crashes:>12d} "
          f"{'PPO' if ppo_crashes < wrr_crashes else 'WRR':>10}")
    
    print("\n" + "="*70)
    
    # PPO advantages
    ppo_wins = 0
    if ppo_reward > wrr_reward:
        ppo_wins += 1
    if ppo_latency < wrr_latency:
        ppo_wins += 1
    if ppo_crashes < wrr_crashes:
        ppo_wins += 1
    
    print(f"\n🏆 PPO wins {ppo_wins}/3 categories vs WRR")
    
    if ppo_wins >= 2:
        print("✅ PPO OUTPERFORMS WRR - AI-BASED LOAD BALANCING IS EFFECTIVE!")
    else:
        print("⚠️ WRR still competitive - more training may help")
    
    return results


if __name__ == "__main__":
    # Import PPOLoadBalancer
    sys.path.insert(0, 'ai_model')
    from ppo_load_balancer import PPOLoadBalancer
    
    results = run_benchmark(n_episodes=10)
    print("\n[✓] Benchmark complete!")
