#!/usr/bin/env python3
"""
Benchmark PPO Fixed vs WRR on Fixed SDN Environment
=====================================================
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from collections import defaultdict

from sdn_env_fixed import (
    make_fixed_env,
    SDNProductionFixedEnv,
    FixedBurstEnv,
    FixedHighNoiseEnv,
    FixedLowSLAEnv,
    FixedDynamicCapacityEnv,
)


class WRRPolicy:
    """Weighted Round Robin - capacity-weighted distribution."""
    
    def __init__(self, capacities=None):
        if capacities is None:
            capacities = np.array([10.0, 50.0, 100.0])
        self.capacities = capacities
        self.weights = capacities / capacities.sum()
    
    def predict(self, obs, deterministic=True):
        return self.weights.copy(), None


class AdaptiveWRRPolicy:
    """Adaptive WRR that adjusts based on current load."""
    
    def __init__(self, capacities=None):
        if capacities is None:
            capacities = np.array([10.0, 50.0, 100.0])
        self.capacities = capacities
        self.base_weights = capacities / capacities.sum()
    
    def predict(self, obs, deterministic=True):
        # Extract queue lengths from observation
        queue_norm = obs[0:3]
        
        adjusted_weights = self.base_weights.copy()
        
        for i in range(3):
            if queue_norm[i] > 0.8:
                adjusted_weights[i] *= 0.5
            elif queue_norm[i] > 0.5:
                adjusted_weights[i] *= 0.8
        
        if adjusted_weights.sum() > 0:
            adjusted_weights /= adjusted_weights.sum()
        else:
            adjusted_weights = self.base_weights
        
        return adjusted_weights, None


class RandomPolicy:
    def predict(self, obs, deterministic=True):
        weights = np.random.dirichlet(np.ones(3))
        return weights, None


def evaluate_policy(env, policy, n_episodes=10, max_steps=200, policy_name="Policy"):
    all_metrics = defaultdict(list)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_metrics = defaultdict(list)
        
        for step in range(max_steps):
            action, _ = policy.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            episode_metrics['latency'].append(info['p99_latency'])
            episode_metrics['packet_loss'].append(info['packet_loss'])
            episode_metrics['throughput'].append(info['throughput'])
            episode_metrics['sla_violation'].append(info['sla_violation'])
            episode_metrics['overload_events'].append(info['overload_events'])
            
            if done:
                break
        
        all_metrics['episode_reward'].append(episode_reward)
        all_metrics['avg_p99_latency'].append(np.mean(episode_metrics['latency']))
        all_metrics['avg_packet_loss'].append(np.mean(episode_metrics['packet_loss']))
        all_metrics['avg_throughput'].append(np.mean(episode_metrics['throughput']))
        all_metrics['total_sla_violations'].append(sum(episode_metrics['sla_violation']))
        all_metrics['total_overload_events'].append(sum(episode_metrics['overload_events']))
    
    results = {
        'policy': policy_name,
        'n_episodes': n_episodes,
        'avg_reward': np.mean(all_metrics['episode_reward']),
        'std_reward': np.std(all_metrics['episode_reward']),
        'avg_p99_latency': np.mean(all_metrics['avg_p99_latency']),
        'avg_packet_loss': np.mean(all_metrics['avg_packet_loss']),
        'avg_throughput': np.mean(all_metrics['avg_throughput']),
        'total_sla_violations': int(np.sum(all_metrics['total_sla_violations'])),
        'total_overload_events': int(np.sum(all_metrics['total_overload_events'])),
    }
    
    return results


def run_benchmark(ppo_model_path=None, scenarios=None, n_episodes=20, max_steps=200, seed=42):
    if scenarios is None:
        scenarios = [
            'SDNFixed-v0',
            'SDNFixedBurst-v0',
            'SDNFixedHighNoise-v0',
            'SDNFixedLowSLA-v0',
            'SDNFixedDynamic-v0',
        ]
    
    print("=" * 80)
    print("FIXED ENVIRONMENT BENCHMARK")
    print("=" * 80)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Episodes per scenario: {n_episodes}")
    print(f"Seed: {seed}")
    print("=" * 80)
    
    np.random.seed(seed)
    all_results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*80}")
        
        env = make_fixed_env(scenario, seed=seed)
        
        policies = {
            'Random': RandomPolicy(),
            'WRR': WRRPolicy(),
            'AdaptiveWRR': AdaptiveWRRPolicy(),
        }
        
        if ppo_model_path and os.path.exists(ppo_model_path):
            try:
                from stable_baselines3 import PPO
                ppo_model = PPO.load(ppo_model_path)
                policies['PPO-Fixed'] = ppo_model
                print(f"Loaded PPO model from: {ppo_model_path}")
            except Exception as e:
                print(f"Warning: Could not load PPO model: {e}")
        
        scenario_results = {}
        
        for policy_name, policy in policies.items():
            print(f"\nEvaluating: {policy_name}")
            print("-" * 40)
            
            results = evaluate_policy(env, policy, n_episodes=n_episodes, max_steps=max_steps, policy_name=policy_name)
            
            print(f"  Avg Reward: {results['avg_reward']:.1f} ± {results['std_reward']:.1f}")
            print(f"  Avg P99 Latency: {results['avg_p99_latency']:.1f}ms")
            print(f"  Avg Packet Loss: {results['avg_packet_loss']:.4f}")
            print(f"  Avg Throughput: {results['avg_throughput']:.3f}")
            print(f"  SLA Violations: {results['total_sla_violations']}")
            print(f"  Overload Events: {results['total_overload_events']}")
            
            scenario_results[policy_name] = results
        
        all_results[scenario] = scenario_results
    
    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Scenario':<20} {'Policy':<15} {'Reward':>10} {'P99 Lat':>10} {'Loss':>8} {'SLA':>5}")
    print("-" * 80)
    
    for scenario, results in all_results.items():
        for policy_name, metrics in results.items():
            print(f"{scenario:<20} {policy_name:<15} "
                  f"{metrics['avg_reward']:>10.1f} "
                  f"{metrics['avg_p99_latency']:>10.1f} "
                  f"{metrics['avg_packet_loss']:>8.4f} "
                  f"{metrics['total_sla_violations']:>5}")
        print("-" * 80)
    
    # Find winner per scenario
    print("\nWINNER PER SCENARIO:")
    for scenario, results in all_results.items():
        best_policy = max(results.items(), key=lambda x: x[1]['avg_reward'])
        print(f"  {scenario}: {best_policy[0]} (reward: {best_policy[1]['avg_reward']:.1f})")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"benchmark_fixed_{timestamp}.json"
    
    serializable_results = {}
    for scenario, policies in all_results.items():
        serializable_results[scenario] = {}
        for policy, metrics in policies.items():
            serializable_results[scenario][policy] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark PPO Fixed vs WRR')
    parser.add_argument('--ppo-model', type=str, default=None, help='Path to trained PPO model')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes per scenario')
    parser.add_argument('--max-steps', type=int, default=200, help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    results = run_benchmark(
        ppo_model_path=args.ppo_model,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )