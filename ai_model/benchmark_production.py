#!/usr/bin/env python3
"""
Benchmark PPO vs WRR on Production SDN Environment
====================================================

Compare PPO (trained on production environment) vs WRR heuristic
across multiple production scenarios.

Key metrics:
- P99 latency (SLA compliance)
- Packet loss
- Throughput
- Overload events
- Burst handling

Author: Research Team
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from collections import defaultdict

# Import production environment
from sdn_env_production import (
    make_production_env,
    SDNProductionEnv,
    BurstTrafficEnv,
    HighNoiseEnv,
    LowSLAEnv,
    DynamicCapacityEnv,
)


class WRRPolicy:
    """Weighted Round Robin - capacity-weighted distribution."""
    
    def __init__(self, capacities=None):
        if capacities is None:
            capacities = np.array([10.0, 50.0, 100.0])
        self.capacities = capacities
        self.weights = capacities / capacities.sum()
    
    def predict(self, obs, deterministic=True):
        """Return capacity-weighted distribution."""
        return self.weights.copy(), None
    
    def set_capacities(self, capacities):
        """Update capacities (for dynamic capacity env)."""
        self.capacities = capacities
        self.weights = capacities / capacities.sum()


class AdaptiveWRRPolicy:
    """Adaptive WRR that adjusts based on current load."""
    
    def __init__(self, capacities=None):
        if capacities is None:
            capacities = np.array([10.0, 50.0, 100.0])
        self.capacities = capacities
        self.base_weights = capacities / capacities.sum()
    
    def predict(self, obs, deterministic=True):
        """
        Adjust weights based on current queue lengths.
        
        obs structure (22 dims):
        - [0:3] queue_norm
        - [3:6] lat_norm
        - [6:9] loss_per_server
        - [9:12] rtt_norm
        - [12:15] trend_onehot
        - [15] recent_throughput
        - [16:19] cap_norm
        - [19] global_load
        - [20] burst_time
        - [21] suspicious
        """
        # Extract queue lengths from observation
        queue_norm = obs[0:3]
        lat_norm = obs[3:6]
        cap_norm = obs[16:19]
        
        # If queue is full on a server, reduce its weight
        adjusted_weights = self.base_weights.copy()
        
        for i in range(3):
            if queue_norm[i] > 0.8:  # Queue > 80%
                adjusted_weights[i] *= 0.5
            elif queue_norm[i] > 0.5:  # Queue > 50%
                adjusted_weights[i] *= 0.8
        
        # Normalize
        if adjusted_weights.sum() > 0:
            adjusted_weights /= adjusted_weights.sum()
        else:
            adjusted_weights = self.base_weights
        
        return adjusted_weights, None


class RandomPolicy:
    """Random load balancing."""
    
    def predict(self, obs, deterministic=True):
        weights = np.random.dirichlet(np.ones(3))
        return weights, None


def evaluate_policy(env, policy, n_episodes=10, max_steps=500, policy_name="Policy"):
    """
    Evaluate a policy on the environment.
    
    Returns:
        dict: Metrics including latency, loss, throughput, SLA violations
    """
    
    all_metrics = defaultdict(list)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_metrics = defaultdict(list)
        
        for step in range(max_steps):
            # Get action from policy
            action, _ = policy.predict(obs)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # Track metrics
            episode_metrics['latency'].append(info['p99_latency'])
            episode_metrics['packet_loss'].append(info['packet_loss'])
            episode_metrics['throughput'].append(info['throughput'])
            episode_metrics['sla_violation'].append(info['sla_violation'])
            episode_metrics['overload_events'].append(info['overload_events'])
            episode_metrics['burst_intensity'].append(info['burst_intensity'])
            
            if done:
                break
        
        # Episode stats
        all_metrics['episode_reward'].append(episode_reward)
        all_metrics['avg_p99_latency'].append(np.mean(episode_metrics['latency']))
        all_metrics['max_p99_latency'].append(np.max(episode_metrics['latency']))
        all_metrics['avg_packet_loss'].append(np.mean(episode_metrics['packet_loss']))
        all_metrics['max_packet_loss'].append(np.max(episode_metrics['packet_loss']))
        all_metrics['avg_throughput'].append(np.mean(episode_metrics['throughput']))
        all_metrics['total_sla_violations'].append(sum(episode_metrics['sla_violation']))
        all_metrics['total_overload_events'].append(sum(episode_metrics['overload_events']))
        all_metrics['burst_handling_rate'].append(
            np.mean([b for b in episode_metrics['burst_intensity'] if b > 0.3]) 
            if any(b > 0.3 for b in episode_metrics['burst_intensity']) else 0
        )
    
    # Aggregate stats
    results = {
        'policy': policy_name,
        'n_episodes': n_episodes,
        'avg_reward': np.mean(all_metrics['episode_reward']),
        'std_reward': np.std(all_metrics['episode_reward']),
        'avg_p99_latency': np.mean(all_metrics['avg_p99_latency']),
        'max_p99_latency': np.max(all_metrics['max_p99_latency']),
        'avg_packet_loss': np.mean(all_metrics['avg_packet_loss']),
        'max_packet_loss': np.max(all_metrics['max_packet_loss']),
        'avg_throughput': np.mean(all_metrics['avg_throughput']),
        'total_sla_violations': int(np.sum(all_metrics['total_sla_violations'])),
        'total_overload_events': int(np.sum(all_metrics['total_overload_events'])),
        'burst_handling_rate': np.mean(all_metrics['burst_handling_rate']),
    }
    
    return results


def run_benchmark(
    ppo_model_path=None,
    scenarios=None,
    n_episodes=20,
    max_steps=500,
    seed=42
):
    """
    Run benchmark comparing PPO vs WRR vs Random on production scenarios.
    
    Args:
        ppo_model_path: Path to trained PPO model
        scenarios: List of scenario names
        n_episodes: Number of episodes per scenario
        max_steps: Max steps per episode
        seed: Random seed
    """
    
    if scenarios is None:
        scenarios = [
            'SDNProduction-v0',
            'SDNBurst-v0',
            'SDNHighNoise-v0',
            'SDNLowSLA-v0',
            'SDNDynamicCapacity-v0',
        ]
    
    print("=" * 80)
    print("PRODUCTION ENVIRONMENT BENCHMARK")
    print("=" * 80)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Episodes per scenario: {n_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Seed: {seed}")
    print("=" * 80)
    
    np.random.seed(seed)
    
    all_results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*80}")
        
        # Create environment
        env = make_production_env(scenario, seed=seed)
        
        # Define policies
        policies = {
            'Random': RandomPolicy(),
            'WRR': WRRPolicy(),
            'AdaptiveWRR': AdaptiveWRRPolicy(),
        }
        
        # Load PPO if available
        if ppo_model_path and os.path.exists(ppo_model_path):
            try:
                from stable_baselines3 import PPO
                ppo_model = PPO.load(ppo_model_path)
                policies['PPO'] = ppo_model
                print(f"Loaded PPO model from: {ppo_model_path}")
            except Exception as e:
                print(f"Warning: Could not load PPO model: {e}")
        
        scenario_results = {}
        
        for policy_name, policy in policies.items():
            print(f"\nEvaluating: {policy_name}")
            print("-" * 40)
            
            # Evaluate
            results = evaluate_policy(
                env, policy, 
                n_episodes=n_episodes, 
                max_steps=max_steps,
                policy_name=policy_name
            )
            
            # Print results
            print(f"  Avg Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"  Avg P99 Latency: {results['avg_p99_latency']:.1f}ms (max: {results['max_p99_latency']:.1f}ms)")
            print(f"  Avg Packet Loss: {results['avg_packet_loss']:.4f} (max: {results['max_packet_loss']:.4f})")
            print(f"  Avg Throughput: {results['avg_throughput']:.3f}")
            print(f"  SLA Violations: {results['total_sla_violations']}")
            print(f"  Overload Events: {results['total_overload_events']}")
            print(f"  Burst Handling: {results['burst_handling_rate']:.3f}")
            
            scenario_results[policy_name] = results
        
        all_results[scenario] = scenario_results
    
    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Create comparison table
    print(f"\n{'Scenario':<20} {'Policy':<12} {'Reward':>10} {'P99 Lat':>10} {'Loss':>8} {'SLA':>5}")
    print("-" * 80)
    
    for scenario, results in all_results.items():
        for policy_name, metrics in results.items():
            print(f"{scenario:<20} {policy_name:<12} "
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
    output_file = f"benchmark_production_{timestamp}.json"
    
    # Convert to serializable format
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


def analyze_wrr_weaknesses(results):
    """
    Analyze where WRR fails compared to other policies.
    
    Returns:
        dict: Scenarios where WRR is NOT the best
    """
    
    weaknesses = {}
    
    for scenario, policies in results.items():
        # Find best policy
        best_policy = max(policies.items(), key=lambda x: x[1]['avg_reward'])
        
        if best_policy[0] != 'WRR':
            weaknesses[scenario] = {
                'best_policy': best_policy[0],
                'best_reward': best_policy[1]['avg_reward'],
                'wrr_reward': policies['WRR']['avg_reward'],
                'improvement': best_policy[1]['avg_reward'] - policies['WRR']['avg_reward'],
                'improvement_pct': (best_policy[1]['avg_reward'] - policies['WRR']['avg_reward']) / abs(policies['WRR']['avg_reward']) * 100,
            }
    
    return weaknesses


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark PPO vs WRR on Production SDN')
    parser.add_argument('--ppo-model', type=str, default=None,
                        help='Path to trained PPO model')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of episodes per scenario')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--scenarios', type=str, nargs='+', default=None,
                        help='Scenarios to evaluate')
    
    args = parser.parse_args()
    
    results = run_benchmark(
        ppo_model_path=args.ppo_model,
        scenarios=args.scenarios,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    
    # Analyze WRR weaknesses
    weaknesses = analyze_wrr_weaknesses(results)
    
    if weaknesses:
        print("\n" + "=" * 80)
        print("WRR WEAKNESSES IDENTIFIED")
        print("=" * 80)
        for scenario, info in weaknesses.items():
            print(f"\n{scenario}:")
            print(f"  Best Policy: {info['best_policy']}")
            print(f"  Improvement: {info['improvement']:.1f} ({info['improvement_pct']:.1f}%)")
    else:
        print("\nWRR is optimal in all scenarios. Consider adding more challenging scenarios.")