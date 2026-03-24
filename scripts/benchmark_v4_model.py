#!/usr/bin/env python3
"""
Benchmark V4 Model on Simulated Environment
=============================================

Compare V4 TFT-AC model vs WRR on SDN simulation environment.
"""

import json
import time
import os
import sys
import numpy as np
from datetime import datetime

# Add path for imports
sys.path.insert(0, '/work')


def run_v4_benchmark(scenario='normal', n_episodes=20, max_steps=200):
    """Run V4 model benchmark."""
    
    print(f"\n{'='*60}")
    print(f"Running V4 Model Benchmark: {scenario}")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*60}\n")
    
    try:
        import torch
        from ai_model.tft_ac_net import TFT_ActorCritic_Model
        from ai_model.sdn_sim_env import SDNLoadBalancerEnv, make_env
        
        # Create environment
        env = make_env('SDN-v0', seed=42)
        
        # Load V4 model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TFT_ActorCritic_Model(
            input_size=44,
            seq_len=5,
            hidden_size=64,
            num_actions=3
        ).to(device)
        
        model_path = 'ai_model/checkpoints/tft_ac_v4_best.pth'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded V4 model from {model_path}")
            print(f"  Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.2%}")
        else:
            print(f"Model not found: {model_path}")
            return None
        
        model.eval()
        
        # Run episodes
        episode_rewards = []
        episode_throughputs = []
        episode_latencies = []
        episode_sla_violations = []
        action_counts = [0, 0, 0]
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            total_reward = 0
            total_throughput = 0
            total_latency = 0
            total_sla_violations = 0
            steps = 0
            
            # History for sequence
            obs_history = [obs] * 5
            
            done = False
            while not done and steps < max_steps:
                # Update history
                obs_history.pop(0)
                obs_history.append(obs)
                
                # Prepare input
                obs_array = np.array(obs_history)  # [5, obs_dim]
                
                # Pad/truncate to 44 features
                if obs_array.shape[1] < 44:
                    padding = np.zeros((5, 44 - obs_array.shape[1]))
                    obs_array = np.concatenate([obs_array, padding], axis=1)
                else:
                    obs_array = obs_array[:, :44]
                
                # Get action from model
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0).to(device)
                    outputs = model(obs_tensor)
                    policy = outputs[4]  # safety_probs
                    action_idx = policy.argmax(dim=1).item()
                
                action_counts[action_idx] += 1
                
                # Convert discrete action to weight vector
                # Action 0 -> h5 (weakest), Action 1 -> h7 (medium), Action 2 -> h8 (strongest)
                # WRR weights: [1, 5, 10] for h5, h7, h8
                if action_idx == 0:
                    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # All to h5
                elif action_idx == 1:
                    action = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # All to h7
                else:
                    action = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # All to h8
                
                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                if 'throughput' in info:
                    total_throughput += info['throughput']
                if 'latency' in info:
                    total_latency += info['latency']
                if 'sla_violations' in info:
                    total_sla_violations += info['sla_violations']
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_throughputs.append(total_throughput / steps if steps > 0 else 0)
            episode_latencies.append(total_latency / steps if steps > 0 else 0)
            episode_sla_violations.append(total_sla_violations)
            
            print(f"  Episode {ep+1}: reward={total_reward:.2f}, steps={steps}")
        
        # Calculate statistics
        results = {
            'scenario': scenario,
            'policy': 'V4_TFT_AC',
            'n_episodes': n_episodes,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_throughput': float(np.mean(episode_throughputs)),
            'mean_latency': float(np.mean(episode_latencies)),
            'mean_sla_violations': float(np.mean(episode_sla_violations)),
            'action_distribution': [c / sum(action_counts) for c in action_counts],
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\nV4 Results:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Throughput: {results['mean_throughput']:.2f}")
        print(f"  Mean Latency: {results['mean_latency']:.2f}")
        print(f"  Action Distribution: [{results['action_distribution'][0]:.1%}, {results['action_distribution'][1]:.1%}, {results['action_distribution'][2]:.1%}]")
        
        return results
        
    except Exception as e:
        print(f"Error running V4 benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_wrr_benchmark(scenario='normal', n_episodes=20, max_steps=200):
    """Run WRR benchmark for comparison."""
    
    print(f"\n{'='*60}")
    print(f"Running WRR Benchmark: {scenario}")
    print(f"{'='*60}\n")
    
    try:
        from ai_model.sdn_sim_env import SDNLoadBalancerEnv, make_env
        
        env = make_env('SDN-v0', seed=42)
        
        # WRR weights based on capacity (h5:10, h7:50, h8:100)
        # Ratio: 1:5:10
        weights = [1, 5, 10]
        weight_sum = sum(weights)
        
        n_episodes = n_episodes
        episode_rewards = []
        episode_throughputs = []
        episode_latencies = []
        episode_sla_violations = []
        action_counts = [0, 0, 0]
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            total_reward = 0
            total_throughput = 0
            total_latency = 0
            total_sla_violations = 0
            steps = 0
            
            done = False
            while not done and steps < max_steps:
                # WRR action selection - use capacity-weighted distribution
                # weights = [1, 5, 10] -> normalized = [0.0625, 0.3125, 0.625]
                action = np.array([1.0, 5.0, 10.0], dtype=np.float32)
                action = action / action.sum()  # Normalize to [0.0625, 0.3125, 0.625]
                
                # Track action distribution (for logging)
                r = np.random.random() * weight_sum
                if r < weights[0]:
                    action_idx = 0
                elif r < weights[0] + weights[1]:
                    action_idx = 1
                else:
                    action_idx = 2
                action_counts[action_idx] += 1
                
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                if 'throughput' in info:
                    total_throughput += info['throughput']
                if 'latency' in info:
                    total_latency += info['latency']
                if 'sla_violations' in info:
                    total_sla_violations += info['sla_violations']
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_throughputs.append(total_throughput / steps if steps > 0 else 0)
            episode_latencies.append(total_latency / steps if steps > 0 else 0)
            episode_sla_violations.append(total_sla_violations)
            
            print(f"  Episode {ep+1}: reward={total_reward:.2f}, steps={steps}")
        
        results = {
            'scenario': scenario,
            'policy': 'WRR',
            'n_episodes': n_episodes,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_throughput': float(np.mean(episode_throughputs)),
            'mean_latency': float(np.mean(episode_latencies)),
            'mean_sla_violations': float(np.mean(episode_sla_violations)),
            'action_distribution': [c / sum(action_counts) for c in action_counts],
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\nWRR Results:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Throughput: {results['mean_throughput']:.2f}")
        print(f"  Action Distribution: [{results['action_distribution'][0]:.1%}, {results['action_distribution'][1]:.1%}, {results['action_distribution'][2]:.1%}]")
        
        return results
        
    except Exception as e:
        print(f"Error running WRR benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run full benchmark comparison."""
    
    print("=" * 60)
    print("V4 MODEL vs WRR BENCHMARK")
    print("=" * 60)
    
    scenarios = ['SDN-v0']  # Default scenario
    
    all_results = {
        'V4_TFT_AC': {},
        'WRR': {}
    }
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*60}")
        
        # Run V4 model benchmark
        v4_results = run_v4_benchmark(scenario=scenario, n_episodes=20, max_steps=200)
        if v4_results:
            all_results['V4_TFT_AC'][scenario] = v4_results
        
        # Run WRR benchmark
        wrr_results = run_wrr_benchmark(scenario=scenario, n_episodes=20, max_steps=200)
        if wrr_results:
            all_results['WRR'][scenario] = wrr_results
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    for scenario in scenarios:
        print(f"\n{scenario}:")
        if scenario in all_results['V4_TFT_AC']:
            v4_r = all_results['V4_TFT_AC'][scenario]['mean_reward']
            v4_s = all_results['V4_TFT_AC'][scenario]['std_reward']
            print(f"  V4 TFT-AC: {v4_r:.2f} ± {v4_s:.2f}")
        if scenario in all_results['WRR']:
            wrr_r = all_results['WRR'][scenario]['mean_reward']
            wrr_s = all_results['WRR'][scenario]['std_reward']
            print(f"  WRR:       {wrr_r:.2f} ± {wrr_s:.2f}")
        
        if scenario in all_results['V4_TFT_AC'] and scenario in all_results['WRR']:
            v4_reward = all_results['V4_TFT_AC'][scenario]['mean_reward']
            wrr_reward = all_results['WRR'][scenario]['mean_reward']
            winner = "V4 TFT-AC" if v4_reward > wrr_reward else "WRR"
            improvement = ((v4_reward - wrr_reward) / abs(wrr_reward) * 100) if wrr_reward != 0 else 0
            print(f"  Winner: {winner} ({improvement:+.1f}%)")
            
            # Statistical significance (t-test)
            from scipy import stats
            # We don't have raw data, so we can't do proper t-test
            # But we can report the difference
            print(f"  Difference: {v4_reward - wrr_reward:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'ai_model/benchmark_v4_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == '__main__':
    main()