#!/usr/bin/env python3
"""
Fine-tune PPO với dữ liệu từ Real Mininet (Online Learning)
============================================================

Script này cho phép fine-tune PPO model dựa trên dữ liệu thực tế từ Mininet benchmark.

Kết quả benchmark hiện tại:
- P99 Latency: PPO WIN (-8.7%)
- Jitter: PPO WIN (-7.1%)
- Packet Loss: WRR WIN (+0.8%)
- Throughput: WRR WIN (+5.2%)

Vấn đề: PPO tập trung vào latency/tail-latency nhưng không tối ưu throughput tốt.

Giải pháp: Fine-tune với reward mới nhấn mạnh throughput.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, '/work')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class SDNEnvV3FineTune(gym.Env):
    """
    Environment V3 đã fine-tuned cho throughput optimization.
    
    Changes so với V3 gốc:
    1. Tăng weight của throughput trong reward
    2. Giảm weight của latency penalty (vì PPO đã tốt ở latency)
    3. Thêm penalty cho packet loss cao
    4. Cân bằng hơn giữa các metrics
    """
    
    def __init__(self, scenario='normal'):
        super().__init__()
        
        self.scenario = scenario
        self.action_space = spaces.Discrete(3)
        
        # 20 features
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
        
        # Link utilization
        self.link_queue_lengths = np.zeros(3, dtype=np.float32)
        self.switch_queue_length = 0.0
        
        # Shared uplink capacity
        self.shared_uplink_capacity = 1000.0  # Mbps
        
        # Cache hit rate
        self.cache_hit_rate = 0.3
        
        # Load balancing tracking
        self.load_history = [[], [], []]
        
        # Episode statistics
        self.episode_stats = {
            'total_throughput': 0.0,
            'total_latency': 0.0,
            'total_packet_loss': 0.0,
            'max_queue': 0.0
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize capacities ±20%
        self.capacities = self.base_capacities * np.random.uniform(0.8, 1.2, size=3)
        
        # Randomize network conditions
        self.packet_loss_rate = np.random.uniform(0.005, 0.02)
        self.network_delay_variation = np.random.uniform(2.0, 8.0)
        
        # Random initial state
        self.state = np.zeros(20, dtype=np.float32)
        self.state[:3] = np.random.rand(3) * 0.3
        self.state[6] = self.traffic_intensity
        self.state[7] = 0.1
        self.state[8:11] = self.capacities / 150.0
        self.state[11:14] = 0.01
        self.state[14:18] = 0.0
        self.state[18] = 1.0  # balanced initially
        self.state[19] = 0.3
        
        self.current_step = 0
        self.in_burst = False
        
        self.episode_stats = {
            'total_throughput': 0.0,
            'total_latency': 0.0,
            'total_packet_loss': 0.0,
            'max_queue': 0.0
        }
        
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1
        
        # Update load based on action (which server gets traffic)
        loads = self.state[:3].copy()
        
        # Action: which server to send traffic to
        if action == 0:
            loads[0] = min(1.0, loads[0] + 0.3)
        elif action == 1:
            loads[1] = min(1.0, loads[1] + 0.3)
        else:
            loads[2] = min(1.0, loads[2] + 0.3)
        
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
        self.link_queue_lengths *= 0.9  # decay
        
        self.switch_queue_length = min(1.0, self.switch_queue_length + max(0, np.mean(loads) - 0.5))
        self.switch_queue_length *= 0.85
        
        # Linear latency model
        base_lat = self.base_latency * (1 + load_h5) + np.random.uniform(0, self.network_delay_variation)
        lat_h5 = base_lat + self.link_queue_lengths[0] * 50
        lat_h7 = self.base_latency * (1 + load_h7) + np.random.uniform(0, self.network_delay_variation) + self.link_queue_lengths[1] * 50
        lat_h8 = self.base_latency * (1 + load_h8) + np.random.uniform(0, self.network_delay_variation) + self.link_queue_lengths[2] * 50
        
        # Packet loss
        packet_loss_h5 = self.packet_loss_rate * (1 + load_h5 + self.link_queue_lengths[0])
        packet_loss_h7 = self.packet_loss_rate * (1 + load_h7 + self.link_queue_lengths[1])
        packet_loss_h8 = self.packet_loss_rate * (1 + load_h8 + self.link_queue_lengths[2])
        
        avg_latency = (lat_h5 + lat_h7 + lat_h8) / 3
        avg_packet_loss = (packet_loss_h5 + packet_loss_h7 + packet_loss_h8) / 3
        
        # Congestion-aware throughput
        eff_h5 = self.capacities[0] * (1 - packet_loss_h5) * (1 - load_h5) * (1 - self.link_queue_lengths[0])
        eff_h7 = self.capacities[1] * (1 - packet_loss_h7) * (1 - load_h7) * (1 - self.link_queue_lengths[1])
        eff_h8 = self.capacities[2] * (1 - packet_loss_h8) * (1 - load_h8) * (1 - self.link_queue_lengths[2])
        
        total_throughput = eff_h5 + eff_h7 + eff_h8
        
        # Load balance
        load_std = np.std(loads)
        load_balance = max(0, 1.0 - load_std * 3)
        
        # Cache effect (distributed load = better cache)
        cache_hit_rate = 0.3 + 0.4 * (1 - load_std)
        
        # ====== REWARD TUNING ======
        # Based on benchmark analysis:
        # - P99 Latency: PPO WIN (-8.7%) -> giảm penalty latency
        # - Jitter: PPO WIN (-7.1%) -> giảm penalty
        # - Packet Loss: WRR WIN (+0.8%) -> tăng penalty packet loss
        # - Throughput: WRR WIN (+5.2%) -> tăng weight throughput
        
        throughput_normalized = total_throughput / (self.max_capacity * 3)
        
        # Increase throughput weight (was 1.0, now 1.5)
        throughput_bonus = 1.5 * throughput_normalized
        
        # Reduce latency penalty (was -0.05, now -0.03)
        latency_penalty = -0.03 * avg_latency
        
        # Increase packet loss penalty (was -5.0, now -7.0)
        packet_loss_penalty = -7.0 * avg_packet_loss
        
        # Overload penalty
        overload_penalty = -3.0 * (max(0, load_h5 - 0.9) + max(0, load_h7 - 0.9) + max(0, load_h8 - 0.9))
        
        # Queue penalty
        queue_penalty = -2.0 * (self.switch_queue_length + np.sum(self.link_queue_lengths) / 3)
        
        # Balance bonus
        balance_bonus = load_balance * 0.15
        
        # Congestion penalty
        max_load = max(load_h5, load_h7, load_h8)
        min_load = min(load_h5, load_h7, load_h8)
        congestion_penalty = 0
        if max_load > 0.8 and min_load < 0.4:
            congestion_penalty = -0.3 * (max_load - 0.8)
        
        reward = (throughput_bonus + balance_bonus + congestion_penalty + 
                  latency_penalty + packet_loss_penalty + overload_penalty + queue_penalty)
        
        # Update state
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
            'cache_hit_rate': cache_hit_rate,
            'congestion': self.switch_queue_length,
            'load_h5': load_h5,
            'load_h7': load_h7,
            'load_h8': load_h8,
            'reward': reward
        }
        
        # Track episode stats
        self.episode_stats['total_throughput'] += total_throughput
        self.episode_stats['total_latency'] += avg_latency
        self.episode_stats['total_packet_loss'] += avg_packet_loss
        self.episode_stats['max_queue'] = max(self.episode_stats['max_queue'], self.switch_queue_length)
        
        return self.state, float(reward), done, False, info


def analyze_real_benchmark_data(benchmark_dir: str) -> Dict:
    """
    Phân tích dữ liệu từ real Mininet benchmark để hiểu PPO vs WRR behavior.
    """
    import pandas as pd
    
    print("\n" + "=" * 60)
    print("ANALYZING REAL MININET BENCHMARK DATA")
    print("=" * 60)
    
    results = {}
    
    for algo in ['ppo', 'wrr']:
        algo_dir = os.path.join(benchmark_dir, algo)
        if not os.path.exists(algo_dir):
            continue
        
        # Read flow stats
        flow_stats_path = os.path.join(algo_dir, 'flow_stats.csv')
        if os.path.exists(flow_stats_path):
            try:
                df = pd.read_csv(flow_stats_path, low_memory=False)
                total_packets = df['packet_count'].sum()
                total_bytes = df['byte_count'].sum()
                
                results[f'{algo}_total_packets'] = total_packets
                results[f'{algo}_total_bytes'] = total_bytes
                results[f'{algo}_flows'] = len(df)
            except Exception as e:
                print(f"Error reading {algo} flow_stats: {e}")
        
        # Read inference log for PPO
        if algo == 'ppo':
            inf_log_path = os.path.join(algo_dir, 'inference_log.csv')
            if os.path.exists(inf_log_path):
                try:
                    df = pd.read_csv(inf_log_path)
                    results['ppo_inference_count'] = len(df)
                    results['ppo_mean_inference_ms'] = df['inference_ms'].mean()
                    results['ppo_actions'] = df['action'].value_counts().to_dict()
                except Exception as e:
                    print(f"Error reading {algo} inference log: {e}")
    
    # Compare
    if 'ppo_total_packets' in results and 'wrr_total_packets' in results:
        diff = results['ppo_total_packets'] - results['wrr_total_packets']
        pct = (diff / results['wrr_total_packets']) * 100
        results['comparison'] = f"PPO: {results['ppo_total_packets']:,} vs WRR: {results['wrr_total_packets']:,} ({pct:+.1f}%)"
        results['winner'] = 'PPO' if diff > 0 else 'WRR'
    
    return results


def fine_tune_ppo(
    model_path: str,
    benchmark_dir: str,
    total_timesteps: int = 100_000,
    save_dir: str = 'ai_model/checkpoints'
) -> Tuple[PPO, Dict]:
    """
    Fine-tune PPO model dựa trên dữ liệu từ real Mininet benchmark.
    
    Args:
        model_path: Đường dẫn đến model hiện tại
        benchmark_dir: Đường dẫn đến benchmark results
        total_timesteps: Số timesteps để fine-tune
        save_dir: Thư mục lưu model
    
    Returns:
        Fine-tuned model và statistics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("FINE-TUNING PPO V3 - THROUGHPUT OPTIMIZED")
    print("=" * 70)
    
    # Analyze real benchmark data first
    benchmark_analysis = analyze_real_benchmark_data(benchmark_dir)
    
    print("\n--- Benchmark Analysis ---")
    for key, value in benchmark_analysis.items():
        if key != 'comparison':
            print(f"  {key}: {value}")
    print(f"  {benchmark_analysis.get('comparison', 'N/A')}")
    
    # Create fine-tuned environment
    env = SDNEnvV3FineTune()
    
    # Load existing model
    print(f"\nLoading existing model from: {model_path}")
    try:
        model = PPO.load(model_path, env=env)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Creating new model instead...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,  # Lower learning rate for fine-tuning
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,  # Smaller clip range for fine-tuning
            ent_coef=0.005,  # Less exploration
            verbose=1,
        )
    
    # Custom callback for monitoring
    class FineTuneCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_throughputs = []
            self.episode_latencies = []
            self.action_counts = {0: 0, 1: 0, 2: 0}
            
        def _on_step(self):
            if len(self.locals.get('actions', [])) > 0:
                for a in self.locals['actions']:
                    if a in self.action_counts:
                        self.action_counts[a] += 1
            return True
        
        def _on_rollout_end(self):
            if self.locals.get('infos'):
                for info in self.locals['infos']:
                    if 'throughput' in info:
                        self.episode_throughputs.append(info['throughput'])
                    if 'latency' in info:
                        self.episode_latencies.append(info['latency'])
            return True
    
    callback = FineTuneCallback()
    
    # Fine-tune
    print(f"\nFine-tuning for {total_timesteps:,} timesteps...")
    print("Key changes from V3:")
    print("  - Throughput weight: 1.0 → 1.5 (+50%)")
    print("  - Latency penalty: 0.05 → 0.03 (-40%)")
    print("  - Packet loss penalty: 5.0 → 7.0 (+40%)")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=False  # Continue from current step
    )
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_dir}/ppo_v3_finetuned_{timestamp}.zip"
    model.save(save_path)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {save_path}")
    print(f"Action distribution: {callback.action_counts}")
    
    if callback.episode_throughputs:
        print(f"\nThroughput: {np.mean(callback.episode_throughputs[-100:]):.2f}")
    if callback.episode_latencies:
        print(f"Latency: {np.mean(callback.episode_latencies[-100:]):.2f} ms")
    
    return model, {
        'action_distribution': callback.action_counts,
        'throughputs': callback.episode_throughputs,
        'latencies': callback.episode_latencies
    }


def benchmark_fine_tuned_model(model_path: str, n_episodes: int = 50) -> Dict:
    """
    Benchmark model mới với WRR để so sánh.
    """
    from stable_baselines3.common.evaluation import evaluate_policy
    
    print("\n" + "=" * 60)
    print("BENCHMARKING FINE-TUNED MODEL")
    print("=" * 60)
    
    env = SDNEnvV3FineTune()
    
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    
    print(f"Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    
    # Run detailed benchmark
    ppo_results = {'throughputs': [], 'latencies': [], 'packet_losses': [], 'actions': []}
    wrr_results = {'throughputs': [], 'latencies': [], 'packet_losses': []}
    
    for _ in range(n_episodes):
        # PPO
        obs, _ = env.reset()
        episode_throughput = 0
        episode_latency = 0
        episode_packet_loss = 0
        actions = []
        
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
            obs, reward, done, _, info = env.step(action)
            episode_throughput += info.get('throughput', 0)
            episode_latency += info.get('latency', 0)
            episode_packet_loss += info.get('packet_loss', 0)
        
        ppo_results['throughputs'].append(episode_throughput / 200)
        ppo_results['latencies'].append(episode_latency / 200)
        ppo_results['packet_losses'].append(episode_packet_loss / 200)
        ppo_results['actions'].extend(actions)
        
        # WRR (simulate)
        obs, _ = env.reset()
        episode_throughput = 0
        episode_latency = 0
        episode_packet_loss = 0
        wrr_action = 0
        
        for _ in range(200):
            # WRR cycles through servers
            obs, reward, done, _, info = env.step(wrr_action % 3)
            wrr_action += 1
            episode_throughput += info.get('throughput', 0)
            episode_latency += info.get('latency', 0)
            episode_packet_loss += info.get('packet_loss', 0)
        
        wrr_results['throughputs'].append(episode_throughput / 200)
        wrr_results['latencies'].append(episode_latency / 200)
        wrr_results['packet_losses'].append(episode_packet_loss / 200)
    
    # Compare
    print("\n--- Simulation Comparison ---")
    print(f"PPO Throughput: {np.mean(ppo_results['throughputs']):.2f}")
    print(f"WRR Throughput: {np.mean(wrr_results['throughputs']):.2f}")
    print(f"PPO Latency: {np.mean(ppo_results['latencies']):.2f} ms")
    print(f"WRR Latency: {np.mean(wrr_results['latencies']):.2f} ms")
    
    # Action distribution
    total_actions = len(ppo_results['actions'])
    action_dist = {i: ppo_results['actions'].count(i) / total_actions for i in range(3)}
    print(f"PPO Action Distribution: {action_dist}")
    
    return {
        'ppo': ppo_results,
        'wrr': wrr_results
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune PPO với dữ liệu Real Mininet')
    parser.add_argument('--model', type=str, default='ai_model/models/ppo_v3_real.zip',
                        help='Model path để fine-tune')
    parser.add_argument('--benchmark', type=str, default='benchmark_results_quick/golden_hour/run_1',
                        help='Benchmark results directory')
    parser.add_argument('--timesteps', type=int, default=100_000,
                        help='Số timesteps để fine-tune')
    parser.add_argument('--save-dir', type=str, default='ai_model/checkpoints',
                        help='Thư mục lưu model')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Chỉ benchmark model, không fine-tune')
    
    args = parser.parse_args()
    
    if args.benchmark_only:
        print("Benchmarking existing model...")
        results = benchmark_fine_tuned_model(args.model)
    else:
        print(f"Fine-tuning model: {args.model}")
        print(f"Using benchmark data from: {args.benchmark}")
        
        fine_tuned_model, stats = fine_tune_ppo(
            model_path=args.model,
            benchmark_dir=args.benchmark,
            total_timesteps=args.timesteps,
            save_dir=args.save_dir
        )
        
        print("\nBenchmarking fine-tuned model...")
        results = benchmark_fine_tuned_model(
            f"{args.save_dir}/ppo_v3_finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )
