#!/usr/bin/env python3
"""
PPO vs WRR Golden Signals Benchmark

So sánh AI vs WRR bằng Golden Signals trong SRE:
1. User Experience: Tail Latency (p95, p99), Jitter, Packet Loss
2. Performance: Goodput, Connection Queuing Time
3. Reliability: Error Rate, Time To Mitigate (TTM)

4 Kịch bản:
1. Golden Hour - Burst Traffic (Elephant + Mice)
2. Video Conference - Low Latency
3. Hardware Degradation - Server Throttling
4. Low Rate DoS - Anomalous Traffic
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdn_sim_env import (
    SDNLoadBalancerEnv, 
    GoldenHourEnv, 
    VideoConferenceEnv,
    HardwareDegradationEnv,
    LowRateDosEnv,
    make_env
)
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════════════════

class WRRBalancer:
    """Weighted Round Robin - baseline truyền thống."""
    def __init__(self):
        self.ratios = np.array([10.0, 50.0, 100.0])
        self.ratios = self.ratios / np.sum(self.ratios)
        self.counter = 0
        
    def predict(self, obs):
        return self.ratios.copy()
    
    def get_weights(self, stats):
        return self.ratios.copy()


class PPOBalancer:
    """PPO-based Load Balancer."""
    def __init__(self, model_path):
        self.model = PPO.load(model_path, device='cpu')
        
    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        weights = action / (np.sum(action) + 1e-8)
        return np.clip(weights, 0.0, 1.0)
    
    def get_weights(self, stats):
        # Convert stats to observation
        obs = np.zeros(6, dtype=np.float32)
        obs[0] = min(1.0, stats.get('h5', {}).get('cpu', 0) / 100.0)
        obs[1] = min(1.0, stats.get('h7', {}).get('cpu', 0) / 100.0)
        obs[2] = min(1.0, stats.get('h8', {}).get('cpu', 0) / 100.0)
        obs[3] = min(1.0, stats.get('h5', {}).get('latency', 10) / 500.0)
        obs[4] = min(1.0, stats.get('h7', {}).get('latency', 10) / 500.0)
        obs[5] = min(1.0, stats.get('h8', {}).get('latency', 10) / 500.0)
        return self.predict(obs)


# ═══════════════════════════════════════════════════════════════════════════
# METRICS TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class GoldenSignalsTracker:
    """Track Golden Signals for SRE-style evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # User Experience
        self.latencies = []  # All latencies
        self.jitters = []    # Latency variation
        self.packet_losses = []  # Packet loss events
        
        # Performance
        self.goodputs = []   # Successful requests
        self.garbage_throughput = []  # Failed/attack traffic
        self.queuing_times = []  # Time in queue
        
        # Reliability
        self.errors = []     # 5xx errors
        self.crashes = []   # Overload events
        self.ttm = []       # Time to mitigate (ms)
        
        # Per-server metrics
        self.server_latencies = {'h5': [], 'h7': [], 'h8': []}
        self.server_loads = {'h5': [], 'h7': [], 'h8': []}
        self.server_throughputs = {'h5': [], 'h7': [], 'h8': []}
        
    def record(self, info, weights, server_names=['h5', 'h7', 'h8']):
        """Record metrics from step info."""
        # Latency
        lat = info.get('latency', 10.0)
        self.latencies.append(lat)
        
        # Per-server latency (based on which got traffic)
        chosen_idx = np.argmax(weights)
        if chosen_idx < len(server_names):
            chosen_server = server_names[chosen_idx]
            self.server_latencies[chosen_server].append(lat)
        
        # Jitter (latency variation)
        if len(self.latencies) > 1:
            jitter = abs(self.latencies[-1] - self.latencies[-2])
            self.jitters.append(jitter)
        
        # Packet loss (simulated when overload)
        if info.get('overload', False):
            self.packet_losses.append(0.05)  # 5% loss during overload
        else:
            self.packet_losses.append(0.0)
        
        # Goodput (successful throughput)
        if not info.get('crash', False):
            self.goodputs.append(info.get('throughput', 0))
        else:
            self.errors.append(1)
        
        # Queuing time (estimated from queue length)
        queue = info.get('queue_length', 0)
        self.queuing_times.append(queue * 0.1)  # 0.1ms per queued item
        
        # Server loads
        for i, srv in enumerate(server_names):
            key = f'load_{srv}' if srv in ['h5', 'h7', 'h8'] else None
            if key and key in info:
                self.server_loads[srv].append(info[key])
        
        # Crashes
        if info.get('crash', False):
            self.crashes.append(1)
    
    def get_summary(self):
        """Compute summary metrics."""
        lat = np.array(self.latencies) if self.latencies else np.array([0])
        jit = np.array(self.jitters) if self.jitters else np.array([0])
        pl = np.array(self.packet_losses) if self.packet_losses else np.array([0])
        gp = np.array(self.goodputs) if self.goodputs else np.array([0])
        qt = np.array(self.queuing_times) if self.queuing_times else np.array([0])
        
        return {
            # User Experience
            'p50_latency': float(np.percentile(lat, 50)),
            'p95_latency': float(np.percentile(lat, 95)),
            'p99_latency': float(np.percentile(lat, 99)),
            'mean_jitter': float(np.mean(jit)) if len(jit) > 0 else 0.0,
            'packet_loss_rate': float(np.mean(pl)) * 100,  # percentage
            
            # Performance
            'mean_goodput': float(np.mean(gp)) if len(gp) > 0 else 0.0,
            'total_goodput': float(np.sum(gp)),
            'mean_queuing_time': float(np.mean(qt)) if len(qt) > 0 else 0.0,
            
            # Reliability
            'error_rate': float(len(self.errors)) / max(1, len(lat)) * 100,  # percentage
            'total_crashes': int(sum(self.crashes)),
            'total_requests': len(lat),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(env, balancer, n_episodes=5, max_steps=500):
    """Run scenario and collect Golden Signals."""
    tracker = GoldenSignalsTracker()
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        tracker.reset()
        
        for step in range(max_steps):
            # Get weights from balancer
            if hasattr(balancer, 'predict'):
                weights = balancer.predict(obs)
            else:
                weights = balancer.get_weights({})
            
            # Step environment
            action = weights.astype(np.float32)
            obs, reward, done, trunc, info = env.step(action)
            
            # Track metrics
            tracker.record(info, weights)
            
            if done:
                break
    
    return tracker.get_summary()


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK SUITE
# ═══════════════════════════════════════════════════════════════════════════

def run_golden_signals_benchmark():
    """Run comprehensive Golden Signals benchmark."""
    
    print("="*80)
    print("  PPO vs WRR - GOLDEN SIGNALS BENCHMARK")
    print("  4 Scenarios × 2 Baselines × 5 Episodes")
    print("="*80)
    
    # Load PPO model
    model_path = 'ai_model/ppo_sdn_load_balancer.zip'
    if not os.path.exists(model_path):
        print(f"[!] Model not found: {model_path}")
        print("[*] Run training first: python ai_model/train_ppo_simple.py")
        return
    
    ppo_balancer = PPOBalancer(model_path)
    wrr_balancer = WRRBalancer()
    
    # Define scenarios
    scenarios = {
        'Golden Hour': {
            'env': 'SDN-v0',
            'description': 'Burst Traffic - Mixed Elephant + Mice flows',
            'expected_advantage': 'PPO tránh h5 cho elephant flows'
        },
        'Video Conference': {
            'env': 'VideoConference-v0', 
            'description': 'Low Latency - Stable routing critical',
            'expected_advantage': 'PPO giảm jitter bằng consistent routing'
        },
        'Hardware Degradation': {
            'env': 'HardwareDegradation-v0',
            'description': 'Server throttling - Adaptive response',
            'expected_advantage': 'PPO phát hiện sớm, drain trước crash'
        },
        'Low Rate DoS': {
            'env': 'LowRateDoS-v0',
            'description': 'Slowloris attack - Tarpit strategy',
            'expected_advantage': 'PPO redirect attack vào h5, bảo vệ h8/h7'
        },
    }
    
    results = {}
    
    for scenario_name, scenario in scenarios.items():
        print(f"\n{'='*80}")
        print(f"  📊 {scenario_name.upper()}")
        print(f"  {scenario['description']}")
        print(f"  Expected: {scenario['expected_advantage']}")
        print("="*80)
        
        env_id = scenario['env']
        
        # Run PPO
        print(f"\n[*] Running PPO (5 episodes)...")
        env = make_env(env_id)
        ppo_metrics = run_scenario(env, ppo_balancer, n_episodes=5)
        
        # Run WRR
        print(f"[*] Running WRR (5 episodes)...")
        env = make_env(env_id)
        wrr_metrics = run_scenario(env, wrr_balancer, n_episodes=5)
        
        results[scenario_name] = {
            'ppo': ppo_metrics,
            'wrr': wrr_metrics,
        }
        
        # Print comparison
        print(f"\n{'Metric':<25} {'WRR':>15} {'PPO':>15} {'Winner':>10}")
        print("-"*60)
        
        # Latency comparison
        p99_wrr = wrr_metrics['p99_latency']
        p99_ppo = ppo_metrics['p99_latency']
        winner = 'PPO' if p99_ppo < p99_wrr else 'WRR'
        print(f"{'p99 Latency (ms)':<25} {p99_wrr:>15.1f} {p99_ppo:>15.1f} {winner:>10}")
        
        # Goodput
        gp_wrr = wrr_metrics['mean_goodput']
        gp_ppo = ppo_metrics['mean_goodput']
        winner = 'PPO' if gp_ppo > gp_wrr else 'WRR'
        print(f"{'Mean Goodput':<25} {gp_wrr:>15.3f} {gp_ppo:>15.3f} {winner:>10}")
        
        # Error Rate
        er_wrr = wrr_metrics['error_rate']
        er_ppo = ppo_metrics['error_rate']
        winner = 'PPO' if er_ppo < er_wrr else 'WRR'
        print(f"{'Error Rate (%)':<25} {er_wrr:>15.2f} {er_ppo:>15.2f} {winner:>10}")
        
        # Crashes
        cr_wrr = wrr_metrics['total_crashes']
        cr_ppo = ppo_metrics['total_crashes']
        winner = 'PPO' if cr_ppo < cr_wrr else 'WRR'
        print(f"{'Total Crashes':<25} {cr_wrr:>15d} {cr_ppo:>15d} {winner:>10}")
        
        # Packet Loss
        pl_wrr = wrr_metrics['packet_loss_rate']
        pl_ppo = ppo_metrics['packet_loss_rate']
        winner = 'PPO' if pl_ppo < pl_wrr else 'WRR'
        print(f"{'Packet Loss (%)':<25} {pl_wrr:>15.2f} {pl_ppo:>15.2f} {winner:>10}")
        
        # Jitter
        jit_wrr = wrr_metrics['mean_jitter']
        jit_ppo = ppo_metrics['mean_jitter']
        winner = 'PPO' if jit_ppo < jit_wrr else 'WRR'
        print(f"{'Mean Jitter (ms)':<25} {jit_wrr:>15.2f} {jit_ppo:>15.2f} {winner:>10}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("  📈 SUMMARY: PPO vs WRR WIN COUNTS")
    print("="*80)
    
    wins = {'ppo': 0, 'wrr': 0}
    metrics_compared = ['p99_latency', 'mean_goodput', 'error_rate', 'total_crashes', 'packet_loss_rate']
    
    for scenario, data in results.items():
        print(f"\n{scenario}:")
        for metric in metrics_compared:
            p_val = data['ppo'][metric]
            w_val = data['wrr'][metric]
            
            # Lower is better for: latency, error, crashes, packet_loss
            if metric in ['p99_latency', 'error_rate', 'total_crashes', 'packet_loss_rate']:
                winner = 'PPO' if p_val < w_val else 'WRR'
            else:  # Higher is better for: goodput
                winner = 'PPO' if p_val > w_val else 'WRR'
            
            if winner == 'PPO':
                wins['ppo'] += 1
            else:
                wins['wrr'] += 1
    
    print(f"\n{'='*80}")
    print(f"  🏆 TOTAL WINS: PPO {wins['ppo']} - {wins['wrr']} WRR")
    
    if wins['ppo'] > wins['wrr']:
        print("  ✅ PPO OUTPERFORMS WRR in Golden Signals!")
    elif wins['ppo'] == wins['wrr']:
        print("  ⚖️ PPO = WRR - PARITY")
    else:
        print("  ⚠️ WRR still competitive in some metrics")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'ai_model/benchmark_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[✓] Results saved to: {results_file}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = run_golden_signals_benchmark()
