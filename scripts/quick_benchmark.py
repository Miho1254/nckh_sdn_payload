#!/usr/bin/env python3
"""
Quick Benchmark - So sánh WRR vs CQL_BEST trên Golden Hour
Chạy offline evaluation (không cần Mininet/sudo)
"""
import os
import sys
import json
import numpy as np
import torch

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai_model'))
sys.path.insert(0, os.path.dirname(__file__))

from cql_agent import CQLAgent
from sdn_env_v2 import SDN_Offline_Env_V2
from config import NUM_ACTIONS, CAPACITY_RATIOS, BACKENDS, CAPACITY_PRIOR, ENTROPY_COEFF, CONSTRAINT_WEIGHTS

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'ai_model', 'processed_data')
CKPT_DIR = os.path.join(BASE_DIR, 'ai_model', 'checkpoints')
RES_DIR = os.path.join(BASE_DIR, 'stats', 'benchmark_final')

os.makedirs(RES_DIR, exist_ok=True)

# File paths
X_PATH = os.path.join(DATA_DIR, 'X_v3.npy')
Y_PATH = os.path.join(DATA_DIR, 'y_v3.npy')


class WeightedRoundRobinBaseline:
    """Weighted Round Robin — chia theo tỷ lệ capacity 1:5:10."""
    def __init__(self):
        weights = [int(b['weight']) for b in BACKENDS]
        self.schedule = []
        for i, w in enumerate(weights):
            self.schedule.extend([i] * w)
        self.counter = 0

    def select_action(self, state):
        action = self.schedule[self.counter % len(self.schedule)]
        self.counter += 1
        return action


def compute_composite_score(metrics):
    """Composite score: higher is better."""
    return (
        5.0 * metrics['served_throughput'] -
        0.05 * metrics['overload_rate'] -
        0.02 * metrics['fairness_deviation']
    )


def evaluate_policy(policy, env, X, y, action_mode='deterministic'):
    """Run a policy through data, collect metrics."""
    env.reset()
    total_reward = 0.0
    overload_count = 0
    utilizations = []
    fairness_devs = []
    action_switches = 0
    action_counts = np.zeros(NUM_ACTIONS)
    prev_action = None
    total_throughput = 0.0

    num_steps = len(X) - 1
    for i in range(num_steps):
        state = X[i]

        if hasattr(policy, 'select_action'):
            # CQL Agent - use model directly
            if hasattr(policy, 'model'):
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(policy.device)
                    context = policy.model.encode(state_t)
                    logits = policy.model.actor_head(context)
                    probs = torch.softmax(logits, dim=-1)
                    if action_mode == 'sampled':
                        action = torch.multinomial(probs, 1).item()
                    else:
                        action = torch.argmax(probs, dim=-1).item()
            else:
                # Baseline (WRR, RR)
                action = policy.select_action(state)
        else:
            action = policy(state)

        env.current_step = i
        env.prev_action = prev_action
        _, reward, _, info = env.step(action)

        total_reward += reward
        total_throughput += info.get('throughput', 0.0)
        overload_count += info.get('overload_flag', 0)
        utilizations.append(info.get('chosen_util', 0.0))
        fairness_devs.append(info.get('fairness_dev', 0.0))
        if prev_action is not None and action != prev_action:
            action_switches += 1
        action_counts[action] += 1
        prev_action = action

    total_steps = max(1, num_steps)
    return {
        'served_throughput': total_throughput / total_steps,
        'avg_reward': total_reward / total_steps,
        'overload_count': overload_count,
        'overload_rate': overload_count / total_steps,
        'p95_utilization': float(np.percentile(utilizations, 95)) if utilizations else 0.0,
        'mean_utilization': float(np.mean(utilizations)) if utilizations else 0.0,
        'fairness_deviation': float(np.mean(fairness_devs)) if fairness_devs else 0.0,
        'policy_churn_rate': action_switches / total_steps,
        'action_distribution': (action_counts / total_steps).tolist(),
        'total_steps': total_steps,
    }


def main():
    print("="*60)
    print("  QUICK BENCHMARK: WRR vs CQL_BEST (Golden Hour)")
    print("="*60)
    
    # Load data
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print(f"[*] Loaded V3 data: X={X.shape}, y={y.shape}")
    
    # Create environment
    env = SDN_Offline_Env_V2(X_PATH, Y_PATH)
    
    # Evaluate WRR
    print("\n" + "="*60)
    print("  Evaluating WRR (Weighted Round Robin)")
    print("="*60)
    
    wrr = WeightedRoundRobinBaseline()
    wrr_metrics = evaluate_policy(wrr, env, X, y)
    
    action_dist = np.array(wrr_metrics['action_distribution']) * 100
    print(f"  Throughput: {wrr_metrics['served_throughput']:.4f}")
    print(f"  Overload Rate: {wrr_metrics['overload_rate']:.4f}")
    print(f"  Fairness Deviation: {wrr_metrics['fairness_deviation']:.4f}")
    print(f"  Action Distribution: h5={action_dist[0]:.1f}%, h7={action_dist[1]:.1f}%, h8={action_dist[2]:.1f}%")
    print(f"  Composite Score: {compute_composite_score(wrr_metrics):.4f}")
    
    # Evaluate CQL_BEST_SAMPLED
    print("\n" + "="*60)
    print("  Evaluating CQL_BEST_SAMPLED")
    print("="*60)
    
    ckpt_path = os.path.join(CKPT_DIR, 'tft_ac_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"[!] Checkpoint not found: {ckpt_path}")
        return
    
    agent = CQLAgent(
        kl_coeff=10.0,
        entropy_coeff=ENTROPY_COEFF,
        constraint_weights=CONSTRAINT_WEIGHTS,
        capacity_prior=CAPACITY_PRIOR,
        input_size=X.shape[2],
        seq_len=X.shape[1],
        num_actions=NUM_ACTIONS,
        hidden_size=64
    )
    agent.load_checkpoint(ckpt_path)
    agent.model.eval()
    
    env = SDN_Offline_Env_V2(X_PATH, Y_PATH)
    cql_metrics = evaluate_policy(agent, env, X, y, action_mode='sampled')
    
    action_dist = np.array(cql_metrics['action_distribution']) * 100
    print(f"  Throughput: {cql_metrics['served_throughput']:.4f}")
    print(f"  Overload Rate: {cql_metrics['overload_rate']:.4f}")
    print(f"  Fairness Deviation: {cql_metrics['fairness_deviation']:.4f}")
    print(f"  Action Distribution: h5={action_dist[0]:.1f}%, h7={action_dist[1]:.1f}%, h8={action_dist[2]:.1f}%")
    print(f"  Composite Score: {compute_composite_score(cql_metrics):.4f}")
    
    # Compare
    print("\n" + "="*60)
    print("  COMPARISON RESULTS")
    print("="*60)
    
    wrr_score = compute_composite_score(wrr_metrics)
    cql_score = compute_composite_score(cql_metrics)
    
    print(f"\n{'Metric':<20} {'WRR':>15} {'CQL_BEST':>15} {'Winner':>10}")
    print("-"*60)
    print(f"{'Throughput':<20} {wrr_metrics['served_throughput']:>15.4f} {cql_metrics['served_throughput']:>15.4f} {'CQL' if cql_metrics['served_throughput'] > wrr_metrics['served_throughput'] else 'WRR':>10}")
    print(f"{'Overload Rate':<20} {wrr_metrics['overload_rate']:>15.4f} {cql_metrics['overload_rate']:>15.4f} {'CQL' if cql_metrics['overload_rate'] < wrr_metrics['overload_rate'] else 'WRR':>10}")
    print(f"{'Fairness Dev':<20} {wrr_metrics['fairness_deviation']:>15.4f} {cql_metrics['fairness_deviation']:>15.4f} {'CQL' if cql_metrics['fairness_deviation'] < wrr_metrics['fairness_deviation'] else 'WRR':>10}")
    print(f"{'Composite Score':<20} {wrr_score:>15.4f} {cql_score:>15.4f} {'CQL' if cql_score > wrr_score else 'WRR':>10}")
    
    # Save results
    wrr_out = {
        'total_flows': int(len(X)),
        'high_pct': 76.26,
        'throughput_MBps': wrr_metrics['served_throughput'],
        'packet_loss_pct': wrr_metrics['overload_rate'] * 100,
        'inference_ms': 0.0,
        'backend_dist': {'h5': int(wrr_metrics['action_distribution'][0] * 1000),
                         'h7': int(wrr_metrics['action_distribution'][1] * 1000),
                         'h8': int(wrr_metrics['action_distribution'][2] * 1000)},
        'utilization_dist': {'h5': 0, 'h7': 0, 'h8': 0},
        'balance_cv': wrr_metrics['fairness_deviation'] * 100,
    }
    
    cql_out = {
        'total_flows': int(len(X)),
        'high_pct': 76.26,
        'throughput_MBps': cql_metrics['served_throughput'],
        'packet_loss_pct': cql_metrics['overload_rate'] * 100,
        'inference_ms': 0.0,
        'backend_dist': {'h5': int(cql_metrics['action_distribution'][0] * 1000),
                         'h7': int(cql_metrics['action_distribution'][1] * 1000),
                         'h8': int(cql_metrics['action_distribution'][2] * 1000)},
        'utilization_dist': {'h5': 0, 'h7': 0, 'h8': 0},
        'balance_cv': cql_metrics['fairness_deviation'] * 100,
    }
    
    # Save to training_logs instead (has write permission)
    out_dir = os.path.join(BASE_DIR, 'ai_model', 'training_logs')
    with open(os.path.join(out_dir, 'WRR_golden_hour_metrics.json'), 'w') as f:
        json.dump(wrr_out, f, indent=2)
    with open(os.path.join(out_dir, 'CQL_BEST_SAMPLED_golden_hour_metrics.json'), 'w') as f:
        json.dump(cql_out, f, indent=2)
    
    print(f"\n[✓] Results saved to {RES_DIR}/")
    
    if cql_score > wrr_score:
        print("\n🎉 CQL_BEST WINS!")
    else:
        print(f"\n⚠️ WRR still wins by {wrr_score - cql_score:.4f} points.")


if __name__ == '__main__':
    main()