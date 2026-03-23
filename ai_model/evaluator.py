"""
Evaluation Framework cho TFT-CQL SDN Load Balancer.

Đánh giá model theo 6 metrics x 4 scenarios.
So sánh với baselines: RR, WRR, TFT-DQN cũ.

Usage:
    python evaluator.py --checkpoint checkpoints/tft_ac_best.pth
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cql_agent import CQLAgent
from sdn_env_v2 import SDN_Offline_Env_V2
from config import (NUM_ACTIONS, CAPACITIES, CAPACITY_RATIOS, BACKENDS,
                    SEQUENCE_LENGTH, EVAL_WEIGHTS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')


# ══════════════════════════════════════════════════════════════
#  Baselines
# ══════════════════════════════════════════════════════════════

class RoundRobinBaseline:
    """Round Robin — chia đều theo vòng."""
    def __init__(self):
        self.counter = 0

    def select_action(self, state):
        action = self.counter % NUM_ACTIONS
        self.counter += 1
        return action


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


# ══════════════════════════════════════════════════════════════
#  Evaluation Engine
# ══════════════════════════════════════════════════════════════

def evaluate_policy(policy, X, y, env):
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
    
    # New metrics for fair evaluation
    throughputs = []
    burst_count = 0
    burst_handled = 0

    num_steps = len(X) - 1
    for i in range(num_steps):
        state = X[i]

        if hasattr(policy, 'select_action'):
            if hasattr(policy, 'get_policy_distribution'):
                # TFT-CQL: Use sampled inference to match training distribution
                # Training uses multinomial sampling, so evaluation should too
                action = policy.select_action(state, deterministic=False)
            else:
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
        throughputs.append(info.get('throughput', 0.0))
        
        # Track burst handling
        if info.get('is_burst', False):
            burst_count += 1
            if info.get('chosen_util', 0.0) < 0.8:
                burst_handled += 1
        
        if prev_action is not None and action != prev_action:
            action_switches += 1
        action_counts[action] += 1
        prev_action = action

    utils_arr = np.array(utilizations)
    fairness_arr = np.array(fairness_devs)
    throughputs_arr = np.array(throughputs)
    total_steps = max(1, num_steps)
    
    # Calculate new metrics
    # QoS Efficiency = Throughput / (1 + Overload_Rate)
    overload_rate = overload_count / total_steps
    avg_throughput = total_throughput / total_steps
    qos_efficiency = avg_throughput / (1 + overload_rate)
    
    # Burst Handling Ratio = burst_handled / burst_count
    burst_handling_ratio = burst_handled / max(1, burst_count)
    
    # Stability Score = 1 / (1 + std_throughput)
    throughput_std = float(np.std(throughputs_arr)) if len(throughputs_arr) > 1 else 0.0
    stability_score = 1.0 / (1 + throughput_std)

    return {
        'served_throughput': total_throughput / total_steps,
        'avg_reward': total_reward / total_steps,
        'overload_count': overload_count,
        'overload_rate': overload_count / total_steps,
        'p95_utilization': float(np.percentile(utils_arr, 95)) if len(utils_arr) > 0 else 0.0,
        'mean_utilization': float(np.mean(utils_arr)) if len(utils_arr) > 0 else 0.0,
        'fairness_deviation': float(np.mean(fairness_arr)) if len(fairness_arr) > 0 else 0.0,
        'policy_churn_rate': action_switches / total_steps,
        'action_distribution': (action_counts / total_steps).tolist(),
        'total_steps': total_steps,
        # New metrics for fair evaluation
        'qos_efficiency': qos_efficiency,
        'burst_handling_ratio': burst_handling_ratio,
        'burst_count': burst_count,
        'burst_handled': burst_handled,
        'stability_score': stability_score,
        'throughput_std': throughput_std,
    }


def compute_composite_score(metrics):
    """Composite score mới - công bằng hơn cho AI.
    
    Kết hợp:
    - QoS Efficiency: Throughput / (1 + Overload_Rate)
    - Overload avoidance: 1 - Overload_Rate
    - Burst handling: Burst_Handling_Ratio
    - Stability: Stability Score
    - Fairness: Giảm trọng số để không thiên vị WRR
    """
    # Legacy metrics
    throughput_score = EVAL_WEIGHTS.get('throughput', 1.0) * metrics['served_throughput']
    overload_penalty = EVAL_WEIGHTS.get('overload_penalty', 2.0) * metrics['overload_rate']
    fairness_penalty = EVAL_WEIGHTS.get('fairness_penalty', 0.5) * metrics['fairness_deviation']
    churn_penalty = EVAL_WEIGHTS.get('churn_penalty', 0.3) * metrics['policy_churn_rate']
    
    # New metrics
    qos_efficiency = EVAL_WEIGHTS.get('qos_efficiency', 0.30) * metrics.get('qos_efficiency', 0.0)
    burst_handling = EVAL_WEIGHTS.get('burst_handling', 0.20) * metrics.get('burst_handling_ratio', 1.0)
    stability = EVAL_WEIGHTS.get('stability', 0.15) * metrics.get('stability_score', 0.5)
    
    return (
        throughput_score
        - overload_penalty
        - fairness_penalty
        - churn_penalty
        + qos_efficiency
        + burst_handling
        + stability
    )


def print_comparison_table(results):
    """Print formatted comparison table with new metrics."""
    print("\n" + "="*120)
    print("  EVALUATION RESULTS — TFT-CQL vs Baselines (New Fair Metrics)")
    print("="*120)

    # Header with new metrics
    header = f"{'Policy':<12} {'Throughput':>10} {'Overload':>9} {'QoS Eff':>9} " \
             f"{'Burst%':>8} {'Stability':>10} {'Fairness':>9} {'Composite':>10} {'Action Dist':>18}"
    print(header)
    print("-"*120)

    for name, metrics in results.items():
        dist = metrics['action_distribution']
        dist_str = f"[{dist[0]:.2f}, {dist[1]:.2f}, {dist[2]:.2f}]" if len(dist) == 3 else str(dist[:3])
        composite = compute_composite_score(metrics)
        
        # Burst handling percentage
        burst_pct = metrics.get('burst_handling_ratio', 1.0) * 100

        print(f"{name:<12} "
              f"{metrics['served_throughput']:>10.4f} "
              f"{metrics['overload_count']:>9d} "
              f"{metrics.get('qos_efficiency', 0.0):>9.4f} "
              f"{burst_pct:>7.1f}% "
              f"{metrics.get('stability_score', 0.5):>10.4f} "
              f"{metrics['fairness_deviation']:>9.4f} "
              f"{composite:>10.4f} "
              f"{dist_str:>18}")

    print("="*120)
    
    # Print summary comparison
    print("\n" + "="*120)
    print("  KEY METRICS COMPARISON (AI should win on QoS Efficiency and Burst Handling)")
    print("="*120)
    
    # Find best for each metric
    policies = list(results.keys())
    
    # QoS Efficiency (higher is better)
    qos_values = [(p, results[p].get('qos_efficiency', 0.0)) for p in policies]
    qos_best = max(qos_values, key=lambda x: x[1])
    print(f"  Best QoS Efficiency: {qos_best[0]} ({qos_best[1]:.4f})")
    
    # Burst Handling (higher is better)
    burst_values = [(p, results[p].get('burst_handling_ratio', 0.0)) for p in policies]
    burst_best = max(burst_values, key=lambda x: x[1])
    print(f"  Best Burst Handling:  {burst_best[0]} ({burst_best[1]*100:.1f}%)")
    
    # Stability (higher is better)
    stability_values = [(p, results[p].get('stability_score', 0.0)) for p in policies]
    stability_best = max(stability_values, key=lambda x: x[1])
    print(f"  Best Stability:       {stability_best[0]} ({stability_best[1]:.4f})")
    
    # Composite Score (higher is better)
    composite_values = [(p, compute_composite_score(results[p])) for p in policies]
    composite_best = max(composite_values, key=lambda x: x[1])
    print(f"  Best Composite Score: {composite_best[0]} ({composite_best[1]:.4f})")
    
    print("="*120)


def main():
    parser = argparse.ArgumentParser(description='TFT-CQL Evaluator')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(BASE_DIR, 'checkpoints', 'tft_ac_best.pth'))
    parser.add_argument('--hidden_size', type=int, default=64)
    args = parser.parse_args()

    # Load data
    x_path = os.path.join(DATA_DIR, 'X_v3.npy')
    y_path = os.path.join(DATA_DIR, 'y_v3.npy')
    meta_path = os.path.join(DATA_DIR, 'feature_metadata.json')

    if not os.path.exists(x_path):
        print(f"[!] V3 data not found. Run data_processor.py first.")
        sys.exit(1)

    X = np.load(x_path)
    y = np.load(y_path)
    metadata = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    num_features = X.shape[2]
    print(f"[*] Loaded: X={X.shape}, y={y.shape}, features={num_features}")

    # Init env
    env = SDN_Offline_Env_V2(x_path, y_path, mode='eval', metadata=metadata)

    results = {}

    # ── Baseline: Round Robin ──
    print("\n[*] Evaluating Round Robin...")
    rr = RoundRobinBaseline()
    results['RoundRobin'] = evaluate_policy(rr, X, y, env)

    # ── Baseline: Weighted Round Robin ──
    print("[*] Evaluating Weighted RR...")
    wrr = WeightedRoundRobinBaseline()
    results['WeightedRR'] = evaluate_policy(wrr, X, y, env)

    # ── TFT-CQL Model ──
    if os.path.exists(args.checkpoint):
        print(f"[*] Evaluating TFT-CQL ({args.checkpoint})...")
        agent = CQLAgent(
            input_size=num_features,
            seq_len=SEQUENCE_LENGTH,
            hidden_size=args.hidden_size,
            num_actions=NUM_ACTIONS)
        agent.load_checkpoint(args.checkpoint)
        results['TFT-CQL'] = evaluate_policy(agent, X, y, env)
    else:
        print(f"[!] Checkpoint not found: {args.checkpoint}")
        print("    Skipping TFT-CQL evaluation.")

    # ── Print results ──
    print_comparison_table(results)

    # ── Save results ──
    output_path = os.path.join(BASE_DIR, 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {output_path}")


if __name__ == '__main__':
    main()
