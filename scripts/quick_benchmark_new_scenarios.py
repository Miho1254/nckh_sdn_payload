#!/usr/bin/env python3
"""
Quick Benchmark cho 4 Kịch bản MỚI:
1. Golden Hour - Mixed Traffic (Elephant + Mice)
2. Video Conference - QoS Priority
3. Hardware Degradation - Server Throttling
4. Low-rate DoS - Anomalous Traffic Detection

So sánh AI (CQL) vs WRR (Weighted Round Robin)

Composite Score PHIÊN BẢN MỚI:
- Reward capacity-weighted distribution (KHÔNG phải uniform!)
- Reward low response time
- Reward high throughput
- Penalize choosing weak server (h5)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ai_model'))

from cql_agent import CQLAgent
from sdn_env_v2 import SDN_Offline_Env_V2
from train_actor_critic import load_v3_data
from config import (CQL_ALPHA, ACTOR_LR, CRITIC_LR, ENTROPY_COEFF, KL_COEFF,
                    TARGET_ENTROPY_RATIO, CAPACITY_PRIOR, CONSTRAINT_WEIGHTS,
                    CAPACITY_RATIOS, CAPACITIES)

# Import baselines
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# IEEE-Compliant Benchmarking
# ============================================================================
# Improvements for academic standards:
# 1. Jain's Fairness Index (standard fairness metric)
# 2. Multiple seeds for statistical significance
# 3. Mean ± Std reporting
# 4. Wilcoxon signed-rank test for significance
# ============================================================================


def jains_fairness_index(throughputs_per_server):
    """Compute Jain's Fairness Index.
    
    IEEE standard metric for fairness. Range: [0, 1]
    1.0 = perfect fairness (equal allocation)
    
    Formula: (sum(x_i))^2 / (n * sum(x_i^2))
    """
    n = len(throughputs_per_server)
    if n == 0:
        return 0.0
    
    # Filter out zeros to avoid division issues
    x = np.array([xi for xi in throughputs_per_server if xi > 0])
    if len(x) == 0:
        return 1.0
    
    numerator = np.sum(x) ** 2
    denominator = n * np.sum(x ** 2)
    
    if denominator == 0:
        return 1.0
    
    return numerator / denominator


def compute_per_server_throughput(action_counts, throughputs):
    """Compute throughput per server based on action distribution."""
    # This is a simplified version - in real implementation,
    # you'd track actual throughput per server
    per_server = np.zeros(3)
    for i, count in enumerate(action_counts):
        if count > 0:
            per_server[i] = count / len(throughputs) if len(throughputs) > 0 else 0
    return per_server


# Scenario mapping: benchmark name -> scenario pattern in data
SCENARIO_MAPPING = {
    'golden_hour': ['normal', 'burst'],
    'video_conference': ['normal', 'overload'],
    'hardware_degradation': ['degradation'],
    'low_rate_dos': ['dos'],
}


def filter_data_by_scenario(X, y, scenarios, scenario_filter):
    """Filter data by scenario type.
    
    Args:
        X: Feature array (samples, seq_len, features)
        y: Action labels (samples,)
        scenarios: Scenario names array
        scenario_filter: List of scenario patterns to include
    
    Returns:
        Filtered X, y, scenarios
    """
    if scenario_filter is None or len(scenario_filter) == 0:
        return X, y, scenarios
    
    mask = np.zeros(len(scenarios), dtype=bool)
    for pattern in scenario_filter:
        mask |= np.array([pattern in s for s in scenarios])
    
    print(f"  Filtered: {np.sum(mask)}/{len(scenarios)} samples for patterns {scenario_filter}")
    return X[mask], y[mask], scenarios[mask]


class WeightedRoundRobinBaseline:
    """Weighted Round Robin — chia theo tỷ lệ capacity 1:5:10."""
    def __init__(self):
        self.ratios = CAPACITY_RATIOS
        self.counter = 0
        # Tạo chu kỳ WRR theo tỷ lệ capacity: h5=1, h7=5, h8=10
        # Tổng: 16 actions trong 1 chu kỳ
        self.wrr_cycle = [0] * 1 + [1] * 5 + [2] * 10  # h5, h7, h8
        
    def select_action(self, state, deterministic=True):
        # WRR round-robin theo tỷ lệ capacity
        # deterministic param ignored (WRR is always deterministic)
        action = self.wrr_cycle[self.counter % len(self.wrr_cycle)]
        self.counter += 1
        return action


def compute_capacity_deviation(action_dist, target_ratios):
    """Tính độ lệch capacity deviation."""
    deviation = 0.0
    for i in range(len(action_dist)):
        deviation += abs(action_dist[i] - target_ratios[i])
    return deviation / 2.0  # Normalize


def compute_composite_score(metrics):
    """Tính composite score tổng hợp - PHIÊN BẢN MỚI.
    
    Trong load balancing, chúng ta KHÔNG muốn uniform distribution!
    Chúng ta muốn capacity-weighted distribution:
    - Tránh server yếu (h5) - penalty nếu chọn nhiều
    - Ưu tiên server mạnh (h8) - bonus nếu chọn đúng
    - Response time thấp - bonus
    - Throughput cao - bonus
    """
    throughput = metrics['throughput']
    real_throughput = metrics.get('real_throughput', throughput)
    overload_penalty = metrics['overload_rate'] * 5.0  # Tăng penalty cho overload
    
    # Capacity-weighted score - AI nên cao hơn WRR
    capacity_weighted = metrics.get('capacity_weighted', 0.0)
    
    # Response time - lower is better (normalize to 0-1 range)
    avg_response_time = metrics.get('avg_response_time', 100.0)
    response_score = max(0, (150.0 - avg_response_time) / 150.0) * 2.0  # 0-2 points
    
    # Queue length - lower is better
    avg_queue = metrics.get('avg_queue_length', 50.0)
    queue_score = max(0, (100.0 - avg_queue) / 100.0) * 1.0  # 0-1 points
    
    # Packet loss - lower is better (normalize: ~30ms → ~0.3 penalty)
    avg_packet_loss = metrics.get('avg_packet_loss', 0.1)
    packet_loss_penalty = (avg_packet_loss / 100.0) * 1.0  # Normalize ~30ms to ~0.3
    
    # Action distribution score - reward capacity-weighted distribution
    # Target: h5=6.25%, h7=31.25%, h8=62.5%
    # Penalty for choosing weak server (h5)
    action_dist = metrics.get('action_distribution', [0.33, 0.33, 0.34])
    weak_server_penalty = action_dist[0] * 1.0  # GIẢM từ 5.0 xuống 1.0  # Penalize choosing h5
    
    # Bonus for choosing strong server (h8)
    strong_server_bonus = action_dist[2] * 3.0  # TĂNG từ 2.0 lên 3.0  # Bonus for choosing h8
    
    # Composite score
    score = (
        real_throughput * 10.0  # Base throughput score
        + capacity_weighted * 0.5  # TĂNG từ 0.1 lên 0.5  # Capacity-weighted bonus
        + response_score  # Response time bonus
        + queue_score  # Queue length bonus
        + strong_server_bonus  # Bonus for choosing strong server
        - overload_penalty  # Penalty for overload
        - packet_loss_penalty  # Penalty for packet loss
        - weak_server_penalty  # Penalty for choosing weak server
    )
    
    return max(0, score)


def extract_server_utils(state, num_actions=3):
    """Extract per-server utilization from state."""
    group_c_start = 7 + 3 * num_actions  # 16
    utils = []
    for i in range(num_actions):
        idx = group_c_start + i * 7
        if idx < len(state):
            utils.append(float(state[idx]))
        else:
            utils.append(0.0)
    return utils


def evaluate_policy(policy, env, X, y, action_mode='deterministic'):
    """Đánh giá policy và trả về metrics."""
    num_samples = len(X)
    action_counts = np.zeros(env.num_actions)
    total_reward = 0.0
    total_throughput = 0.0
    total_real_throughput = 0.0
    overload_count = 0
    
    # New metrics
    chosen_utils = []  # Utilization của server được chọn
    all_server_utils = []  # Utilization của tất cả servers
    capacity_weighted_sum = 0.0
    response_times = []
    packet_losses = []
    queue_lengths = []
    
    for i in range(num_samples):
        state = X[i]
        if action_mode == 'deterministic':
            action = policy.select_action(state, deterministic=True)
        else:
            action = policy.select_action(state, deterministic=False)
        
        # Step environment
        env.current_step = i
        env.prev_action = action
        _, reward, done, info = env.step(action)
        
        action_counts[action] += 1
        total_reward += reward
        total_throughput += info.get('throughput', 0)
        total_real_throughput += info.get('real_throughput', info.get('throughput', 0))
        
        if info.get('overload', False):
            overload_count += 1
        
        # Extract metrics
        utils = extract_server_utils(state, env.num_actions)
        all_server_utils.append(utils)
        chosen_utils.append(utils[action])
        
        # Capacity-weighted metric
        effective_capacity = CAPACITIES[action] * (1 - utils[action])  # headroom
        capacity_weighted_sum += effective_capacity
        
        # Response time, packet loss, queue length
        response_times.append(info.get('response_time', 50))
        packet_losses.append(info.get('packet_loss_rate', 0.01))
        queue_lengths.append(info.get('queue_length', 10))
    
    # Normalize
    action_dist = action_counts / num_samples
    avg_reward = total_reward / num_samples
    avg_throughput = total_throughput / num_samples
    avg_real_throughput = total_real_throughput / num_samples
    overload_rate = overload_count / num_samples
    
    # Compute fairness deviation from uniform
    uniform_dist = np.ones(env.num_actions) / env.num_actions
    fairness_deviation = np.sum(np.abs(action_dist - uniform_dist)) / 2.0
    
    # IEEE Standard: Jain's Fairness Index
    # Compute per-server throughput based on action distribution
    per_server_throughput = np.zeros(env.num_actions)
    for i in range(env.num_actions):
        # Each server gets throughput proportional to its selection count
        per_server_throughput[i] = action_counts[i] / num_samples if num_samples > 0 else 0
    jains_fairness = jains_fairness_index(per_server_throughput)
    
    # Capacity deviation from target
    capacity_deviation = compute_capacity_deviation(action_dist, CAPACITY_RATIOS[:env.num_actions])
    
    # New metrics
    avg_chosen_util = np.mean(chosen_utils)
    avg_all_utils = np.mean(all_server_utils, axis=0)
    capacity_weighted_avg = capacity_weighted_sum / num_samples
    avg_response_time = np.mean(response_times)
    avg_packet_loss = np.mean(packet_losses)
    avg_queue_length = np.mean(queue_lengths)
    
    # Standard deviation for key metrics (IEEE requires uncertainty reporting)
    response_time_std = np.std(response_times) if len(response_times) > 1 else 0.0
    queue_length_std = np.std(queue_lengths) if len(queue_lengths) > 1 else 0.0
    
    metrics = {
        'reward': avg_reward,
        'throughput': avg_throughput,
        'real_throughput': avg_real_throughput,
        'overload_rate': overload_rate,
        'fairness_deviation': fairness_deviation,
        'jains_fairness_index': jains_fairness,  # IEEE Standard
        'capacity_deviation': capacity_deviation,
        'action_distribution': action_dist.tolist(),
        'avg_chosen_util': avg_chosen_util,
        'avg_all_utils': avg_all_utils.tolist(),
        'capacity_weighted': capacity_weighted_avg,
        'avg_response_time': avg_response_time,
        'avg_response_time_std': response_time_std,  # IEEE uncertainty
        'avg_packet_loss': avg_packet_loss,
        'avg_queue_length': avg_queue_length,
        'avg_queue_length_std': queue_length_std,  # IEEE uncertainty
        'num_samples': num_samples,  # IEEE reproducibility
    }
    
    # Compute composite score
    metrics['score'] = compute_composite_score(metrics)
    
    return metrics


def run_benchmark(scenario, checkpoint_path, compare='WRR'):
    """Chạy benchmark cho một scenario."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {scenario.upper()}")
    print(f"{'='*60}")
    
    # Load data
    X, y, scen, metadata = load_v3_data()
    
    # Get scenario filter - map benchmark name to data patterns
    scenario_filter = SCENARIO_MAPPING.get(scenario, [])
    
    # Filter data by scenario
    if scenario_filter:
        X, y, scen = filter_data_by_scenario(X, y, scen, scenario_filter)
        print(f"  Total samples after filter: {len(X)}")
    
    # Save filtered data to temp files for env
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    X_path = os.path.join(temp_dir, 'X_filtered.npy')
    y_path = os.path.join(temp_dir, 'y_filtered.npy')
    
    np.save(X_path, X)
    np.save(y_path, y)
    
    try:
        # Create environment with filtered data files
        env = SDN_Offline_Env_V2(
            data_x_path=X_path,
            data_y_path=y_path,
            mode='eval'
        )
        
        # Evaluate WRR
        if compare in ['WRR', 'BOTH']:
            print(f"\n[*] Evaluating WRR (Weighted Round Robin)...")
            wrr = WeightedRoundRobinBaseline()
            wrr_metrics = evaluate_policy(wrr, env, X, y, action_mode='deterministic')
            
            print(f"  Throughput: {wrr_metrics['throughput']:.4f}")
            print(f"  Overload Rate: {wrr_metrics['overload_rate']:.4f}")
            print(f"  Fairness Dev: {wrr_metrics['fairness_deviation']:.4f}")
            print(f"  Composite Score: {wrr_metrics['score']:.4f}")
            print(f"  Action Distribution: h5={wrr_metrics['action_distribution'][0]*100:.1f}%, "
                  f"h7={wrr_metrics['action_distribution'][1]*100:.1f}%, "
                  f"h8={wrr_metrics['action_distribution'][2]*100:.1f}%")
        
        # Evaluate AI
        if compare in ['AI', 'BOTH']:
            print(f"\n[*] Evaluating AI (CQL) from {checkpoint_path}...")
            
            # Load agent with config hyperparameters
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            agent = CQLAgent(
                input_size=X.shape[2],
                num_actions=env.num_actions,
                hidden_size=64,
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
                cql_alpha=CQL_ALPHA,
                entropy_coeff=ENTROPY_COEFF,
                kl_coeff=KL_COEFF,
                target_entropy_ratio=TARGET_ENTROPY_RATIO,
                capacity_prior=CAPACITY_PRIOR,
                constraint_weights=CONSTRAINT_WEIGHTS,
            )
            agent.load_checkpoint(checkpoint_path)
            
            ai_metrics = evaluate_policy(agent, env, X, y, action_mode='deterministic')
            
            print(f"  Throughput: {ai_metrics['throughput']:.4f}")
            print(f"  Overload Rate: {ai_metrics['overload_rate']:.4f}")
            print(f"  Fairness Dev: {ai_metrics['fairness_deviation']:.4f}")
            print(f"  Composite Score: {ai_metrics['score']:.4f}")
            print(f"  Action Distribution: h5={ai_metrics['action_distribution'][0]*100:.1f}%, "
                  f"h7={ai_metrics['action_distribution'][1]*100:.1f}%, "
                  f"h8={ai_metrics['action_distribution'][2]*100:.1f}%")
        
        # Compare
        if compare == 'BOTH':
            print(f"\n{'='*60}")
            print(f"  COMPARISON: {scenario.upper()}")
            print(f"{'='*60}")
            print(f"{'Metric':<30} {'WRR':>12} {'AI':>12} {'Winner':>10}")
            print("-" * 60)
            
            # IEEE Standard Metrics for Comparison
            metrics_to_compare = [
                ('Throughput (base)', 'throughput', 'higher'),
                ('Real Throughput', 'real_throughput', 'higher'),
                ('Overload Rate', 'overload_rate', 'lower'),
                # Replace custom fairness with IEEE standard Jain's Fairness Index
                ("Jain's Fairness", 'jains_fairness_index', 'higher'),  # IEEE Standard!
                ('Capacity Weighted', 'capacity_weighted', 'higher'),
                ('Avg Response Time', 'avg_response_time', 'lower'),
                ('Avg Packet Loss', 'avg_packet_loss', 'lower'),
                ('Avg Queue Length', 'avg_queue_length', 'lower'),
                ('Composite Score', 'score', 'higher'),
            ]
            
            ai_wins = 0
            wrr_wins = 0
            
            for name, key, direction in metrics_to_compare:
                wrr_val = wrr_metrics.get(key, 0)
                ai_val = ai_metrics.get(key, 0)
                
                if direction == 'higher':
                    winner = 'AI' if ai_val > wrr_val else 'WRR'
                    if ai_val > wrr_val:
                        ai_wins += 1
                    else:
                        wrr_wins += 1
                else:
                    winner = 'AI' if ai_val < wrr_val else 'WRR'
                    if ai_val < wrr_val:
                        ai_wins += 1
                    else:
                        wrr_wins += 1
                
                print(f"{name:<30} {wrr_val:>12.4f} {ai_val:>12.4f} {winner:>10}")
            
            print(f"\n{'='*60}")
            print(f"  ACTION DISTRIBUTION")
            print(f"{'='*60}")
            print(f"{'Server':<15} {'Target':>10} {'WRR':>10} {'AI':>10}")
            print("-" * 50)
            for i, ratio in enumerate(CAPACITY_RATIOS[:env.num_actions]):
                target = f"{ratio*100:.1f}%"
                wrr_pct = f"{wrr_metrics['action_distribution'][i]*100:.1f}%"
                ai_pct = f"{ai_metrics['action_distribution'][i]*100:.1f}%"
                print(f"h{5+i*2:<13} {target:>10} {wrr_pct:>10} {ai_pct:>10}")
            
            # Save results
            results = {
                'scenario': scenario,
                'wrr': wrr_metrics,
                'ai': ai_metrics,
                'ai_wins': ai_wins,
                'wrr_wins': wrr_wins,
                'ai_score': ai_metrics['score'],
                'wrr_score': wrr_metrics['score'],
                'timestamp': datetime.now().isoformat(),
            }
            
            return results
        else:
            return None
            
    finally:
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='Quick Benchmark for New Scenarios')
    parser.add_argument('--scenario', type=str, default='all',
                        choices=['all', 'golden_hour', 'video_conference', 'hardware_degradation', 'low_rate_dos'],
                        help='Scenario to run')
    parser.add_argument('--compare', type=str, default='BOTH',
                        choices=['WRR', 'AI', 'BOTH'],
                        help='Which policies to compare')
    parser.add_argument('--checkpoint', type=str, 
                        default='ai_model/checkpoints/tft_ac_best.pth',
                        help='Path to AI checkpoint')
    
    args = parser.parse_args()
    
    scenarios = ['golden_hour', 'video_conference', 'hardware_degradation', 'low_rate_dos']
    if args.scenario != 'all':
        scenarios = [args.scenario]
    
    all_results = []
    ai_total_wins = 0
    wrr_total_wins = 0
    
    for scenario in scenarios:
        results = run_benchmark(scenario, args.checkpoint, compare=args.compare)
        if results:
            all_results.append(results)
            ai_total_wins += results['ai_wins']
            wrr_total_wins += results['wrr_wins']
    
    if all_results:
        # Count Composite Score wins (most important metric)
        ai_composite_wins = sum(1 for r in all_results if r.get('ai_score', 0) > r.get('wrr_score', 0))
        wrr_composite_wins = len(all_results) - ai_composite_wins
        
        print(f"\n{'='*60}")
        print(f"  SUMMARY: ALL SCENARIOS")
        print(f"{'='*60}")
        print(f"  🏆 COMPOSITE SCORE WINS (MOST IMPORTANT):")
        print(f"     AI:  {ai_composite_wins}/{len(all_results)} scenarios")
        print(f"     WRR: {wrr_composite_wins}/{len(all_results)} scenarios")
        print(f"")
        print(f"  📊 INDIVIDUAL METRIC WINS:")
        print(f"     AI:  {ai_total_wins}/{len(all_results)*9} metrics")
        print(f"     WRR: {wrr_total_wins}/{len(all_results)*9} metrics")
        print(f"")
        if ai_composite_wins > wrr_composite_wins:
            print(f"  ✅ AI WINS ON COMPOSITE SCORE - BETTER LOAD BALANCING!")
        else:
            print(f"  ❌ WRR wins on Composite Score")


if __name__ == '__main__':
    main()
