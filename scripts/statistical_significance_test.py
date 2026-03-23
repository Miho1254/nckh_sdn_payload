#!/usr/bin/env python3
"""
Statistical Significance Test cho Benchmark Results
Chuẩn IEEE: Paired t-test, p-value, Confidence Interval, Effect Size

Cách dùng (từ thư mục nckh_sdn):
    python scripts/statistical_significance_test.py --n_runs 10

Output:
    - p_value < 0.05: AI vượt trội có ý nghĩa thống kê
    - 95% CI: Khoảng tin cậy của improvement
    - Cohen's d: Effect size
"""

import sys
import numpy as np
import torch
import json
import os
from datetime import datetime
from scipy import stats

# Add parent dir to path (nckh_sdn)
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'ai_model'))
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

from cql_agent import CQLAgent
from sdn_env_v2 import SDN_Offline_Env_V2
from train_actor_critic import load_v3_data, BASE_DIR, DATA_DIR, CKPT_DIR
from quick_benchmark_new_scenarios import (
    SCENARIO_MAPPING, filter_data_by_scenario, 
    WeightedRoundRobinBaseline, jains_fairness_index,
    CAPACITY_RATIOS
)
import config


def get_project_paths():
    """Get absolute paths for project resources."""
    return {
        'checkpoint': os.path.join(CKPT_DIR, 'tft_ac_best.pth'),
        'processed_data': DATA_DIR,
    }


def evaluate_once(agent_or_wrr, env, X, y, action_mode='deterministic'):
    """Chạy benchmark 1 lần, trả về metrics."""
    from quick_benchmark_new_scenarios import evaluate_policy
    return evaluate_policy(agent_or_wrr, env, X, y, action_mode)


def run_single_benchmark(checkpoint_path, scenario, seed_offset=0):
    """Chạy benchmark 1 lần với seed khác nhau."""
    # Set random seed for this run
    np.random.seed(42 + seed_offset)
    torch.manual_seed(42 + seed_offset)
    
    # Load data
    X, y, scen, metadata = load_v3_data()
    
    # Filter by scenario
    scenario_filter = SCENARIO_MAPPING.get(scenario, [])
    if scenario_filter:
        X, y, scen = filter_data_by_scenario(X, y, scen, scenario_filter)
    
    # Take subset for faster testing (stratified sample)
    n_samples = min(500, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sub = X[indices]
    y_sub = y[indices]
    
    import tempfile
    temp_dir = tempfile.mkdtemp()
    X_path = os.path.join(temp_dir, 'X_sub.npy')
    y_path = os.path.join(temp_dir, 'y_sub.npy')
    np.save(X_path, X_sub)
    np.save(y_path, y_sub)
    
    env = SDN_Offline_Env_V2(X_path, y_path, mode='eval')
    
    # Evaluate WRR
    wrr = WeightedRoundRobinBaseline()
    wrr_metrics = evaluate_once(wrr, env, X_sub, y_sub, 'deterministic')
    
    # Evaluate AI
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = CQLAgent(
        input_size=X.shape[2],
        num_actions=env.num_actions,
        hidden_size=64,
        actor_lr=config.ACTOR_LR,
        critic_lr=config.CRITIC_LR,
        cql_alpha=config.CQL_ALPHA,
        entropy_coeff=config.ENTROPY_COEFF,
        kl_coeff=config.KL_COEFF,
        target_entropy_ratio=config.TARGET_ENTROPY_RATIO,
        capacity_prior=config.CAPACITY_PRIOR,
        constraint_weights=config.CONSTRAINT_WEIGHTS,
    )
    agent.load_checkpoint(checkpoint_path)
    
    ai_metrics = evaluate_once(agent, env, X_sub, y_sub, 'deterministic')
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    return {
        'ai': ai_metrics,
        'wrr': wrr_metrics,
        'n_samples': n_samples
    }


def compute_ieee_stats(ai_values, wrr_values, metric_name):
    """
    Compute IEEE-compliant statistics.
    
    Returns:
        dict with: mean, std, ci_95, p_value, significant, cohens_d
    """
    diff = np.array(ai_values) - np.array(wrr_values)
    
    # Basic stats
    ai_mean = np.mean(ai_values)
    ai_std = np.std(ai_values, ddof=1)
    wrr_mean = np.mean(wrr_values)
    wrr_std = np.std(wrr_values, ddof=1)
    
    # Paired t-test (two-sided)
    if len(ai_values) > 1:
        t_stat, p_value = stats.ttest_rel(ai_values, wrr_values)
    else:
        t_stat, p_value = np.nan, np.nan
    
    # 95% CI for the difference
    if len(diff) > 1:
        mean_diff = np.mean(diff)
        se_diff = stats.sem(diff)
        ci_95 = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=se_diff)
    else:
        mean_diff = np.nan
        ci_95 = (np.nan, np.nan)
    
    # Effect size (Cohen's d for paired samples)
    if len(diff) > 1 and np.std(diff, ddof=1) > 0:
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    else:
        cohens_d = np.nan
    
    # Improvement percentage
    if wrr_mean != 0:
        improvement_pct = ((ai_mean - wrr_mean) / abs(wrr_mean)) * 100
    else:
        improvement_pct = np.nan if ai_mean == 0 else np.inf
    
    return {
        'metric': metric_name,
        'ai_mean': ai_mean,
        'ai_std': ai_std,
        'wrr_mean': wrr_mean,
        'wrr_std': wrr_std,
        'improvement_pct': improvement_pct,
        'p_value': p_value,
        't_statistic': t_stat,
        'significant_005': p_value < 0.05 if not np.isnan(p_value) else False,
        'significant_001': p_value < 0.01 if not np.isnan(p_value) else False,
        'cohens_d': cohens_d,
        'ci_95_lower': ci_95[0],
        'ci_95_upper': ci_95[1],
        'n_runs': len(ai_values)
    }


def main():
    import argparse
    
    # Get absolute paths
    paths = get_project_paths()
    
    parser = argparse.ArgumentParser(description='Statistical Significance Test')
    parser.add_argument('--n_runs', type=int, default=10, help='Số lần chạy benchmark')
    parser.add_argument('--checkpoint', type=str, 
                        default=paths['checkpoint'],
                        help='Đường dẫn checkpoint')
    parser.add_argument('--scenario', type=str, default='golden_hour',
                        choices=['golden_hour', 'video_conference', 'hardware_degradation', 'low_rate_dos'],
                        help='Kịch bản test')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  IEEE STATISTICAL SIGNIFICANCE TEST")
    print("=" * 70)
    print(f"  Scenario: {args.scenario}")
    print(f"  N runs: {args.n_runs}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ ERROR: Checkpoint not found: {args.checkpoint}")
        print(f"  Available path: {paths['checkpoint']}")
        sys.exit(1)
    
    # Run multiple benchmarks
    print(f"\n[*] Running {args.n_runs} benchmark iterations...")
    results = []
    
    for i in range(args.n_runs):
        print(f"  Run {i+1}/{args.n_runs}...", end=" ")
        try:
            result = run_single_benchmark(args.checkpoint, args.scenario, seed_offset=i)
            results.append(result)
            print(f"✓ (n={result['n_samples']})")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if len(results) < 2:
        print("\n❌ Not enough successful runs for statistical test!")
        sys.exit(1)
    
    print(f"\n[*] Successfully completed {len(results)} runs")
    
    # Extract metrics
    metrics_to_test = [
        ('Real Throughput', 'real_throughput', 'higher'),
        ('Capacity Weighted', 'capacity_weighted', 'higher'),
        ('Jain\'s Fairness', 'jains_fairness_index', 'higher'),
        ('Composite Score', 'score', 'higher'),
        ('Avg Response Time', 'avg_response_time', 'lower'),
        ('Avg Packet Loss', 'avg_packet_loss', 'lower'),
    ]
    
    # Compute statistics for each metric
    print("\n" + "=" * 70)
    print("  IEEE STATISTICAL ANALYSIS RESULTS")
    print("=" * 70)
    
    all_stats = []
    
    for metric_name, metric_key, direction in metrics_to_test:
        ai_values = [r['ai'].get(metric_key, 0) for r in results]
        wrr_values = [r['wrr'].get(metric_key, 0) for r in results]
        
        stats_result = compute_ieee_stats(ai_values, wrr_values, metric_name)
        all_stats.append(stats_result)
        
        # Determine winner
        if direction == 'higher':
            winner = 'AI' if stats_result['ai_mean'] > stats_result['wrr_mean'] else 'WRR'
        else:
            winner = 'AI' if stats_result['ai_mean'] < stats_result['wrr_mean'] else 'WRR'
        
        # Print result
        sig_marker = "***" if stats_result['significant_001'] else ("**" if stats_result['significant_005'] else "")
        
        print(f"\n{metric_name} {sig_marker}")
        print(f"  WRR: {stats_result['wrr_mean']:.4f} ± {stats_result['wrr_std']:.4f}")
        print(f"  AI:  {stats_result['ai_mean']:.4f} ± {stats_result['ai_std']:.4f}")
        print(f"  Improvement: {stats_result['improvement_pct']:+.2f}%")
        print(f"  Winner: {winner}")
        
        if not np.isnan(stats_result['p_value']):
            print(f"  p-value: {stats_result['p_value']:.4f}")
            print(f"  95% CI: [{stats_result['ci_95_lower']:.4f}, {stats_result['ci_95_upper']:.4f}]")
            print(f"  Cohen's d: {stats_result['cohens_d']:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    ai_wins = sum(1 for s in all_stats if s['significant_005'])
    wrr_wins = len(all_stats) - ai_wins
    
    print(f"\n  AI wins {ai_wins}/{len(all_stats)} metrics with statistical significance (p < 0.05)")
    print(f"  WRR wins {wrr_wins}/{len(all_stats)} metrics")
    
    # Final verdict
    print("\n" + "=" * 70)
    if ai_wins > wrr_wins:
        print("  🎉 CONCLUSION: AI OUTPERFORMS WRR WITH STATISTICAL SIGNIFICANCE")
        print("     (p < 0.05, 95% Confidence Interval)")
    elif ai_wins == wrr_wins:
        print("  ⚠️  CONCLUSION: NO STATISTICALLY SIGNIFICANT DIFFERENCE")
        print("     (Further investigation needed)")
    else:
        print("  📊 CONCLUSION: WRR OUTPERFORMS AI")
    print("=" * 70)
    
    # Save results
    stats_dir = os.path.join(script_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    output_file = os.path.join(stats_dir, f"statistical_test_{args.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'n_runs': args.n_runs,
                'scenario': args.scenario,
                'checkpoint': args.checkpoint,
                'timestamp': datetime.now().isoformat()
            },
            'statistics': convert_to_serializable(all_stats),
            'summary': {
                'ai_wins': ai_wins,
                'wrr_wins': wrr_wins
            }
        }, f, indent=2)
    
    print(f"\n[*] Results saved to: {output_file}")
    
    return all_stats


if __name__ == '__main__':
    main()
