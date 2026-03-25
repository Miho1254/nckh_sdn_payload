#!/usr/bin/env python3
"""
Generate visualization charts for benchmark results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

def load_results(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def calculate_statistical_significance(results, scenario, policy1, policy2):
    """Calculate t-test and effect size between two policies."""
    # Extract rewards
    r1 = results[scenario][policy1]['avg_reward']
    r2 = results[scenario][policy2]['avg_reward']
    s1 = results[scenario][policy1]['std_reward']
    s2 = results[scenario][policy2]['std_reward']
    n = results[scenario][policy1]['n_episodes']
    
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind_from_stats(
        mean1=r1, std1=s1, nobs1=n,
        mean2=r2, std2=s2, nobs2=n
    )
    
    # Cohen's d effect size
    pooled_std = np.sqrt((s1**2 + s2**2) / 2)
    cohens_d = (r1 - r2) / pooled_std
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }

def plot_reward_comparison(results, output_path):
    """Plot reward comparison across scenarios."""
    scenarios = list(results.keys())
    policies = ['WRR', 'AdaptiveWRR']
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    wrr_rewards = [results[s]['WRR']['avg_reward'] for s in scenarios]
    wrr_stds = [results[s]['WRR']['std_reward'] for s in scenarios]
    awrr_rewards = [results[s]['AdaptiveWRR']['avg_reward'] for s in scenarios]
    awrr_stds = [results[s]['AdaptiveWRR']['std_reward'] for s in scenarios]
    
    bars1 = ax.bar(x - width/2, wrr_rewards, width, label='WRR', 
                   yerr=wrr_stds, capsize=5, color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, awrr_rewards, width, label='AdaptiveWRR',
                   yerr=awrr_stds, capsize=5, color='#3498db', alpha=0.8)
    
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_title('WRR vs AdaptiveWRR: Reward Comparison (100 episodes)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('SDNFixed', '') for s in scenarios], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add significance markers
    for i, scenario in enumerate(scenarios):
        stat = calculate_statistical_significance(results, scenario, 'WRR', 'AdaptiveWRR')
        if stat['significant']:
            marker = '*' if stat['p_value'] < 0.01 else '†'
            ax.text(i, min(wrr_rewards[i], awrr_rewards[i]) - 100, marker, 
                   ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_latency_comparison(results, output_path):
    """Plot P99 latency comparison."""
    scenarios = list(results.keys())
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    wrr_latency = [results[s]['WRR']['avg_p99_latency'] for s in scenarios]
    awrr_latency = [results[s]['AdaptiveWRR']['avg_p99_latency'] for s in scenarios]
    
    bars1 = ax.bar(x - width/2, wrr_latency, width, label='WRR', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, awrr_latency, width, label='AdaptiveWRR', color='#3498db', alpha=0.8)
    
    ax.set_ylabel('P99 Latency (ms)', fontsize=12)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_title('P99 Latency Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('SDNFixed', '') for s in scenarios], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_sla_violations(results, output_path):
    """Plot SLA violations comparison."""
    scenarios = list(results.keys())
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    wrr_sla = [results[s]['WRR']['total_sla_violations'] for s in scenarios]
    awrr_sla = [results[s]['AdaptiveWRR']['total_sla_violations'] for s in scenarios]
    
    bars1 = ax.bar(x - width/2, wrr_sla, width, label='WRR', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, awrr_sla, width, label='AdaptiveWRR', color='#3498db', alpha=0.8)
    
    ax.set_ylabel('Total SLA Violations', fontsize=12)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_title('SLA Violations Comparison (100 episodes)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('SDNFixed', '') for s in scenarios], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_winner_summary(results, output_path):
    """Plot summary of winners per scenario."""
    scenarios = list(results.keys())
    
    winners = []
    for scenario in scenarios:
        wrr_reward = results[scenario]['WRR']['avg_reward']
        awrr_reward = results[scenario]['AdaptiveWRR']['avg_reward']
        winner = 'WRR' if wrr_reward > awrr_reward else 'AdaptiveWRR'
        delta = abs(wrr_reward - awrr_reward)
        winners.append({
            'scenario': scenario.replace('SDNFixed', ''),
            'winner': winner,
            'delta': delta,
            'wrr_reward': wrr_reward,
            'awrr_reward': awrr_reward
        })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71' if w['winner'] == 'WRR' else '#3498db' for w in winners]
    bars = ax.bar([w['scenario'] for w in winners], [w['delta'] for w in winners], color=colors)
    
    ax.set_ylabel('Reward Delta (WRR - AdaptiveWRR)', fontsize=12)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_title('Winner per Scenario (Green=WRR, Blue=AdaptiveWRR)', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add winner labels
    for bar, w in zip(bars, winners):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{w['winner']}\nΔ={abs(w['delta']):.1f}",
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_statistical_report(results, output_path):
    """Generate statistical significance report."""
    report = []
    report.append("# Statistical Significance Report\n")
    report.append("## Methodology\n")
    report.append("- Two-sample t-test for comparing WRR vs AdaptiveWRR")
    report.append("- Cohen's d for effect size")
    report.append("- Significance level: α = 0.05\n")
    report.append("## Results\n")
    report.append("| Scenario | WRR Reward | AdaptiveWRR Reward | Delta | t-stat | p-value | Cohen's d | Significant |")
    report.append("|----------|------------|---------------------|-------|--------|---------|-----------|-------------|")
    
    for scenario in results.keys():
        stat = calculate_statistical_significance(results, scenario, 'WRR', 'AdaptiveWRR')
        wrr_r = results[scenario]['WRR']['avg_reward']
        awrr_r = results[scenario]['AdaptiveWRR']['avg_reward']
        delta = wrr_r - awrr_r
        
        sig_marker = "✓" if stat['significant'] else "✗"
        report.append(f"| {scenario.replace('SDNFixed', '')} | {wrr_r:.1f} | {awrr_r:.1f} | {delta:.1f} | {stat['t_stat']:.2f} | {stat['p_value']:.4f} | {stat['cohens_d']:.2f} | {sig_marker} |")
    
    report.append("\n## Interpretation\n")
    report.append("- **Cohen's d**: Small (< 0.2), Medium (0.2-0.8), Large (> 0.8)")
    report.append("- **p-value < 0.05**: Statistically significant difference")
    report.append("- **Delta > 0**: WRR wins, **Delta < 0**: AdaptiveWRR wins")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Saved: {output_path}")

def main():
    import glob
    
    # Find latest benchmark file
    benchmark_files = sorted(glob.glob('benchmark_fixed_*.json'))
    if not benchmark_files:
        print("No benchmark files found!")
        return
    
    latest_file = benchmark_files[-1]
    print(f"Loading: {latest_file}")
    
    results = load_results(latest_file)
    
    # Create output directory
    output_dir = Path('benchmark_charts')
    output_dir.mkdir(exist_ok=True)
    
    # Generate charts
    plot_reward_comparison(results, output_dir / 'reward_comparison.png')
    plot_latency_comparison(results, output_dir / 'latency_comparison.png')
    plot_sla_violations(results, output_dir / 'sla_violations.png')
    plot_winner_summary(results, output_dir / 'winner_summary.png')
    generate_statistical_report(results, output_dir / 'statistical_report.md')
    
    print("\nAll charts generated successfully!")

if __name__ == '__main__':
    main()