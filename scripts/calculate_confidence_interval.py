#!/usr/bin/env python3
"""
Tính 95% Confidence Interval cho benchmark metrics.
Công thức: CI = mean ± t * (std / sqrt(n))
Với t = 2.776 cho n=5 (df=4, 95% CI)
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

# T-value cho 95% CI với df=4 (n=5 runs)
T_VALUE_95 = 2.776


def calculate_ci_95(values: List[float]) -> Tuple[float, float, float]:
    """
    Tính 95% Confidence Interval.
    
    Args:
        values: List of values
        
    Returns:
        (mean, lower_bound, upper_bound)
    """
    if len(values) < 2:
        return (values[0], values[0], values[0]) if values else (0, 0, 0)
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    n = len(values)
    
    # Standard error
    se = std / np.sqrt(n)
    
    # Margin of error
    me = T_VALUE_95 * se
    
    return (mean, mean - me, mean + me)


def format_ci_95(mean: float, lower: float, upper: float, unit: str = "") -> str:
    """Format CI với 95% confidence."""
    return f"{mean:.2f} [{lower:.2f} - {upper:.2f}]{unit}"


def analyze_scenario_with_ci(root_dir: str, scenario: str) -> Dict:
    """Analyze một scenario với 95% CI."""
    scenario_dir = os.path.join(root_dir, scenario)
    if not os.path.exists(scenario_dir):
        return None
    
    results = {
        'scenario': scenario,
        'ppo': {},
        'wrr': {}
    }
    
    for algo in ['ppo', 'wrr']:
        run_dirs = sorted([d for d in os.listdir(scenario_dir) if d.startswith('run_')])
        metrics = {
            'throughputs': [],
            'p99_latencies': [],
            'jitters': [],
            'packet_losses': []
        }
        
        for run_dir in run_dirs:
            run_path = os.path.join(scenario_dir, run_dir, algo)
            if not os.path.exists(run_path):
                continue
            
            # Find stress.log
            stress_logs = list(Path(run_path).glob('h*_stress.log'))
            if not stress_logs:
                continue
            
            # Parse stress.log
            for log_path in stress_logs:
                with open(log_path, 'r') as f:
                    content = f.read()
                
                # Extract http.requests
                req_matches = re.findall(r'http\.requests:.*?([\d.]+)\s*$', content, re.MULTILINE)
                total_requests = sum(int(float(m)) for m in req_matches)
                
                # Extract errors
                error_matches = re.findall(r'errors\.\w+:.*?([\d.]+)\s*$', content, re.MULTILINE)
                total_errors = sum(int(float(m)) for m in error_matches)
                
                # Extract 200 responses
                code_200_matches = re.findall(r'http\.codes\.200:.*?([\d.]+)\s*$', content, re.MULTILINE)
                total_200 = sum(int(float(m)) for m in code_200_matches)
                
                # Extract p99
                p99_matches = re.findall(r'p99:.*?([\d.]+)\s*$', content, re.MULTILINE)
                p99_values = [float(m) for m in p99_matches if float(m) > 100]
                
                # Extract mean latency
                mean_matches = re.findall(r'mean:.*?([\d.]+)\s*$', content, re.MULTILINE)
                mean_values = [float(m) for m in mean_matches if float(m) > 100]
                
                # Extract median latency
                median_matches = re.findall(r'median:.*?([\d.]+)\s*$', content, re.MULTILINE)
                median_values = [float(m) for m in median_matches if float(m) > 100]
                
                # Calculate jitter
                if len(p99_values) >= 2:
                    jitter = float(np.std(p99_values))
                    metrics['jitters'].append(jitter)
                
                # Calculate packet loss
                if total_requests > 0:
                    pl = (total_errors / total_requests) * 100
                    metrics['packet_losses'].append(pl)
                
                # Store throughput (200 responses)
                metrics['throughputs'].append(total_200)
                
                # Store p99 mean for this run
                if p99_values:
                    metrics['p99_latencies'].append(np.mean(p99_values))
                
                # Store mean latency
                if mean_values:
                    metrics['mean_latencies'] = mean_values
        
        # Calculate CI cho từng metric
        if metrics['throughputs']:
            mean, lower, upper = calculate_ci_95(metrics['throughputs'])
            results[algo]['throughput_ci'] = (mean, lower, upper)
        
        if metrics['p99_latencies']:
            mean, lower, upper = calculate_ci_95(metrics['p99_latencies'])
            results[algo]['p99_ci'] = (mean, lower, upper)
        
        if metrics['jitters']:
            mean, lower, upper = calculate_ci_95(metrics['jitters'])
            results[algo]['jitter_ci'] = (mean, lower, upper)
        
        if metrics['packet_losses']:
            mean, lower, upper = calculate_ci_95(metrics['packet_losses'])
            results[algo]['packet_loss_ci'] = (mean, lower, upper)
    
    return results


def print_ci_report(root_dir: str, scenario: str = 'golden_hour'):
    """In báo cáo 95% CI."""
    print("=" * 80)
    print(f"95% CONFIDENCE INTERVAL - {scenario}")
    print("=" * 80)
    
    results = analyze_scenario_with_ci(root_dir, scenario)
    if not results:
        print("No results found!")
        return
    
    print("\n┌─────────────────┬──────────────────────────────────┬──────────────────────────────────┐")
    print("│ Metric          │ PPO (95% CI)                     │ WRR (95% CI)                     │")
    print("├─────────────────┼──────────────────────────────────┼──────────────────────────────────┤")
    
    # Throughput
    if 'throughput_ci' in results['ppo']:
        ppo_mean, ppo_lower, ppo_upper = results['ppo']['throughput_ci']
        wrr_mean, wrr_lower, wrr_upper = results['wrr']['throughput_ci']
        ppo_diff = ((ppo_mean - wrr_mean) / wrr_mean * 100) if wrr_mean > 0 else 0
        winner = "PPO" if ppo_mean > wrr_mean else "WRR"
        print(f"│ Throughput      │ {ppo_mean:,.0f} [{ppo_lower:,.0f} - {ppo_upper:,.0f}] │ {wrr_mean:,.0f} [{wrr_lower:,.0f} - {wrr_upper:,.0f}] │ {winner} ({ppo_diff:+.1f}%) │")
    
    # P99 Latency
    if 'p99_ci' in results['ppo']:
        ppo_mean, ppo_lower, ppo_upper = results['ppo']['p99_ci']
        wrr_mean, wrr_lower, wrr_upper = results['wrr']['p99_ci']
        ppo_diff = ((ppo_mean - wrr_mean) / wrr_mean * 100) if wrr_mean > 0 else 0
        winner = "WRR" if wrr_mean < ppo_mean else "PPO"
        print(f"│ P99 Latency     │ {ppo_mean:.2f} [{ppo_lower:.2f} - {ppo_upper:.2f}] ms │ {wrr_mean:.2f} [{wrr_lower:.2f} - {wrr_upper:.2f}] ms │ {winner} ({ppo_diff:+.1f}%) │")
    
    # Jitter
    if 'jitter_ci' in results['ppo']:
        ppo_mean, ppo_lower, ppo_upper = results['ppo']['jitter_ci']
        wrr_mean, wrr_lower, wrr_upper = results['wrr']['jitter_ci']
        ppo_diff = ((ppo_mean - wrr_mean) / wrr_mean * 100) if wrr_mean > 0 else 0
        winner = "WRR" if wrr_mean < ppo_mean else "PPO"
        print(f"│ Jitter          │ {ppo_mean:.2f} [{ppo_lower:.2f} - {ppo_upper:.2f}] ms │ {wrr_mean:.2f} [{wrr_lower:.2f} - {wrr_upper:.2f}] ms │ {winner} ({ppo_diff:+.1f}%) │")
    
    # Packet Loss
    if 'packet_loss_ci' in results['ppo']:
        ppo_mean, ppo_lower, ppo_upper = results['ppo']['packet_loss_ci']
        wrr_mean, wrr_lower, wrr_upper = results['wrr']['packet_loss_ci']
        ppo_diff = ((ppo_mean - wrr_mean) / wrr_mean * 100) if wrr_mean > 0 else 0
        winner = "PPO" if ppo_mean < wrr_mean else "WRR"
        print(f"│ Packet Loss     │ {ppo_mean:.2f} [{ppo_lower:.2f} - {ppo_upper:.2f}] %   │ {wrr_mean:.2f} [{wrr_lower:.2f} - {wrr_upper:.2f}] %   │ {winner} ({ppo_diff:+.1f}%) │")
    
    print("└─────────────────┴──────────────────────────────────┴──────────────────────────────────┘")
    
    # Interpretation
    print("\n*Ghi chú: 95% CI được tính với t-value = 2.776 (df=4, n=5 runs)")
    print("*Nếu 2 CI không chồng lấn, sự khác biệt có ý nghĩa thống kê (p<0.05)")
    
    return results


def main():
    import sys
    
    if len(sys.argv) < 2:
        results_dir = 'benchmark_results_final_n5'
    else:
        results_dir = sys.argv[1]
    
    scenarios = ['golden_hour', 'video_conference', 'low_rate_dos', 'hardware_degradation']
    
    for scenario in scenarios:
        print_ci_report(results_dir, scenario)
        print()


if __name__ == '__main__':
    main()
