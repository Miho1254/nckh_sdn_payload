#!/usr/bin/env python3
"""
Extract detailed metrics (p99, jitter, packet_loss) from artillery benchmark logs.
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def parse_artillery_log(log_path: str) -> Dict:
    """Parse a single artillery stress log file."""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract p99 latency
    p99_matches = re.findall(r'p99:\s*\.*\s*([\d.]+)', content)
    p99_values = [float(m) for m in p99_matches if float(m) > 0]
    
    # Extract mean latency
    mean_matches = re.findall(r'mean:\s*\.*\s*([\d.]+)', content)
    mean_values = [float(m) for m in mean_matches if float(m) > 0]
    
    # Extract median latency
    median_matches = re.findall(r'median:\s*\.*\s*([\d.]+)', content)
    median_values = [float(m) for m in median_matches if float(m) > 0]
    
    # Extract total requests and errors
    req_matches = re.findall(r'http\.requests:\s*\.*\s*(\d+)', content)
    total_requests = sum(int(m) for m in req_matches)
    
    error_matches = re.findall(r'errors\.\w+:\s*\.*\s*(\d+)', content)
    total_errors = sum(int(m) for m in error_matches)
    
    # Extract 200 responses
    code_200_matches = re.findall(r'http\.codes\.200:\s*\.*\s*(\d+)', content)
    total_200 = sum(int(m) for m in code_200_matches)
    
    return {
        'p99': p99_values,
        'mean': mean_values,
        'median': median_values,
        'total_requests': total_requests,
        'total_errors': total_errors,
        'total_200': total_200
    }

def calculate_jitter(values: List[float]) -> float:
    """Calculate jitter as standard deviation of latency values."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values))

def calculate_packet_loss(total_requests: int, total_errors: int) -> float:
    """Calculate packet loss percentage."""
    if total_requests == 0:
        return 0.0
    return (total_errors / total_requests) * 100

def extract_all_metrics(results_dir: str) -> Dict:
    """Extract and aggregate metrics from all artillery logs in directory."""
    results_path = Path(results_dir)
    
    metrics = {
        'p99_latencies': [],
        'mean_latencies': [],
        'medians': [],
        'jitters': [],
        'packet_losses': [],
        'total_requests': 0,
        'total_errors': 0,
        'total_200': 0
    }
    
    # Find all stress log files
    stress_logs = list(results_path.glob('*_stress.log'))
    
    for log_path in stress_logs:
        data = parse_artillery_log(str(log_path))
        if data:
            # Collect p99 values (use mean of all p99 measurements in this log)
            if data['p99']:
                metrics['p99_latencies'].extend(data['p99'])
            
            if data['mean']:
                metrics['mean_latencies'].extend(data['mean'])
            
            if data['median']:
                metrics['medians'].extend(data['median'])
            
            # Calculate jitter for this log
            if len(data['p99']) >= 2:
                jitter = calculate_jitter(data['p99'])
                metrics['jitters'].append(jitter)
            
            # Packet loss for this log
            pl = calculate_packet_loss(data['total_requests'], data['total_errors'])
            if pl > 0:
                metrics['packet_losses'].append(pl)
            
            metrics['total_requests'] += data['total_requests']
            metrics['total_errors'] += data['total_errors']
            metrics['total_200'] += data['total_200']
    
    return metrics

def analyze_run(run_dir: str) -> Dict:
    """Analyze a single run directory (contains ppo/ and wrr/ subdirs)."""
    results = {}
    
    for algo in ['ppo', 'wrr']:
        algo_dir = os.path.join(run_dir, algo)
        if os.path.exists(algo_dir):
            metrics = extract_all_metrics(algo_dir)
            
            # Calculate aggregated statistics
            results[algo] = {
                'p99_mean': np.mean(metrics['p99_latencies']) if metrics['p99_latencies'] else 0,
                'p99_std': np.std(metrics['p99_latencies']) if metrics['p99_latencies'] else 0,
                'p99_min': np.min(metrics['p99_latencies']) if metrics['p99_latencies'] else 0,
                'p99_max': np.max(metrics['p99_latencies']) if metrics['p99_latencies'] else 0,
                'mean_latency': np.mean(metrics['mean_latencies']) if metrics['mean_latencies'] else 0,
                'jitter_mean': np.mean(metrics['jitters']) if metrics['jitters'] else 0,
                'jitter_std': np.std(metrics['jitters']) if metrics['jitters'] else 0,
                'packet_loss_mean': np.mean(metrics['packet_losses']) if metrics['packet_losses'] else 0,
                'packet_loss_max': np.max(metrics['packet_losses']) if metrics['packet_losses'] else 0,
                'total_requests': metrics['total_requests'],
                'total_errors': metrics['total_errors'],
                'total_200': metrics['total_200']
            }
    
    return results

def analyze_benchmark_results(root_dir: str, scenario: str = 'golden_hour') -> pd.DataFrame:
    """Analyze benchmark results and return comparison DataFrame."""
    all_results = []
    
    # Find all run directories
    scenario_dir = os.path.join(root_dir, scenario)
    if not os.path.exists(scenario_dir):
        print(f"Scenario directory not found: {scenario_dir}")
        return None
    
    run_dirs = sorted([d for d in os.listdir(scenario_dir) if d.startswith('run_')])
    
    for run_dir in run_dirs:
        run_path = os.path.join(scenario_dir, run_dir)
        results = analyze_run(run_path)
        
        if results:
            all_results.append({
                'run': run_dir,
                **results.get('ppo', {}),
                **{f'wrr_{k}': v for k, v in results.get('wrr', {}).items()}
            })
    
    if not all_results:
        return None
    
    df = pd.DataFrame(all_results)
    return df

def print_comparison(results_dir: str, scenario: str = 'golden_hour'):
    """Print detailed comparison of PPO vs WRR metrics."""
    print("=" * 80)
    print(f"BENCHMARK DETAILED METRICS - {scenario}")
    print("=" * 80)
    
    df = analyze_benchmark_results(results_dir, scenario)
    
    if df is None or df.empty:
        print("No results found!")
        return
    
    # Calculate means across all runs
    ppo_p99 = df['p99_mean'].mean()
    ppo_jitter = df['jitter_mean'].mean()
    ppo_packet_loss = df['packet_loss_mean'].mean()
    ppo_throughput = df['total_200'].mean()
    
    wrr_p99 = df['wrr_p99_mean'].mean()
    wrr_jitter = df['wrr_jitter_mean'].mean()
    wrr_packet_loss = df['wrr_packet_loss_mean'].mean()
    wrr_throughput = df['wrr_total_200'].mean()
    
    print("\n┌─────────────────┬────────────────┬────────────────┬───────────────┐")
    print("│ Metric          │ PPO            │ WRR            │ Winner        │")
    print("├─────────────────┼────────────────┼────────────────┼───────────────┤")
    
    # P99 Latency (lower is better)
    p99_winner = "WRR" if wrr_p99 < ppo_p99 else "PPO"
    p99_diff = ((ppo_p99 - wrr_p99) / wrr_p99 * 100) if wrr_p99 > 0 else 0
    print(f"│ P99 Latency     │ {ppo_p99:>10.2f} ms │ {wrr_p99:>10.2f} ms │ {p99_winner} ({p99_diff:+.1f}%)   │")
    
    # Jitter (lower is better)
    jitter_winner = "WRR" if wrr_jitter < ppo_jitter else "PPO"
    jitter_diff = ((ppo_jitter - wrr_jitter) / wrr_jitter * 100) if wrr_jitter > 0 else 0
    print(f"│ Jitter          │ {ppo_jitter:>10.2f} ms │ {wrr_jitter:>10.2f} ms │ {jitter_winner} ({jitter_diff:+.1f}%)   │")
    
    # Packet Loss (lower is better)
    pl_winner = "WRR" if wrr_packet_loss < ppo_packet_loss else "PPO"
    pl_diff = ((ppo_packet_loss - wrr_packet_loss) / wrr_packet_loss * 100) if wrr_packet_loss > 0 else 0
    print(f"│ Packet Loss     │ {ppo_packet_loss:>10.2f} %   │ {wrr_packet_loss:>10.2f} %   │ {pl_winner} ({pl_diff:+.1f}%)   │")
    
    # Throughput (higher is better)
    tp_winner = "PPO" if ppo_throughput > wrr_throughput else "WRR"
    tp_diff = ((ppo_throughput - wrr_throughput) / wrr_throughput * 100) if wrr_throughput > 0 else 0
    print(f"│ Throughput      │ {ppo_throughput:>10.0f} reqs │ {wrr_throughput:>10.0f} reqs │ {tp_winner} ({tp_diff:+.1f}%)   │")
    
    print("└─────────────────┴────────────────┴────────────────┴───────────────┘")
    
    print("\n--- Per-run breakdown ---")
    print(df[['run', 'p99_mean', 'wrr_p99_mean', 'jitter_mean', 'wrr_jitter_mean', 
              'packet_loss_mean', 'wrr_packet_loss_mean']].to_string(index=False))
    
    return df

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        results_dir = 'benchmark_results_quick'
    else:
        results_dir = sys.argv[1]
    
    scenario = sys.argv[2] if len(sys.argv) > 2 else 'golden_hour'
    
    df = print_comparison(results_dir, scenario)
    
    if df is not None:
        # Save detailed results
        output_file = f'{results_dir}_{scenario}_detailed_metrics.csv'
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
