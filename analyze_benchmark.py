#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import os

print('=' * 70)
print('BENCHMARK RESULTS ANALYSIS - Real Mininet/Ryu')
print('=' * 70)

scenarios = ['golden_hour', 'hardware_degradation', 'low_rate_dos', 'video_conference']
algo_results = {'ppo': [], 'wrr': []}

for scenario in scenarios:
    for run in range(1, 6):
        for algo in ['ppo', 'wrr']:
            flow_file = f'/work/benchmark_results_final_n5/{scenario}/run_{run}/{algo}/flow_stats.csv'
            if os.path.exists(flow_file):
                try:
                    df = pd.read_csv(flow_file)
                    if len(df) > 0:
                        total_packets = df['packet_count'].sum() if 'packet_count' in df.columns else 0
                        total_bytes = df['byte_count'].sum() if 'byte_count' in df.columns else 0
                        algo_results[algo].append({
                            'scenario': scenario,
                            'run': run,
                            'packets': total_packets,
                            'bytes': total_bytes
                        })
                except Exception as e:
                    print(f"Error reading {flow_file}: {e}")

print(f'\nPPO runs collected: {len(algo_results["ppo"])}')
print(f'WRR runs collected: {len(algo_results["wrr"])}')

if algo_results['ppo'] and algo_results['wrr']:
    ppo_packets = [r['packets'] for r in algo_results['ppo']]
    wrr_packets = [r['packets'] for r in algo_results['wrr']]
    
    ppo_bytes = [r['bytes'] for r in algo_results['ppo']]
    wrr_bytes = [r['bytes'] for r in algo_results['wrr']]
    
    print(f'\n--- Throughput Comparison ---')
    print(f'PPO: {np.mean(ppo_packets):.0f} ± {np.std(ppo_packets):.0f} packets')
    print(f'WRR: {np.mean(wrr_packets):.0f} ± {np.std(wrr_packets):.0f} packets')
    
    print(f'\nPPO: {np.mean(ppo_bytes)/1e6:.2f} ± {np.std(ppo_bytes)/1e6:.2f} MB')
    print(f'WRR: {np.mean(wrr_bytes)/1e6:.2f} ± {np.std(wrr_bytes)/1e6:.2f} MB')
    
    t_stat, p_value = stats.ttest_ind(ppo_packets, wrr_packets)
    cohen_d = (np.mean(ppo_packets) - np.mean(wrr_packets)) / np.sqrt((np.std(ppo_packets)**2 + np.std(wrr_packets)**2) / 2)
    
    print(f'\n--- Statistical Tests ---')
    print(f't-test: t={t_stat:.3f}, p={p_value:.4f}')
    print(f"Cohen's d: {cohen_d:.3f}")
    sig = "YES" if p_value < 0.05 else "NO"
    print(f'Significant (p<0.05): {sig}')
    
    winner = 'PPO' if np.mean(ppo_packets) > np.mean(wrr_packets) else 'WRR'
    diff_pct = abs(np.mean(ppo_packets) - np.mean(wrr_packets)) / np.mean(wrr_packets) * 100
    print(f'\nWinner: {winner} (+{diff_pct:.1f}%)')

# Analyze PPO action distribution from inference logs
print('\n' + '=' * 70)
print('PPO ACTION DISTRIBUTION ANALYSIS')
print('=' * 70)

action_counts = {0: 0, 1: 0, 2: 0}
total_actions = 0

for scenario in scenarios:
    for run in range(1, 6):
        inf_file = f'/work/benchmark_results_final_n5/{scenario}/run_{run}/ppo/inference_log.csv'
        if os.path.exists(inf_file):
            try:
                df = pd.read_csv(inf_file)
                if 'action' in df.columns:
                    for a in df['action']:
                        if a in action_counts:
                            action_counts[a] += 1
                            total_actions += 1
            except:
                pass

if total_actions > 0:
    print(f'\nTotal actions analyzed: {total_actions}')
    for a, count in action_counts.items():
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f'Action {a}: {count} ({pct:.1f}%)')

print('\n' + '=' * 70)
print('PER-SCENARIO BREAKDOWN')
print('=' * 70)

for scenario in scenarios:
    ppo_scenario = [r for r in algo_results['ppo'] if r['scenario'] == scenario]
    wrr_scenario = [r for r in algo_results['wrr'] if r['scenario'] == scenario]
    
    if ppo_scenario and wrr_scenario:
        ppo_mean = np.mean([r['packets'] for r in ppo_scenario])
        wrr_mean = np.mean([r['packets'] for r in wrr_scenario])
        diff = (ppo_mean - wrr_mean) / wrr_mean * 100 if wrr_mean > 0 else 0
        winner = 'PPO' if ppo_mean > wrr_mean else 'WRR'
        print(f'{scenario}: PPO={ppo_mean:.0f}, WRR={wrr_mean:.0f}, Diff={diff:+.1f}%, Winner={winner}')
