#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import os

print('=' * 70)
print('PPO V3 BENCHMARK RESULTS - Real Mininet/Ryu')
print('=' * 70)

scenarios = ['golden_hour', 'hardware_degradation', 'low_rate_dos', 'video_conference']
algo_results = {'ppo': [], 'wrr': []}

for scenario in scenarios:
    for run in range(1, 6):
        for algo in ['ppo', 'wrr']:
            flow_file = f'/work/benchmark_results_final_n5/{scenario}/run_{run}/{algo}/flow_stats.csv'
            if os.path.exists(flow_file):
                try:
                    df = pd.read_csv(flow_file, low_memory=False)
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
                    pass

print(f"PPO runs: {len(algo_results['ppo'])}, WRR runs: {len(algo_results['wrr'])}")

if algo_results['ppo'] and algo_results['wrr']:
    ppo_packets = [r['packets'] for r in algo_results['ppo']]
    wrr_packets = [r['packets'] for r in algo_results['wrr']]
    
    print(f'\nPPO: {np.mean(ppo_packets):.0f} ± {np.std(ppo_packets):.0f} packets')
    print(f'WRR: {np.mean(wrr_packets):.0f} ± {np.std(wrr_packets):.0f} packets')
    
    t_stat, p_value = stats.ttest_ind(ppo_packets, wrr_packets)
    cohen_d = (np.mean(ppo_packets) - np.mean(wrr_packets)) / np.sqrt((np.std(ppo_packets)**2 + np.std(wrr_packets)**2) / 2)
    
    print(f'\nStatistical Tests:')
    print(f'  p-value: {p_value:.4f}')
    print(f'  Cohen d: {cohen_d:.3f}')
    
    winner = 'PPO V3' if np.mean(ppo_packets) > np.mean(wrr_packets) else 'WRR'
    diff = abs(np.mean(ppo_packets) - np.mean(wrr_packets)) / np.mean(wrr_packets) * 100
    print(f'\nWinner: {winner} ({diff:+.1f}%)')

# PPO action distribution
print('\n' + '=' * 70)
print('PPO ACTION DISTRIBUTION')
print('=' * 70)

action_counts = {0: 0, 1: 0, 2: 0}
total = 0

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
                            total += 1
            except:
                pass

if total > 0:
    print(f'Total actions: {total}')
    for a, c in action_counts.items():
        print(f'  Action {a}: {c} ({c/total*100:.1f}%)')

print('\n' + '=' * 70)
print('PER-SCENARIO BREAKDOWN')
print('=' * 70)

for scenario in scenarios:
    ppo_s = [r for r in algo_results['ppo'] if r['scenario'] == scenario]
    wrr_s = [r for r in algo_results['wrr'] if r['scenario'] == scenario]
    if ppo_s and wrr_s:
        ppo_m = np.mean([r['packets'] for r in ppo_s])
        wrr_m = np.mean([r['packets'] for r in wrr_s])
        diff = (ppo_m - wrr_m) / wrr_m * 100
        win = 'PPO' if ppo_m > wrr_m else 'WRR'
        print(f'{scenario}: PPO={ppo_m/1e6:.2f}M, WRR={wrr_m/1e6:.2f}M, Diff={diff:+.1f}%, Winner={win}')
