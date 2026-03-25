#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats

results = []
scenarios = ['golden_hour', 'video_conference', 'hardware_degradation', 'low_rate_dos']

for scenario in scenarios:
    for run in range(1, 6):
        for algo in ['ppo', 'wrr']:
            flow_path = f'benchmark_results_final_n5/{scenario}/run_{run}/{algo}/flow_stats.csv'
            try:
                df = pd.read_csv(flow_path, low_memory=False)
                packets = df['packet_count'].sum()
                results.append({'scenario': scenario, 'run': run, 'algo': algo, 'packets': packets})
            except Exception as e:
                print(f"Error reading {flow_path}: {e}")

df = pd.DataFrame(results)

print('=' * 60)
print('BENCHMARK RESULTS SUMMARY - PPO V3 vs WRR')
print('=' * 60)
print()

ppo_wins = 0
wrr_wins = 0

for scenario in scenarios:
    sc = df[df['scenario'] == scenario]
    ppo_data = sc[sc['algo'] == 'ppo']['packets'].values
    wrr_data = sc[sc['algo'] == 'wrr']['packets'].values
    
    if len(ppo_data) == 5 and len(wrr_data) == 5:
        ppo_mean = np.mean(ppo_data)
        wrr_mean = np.mean(wrr_data)
        diff_pct = ((ppo_mean - wrr_mean) / wrr_mean) * 100
        
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(ppo_data, wrr_data)
        
        winner = 'PPO' if ppo_mean > wrr_mean else 'WRR'
        if winner == 'PPO':
            ppo_wins += 1
        else:
            wrr_wins += 1
        
        sig = "(SIGNIFICANT)" if p_val < 0.05 else ""
        print(f"{scenario}:")
        print(f"  PPO: {ppo_mean:>12,.0f} ± {np.std(ppo_data):>10,.0f} packets")
        print(f"  WRR: {wrr_mean:>12,.0f} ± {np.std(wrr_data):>10,.0f} packets")
        print(f"  Winner: {winner:>4} ({diff_pct:>+6.1f}%)  p={p_val:.4f} {sig}")
        print()

print('=' * 60)
print(f"OVERALL: PPO won {ppo_wins}/4 scenarios, WRR won {wrr_wins}/4 scenarios")

# Overall statistics
ppo_all = df[df['algo'] == 'ppo']['packets']
wrr_all = df[df['algo'] == 'wrr']['packets']
print()
print(f"Overall PPO: {ppo_all.mean():,.0f} ± {ppo_all.std():,.0f} packets")
print(f"Overall WRR: {wrr_all.mean():,.0f} ± {wrr_all.std():,.0f} packets")
print(f"Difference: {((ppo_all.mean() - wrr_all.mean()) / wrr_all.mean()) * 100:+.1f}%")

# Check inference logs
print()
print("=" * 60)
print("INFERENCE LOG STATUS")
print("=" * 60)
for scenario in scenarios:
    for run in range(1, 6):
        inf_path = f'benchmark_results_final_n5/{scenario}/run_{run}/ppo/inference_log.csv'
        try:
            with open(inf_path, 'r') as f:
                lines = f.readlines()
                data_rows = len(lines) - 1  # exclude header
                print(f"{scenario}/run_{run}: {data_rows} inference records")
        except Exception as e:
            print(f"{scenario}/run_{run}: Error - {e}")
