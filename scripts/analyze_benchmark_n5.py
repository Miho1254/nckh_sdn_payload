#!/usr/bin/env python3
"""
Analyze benchmark results from 5 runs x 4 scenarios x 2 algorithms.
Tính trung bình và so sánh với kết quả chuẩn trong báo cáo.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Kết quả chuẩn từ báo cáo (Bảng 1)
BASELINE_RESULTS = {
    'golden_hour': {'wrr': 11099200, 'ppo': 10072157, 'diff': -9.3},
    'video_conference': {'wrr': 10926702, 'ppo': 9983185, 'diff': -8.6},
    'low_rate_dos': {'wrr': 7944897, 'ppo': 7328959, 'diff': -7.8},
    'hardware_degradation': {'wrr': 7746436, 'ppo': 8685429, 'diff': 12.1},
}

def calculate_total_packets(flow_stats_path):
    """Tính tổng packet_count từ flow_stats.csv"""
    if not os.path.exists(flow_stats_path):
        return None
    try:
        df = pd.read_csv(flow_stats_path, low_memory=False)
        # Lọc chỉ các flow có packet_count > 0 (bỏ header và flow rỗng)
        valid_packets = df[df['packet_count'] > 0]['packet_count'].sum()
        return int(valid_packets)
    except Exception as e:
        print(f"  Error reading {flow_stats_path}: {e}")
        return None

def analyze_scenario(scenario_dir, scenario_name):
    """Analyze một scenario với 5 runs"""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    
    runs_data = []
    
    for run_num in range(1, 6):
        run_dir = os.path.join(scenario_dir, f"run_{run_num}")
        
        # WRR
        wrr_flow = os.path.join(run_dir, 'wrr', 'flow_stats.csv')
        wrr_packets = calculate_total_packets(wrr_flow)
        
        # PPO
        ppo_flow = os.path.join(run_dir, 'ppo', 'flow_stats.csv')
        ppo_packets = calculate_total_packets(ppo_flow)
        
        if wrr_packets is not None and ppo_packets is not None:
            diff_pct = ((ppo_packets - wrr_packets) / wrr_packets * 100) if wrr_packets > 0 else 0
            runs_data.append({
                'run': run_num,
                'wrr_packets': wrr_packets,
                'ppo_packets': ppo_packets,
                'diff_pct': diff_pct
            })
            print(f"  Run {run_num}: WRR={wrr_packets:,}, PPO={ppo_packets:,}, Diff={diff_pct:+.2f}%")
        else:
            print(f"  Run {run_num}: MISSING DATA (WRR={wrr_packets}, PPO={ppo_packets})")
    
    if not runs_data:
        return None
    
    # Tính trung bình
    df = pd.DataFrame(runs_data)
    wrr_mean = df['wrr_packets'].mean()
    ppo_mean = df['ppo_packets'].mean()
    wrr_std = df['wrr_packets'].std()
    ppo_std = df['ppo_packets'].std()
    
    diff_pct_mean = ((ppo_mean - wrr_mean) / wrr_mean * 100) if wrr_mean > 0 else 0
    
    print(f"\n  TRUNG BÌNH 5 RUNS:")
    print(f"    WRR: {wrr_mean:,.0f} ± {wrr_std:,.0f}")
    print(f"    PPO: {ppo_mean:,.0f} ± {ppo_std:,.0f}")
    print(f"    Chênh lệch: {diff_pct_mean:+.2f}%")
    
    # So sánh với baseline
    baseline = BASELINE_RESULTS.get(scenario_name, {})
    if baseline:
        print(f"\n  SO SÁNH VỚI BÁO CÁO:")
        print(f"    WRR báo cáo: {baseline['wrr']:,}")
        print(f"    PPO báo cáo: {baseline['ppo']:,}")
        print(f"    Diff báo cáo: {baseline['diff']:+.1f}%")
        
        wrr_deviation = ((wrr_mean - baseline['wrr']) / baseline['wrr'] * 100) if baseline['wrr'] > 0 else 0
        ppo_deviation = ((ppo_mean - baseline['ppo']) / baseline['ppo'] * 100) if baseline['ppo'] > 0 else 0
        
        print(f"\n  ĐỘ LỆCH so với báo cáo:")
        print(f"    WRR: {wrr_deviation:+.2f}%")
        print(f"    PPO: {ppo_deviation:+.2f}%")
        
        if abs(wrr_deviation) > 5 or abs(ppo_deviation) > 5:
            print(f"  ⚠️ CẢNH BÁO: Độ lệch > 5% - có thể benchmark đã làm sai!")
    
    return {
        'scenario': scenario_name,
        'wrr_mean': wrr_mean,
        'wrr_std': wrr_std,
        'ppo_mean': ppo_mean,
        'ppo_std': ppo_std,
        'diff_pct': diff_pct_mean
    }

def main():
    results_dir = 'benchmark_results_final_n5'
    
    scenarios = ['golden_hour', 'video_conference', 'low_rate_dos', 'hardware_degradation']
    
    all_results = []
    
    for scenario in scenarios:
        scenario_dir = os.path.join(results_dir, scenario)
        if os.path.exists(scenario_dir):
            result = analyze_scenario(scenario_dir, scenario)
            if result:
                all_results.append(result)
    
    # Tổng hợp
    print(f"\n{'='*60}")
    print("TỔNG HỢP KẾT QUẢ (5 runs trung bình)")
    print(f"{'='*60}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        print(f"\n| Kịch bản | WRR (trung bình) | PPO (trung bình) | Chênh lệch |")
        print(f"|----------|------------------|------------------|------------|")
        
        ppo_wins = 0
        for _, row in df.iterrows():
            winner = "PPO" if row['ppo_mean'] > row['wrr_mean'] else "WRR"
            if winner == "PPO":
                ppo_wins += 1
            print(f"| {row['scenario']:<15} | {row['wrr_mean']:>15,.0f} | {row['ppo_mean']:>15,.0f} | {row['diff_pct']:>+10.1f}% |")
        
        total_wrr = df['wrr_mean'].sum()
        total_ppo = df['ppo_mean'].sum()
        total_diff = ((total_ppo - total_wrr) / total_wrr * 100) if total_wrr > 0 else 0
        
        print(f"| {'TỔNG':<15} | {total_wrr:>15,.0f} | {total_ppo:>15,.0f} | {total_diff:>+10.1f}% |")
        
        print(f"\nTổng kết: PPO thắng {ppo_wins}/4 kịch bản ({ppo_wins*25}%), WRR thắng {4-ppo_wins}/4 kịch bản ({(4-ppo_wins)*25}%)")
        
        # So sánh với báo cáo
        print(f"\n{'='*60}")
        print("SO SÁNH VỚI BÁO CÁO (Bảng 1)")
        print(f"{'='*60}")
        
        print(f"\nBáo cáo:")
        print(f"  WRR: 37,717,235 packets")
        print(f"  PPO: 36,069,726 packets")
        print(f"  Diff: -4.4%")
        
        print(f"\nBenchmark 5 runs (trung bình):")
        print(f"  WRR: {total_wrr:,.0f} packets")
        print(f"  PPO: {total_ppo:,.0f} packets")
        print(f"  Diff: {total_diff:+.1f}%")
        
        wrr_dev = ((total_wrr - 37717235) / 37717235 * 100)
        ppo_dev = ((total_ppo - 36069726) / 36069726 * 100)
        
        print(f"\nĐộ lệch:")
        print(f"  WRR: {wrr_dev:+.2f}%")
        print(f"  PPO: {ppo_dev:+.2f}%")
        
        if abs(wrr_dev) < 5 and abs(ppo_dev) < 5:
            print(f"\n✅ ĐẠT YÊU CẦU: Độ lệch < 5% - Kết quả đáng tin cậy!")
        else:
            print(f"\n⚠️ CẢNH BÁO: Độ lệch > 5% - Kiểm tra lại benchmark!")
    
    return all_results

if __name__ == '__main__':
    main()
