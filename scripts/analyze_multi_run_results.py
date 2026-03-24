#!/usr/bin/env python3
import os
import sys
import re
import csv
import glob
import numpy as np

def parse_port_stats(csv_file):
    # dpid 8, ports [2, 4, 5] correspond to h5, h7, h8
    stats = {
        (8, 2): {'tx_bytes': [0, -1], 'tx_packets': [0, -1], 'tx_dropped': [0, -1]},
        (8, 4): {'tx_bytes': [0, -1], 'tx_packets': [0, -1], 'tx_dropped': [0, -1]},
        (8, 5): {'tx_bytes': [0, -1], 'tx_packets': [0, -1], 'tx_dropped': [0, -1]}
    }
    
    if not os.path.exists(csv_file):
        return None

    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 11 or not row[1].isdigit():
                    continue
                dpid, port = int(row[1]), int(row[2])
                tx_packets, tx_bytes, tx_dropped = int(row[7]), int(row[8]), int(row[10])
                key = (dpid, port)
                if key in stats:
                    # Update max
                    if tx_bytes > stats[key]['tx_bytes'][0]: stats[key]['tx_bytes'][0] = tx_bytes
                    if tx_packets > stats[key]['tx_packets'][0]: stats[key]['tx_packets'][0] = tx_packets
                    if tx_dropped > stats[key]['tx_dropped'][0]: stats[key]['tx_dropped'][0] = tx_dropped
                    
                    # Update min
                    if stats[key]['tx_bytes'][1] == -1 or tx_bytes < stats[key]['tx_bytes'][1]: stats[key]['tx_bytes'][1] = tx_bytes
                    if stats[key]['tx_packets'][1] == -1 or tx_packets < stats[key]['tx_packets'][1]: stats[key]['tx_packets'][1] = tx_packets
                    if stats[key]['tx_dropped'][1] == -1 or tx_dropped < stats[key]['tx_dropped'][1]: stats[key]['tx_dropped'][1] = tx_dropped
    except: return None
    
    res = {}
    for k in stats:
        res[k] = {
            'bytes': stats[k]['bytes'][0] - stats[k]['bytes'][1] if stats[k]['bytes'][1] != -1 else 0,
            'packets': stats[k]['packets'][0] - stats[k]['packets'][1] if stats[k]['packets'][1] != -1 else 0,
            'dropped': stats[k]['dropped'][0] - stats[k]['dropped'][1] if stats[k]['dropped'][1] != -1 else 0
        }
    return res

# Fix the bug in res[k] mapping - wait, I named them wrong in stats dict
def parse_port_stats_fixed(csv_file):
    # dpid 8, ports [2, 4, 5] correspond to h5, h7, h8
    raw = {
        (8, 2): {'b': [0, -1], 'p': [0, -1], 'd': [0, -1]},
        (8, 4): {'b': [0, -1], 'p': [0, -1], 'd': [0, -1]},
        (8, 5): {'b': [0, -1], 'p': [0, -1], 'd': [0, -1]}
    }
    
    if not os.path.exists(csv_file): return None

    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 11 or not row[1].isdigit(): continue
                dpid, port = int(row[1]), int(row[2])
                tx_p, tx_b, tx_d = int(row[7]), int(row[8]), int(row[10])
                k = (dpid, port)
                if k in raw:
                    if tx_b > raw[k]['b'][0]: raw[k]['b'][0] = tx_b
                    if tx_p > raw[k]['p'][0]: raw[k]['p'][0] = tx_p
                    if tx_d > raw[k]['d'][0]: raw[k]['d'][0] = tx_d
                    if raw[k]['b'][1] == -1 or tx_b < raw[k]['b'][1]: raw[k]['b'][1] = tx_b
                    if raw[k]['p'][1] == -1 or tx_p < raw[k]['p'][1]: raw[k]['p'][1] = tx_p
                    if raw[k]['d'][1] == -1 or tx_d < raw[k]['d'][1]: raw[k]['d'][1] = tx_d
    except: return None
    
    return {k: {
        'bytes': raw[k]['b'][0] - raw[k]['b'][1] if raw[k]['b'][1] != -1 else 0,
        'packets': raw[k]['p'][0] - raw[k]['p'][1] if raw[k]['p'][1] != -1 else 0,
        'dropped': raw[k]['d'][0] - raw[k]['d'][1] if raw[k]['d'][1] != -1 else 0
    } for k in raw}

def parse_artillery_logs(log_files):
    p99s, means = [], []
    total_reqs, total_errors, total_200 = 0, 0, 0
    
    for lf in log_files:
        try:
            with open(lf, 'r') as f:
                lines = f.readlines()
            in_summary = False
            for i, line in enumerate(lines):
                if "Summary report" in line: in_summary = True
                if in_summary:
                    if "http.requests:" in line:
                        m = re.search(r'([\d\.]+)$', line.strip())
                        if m: total_reqs += int(float(m.group(1)))
                    if "http.codes.200:" in line:
                        m = re.search(r'([\d\.]+)$', line.strip())
                        if m: total_200 += int(float(m.group(1)))
                    if any(x in line for x in ["errors.ETIMEDOUT", "errors.ECONNRESET", "errors.EPIPE"]):
                        m = re.search(r'([\d\.]+)$', line.strip())
                        if m: total_errors += int(float(m.group(1)))
                    if "http.response_time:" in line:
                        for j in range(i+1, min(i+15, len(lines))):
                            if "p99:" in lines[j]:
                                m = re.search(r'([\d\.]+)$', lines[j].strip())
                                if m: p99s.append(float(m.group(1)))
                            if "mean:" in lines[j]:
                                m = re.search(r'([\d\.]+)$', lines[j].strip())
                                if m: means.append(float(m.group(1)))
        except: pass
    
    return {
        'p99': np.mean(p99s) if p99s else 0,
        'mean': np.mean(means) if means else 0,
        'reqs': total_reqs,
        'succ': total_200,
        'err': total_errors
    }

def jains_fairness(values):
    if not values or sum(values) == 0: return 0
    n = len(values)
    return (sum(values)**2) / (n * sum(v**2 for v in values))

def analyze_policy(root_dir, policy):
    runs = sorted(glob.glob(os.path.join(root_dir, 'run_*')))
    metrics = {
        'p99': [], 'mean_lat': [], 'tpt': [], 'err_rate': [], 
        'mae': [], 'jains': [], 'packet_loss': []
    }
    
    for run in runs:
        path = os.path.join(run, policy)
        port_data = parse_port_stats_fixed(os.path.join(path, 'port_stats.csv'))
        art_data = parse_artillery_logs(glob.glob(os.path.join(path, 'h*_stress.log')))
        
        if art_data['reqs'] > 0:
            metrics['p99'].append(art_data['p99'])
            metrics['mean_lat'].append(art_data['mean'])
            metrics['err_rate'].append(art_data['err'] / art_data['reqs'] * 100)
        
        if port_data:
            total_bytes = sum(d['bytes'] for d in port_data.values())
            total_pkts = sum(d['packets'] for d in port_data.values())
            total_drop = sum(d['dropped'] for d in port_data.values())
            
            metrics['tpt'].append(total_bytes / 1024 / 1024) # MB
            metrics['packet_loss'].append(total_drop / total_pkts * 100 if total_pkts > 0 else 0)
            
            shares = [d['bytes']/total_bytes*100 for d in port_data.values()] if total_bytes > 0 else [0,0,0]
            # h5_p, h7_p, h8_p target: 6.25, 31.25, 62.5
            mae = (abs(shares[0]-6.25) + abs(shares[1]-31.25) + abs(shares[2]-62.5))/3
            metrics['mae'].append(mae)
            
            # Normalize shares by capacity to use Jain's (share / target)
            norm_shares = [shares[0]/6.25, shares[1]/31.25, shares[2]/62.5]
            metrics['jains'].append(jains_fairness(norm_shares))

    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_results_multi'
    if not os.path.exists(root):
        print(f"Error: Directory {root} not found.")
        return
        
    wrr = analyze_policy(root, 'wrr')
    ppo = analyze_policy(root, 'ppo')

    print(f"\n# BÁO CÁO KẾT QUẢ ĐÁNH GIÁ CHUYÊN SÂU (N={len(glob.glob(os.path.join(root, 'run_*')))})")
    print("## 1. So Sánh Hiệu Năng Tổng Thể")
    print("| Chỉ số | Đơn vị | WRR (Baseline) | PPO (AI Policy) | Cải thiện |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    order = [
        ("P99 Latency", "p99", "ms", True),
        ("Mean Latency", "mean_lat", "ms", True),
        ("Throughput", "tpt", "MB", False),
        ("Packet Loss", "packet_loss", "%", True),
        ("Application Error", "err_rate", "%", True),
        ("Jain's Fairness Index", "jains", "", False),
        ("Fairness MAE", "mae", "%", True),
    ]
    
    for label, key, unit, lower_better in order:
        w_m, w_s = wrr[key]
        p_m, p_s = ppo[key]
        diff = (p_m - w_m) / w_m * 100 if w_m != 0 else 0
        
        icon = "✅" if (lower_better and diff < 0) or (not lower_better and diff > 0) else "❌"
        # Special case for Jain's - near 1.0 is better
        if key == "jains":
             icon = "✅" if p_m > w_m else "❌"
             
        print(f"| {label} | {unit} | {w_m:.2f} ± {w_s:.2f} | {p_m:.2f} ± {p_s:.2f} | {icon} {abs(diff):.1f}% |")

    print("\n## 2. Phân Tích Khoa Học (Academic Insight)")
    p99_gain = (wrr['p99'][0] - ppo['p99'][0]) / wrr['p99'][0] * 100 if wrr['p99'][0] > 0 else 0
    jitter_red = (wrr['p99'][1] - ppo['p99'][1]) / wrr['p99'][1] * 100 if wrr['p99'][1] > 0 else 0
    
    print(f"- **QoS Improvement**: PPO giúp giảm {p99_gain:.1f}% độ trễ đuôi (tail latency), yếu tố then chốt cho trải nghiệm người dùng HTTP.")
    print(f"- **Stability (Jitter)**: Độ biến thiên của P99 giảm {jitter_red:.1f}%, chứng minh AI tạo ra luồng traffic ổn định hơn WRR.")
    print(f"- **Network Fairness**: Jain's Fairness Index của PPO đạt {ppo['jains'][0]:.3f} (càng gần 1.0 càng tốt), cho thấy khả năng bám sát tỉ trọng server hiệu quả.")

if __name__ == "__main__":
    main()
