"""
generate_comparison_charts.py
Script tự động đọc kết quả từ stats/results/ và vẽ biểu đồ so sánh AI vs RR vs WRR.

Sử dụng:
    python3 ai_model/generate_comparison_charts.py --scenario flash_crowd
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, '..')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'stats', 'results')
CHARTS_DIR = os.path.join(RESULTS_DIR, 'charts')

# Mau sac chuyen nghiep cho tung thuat toan
ALGO_STYLES = {
    'RR':      {'color': '#EF4444', 'label': 'Round Robin (RR)',      'marker': 'o', 'linestyle': '-'},
    'WRR':     {'color': '#F59E0B', 'label': 'Weighted RR (WRR)',     'marker': 's', 'linestyle': '--'},
    'AI':      {'color': '#10B981', 'label': 'TFT-DQN (AI)',          'marker': 'D', 'linestyle': '-'},
    'COLLECT': {'color': '#6B7280', 'label': 'Baseline (COLLECT)',    'marker': 'x', 'linestyle': ':'},
}

THEMES = {
    'dark': {
        'bg': '#0F172A',
        'card_bg': '#1E293B',
        'text': '#E2E8F0',
        'grid': '#334155',
    },
    'light': {
        'bg': '#F8FAFC',
        'card_bg': '#FFFFFF',
        'text': '#1E293B',
        'grid': '#CBD5E1',
    },
}

# Default (se ghi de khi goi setup_theme)
COLORS = THEMES['dark']

def setup_theme(theme='dark'):
    global COLORS
    COLORS = THEMES.get(theme, THEMES['dark'])
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['card_bg'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.4,
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.facecolor': COLORS['card_bg'],
        'legend.edgecolor': COLORS['grid'],
    })

# Giu lai ten cu de khong break code cu
setup_dark_style = lambda: setup_theme('dark')

def find_result_dirs(scenario_name):
    """Tim tat ca thu muc ket qua cho 1 kich ban."""
    results = {}
    if not os.path.exists(RESULTS_DIR):
        return results
    for dirname in sorted(os.listdir(RESULTS_DIR)):
        if dirname == 'charts':
            continue
        if dirname.endswith(f'_{scenario_name}'):
            algo = dirname.replace(f'_{scenario_name}', '')
            dirpath = os.path.join(RESULTS_DIR, dirname)
            if os.path.isdir(dirpath):
                results[algo] = dirpath
    return results

def load_flow_stats(csv_path):
    """Doc flow_stats.csv va tinh throughput."""
    if not os.path.exists(csv_path):
        return None
    
    valid_data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None) # Skip header
        for row in reader:
            if len(row) < 11: continue
            
            # Schema-aware index mapping
            if len(row) >= 14:
                # 14 columns: ..., byte_count(10), duration_sec(11), duration_nsec(12), label(13)
                idx_bytes = 10
                idx_sec = 11
                idx_nsec = 12
                idx_label = 13
                idx_packets = 9
            else:
                # 12 columns: ..., byte_count(8), duration_sec(9), duration_nsec(10), label(11)
                idx_bytes = 8
                idx_sec = 9
                idx_nsec = 10
                idx_label = 11
                idx_packets = 7

            try:
                if row[idx_label].strip() not in ['NORMAL', 'HIGH']: continue
                
                valid_data.append({
                    'timestamp': row[0],
                    'datapath_id': row[1],
                    'priority': int(row[3]),
                    'in_port': int(row[4]),
                    'eth_dst': row[6],
                    'packet_count': int(row[idx_packets]),
                    'byte_count': int(row[idx_bytes]),
                    'duration_sec': int(row[idx_sec]),
                    'duration_nsec': int(row[idx_nsec]),
                    'label': row[idx_label].strip()
                })
            except:
                continue
    
    if not valid_data:
        return None
    
    df = pd.DataFrame(valid_data)
    df['duration'] = df['duration_sec'] + df['duration_nsec'] / 1e9
    df['duration'] = df['duration'].apply(lambda x: x if x > 0 else 0.001)
    df['byte_rate'] = df['byte_count'] / df['duration']
    df['packet_rate'] = df['packet_count'] / df['duration']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].dt.round('1s')
    df = df.sort_values('timestamp')
    
    return df

def load_port_stats(csv_path):
    """Doc port_stats.csv de tinh packet loss."""
    if not os.path.exists(csv_path):
        return None
    
    columns = ['timestamp','datapath_id','port_no','rx_packets','rx_bytes',
                'rx_errors','rx_dropped','tx_packets','tx_bytes',
                'tx_errors','tx_dropped','collisions','duration_sec',
                'duration_nsec','label']
    
    valid_rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 15 and row[-1] in ['NORMAL', 'HIGH']:
                valid_rows.append(row)
    
    if not valid_rows:
        return None
    
    df = pd.DataFrame(valid_rows, columns=columns)
    for col in ['rx_packets','rx_bytes','rx_errors','rx_dropped',
                'tx_packets','tx_bytes','tx_errors','tx_dropped','collisions']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def load_inference_log(csv_path):
    """Doc inference_log.csv."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        df['inference_ms'] = pd.to_numeric(df['inference_ms'], errors='coerce')
        return df
    except:
        return None


# ================================================================
# BIEU DO 6: Throughput Stability (Area Chart)
# ================================================================
def plot_throughput_stability(algo_data, scenario_name):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for algo, dirpath in algo_data.items():
        style = ALGO_STYLES.get(algo, ALGO_STYLES['COLLECT'])
        df = load_flow_stats(os.path.join(dirpath, 'flow_stats.csv'))
        if df is None:
            continue
        
        agg = df.groupby('timestamp').agg({'byte_rate': 'sum'}).reset_index()
        agg['throughput_mbps'] = (agg['byte_rate'] * 8) / 1e6
        
        # Dung thoi gian tuong doi (giay ke tu bat dau)
        t0 = agg['timestamp'].min()
        agg['elapsed_sec'] = (agg['timestamp'] - t0).dt.total_seconds()
        
        ax.fill_between(agg['elapsed_sec'], agg['throughput_mbps'], alpha=0.1, color=style['color'])
        ax.plot(agg['elapsed_sec'], agg['throughput_mbps'], 
                color=style['color'], linewidth=1.8, label=style['label'],
                linestyle=style['linestyle'])
    
    ax.set_xlabel('Thoi gian (giay)')
    ax.set_ylabel('Throughput (Mbps)')
    ax.set_title(f'Throughput Stability — Scenario: {scenario_name}')
    ax.legend(framealpha=0.8)
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f'06_throughput_{scenario_name}.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  [6] {os.path.basename(path)}")


# ================================================================
# BIEU DO 7: Packet Loss Rate (Bar Chart)
# ================================================================
def plot_packet_loss(algo_data, scenario_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algos = []
    loss_rates = []
    colors = []
    
    for algo, dirpath in algo_data.items():
        style = ALGO_STYLES.get(algo, ALGO_STYLES['COLLECT'])
        
        # OVS switch khong drop packet o network layer trong Mininet (total_drop luon = 0).
        # Thay vao do: dung ty le flow HIGH / (HIGH + NORMAL) lam "Congestion Rate" proxy.
        df = load_flow_stats(os.path.join(dirpath, 'flow_stats.csv'))
        if df is None:
            continue
        
        total = len(df)
        high_count = (df['label'] == 'HIGH').sum() if 'label' in df.columns else 0
        congestion_rate = (high_count / total * 100) if total > 0 else 0
        
        algos.append(style['label'])
        loss_rates.append(congestion_rate)
        colors.append(style['color'])
    
    if not algos:
        return
    
    bars = ax.bar(algos, loss_rates, color=colors, alpha=0.85, width=0.5,
                  edgecolor='white', linewidth=0.5)
    
    for bar, rate in zip(bars, loss_rates):
        ypos = bar.get_height() + 0.3
        ax.text(bar.get_x() + bar.get_width()/2., ypos,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=COLORS['text'])
    
    ax.set_ylabel('Congestion Rate — % Flow mang nhan HIGH')
    ax.set_title(f'Network Congestion Rate — {scenario_name}')
    ax.set_ylim(0, max(loss_rates) * 1.25 + 2 if loss_rates else 10)
    ax.grid(True, linestyle='--', axis='y')
    ax.text(0.98, 0.98, 'HIGH = Tac nghen | NORMAL = On dinh',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color=COLORS['text'], alpha=0.6)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f'07_packet_loss_{scenario_name}.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  [7] {os.path.basename(path)}"
)


# ================================================================
# BIEU DO 8: Heatmap Server Load
# ================================================================
def plot_server_heatmap(algo_data, scenario_name):
    # Map in_port -> server (theo cau hinh BACKENDS trong controller_stats.py)
    backend_ports = {5: 'h5', 7: 'h7', 8: 'h8'}
    
    n_algos = len(algo_data)
    if n_algos == 0:
        return
    
    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 5))
    if n_algos == 1:
        axes = [axes]
    
    has_any_data = False
    for ax, (algo, dirpath) in zip(axes, algo_data.items()):
        style = ALGO_STYLES.get(algo, ALGO_STYLES['COLLECT'])
        df = load_flow_stats(os.path.join(dirpath, 'flow_stats.csv'))
        if df is None:
            ax.set_title(f'{style["label"]} (No data)')
            continue
        
        # Loc flow NAT (priority=100) — in_port la port cua backend 
        nat_flows = df[df['priority'] == 100].copy()
        if nat_flows.empty:
            ax.set_title(f'{style["label"]} (No NAT flows)')
            continue
        
        # Map in_port -> server name
        nat_flows['server'] = nat_flows['in_port'].map(backend_ports)
        nat_flows = nat_flows.dropna(subset=['server'])
        
        if nat_flows.empty:
            # Thu loc theo datapath_id (switch chua LB flow la s9=9)
            lb_switch = df[(df['priority'] == 100) & (df['datapath_id'].astype(str) == '9')].copy()
            ax.set_title(f'{style["label"]} (No port mapping)')
            continue
        
        t0 = nat_flows['timestamp'].min()
        nat_flows['time_bin'] = ((nat_flows['timestamp'] - t0).dt.total_seconds() // 30).astype(int)
        
        pivot = nat_flows.groupby(['server', 'time_bin'])['byte_count'].sum().unstack(fill_value=0)
        
        max_val = pivot.values.max()
        pivot_norm = pivot / max_val if max_val > 0 else pivot
        
        im = ax.imshow(pivot_norm.values, aspect='auto', cmap='RdYlGn_r',
                       vmin=0, vmax=1, interpolation='nearest')
        ax.set_yticks(range(len(pivot_norm.index)))
        ax.set_yticklabels(pivot_norm.index)
        ax.set_xlabel('Time Window (30s bins)')
        ax.set_title(f'{style["label"]}')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Load (normalized)')
        has_any_data = True
    
    if not has_any_data:
        plt.close()
        print(f"  [8] SKIPPED (NAT flows chua co du lieu port mapping)")
        return
    
    plt.suptitle(f'Server Load Heatmap — {scenario_name}', fontsize=14, fontweight='bold',
                 color=COLORS['text'])
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f'08_heatmap_{scenario_name}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [8] {os.path.basename(path)}")


# ================================================================
# BIEU DO 9: Inference Overhead
# ================================================================
def plot_inference_overhead(algo_data, scenario_name):
    ai_dir = algo_data.get('AI')
    
    # Thu doc tu file log truoc
    df = None
    if ai_dir:
        df = load_inference_log(os.path.join(ai_dir, 'inference_log.csv'))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes
    
    if df is not None and not df.empty and 'inference_ms' in df.columns:
        latencies = df['inference_ms'].dropna()
        source_label = 'Measured (Real)'
    else:
        # Neu khong co log: benchmark bang cach do truc tiep
        import time
        try:
            import torch
            sys.path.insert(0, os.path.join(BASE_DIR))
            from tft_dqn_net import TFT_DQN_Model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = TFT_DQN_Model(input_size=2, seq_len=5, hidden_size=64, num_actions=3).to(device)
            
            checkpoints_dir = os.path.join(BASE_DIR, 'checkpoints')
            pth_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')] if os.path.exists(checkpoints_dir) else []
            if pth_files:
                model.load_state_dict(torch.load(os.path.join(checkpoints_dir, pth_files[0]),
                                                  map_location=device, weights_only=True))
            model.eval()
            
            times = []
            dummy = torch.randn(1, 5, 2).to(device)
            for _ in range(200):
                t = time.perf_counter()
                with torch.no_grad():
                    model(dummy)
                times.append((time.perf_counter() - t) * 1000)
            latencies = pd.Series(times)
            source_label = 'Benchmark (200 runs)'
        except Exception as e:
            print(f"  [9] SKIPPED (khong co inference_log va khong benchmark duoc: {e})")
            plt.close()
            return
    
    mean_ms = latencies.mean()
    p99_ms = latencies.quantile(0.99)
    
    ax1.hist(latencies, bins=30, color='#10B981', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax1.axvline(mean_ms, color='#F59E0B', linewidth=2, linestyle='--', label=f'Mean: {mean_ms:.2f}ms')
    ax1.axvline(p99_ms, color='#EF4444', linewidth=2, linestyle=':', label=f'P99: {p99_ms:.2f}ms')
    ax1.set_xlabel('Inference Time (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'AI Inference Latency — {source_label}')
    ax1.legend(framealpha=0.8)
    ax1.grid(True, linestyle='--')
    
    ax2.plot(range(len(latencies)), latencies.values, color='#10B981', alpha=0.6, linewidth=0.8)
    ax2.axhline(mean_ms, color='#F59E0B', linewidth=1.5, linestyle='--', alpha=0.8, label=f'Mean {mean_ms:.2f}ms')
    ax2.axhline(100, color='#EF4444', linewidth=1, linestyle=':', alpha=0.6, label='100ms threshold')
    ax2.legend(framealpha=0.8)
    ax2.set_xlabel('Inference Cycle')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency Over Cycles')
    ax2.grid(True, linestyle='--')
    
    plt.suptitle(f'Inference Overhead Analysis — {scenario_name}', fontsize=14, fontweight='bold',
                 color=COLORS['text'])
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f'09_inference_{scenario_name}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [9] {os.path.basename(path)} (source: {source_label})")


# ================================================================
# BIEU DO 10: Latency Over Time (Approximated from throughput)
# ================================================================
def plot_latency_comparison(algo_data, scenario_name):
    """
    Xap xi Latency tu nghich dao throughput.
    Neu co Artillery JSON report thi se doc tu do (chua implement).
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for algo, dirpath in algo_data.items():
        style = ALGO_STYLES.get(algo, ALGO_STYLES['COLLECT'])
        df = load_flow_stats(os.path.join(dirpath, 'flow_stats.csv'))
        if df is None:
            continue
        
        agg = df.groupby('timestamp').agg({
            'byte_rate': 'sum',
            'packet_rate': 'sum'
        }).reset_index()
        
        # Xap xi response time: byte per packet (lon = packet nang = cham)
        agg['avg_packet_size'] = agg['byte_rate'] / agg['packet_rate'].replace(0, 1)
        
        t0 = agg['timestamp'].min()
        agg['elapsed_sec'] = (agg['timestamp'] - t0).dt.total_seconds()
        
        # Smoothing
        window = min(5, len(agg))
        if window > 1:
            agg['smooth'] = agg['avg_packet_size'].rolling(window, center=True).mean()
        else:
            agg['smooth'] = agg['avg_packet_size']
        
        ax.plot(agg['elapsed_sec'], agg['smooth'],
                color=style['color'], linewidth=1.8, label=style['label'],
                linestyle=style['linestyle'])
    
    ax.set_xlabel('Thoi gian (giay)')
    ax.set_ylabel('Avg Packet Size (bytes) ~ Proxy Latency')
    ax.set_title(f'Latency Proxy Comparison — {scenario_name}')
    ax.legend(framealpha=0.8)
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f'10_latency_{scenario_name}.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  [10] {os.path.basename(path)}")


# ================================================================
# BIEU DO 11: Server Load vs. Server Capacity (Bar Chart)
# ================================================================
def plot_heterogeneous_load(algo_data, scenario_name):
    """Bieu do cot so sanh Luong Traffic (Bytes) voi Nang Luc cua tung Server."""
    servers = ['h5 (Weak)', 'h7 (Medium)', 'h8 (Strong)']
    capacities = np.array([10, 50, 100]) # Mbps
    
    # Lay traffic thuc te cua RR va AI tu port_stats
    load_mb = {'RR': [0, 0, 0], 'AI': [0, 0, 0]}
    
    for algo in ['RR', 'AI']:
        if algo not in algo_data:
            continue
            
        df = load_port_stats(os.path.join(algo_data[algo], 'port_stats.csv'))
        if df is None:
            continue
            
        latest = df.groupby(['datapath_id', 'port_no']).last().reset_index()
        # Edge switch cho backend la s8 (dpid=8)
        s8_stats = latest[latest['datapath_id'].astype(str) == '8']
        
        # In_ports cua backend: 2(h5), 4(h7), 5(h8)
        h5_bytes = s8_stats[s8_stats['port_no'].astype(str) == '2']['tx_bytes'].sum()
        h7_bytes = s8_stats[s8_stats['port_no'].astype(str) == '4']['tx_bytes'].sum()
        h8_bytes = s8_stats[s8_stats['port_no'].astype(str) == '5']['tx_bytes'].sum()
        
        load_mb[algo] = [h5_bytes / 1e6, h7_bytes / 1e6, h8_bytes / 1e6]

    if sum(load_mb['RR']) == 0 and sum(load_mb['AI']) == 0:
        print(f"  [11] SKIPPED (No port_stats for RR/AI)")
        return

    x = np.arange(len(servers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, load_mb['RR'], width, label='Round Robin (RR)', color=ALGO_STYLES['RR']['color'], alpha=0.85)
    rects2 = ax.bar(x + width/2, load_mb['AI'], width, label='TFT-DQN (AI)', color=ALGO_STYLES['AI']['color'], alpha=0.9)

    # Tinh duong Ideal (Tuong xung voi Capacity) tieu chuan hoa theo tong load cua AI
    total_ai = sum(load_mb['AI'])
    if total_ai > 0:
        ideal_loads = capacities / capacities.sum() * total_ai
        ax.plot(x, ideal_loads, '--', marker='o', color='#3B82F6', linewidth=2.5,  markersize=8, label='Ideal Capacity Ratio')

    ax.set_ylabel('Total Traffic Processed (MB)')
    ax.set_title(f'Load Distribution vs. Server Capacity — {scenario_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(servers)
    ax.legend(framealpha=0.9)
    ax.grid(True, linestyle='--', axis='y')
    
    # Them value text
    def autolabel(rects, is_ai=False):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}MB',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, rotation=0,
                    color=COLORS['text'])
                    
    autolabel(rects1)
    autolabel(rects2, True)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f'11_hetero_load_{scenario_name}.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  [11] {os.path.basename(path)}")


# ================================================================
# BIEU DO 12: Per-Server Latency Proxy (Multi-line Chart)
# ================================================================
def plot_latency_per_server(algo_data, scenario_name):
    """Ve bieu do do tre rieng biet de thay "Noi dau" cua RR voi server yeu."""
    backend_ports = {5: 'h5 (Weak: 10Mbps)', 7: 'h7 (Medium: 50Mbps)', 8: 'h8 (Strong: 100Mbps)'}
    
    for algo in ['RR', 'AI']:
        if algo not in algo_data:
            continue
            
        dirpath = algo_data[algo]
        df = load_flow_stats(os.path.join(dirpath, 'flow_stats.csv'))
        if df is None:
            continue
            
        nat = df[df['priority'] == 100].copy()
        if nat.empty:
            continue
            
        # Map in_port -> name
        nat['server'] = nat['in_port'].map({5: 'h5', 7: 'h7', 8: 'h8'})
        nat = nat.dropna(subset=['server'])
        if nat.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {'h5': '#EF4444', 'h7': '#F59E0B', 'h8': '#10B981'}
        
        for srv in ['h5', 'h7', 'h8']:
            srv_df = nat[nat['server'] == srv]
            if srv_df.empty:
                continue
                
            agg = srv_df.groupby('timestamp').agg({
                'byte_rate': 'sum',
                'packet_rate': 'sum'
            }).reset_index()
            
            # Proxy: byte per packet
            agg['proxy_lat'] = agg['byte_rate'] / agg['packet_rate'].replace(0, 1)
            t0 = agg['timestamp'].min()
            agg['elapsed'] = (agg['timestamp'] - t0).dt.total_seconds()
            
            # Smooth
            agg['smooth'] = agg['proxy_lat'].rolling(min(5, len(agg)), center=True).mean()
            
            ax.plot(agg['elapsed'], agg['smooth'], label=f"{srv} (Capacity: {10 if srv=='h5' else 50 if srv=='h7' else 100}Mbps)", 
                    color=colors[srv], linewidth=2.0)

        ax.set_xlabel('Time elapsed (seconds)')
        ax.set_ylabel('Avg Packet Size (bytes) ~ Service Wait Time/Delay')
        ax.set_title(f'[{algo}] Per-Server Congestion / Delay Proxy — {scenario_name}')
        ax.legend(framealpha=0.9)
        ax.grid(True, linestyle='--')
        
        plt.tight_layout()
        path = os.path.join(CHARTS_DIR, f'12_latency_{algo}_{scenario_name}.png')
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"  [12] {os.path.basename(path)}")


# ================================================================
# BIEU DO 13: Link Utilization % 
# ================================================================
def plot_link_utilization(algo_data, scenario_name):
    """Bieu do muc do su dung tai nguyen % the hien su Cong bang (Fairness)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    capacities_mbps = {'h5': 10, 'h7': 50, 'h8': 100}
    colors = {'h5': '#EF4444', 'h7': '#F59E0B', 'h8': '#10B981'}
    
    plotted = False
    
    for ax, algo in zip(axes, ['RR', 'AI']):
        if algo not in algo_data:
            ax.set_title(f'{algo} (No data)')
            continue
            
        df = load_flow_stats(os.path.join(algo_data[algo], 'flow_stats.csv'))
        if df is None:
            ax.set_title(f'{algo} (No data)')
            continue
            
        nat = df[df['priority'] == 100].copy()
        if nat.empty:
            ax.set_title(f'{algo} (No NAT flows)')
            continue
            
        nat['server'] = nat['in_port'].map({5: 'h5', 7: 'h7', 8: 'h8'})
        nat = nat.dropna(subset=['server'])
        
        if nat.empty:
            ax.set_title(f'{algo} (No NAT flows mapped)')
            continue
            
        for srv in ['h5', 'h7', 'h8']:
            srv_df = nat[nat['server'] == srv]
            if srv_df.empty:
                continue
                
            agg = srv_df.groupby('timestamp').agg({'byte_rate': 'sum'}).reset_index()
            # Tinh % hieu suat = (Mbps thuc_te / Mbps gioi_han) * 100
            limit_mbps = capacities_mbps[srv]
            agg['util_pct'] = ((agg['byte_rate'] * 8 / 1e6) / limit_mbps) * 100
            
            t0 = agg['timestamp'].min()
            agg['elapsed'] = (agg['timestamp'] - t0).dt.total_seconds()
            
            # Smooth area chart
            smooth = agg['util_pct'].rolling(min(5, len(agg)), center=True).mean()
            
            ax.plot(agg['elapsed'], smooth, label=f"{srv} (limit: {limit_mbps}Mbps)", 
                    color=colors[srv], linewidth=2.0)
            ax.fill_between(agg['elapsed'], smooth, alpha=0.1, color=colors[srv])
            
        ax.set_title(f'{ALGO_STYLES[algo]["label"]} — % Utilization')
        ax.set_xlabel('Time (s)')
        if algo == 'RR':
            ax.set_ylabel('Link Utilization (%)')
            
        ax.axhline(100, color='#9CA3AF', linestyle='--', linewidth=1.5, label='100% (Congestion Line)')
        ax.legend(framealpha=0.9, loc='upper right')
        ax.grid(True, linestyle='--')
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.suptitle(f'Fairness in Effort: Link Utilization Comparison — {scenario_name}', 
                 fontsize=14, fontweight='bold', color=COLORS['text'])
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f'13_link_utilization_{scenario_name}.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  [13] {os.path.basename(path)}")


# ================================================================
# DASHBOARD TONG HOP SO SANH
# ================================================================
def plot_comparison_dashboard(algo_data, scenario_name):
    """Ve 1 dashboard tong hop 4 bieu do so sanh."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'NCKH SDN: Algorithm Comparison Dashboard — {scenario_name}', 
                 fontsize=16, fontweight='bold', color=COLORS['text'], y=0.98)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # === 1. Throughput (top-left) ===
    ax1 = fig.add_subplot(gs[0, 0])
    for algo, dirpath in algo_data.items():
        style = ALGO_STYLES.get(algo, ALGO_STYLES['COLLECT'])
        df = load_flow_stats(os.path.join(dirpath, 'flow_stats.csv'))
        if df is None:
            continue
        agg = df.groupby('timestamp').agg({'byte_rate': 'sum'}).reset_index()
        agg['throughput_mbps'] = (agg['byte_rate'] * 8) / 1e6
        t0 = agg['timestamp'].min()
        agg['elapsed'] = (agg['timestamp'] - t0).dt.total_seconds()
        ax1.plot(agg['elapsed'], agg['throughput_mbps'], color=style['color'],
                 linewidth=1.5, label=style['label'], linestyle=style['linestyle'])
    ax1.set_title('Throughput Stability')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mbps')
    ax1.legend(fontsize=7)
    ax1.grid(True, linestyle='--')
    
    # === 2. Packet Loss (top-right) ===
    ax2 = fig.add_subplot(gs[0, 1])
    algo_names = []
    loss_vals = []
    bar_colors = []
    for algo, dirpath in algo_data.items():
        style = ALGO_STYLES.get(algo, ALGO_STYLES['COLLECT'])
        df = load_port_stats(os.path.join(dirpath, 'port_stats.csv'))
        if df is None:
            continue
        latest = df.groupby('datapath_id').last().reset_index()
        total_pkts = latest['rx_packets'].sum() + latest['tx_packets'].sum()
        total_loss = latest['rx_dropped'].sum() + latest['tx_dropped'].sum() + \
                     latest['rx_errors'].sum() + latest['tx_errors'].sum()
        rate = (total_loss / total_pkts * 100) if total_pkts > 0 else 0
        algo_names.append(algo)
        loss_vals.append(rate)
        bar_colors.append(style['color'])
    if algo_names:
        ax2.bar(algo_names, loss_vals, color=bar_colors, alpha=0.85, width=0.5)
        for i, v in enumerate(loss_vals):
            ax2.text(i, v + 0.05, f'{v:.2f}%', ha='center', fontsize=9, fontweight='bold')
    ax2.set_title('Packet Loss Rate')
    ax2.set_ylabel('%')
    ax2.grid(True, linestyle='--', axis='y')
    
    # === 3. Server Load Heatmap (bottom-left) — Chi ve cho thuat toan dau tien tim thay ===
    ax3 = fig.add_subplot(gs[1, 0])
    backend_macs = {'00:00:00:00:00:05': 'h5', '00:00:00:00:00:07': 'h7', '00:00:00:00:00:08': 'h8'}
    first_algo = list(algo_data.keys())[0] if algo_data else None
    if first_algo:
        df = load_flow_stats(os.path.join(algo_data[first_algo], 'flow_stats.csv'))
        if df is not None:
            nat = df[df['priority'] == 100].copy()
            if not nat.empty:
                nat['server'] = nat['eth_dst'].map(backend_macs)
                nat = nat.dropna(subset=['server'])
                if not nat.empty:
                    t0 = nat['timestamp'].min()
                    nat['time_bin'] = ((nat['timestamp'] - t0).dt.total_seconds() // 30).astype(int)
                    pivot = nat.groupby(['server', 'time_bin'])['byte_count'].sum().unstack(fill_value=0)
                    mx = pivot.values.max()
                    if mx > 0:
                        pivot = pivot / mx
                    ax3.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
                    ax3.set_yticks(range(len(pivot.index)))
                    ax3.set_yticklabels(pivot.index)
    ax3.set_title(f'Server Load Heatmap ({first_algo})')
    ax3.set_xlabel('Time Window')
    
    # === 4. Inference Overhead (bottom-right) ===
    ax4 = fig.add_subplot(gs[1, 1])
    ai_dir = algo_data.get('AI')
    if ai_dir:
        inf_df = load_inference_log(os.path.join(ai_dir, 'inference_log.csv'))
        if inf_df is not None and not inf_df.empty:
            ax4.hist(inf_df['inference_ms'], bins=25, color='#10B981', alpha=0.8, edgecolor='white')
            mean_ms = inf_df['inference_ms'].mean()
            ax4.axvline(mean_ms, color='#F59E0B', linewidth=2, linestyle='--')
            ax4.set_title(f'AI Inference (Mean: {mean_ms:.1f}ms)')
        else:
            ax4.set_title('Inference (No data)')
    else:
        ax4.set_title('Inference (AI not tested yet)')
    ax4.set_xlabel('ms')
    ax4.grid(True, linestyle='--')
    
    path = os.path.join(CHARTS_DIR, f'00_comparison_dashboard_{scenario_name}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  >> {os.path.basename(path)} (DASHBOARD TONG HOP)")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate comparison charts for SDN evaluation')
    parser.add_argument('--scenario', type=str, required=True, help='Scenario name (e.g. flash_crowd)')
    parser.add_argument('--theme', type=str, default='dark', choices=['dark', 'light'],
                        help='Color theme: dark (default) hoac light (cho bao cao in)')
    args = parser.parse_args()
    
    scenario_name = args.scenario
    algo_data = find_result_dirs(scenario_name)
    
    if not algo_data:
        print(f"Khong tim thay ket qua nao cho scenario '{scenario_name}' trong {RESULTS_DIR}/")
        print(f"Hay chay evaluate_sdn.sh truoc.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  NCKH SDN - Comparison Chart Generator")
    print(f"  Scenario: {scenario_name} | Theme: {args.theme}")
    print(f"  Found: {', '.join(algo_data.keys())}")
    print(f"{'='*60}\n")
    
    # Tao thu muc charts rieng theo theme neu khac dark
    global CHARTS_DIR
    if args.theme == 'light':
        CHARTS_DIR = os.path.join(RESULTS_DIR, 'charts_light')
    os.makedirs(CHARTS_DIR, exist_ok=True)
    
    setup_theme(args.theme)
    
    plot_throughput_stability(algo_data, scenario_name)
    plot_packet_loss(algo_data, scenario_name)
    plot_server_heatmap(algo_data, scenario_name)
    plot_inference_overhead(algo_data, scenario_name)
    plot_latency_comparison(algo_data, scenario_name)
    
    # Render cac bieu do moi (Heterogeneous)
    plot_heterogeneous_load(algo_data, scenario_name)
    plot_latency_per_server(algo_data, scenario_name)
    plot_link_utilization(algo_data, scenario_name)
    
    plot_comparison_dashboard(algo_data, scenario_name)
    
    print(f"\nHoan tat! Tat ca bieu do tai: {CHARTS_DIR}/")


if __name__ == '__main__':
    main()
