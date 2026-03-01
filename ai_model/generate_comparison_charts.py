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

COLORS = {
    'bg': '#0F172A',
    'card_bg': '#1E293B',
    'text': '#E2E8F0',
    'grid': '#334155',
}

def setup_dark_style():
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['card_bg'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
    })

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
    
    valid_rows = []
    columns = ['timestamp','datapath_id','table_id','priority','in_port',
                'eth_src','eth_dst','packet_count','byte_count',
                'duration_sec','duration_nsec','label']
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 12 and row[-1] in ['NORMAL', 'HIGH']:
                valid_rows.append(row)
    
    if not valid_rows:
        return None
    
    df = pd.DataFrame(valid_rows, columns=columns)
    for col in ['duration_sec','duration_nsec','packet_count','byte_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['duration_sec','packet_count'])
    
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
        df = load_port_stats(os.path.join(dirpath, 'port_stats.csv'))
        if df is None:
            continue
        
        # Lay snapshot cuoi cung (cumulative counters)
        latest = df.groupby('datapath_id').last().reset_index()
        total_rx = latest['rx_packets'].sum()
        total_tx = latest['tx_packets'].sum()
        total_dropped = latest['rx_dropped'].sum() + latest['tx_dropped'].sum()
        total_errors = latest['rx_errors'].sum() + latest['tx_errors'].sum()
        
        total_pkts = total_rx + total_tx
        loss_rate = ((total_dropped + total_errors) / total_pkts * 100) if total_pkts > 0 else 0
        
        algos.append(style['label'])
        loss_rates.append(loss_rate)
        colors.append(style['color'])
    
    if not algos:
        return
    
    bars = ax.bar(algos, loss_rates, color=colors, alpha=0.85, width=0.5,
                  edgecolor='white', linewidth=0.5)
    
    for bar, rate in zip(bars, loss_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{rate:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=COLORS['text'])
    
    ax.set_ylabel('Packet Loss Rate (%)')
    ax.set_title(f'Packet Loss Rate Comparison — {scenario_name}')
    ax.grid(True, linestyle='--', axis='y')
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f'07_packet_loss_{scenario_name}.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  [7] {os.path.basename(path)}")


# ================================================================
# BIEU DO 8: Heatmap Server Load
# ================================================================
def plot_server_heatmap(algo_data, scenario_name):
    backend_macs = {
        '00:00:00:00:00:05': 'h5',
        '00:00:00:00:00:07': 'h7',
        '00:00:00:00:00:08': 'h8',
    }
    
    n_algos = len(algo_data)
    if n_algos == 0:
        return
    
    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 5))
    if n_algos == 1:
        axes = [axes]
    
    for ax, (algo, dirpath) in zip(axes, algo_data.items()):
        style = ALGO_STYLES.get(algo, ALGO_STYLES['COLLECT'])
        df = load_flow_stats(os.path.join(dirpath, 'flow_stats.csv'))
        if df is None:
            ax.set_title(f'{style["label"]} (No data)')
            continue
        
        # Loc flow NAT Load Balancer (priority=100)
        nat_flows = df[df['priority'] == 100].copy()
        if nat_flows.empty:
            ax.set_title(f'{style["label"]} (No NAT flows)')
            continue
        
        # Map MAC -> server name
        nat_flows['server'] = nat_flows['eth_dst'].map(backend_macs)
        nat_flows = nat_flows.dropna(subset=['server'])
        
        if nat_flows.empty:
            ax.set_title(f'{style["label"]} (No backend flows)')
            continue
        
        # Gom theo thoi gian (moi 30 giay) va server
        t0 = nat_flows['timestamp'].min()
        nat_flows['time_bin'] = ((nat_flows['timestamp'] - t0).dt.total_seconds() // 30).astype(int)
        
        pivot = nat_flows.groupby(['server', 'time_bin'])['byte_count'].sum().unstack(fill_value=0)
        
        # Chuan hoa ve 0-1 de mau sac dong nhat
        max_val = pivot.values.max()
        if max_val > 0:
            pivot_norm = pivot / max_val
        else:
            pivot_norm = pivot
        
        im = ax.imshow(pivot_norm.values, aspect='auto', cmap='RdYlGn_r',
                       vmin=0, vmax=1, interpolation='nearest')
        ax.set_yticks(range(len(pivot_norm.index)))
        ax.set_yticklabels(pivot_norm.index)
        ax.set_xlabel('Time Window (30s bins)')
        ax.set_title(f'{style["label"]}')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Load (normalized)')
    
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
    if not ai_dir:
        print("  [9] SKIPPED (khong co du lieu AI)")
        return
    
    df = load_inference_log(os.path.join(ai_dir, 'inference_log.csv'))
    if df is None or df.empty:
        print("  [9] SKIPPED (inference_log.csv rong)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(df['inference_ms'], bins=30, color='#10B981', alpha=0.8, edgecolor='white', linewidth=0.5)
    mean_ms = df['inference_ms'].mean()
    p99_ms = df['inference_ms'].quantile(0.99)
    ax1.axvline(mean_ms, color='#F59E0B', linewidth=2, linestyle='--', label=f'Mean: {mean_ms:.2f}ms')
    ax1.axvline(p99_ms, color='#EF4444', linewidth=2, linestyle=':', label=f'P99: {p99_ms:.2f}ms')
    ax1.set_xlabel('Inference Time (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('AI Inference Latency Distribution')
    ax1.legend(framealpha=0.8)
    ax1.grid(True, linestyle='--')
    
    # Timeline
    ax2 = axes[1]
    ax2.plot(range(len(df)), df['inference_ms'], color='#10B981', alpha=0.5, linewidth=0.8)
    ax2.axhline(mean_ms, color='#F59E0B', linewidth=1.5, linestyle='--', alpha=0.8)
    # Danh dau cac lan chuyen backend
    switches = df[df['switched'] == 1]
    if not switches.empty:
        ax2.scatter(switches.index, switches['inference_ms'], color='#EF4444', 
                   s=30, zorder=5, label=f'Backend Switch ({len(switches)}x)')
        ax2.legend(framealpha=0.8)
    ax2.set_xlabel('Inference Cycle')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Inference Latency Over Time')
    ax2.grid(True, linestyle='--')
    
    plt.suptitle(f'Inference Overhead Analysis — {scenario_name}', fontsize=14, fontweight='bold',
                 color=COLORS['text'])
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f'09_inference_{scenario_name}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [9] {os.path.basename(path)}")


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
    args = parser.parse_args()
    
    scenario_name = args.scenario
    algo_data = find_result_dirs(scenario_name)
    
    if not algo_data:
        print(f"Khong tim thay ket qua nao cho scenario '{scenario_name}' trong {RESULTS_DIR}/")
        print(f"Hay chay evaluate_sdn.sh truoc.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  NCKH SDN - Comparison Chart Generator")
    print(f"  Scenario: {scenario_name}")
    print(f"  Found: {', '.join(algo_data.keys())}")
    print(f"{'='*60}\n")
    
    os.makedirs(CHARTS_DIR, exist_ok=True)
    setup_dark_style()
    
    # Ve tung bieu do rieng
    plot_throughput_stability(algo_data, scenario_name)
    plot_packet_loss(algo_data, scenario_name)
    plot_server_heatmap(algo_data, scenario_name)
    plot_inference_overhead(algo_data, scenario_name)
    plot_latency_comparison(algo_data, scenario_name)
    
    # Ve dashboard tong hop
    plot_comparison_dashboard(algo_data, scenario_name)
    
    print(f"\nHoan tat! Tat ca bieu do tai: {CHARTS_DIR}/")


if __name__ == '__main__':
    main()
