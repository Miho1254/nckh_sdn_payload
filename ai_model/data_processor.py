import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLOW_CSV = os.path.join(BASE_DIR, '../stats/flow_stats.csv')
PORT_CSV = os.path.join(BASE_DIR, '../stats/port_stats.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_data')

def load_flow_features(csv_path):
    import csv
    print("1. Đang tải dữ liệu Flow Stats từ CSV...")
    valid_data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return pd.DataFrame()
            
        # Map column indices dynamically
        col_map = {name.strip(): idx for idx, name in enumerate(header)}
        
        idx_packets = col_map.get('packet_count', -1)
        idx_bytes = col_map.get('byte_count', -1)
        idx_sec = col_map.get('duration_sec', -1)
        idx_nsec = col_map.get('duration_nsec', -1)
        idx_label = col_map.get('label', -1)
        idx_scenario = col_map.get('scenario', -1)
        idx_ts = col_map.get('timestamp', 0)

        for row in reader:
            if len(row) <= max(idx_packets, idx_bytes, idx_sec, idx_nsec, idx_label, idx_ts): 
                continue
                
            label = row[idx_label].strip() if idx_label != -1 else 'UNKNOWN'
            if label not in ['NORMAL', 'HIGH']:
                continue

            try:
                scenario_val = row[idx_scenario].strip() if idx_scenario != -1 and len(row) > idx_scenario else 'UNKNOWN'
                valid_data.append({
                    'timestamp': row[idx_ts],
                    'packet_count': int(row[idx_packets]),
                    'byte_count': int(row[idx_bytes]),
                    'duration_sec': int(row[idx_sec]),
                    'duration_nsec': int(row[idx_nsec]),
                    'label': label,
                    'scenario': scenario_val
                })
            except:
                continue
                
    df = pd.DataFrame(valid_data)
    
    if df.empty:
        print("❌ LỖI: Dữ liệu tải lên hoàn toàn rỗng. File flow_stats.csv không chứa data hợp lệ!")
        print("   Hãy đảm bảo bạn đã chạy Mininet (non_stop_experiment.sh) và sinh ra traffic trước khi train AI.")
        exit(1)
        
    df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce')
    df['duration_nsec'] = pd.to_numeric(df['duration_nsec'], errors='coerce')
    df['packet_count'] = pd.to_numeric(df['packet_count'], errors='coerce')
    df['byte_count'] = pd.to_numeric(df['byte_count'], errors='coerce')
    df = df.dropna(subset=['duration_sec', 'packet_count'])
    
    df['duration'] = df['duration_sec'] + df['duration_nsec'] / 1e9
    df['duration'] = df['duration'].apply(lambda x: x if x > 0 else 0.001)
    
    df['byte_rate'] = df['byte_count'] / df['duration']
    df['packet_rate'] = df['packet_count'] / df['duration']
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].dt.round('1s') 
    df = df.sort_values('timestamp')
    
    return df

# Dynamically import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BACKENDS

def load_port_server_loads(csv_path):
    """Đọc port_stats.csv và trích xuất tải per-server (tự động theo dpid/port trong config)."""
    import csv
    print("1b. Đang tải dữ liệu Port Stats (per-server load)...")
    
    if not os.path.exists(csv_path):
        print("   -> port_stats.csv KHÔNG TÌM THẤY. Sẽ dùng giá trị mặc định.")
        return None
    
    records = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 13: continue
            try:
                dpid = int(row[1])
                port = int(row[2])
                tx_bytes = int(row[8])
                ts = row[0]
                records.append({
                    'timestamp': ts, 
                    'dpid': dpid, 
                    'port': port, 
                    'tx_bytes': tx_bytes
                })
            except:
                continue
    
    if not records:
        return None
        
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].dt.round('1s')
    
    # Trích xuất tải cho từng backend node từ config
    all_loads = None
    
    for node in BACKENDS:
        name = node['name']
        dpid = node['dpid']
        port = node['port']
        
        # Filter data cho node này
        node_df = df[(df['dpid'] == dpid) & (df['port'] == port)]
        node_df = node_df[['timestamp', 'tx_bytes']].rename(columns={'tx_bytes': f'load_{name}'})
        
        if all_loads is None:
            all_loads = node_df
        else:
            all_loads = all_loads.merge(node_df, on='timestamp', how='outer')
            
    if all_loads is not None:
        all_loads = all_loads.fillna(0)
        all_loads = all_loads.sort_values('timestamp')
        
        # Tính delta (difference giữa các bước thời gian kế tiếp)
        load_cols = [f'load_{node["name"]}' for node in BACKENDS]
        for col in load_cols:
            all_loads[col] = all_loads[col].diff().fillna(0).clip(lower=0)
    
    print(f"   -> Đã trích xuất {len(all_loads) if all_loads is not None else 0} mẫu tải per-server từ port_stats.")
    return all_loads

def aggregate_and_visualize(df):
    print("2. Đang tổng hợp dữ liệu toàn mạng và vẽ biểu đồ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if 'scenario' not in df.columns:
        df['scenario'] = 'UNKNOWN'
        
    agg_df = df.groupby('timestamp').agg({
        'byte_rate': 'sum',
        'packet_rate': 'sum',
        'label': 'last',
        'scenario': 'last'
    }).reset_index()
    
    plt.figure(figsize=(12, 5))
    times = agg_df['timestamp']
    throughput_mbps = (agg_df['byte_rate'] * 8) / 1e6
    
    plt.plot(times, throughput_mbps, label='Network Throughput (Mbps)', color='#1f77b4', linewidth=2)
    
    high_traffic = agg_df[agg_df['label'] == 'HIGH']
    if not high_traffic.empty:
        high_times = high_traffic['timestamp']
        high_mbps = (high_traffic['byte_rate'] * 8) / 1e6
        plt.scatter(high_times, high_mbps, color='#d62728', label='HIGH Traffic (Burst)', zorder=5, s=50)
        plt.axvspan(high_times.min(), high_times.max(), color='red', alpha=0.1)
        
    plt.title('Fat-Tree SDN: Network Throughput Analysis')
    plt.xlabel('Timeline')
    plt.ylabel('Throughput (Mbps)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, 'throughput_visualization.png')
    plt.savefig(plot_path, dpi=300)
    print(f"   -> Đã lưu biểu đồ tại: {plot_path}")
    
    return agg_df

def create_time_series_windows(agg_df, server_loads_df, sequence_length=5):
    """
    Tạo cửa sổ trượt cho TFT.
    Features: [byte_rate, packet_rate, load_hX...]
    """
    print(f"3. Cắt Windowing dữ liệu (Sequence Length = {sequence_length})...")
    
    load_cols = [f'load_{node["name"]}' for node in BACKENDS]
    features = ['byte_rate', 'packet_rate'] + load_cols
    
    # Merge per-server loads vào aggregated data
    if server_loads_df is not None and len(server_loads_df) > 0:
        agg_df = agg_df.merge(server_loads_df, on='timestamp', how='left')
        agg_df[load_cols] = agg_df[load_cols].fillna(0)
    else:
        # Fallback: tạo cột load giả = 0
        for col in load_cols:
            agg_df[col] = 0.0
        print(f"   -> FALLBACK: {len(load_cols)} server loads internalize to 0 (no port_stats)")
    
    print(f"   -> State Vector: {len(features)} features {features}")
    
    # Chuẩn hóa Data theo Hằng số vật lý (Global Scaling) thay vì MinMaxScaler
    from config import SCALING_BYTE_RATE, SCALING_PKT_RATE, SCALING_LOAD
    
    agg_df['byte_rate'] = (agg_df['byte_rate'] / SCALING_BYTE_RATE).clip(0, 1)
    agg_df['packet_rate'] = (agg_df['packet_rate'] / SCALING_PKT_RATE).clip(0, 1)
    
    for col in load_cols:
        agg_df[col] = (agg_df[col] / SCALING_LOAD).clip(0, 1)
    
    # Mã hóa nhãn: NORMAL=0, HIGH=1
    agg_df['label_encoded'] = (agg_df['label'] == 'HIGH').astype(int)
    
    X, y = [], []
    data_scaled = agg_df[features].values
    labels = agg_df['label_encoded'].values
    
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i : i + sequence_length])
        y.append(labels[i + sequence_length])
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"   -> Tạo thành công {len(X)} sequences.")
    print(f"   -> Shape của tập dữ liệu vào (Input X): {X.shape}")
    print(f"   -> Shape của nhãn (Label y): {y.shape}")
    
    np.save(os.path.join(OUTPUT_DIR, 'X_sequences.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_labels.npy'), y)
    print("   -> Đã lưu các Tensor numpy (npy).")
    
    return X, y


def create_time_series_windows_v3(agg_df, server_loads_df, sequence_length=5):
    """
    V3 Feature Pipeline cho TFT-CQL Actor-Critic.
    4 nhóm feature: Global traffic, Per-server raw, Normalized risk, Policy/context.
    """
    from config import (SCALING_BYTE_RATE, SCALING_PKT_RATE, SCALING_LOAD,
                        CAPACITIES, EWMA_ALPHA, ROLLING_WINDOW,
                        REGIME_HIGH_MIN, CAPACITY_RATIOS)
    import json

    print(f"3v3. Feature Engineering V3 (Sequence Length = {sequence_length})...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    num_servers = len(BACKENDS)
    load_cols = [f'load_{node["name"]}' for node in BACKENDS]
    server_names = [node["name"] for node in BACKENDS]

    # ── Merge server loads ──
    if server_loads_df is not None and len(server_loads_df) > 0:
        agg_df = agg_df.merge(server_loads_df, on='timestamp', how='left')
        agg_df[load_cols] = agg_df[load_cols].fillna(0)
    else:
        for col in load_cols:
            agg_df[col] = 0.0

    # ── Normalize base features ──
    agg_df['byte_rate_norm'] = (agg_df['byte_rate'] / SCALING_BYTE_RATE).clip(0, 1)
    agg_df['packet_rate_norm'] = (agg_df['packet_rate'] / SCALING_PKT_RATE).clip(0, 1)
    for col in load_cols:
        agg_df[f'{col}_norm'] = (agg_df[col] / SCALING_LOAD).clip(0, 1)

    # ── NHÓM A: Global traffic (7 features) ──
    agg_df['delta_byte_rate'] = agg_df['byte_rate_norm'].diff().fillna(0)
    agg_df['delta_packet_rate'] = agg_df['packet_rate_norm'].diff().fillna(0)
    agg_df['ewma_byte_rate'] = agg_df['byte_rate_norm'].ewm(alpha=EWMA_ALPHA, adjust=False).mean()
    agg_df['traffic_volatility'] = agg_df['byte_rate_norm'].rolling(window=max(3, ROLLING_WINDOW), min_periods=1).std().fillna(0)
    agg_df['regime_flag'] = (agg_df['byte_rate_norm'] >= REGIME_HIGH_MIN).astype(float)

    group_a = ['byte_rate_norm', 'packet_rate_norm', 'delta_byte_rate',
               'delta_packet_rate', 'ewma_byte_rate', 'traffic_volatility', 'regime_flag']

    # ── NHÓM B: Per-server raw load (3 per server) ──
    group_b = []
    for name in server_names:
        col = f'load_{name}_norm'
        prev_col = f'prev_load_{name}'
        delta_col = f'delta_load_{name}'
        agg_df[prev_col] = agg_df[col].shift(1).fillna(0)
        agg_df[delta_col] = agg_df[col] - agg_df[prev_col]
        group_b.extend([col, prev_col, delta_col])

    # ── NHÓM C: Per-server normalized risk (7 per server) ──
    group_c = []
    for i, name in enumerate(server_names):
        load_col = f'load_{name}_norm'
        cap = CAPACITIES[i]

        util_col = f'util_{name}'
        agg_df[util_col] = (agg_df[load_col] * SCALING_LOAD) / (cap * 1e6) if cap > 0 else 0.0
        agg_df[util_col] = agg_df[util_col].clip(0, 1)

        util_delta_col = f'util_delta_{name}'
        agg_df[util_delta_col] = agg_df[util_col].diff().fillna(0)

        headroom_col = f'headroom_{name}'
        agg_df[headroom_col] = (1.0 - agg_df[util_col]).clip(0, 1)

        headroom_ratio_col = f'headroom_ratio_{name}'
        agg_df[headroom_ratio_col] = agg_df[headroom_col] * (cap / float(np.max(CAPACITIES)))

        congestion_col = f'congestion_proxy_{name}'
        agg_df[congestion_col] = (agg_df[util_col] ** 2).clip(0, 1)

        roll_mean_col = f'roll_mean_util_{name}'
        agg_df[roll_mean_col] = agg_df[util_col].rolling(window=ROLLING_WINDOW, min_periods=1).mean()

        roll_max_col = f'roll_max_util_{name}'
        agg_df[roll_max_col] = agg_df[util_col].rolling(window=ROLLING_WINDOW, min_periods=1).max()

        group_c.extend([util_col, util_delta_col, headroom_col, headroom_ratio_col,
                        congestion_col, roll_mean_col, roll_max_col])

    # ── NHÓM D: Policy/context features (num_servers + 2) ──
    group_d = []
    for name in server_names:
        assign_col = f'assign_ratio_{name}'
        agg_df[assign_col] = CAPACITY_RATIOS[server_names.index(name)]
        group_d.append(assign_col)

    agg_df['action_churn'] = 0.0
    group_d.append('action_churn')

    cap_vec_cols = []
    for i, name in enumerate(server_names):
        cap_col = f'rel_capacity_{name}'
        agg_df[cap_col] = CAPACITIES[i] / float(np.max(CAPACITIES))
        cap_vec_cols.append(cap_col)
    group_d.extend(cap_vec_cols)

    # ── Build features list ──
    all_features = group_a + group_b + group_c + group_d
    print(f"   -> V3 State Vector: {len(all_features)} features")

    # ── Label encoding ──
    agg_df['label_encoded'] = (agg_df['label'] == 'HIGH').astype(int) if 'label' in agg_df.columns else 0

    # ── Window slicing ──
    data_matrix = agg_df[all_features].values.astype(np.float32)
    labels = agg_df['label_encoded'].values
    scenarios = agg_df['scenario'].values if 'scenario' in agg_df.columns else np.array(['UNKNOWN'] * len(agg_df))

    X, y, scen = [], [], []
    for i in range(len(data_matrix) - sequence_length):
        X.append(data_matrix[i: i + sequence_length])
        y.append(labels[i + sequence_length])
        scen.append(scenarios[i + sequence_length])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    scen = np.array(scen, dtype=str)

    print(f"   -> Tạo thành công {len(X)} sequences V3.")
    print(f"   -> Shape X_v3: {X.shape}")
    print(f"   -> Shape y_v3: {y.shape}")

    # ── Save ──
    np.save(os.path.join(OUTPUT_DIR, 'X_v3.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_v3.npy'), y)
    np.save(os.path.join(OUTPUT_DIR, 'scenarios_v3.npy'), scen)

    # ── Feature metadata ──
    metadata = {
        "version": "v3",
        "num_features": len(all_features),
        "sequence_length": sequence_length,
        "feature_names": all_features,
        "feature_groups": {
            "A_global_traffic": group_a,
            "B_per_server_raw": group_b,
            "C_per_server_risk": group_c,
            "D_policy_context": group_d,
        },
        "num_servers": num_servers,
        "server_names": server_names,
        "capacities": CAPACITIES.tolist(),
    }
    meta_path = os.path.join(OUTPUT_DIR, 'feature_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   -> Feature metadata đã lưu: {meta_path}")

    return X, y, metadata

if __name__ == '__main__':
    print(f"=== TỔNG HỢP & TIỀN XỬ LÝ DỮ LIỆU SDN ({2 + len(BACKENDS)}-FEATURE) ===")
    
    if not os.path.exists(FLOW_CSV):
        print(f"❌ Không tìm thấy file data: {FLOW_CSV}")
        print("Vui lòng chạy Mininet & Artillery để thu thập dữ liệu trước!")
        exit(1)
        
    df_raw = load_flow_features(FLOW_CSV)
    server_loads = load_port_server_loads(PORT_CSV)
    df_agg = aggregate_and_visualize(df_raw)
    
    # V2 (legacy) — giữ backward compatibility
    X, y = create_time_series_windows(df_agg.copy(), server_loads, sequence_length=5)
    print(f"✅ Hoàn tất Pipeline V2 ({2 + len(BACKENDS)} features)!")
    
    # V3 (mới) — 42 features cho TFT-CQL
    X_v3, y_v3, meta = create_time_series_windows_v3(df_agg.copy(), server_loads, sequence_length=5)
    print(f"✅ Hoàn tất Pipeline V3 ({meta['num_features']} features)!")
