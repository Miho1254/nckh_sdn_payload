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
        for row in reader:
            if len(row) < 11: continue
            
            if len(row) >= 14:
                idx_bytes = 10
                idx_sec = 11
                idx_nsec = 12
                idx_label = 13
                idx_packets = 9
            else:
                idx_bytes = 8
                idx_sec = 9
                idx_nsec = 10
                idx_label = 11
                idx_packets = 7

            try:
                if row[idx_label].strip() not in ['NORMAL', 'HIGH']: continue
                
                valid_data.append({
                    'timestamp': row[0],
                    'packet_count': int(row[idx_packets]),
                    'byte_count': int(row[idx_bytes]),
                    'duration_sec': int(row[idx_sec]),
                    'duration_nsec': int(row[idx_nsec]),
                    'label': row[idx_label].strip()
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

def load_port_server_loads(csv_path):
    """Đọc port_stats.csv và trích xuất tải per-server (tx_bytes trên s8 ports)."""
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
    
    # s8 ports: 2=h5, 4=h7, 5=h8
    s8 = df[df['dpid'] == 8]
    
    h5_load = s8[s8['port'] == 2][['timestamp', 'tx_bytes']].rename(columns={'tx_bytes': 'load_h5'})
    h7_load = s8[s8['port'] == 4][['timestamp', 'tx_bytes']].rename(columns={'tx_bytes': 'load_h7'})
    h8_load = s8[s8['port'] == 5][['timestamp', 'tx_bytes']].rename(columns={'tx_bytes': 'load_h8'})
    
    # Merge trên timestamp
    loads = h5_load.merge(h7_load, on='timestamp', how='outer')
    loads = loads.merge(h8_load, on='timestamp', how='outer')
    loads = loads.fillna(0)
    loads = loads.sort_values('timestamp')
    
    # Tính delta (difference giữa các bước thời gian kế tiếp)
    for col in ['load_h5', 'load_h7', 'load_h8']:
        loads[col] = loads[col].diff().fillna(0).clip(lower=0)
    
    print(f"   -> Đã trích xuất {len(loads)} mẫu tải per-server từ port_stats.")
    return loads

def aggregate_and_visualize(df):
    print("2. Đang tổng hợp dữ liệu toàn mạng và vẽ biểu đồ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    agg_df = df.groupby('timestamp').agg({
        'byte_rate': 'sum',
        'packet_rate': 'sum',
        'label': 'last'
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
    Features: [byte_rate, packet_rate, load_h5, load_h7, load_h8]
    """
    print(f"3. Cắt Windowing dữ liệu (Sequence Length = {sequence_length})...")
    
    # Merge per-server loads vào aggregated data
    if server_loads_df is not None and len(server_loads_df) > 0:
        agg_df = agg_df.merge(server_loads_df, on='timestamp', how='left')
        agg_df[['load_h5', 'load_h7', 'load_h8']] = agg_df[['load_h5', 'load_h7', 'load_h8']].fillna(0)
        features = ['byte_rate', 'packet_rate', 'load_h5', 'load_h7', 'load_h8']
        print("   -> State Vector: 5 features [byte_rate, packet_rate, load_h5, load_h7, load_h8]")
    else:
        # Fallback: tạo cột load giả = 0 (sẽ chỉ dùng byte_rate + packet_rate)
        agg_df['load_h5'] = 0.0
        agg_df['load_h7'] = 0.0
        agg_df['load_h8'] = 0.0
        features = ['byte_rate', 'packet_rate', 'load_h5', 'load_h7', 'load_h8']
        print("   -> FALLBACK: load per-server = 0 (không có port_stats)")
    
    # Chuẩn hóa Data (Min-Max Scaling) về [0, 1]
    scaler = MinMaxScaler()
    agg_df[features] = scaler.fit_transform(agg_df[features])
    
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

if __name__ == '__main__':
    print("=== TỔNG HỢP & TIỀN XỬ LÝ DỮ LIỆU SDN (5-FEATURE) ===")
    
    if not os.path.exists(FLOW_CSV):
        print(f"❌ Không tìm thấy file data: {FLOW_CSV}")
        print("Vui lòng chạy Mininet & Artillery để thu thập dữ liệu trước!")
        exit(1)
        
    df_raw = load_flow_features(FLOW_CSV)
    server_loads = load_port_server_loads(PORT_CSV)
    df_agg = aggregate_and_visualize(df_raw)
    X, y = create_time_series_windows(df_agg, server_loads, sequence_length=5)
    
    print("✅ Hoàn tất toàn bộ Pipeline Tiền xử lý (5 features)!")
