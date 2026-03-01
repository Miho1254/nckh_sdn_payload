import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_CSV = os.path.join(BASE_DIR, '../stats/flow_stats.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_data')

def load_and_engineer_features(csv_path):
    import csv
    print("1. Đang tải dữ liệu từ CSV...")
    
    valid_rows = []
    columns = [
        'timestamp', 'datapath_id', 'table_id', 'priority', 'in_port', 
        'eth_src', 'eth_dst', 'packet_count', 'byte_count', 
        'duration_sec', 'duration_nsec', 'label'
    ]
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 12 and row[-1] in ['NORMAL', 'HIGH']:
                valid_rows.append(row)
                
    df = pd.DataFrame(valid_rows, columns=columns)
    
    # Ép kiểu dữ liệu (tránh parse sai cột số thành object do bị dính header rác)
    df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce')
    df['duration_nsec'] = pd.to_numeric(df['duration_nsec'], errors='coerce')
    df['packet_count'] = pd.to_numeric(df['packet_count'], errors='coerce')
    df['byte_count'] = pd.to_numeric(df['byte_count'], errors='coerce')
    
    # Loại bỏ lần nữa nếu ép kiểu ra NaN
    df = df.dropna(subset=['duration_sec', 'packet_count'])
    
    # Feature Engineering
    # Tính thời gian tồn tại thực sự của flow (giây)
    df['duration'] = df['duration_sec'] + df['duration_nsec'] / 1e9
    df['duration'] = df['duration'].apply(lambda x: x if x > 0 else 0.001) # Tránh chia cho 0
    
    # Tính tốc độ bit và gói tin (Byte/s và Packet/s)
    df['byte_rate'] = df['byte_count'] / df['duration']
    df['packet_rate'] = df['packet_count'] / df['duration']
    
    # Đổi sang datetime và làm tròn về hàng giây để gom nhóm chuẩn hơn
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].dt.round('1s') 
    df = df.sort_values('timestamp')
    
    return df

def aggregate_and_visualize(df):
    print("2. Đang tổng hợp dữ liệu toàn mạnng và vẽ biểu đồ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Gom tất cả các flow có cùng timestamp để tính tổng throughput của mạng
    agg_df = df.groupby('timestamp').agg({
        'byte_rate': 'sum',
        'packet_rate': 'sum',
        'label': 'last' # Nhãn HIGH hoặc NORMAL
    }).reset_index()
    
    # -- Data Visualization --
    plt.figure(figsize=(12, 5))
    
    # Lấy ra các mốc thời gian và điểm dữ liệu
    times = agg_df['timestamp']
    throughput_mbps = (agg_df['byte_rate'] * 8) / 1e6 # Đổi ra Mbps
    
    plt.plot(times, throughput_mbps, label='Network Throughput (Mbps)', color='#1f77b4', linewidth=2)
    
    # Điểm nhấn đợt tấn công Artillery (Label: HIGH)
    high_traffic = agg_df[agg_df['label'] == 'HIGH']
    if not high_traffic.empty:
        high_times = high_traffic['timestamp']
        high_mbps = (high_traffic['byte_rate'] * 8) / 1e6
        plt.scatter(high_times, high_mbps, color='#d62728', label='⚡ HIGH Traffic (Burst)', zorder=5, s=50)
        plt.axvspan(high_times.min(), high_times.max(), color='red', alpha=0.1) # Tô màu vùng bị stress
        
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

def create_time_series_windows(agg_df, sequence_length=5):
    """
    Tạo cửa sổ trượt (Sliding Window) cho model Sequential như TFT.
    Ví dụ sequence_length=5: Model sẽ nhìn vào 5 timestep quá khứ để dự đoán.
    """
    print(f"3. Cắt Windowing dữ liệu (Sequence Length = {sequence_length})...")
    
    features = ['byte_rate', 'packet_rate']
    
    # Chuẩn hóa Data (Min-Max Scaling) về scale [0, 1] cho Neural Network
    scaler = MinMaxScaler()
    agg_df[features] = scaler.fit_transform(agg_df[features])
    
    # Mã hóa nhãn: NORMAL=0, HIGH=1
    agg_df['label_encoded'] = (agg_df['label'] == 'HIGH').astype(int)
    
    X, y = [], []
    data_scaled = agg_df[features].values
    labels = agg_df['label_encoded'].values
    
    # Trượt window
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i : i + sequence_length])
        y.append(labels[i + sequence_length]) # Nhãn của timestep tiếp theo
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"   -> Tạo thành công {len(X)} sequences.")
    print(f"   -> Shape của tập dữ liệu vào (Input X): {X.shape}")
    print(f"   -> Shape của nhãn (Label y): {y.shape}")
    
    # Lưu dưới dạng Binary cho Numpy để Pytorch DataLoader ăn vào dễ dàng
    np.save(os.path.join(OUTPUT_DIR, 'X_sequences.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_labels.npy'), y)
    print("   -> Đã lưu các Tensor numpy (npy).")
    
    return X, y

if __name__ == '__main__':
    print("=== TỔNG HỢP & TIỀN XỬ LÝ DỮ LIỆU SDN ===")
    
    if not os.path.exists(STATS_CSV):
        print(f"❌ Không tìm thấy file data: {STATS_CSV}")
        print("Vui lòng chạy Mininet & Artillery để thu thập dữ liệu trước!")
        exit(1)
        
    df_raw = load_and_engineer_features(STATS_CSV)
    df_agg = aggregate_and_visualize(df_raw)
    X, y = create_time_series_windows(df_agg, sequence_length=5)
    
    print("✅ Hoàn tất toàn bộ Pipeline Tiền xử lý!")
