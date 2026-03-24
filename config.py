import numpy as np

# ── ĐỊNH NGHĨA KHÔNG GIAN BÀI TOÁN - ĐIỀU PHỐI ĐỘNG ─────────────────
# VIP (Virtual IP) cho Mạng
VIP = "10.0.0.100"
VMAC = "00:00:00:00:01:00"

# Cấu hình Database
DB_NODE = "h6"
DB_IP = "10.0.0.6"

# Chu kì Controller query dữ liệu từ switch (giây)
POLL_INTERVAL = 10  

# ── DANH SÁCH BACKEND SERVER (NODE) ──────────────────────────────────
# Bạn có thể thêm, bớt bất kỳ số lượng Server nào ở đây.
# Mỗi mục yêu cầu: 'name', 'ip', 'mac', 'port', 'dpid' (thuộc Edge switch nào), và 'weight' (công suất phần cứng quy đối)
BACKENDS = [
    {"name": "h5", "ip": "10.0.0.5", "mac": "00:00:00:00:00:05", "dpid": 8, "port": 2, "weight": 1},  # Nút yếu (ví dụ: 10Mbps)
    {"name": "h7", "ip": "10.0.0.7", "mac": "00:00:00:00:00:07", "dpid": 8, "port": 4, "weight": 5},  # Nút trung bình (ví dụ: 50Mbps)
    {"name": "h8", "ip": "10.0.0.8", "mac": "00:00:00:00:00:08", "dpid": 8, "port": 5, "weight": 10}, # Nút trâu bò (ví dụ: 100Mbps)
]

# ── THÔNG SỐ CHUẨN HÓA AI (SCALING CONSTANTS) ───────────────────────
# Tăng độ nhạy (Sensitivity): Chuẩn hóa theo 10Mbps thay vì 100Mbps
# vì traffic thực tế trong kịch bản đang dao động quanh mức 1-5Mbps.
SCALING_BYTE_RATE = 1.0e7 
# Packet Rate: chuẩn hóa theo 1k packets/s
SCALING_PKT_RATE = 1.0e3
# Tải server: chuẩn hóa theo delta bytes ~1MB/s
SCALING_LOAD = 1.0e6

# Tự động xuất tỉ lệ công suất để dùng cho AI State Normalization và Capacity-aware routing
CAPACITIES = np.array([node["weight"] for node in BACKENDS], dtype=float)
NUM_ACTIONS = len(BACKENDS)

# Danh sách Client (h9-h16) dùng để bắn traffic và thông số kết nối
TEST_CLIENTS_DETAILS = [
    {"name": "h9", "dpid": 9, "port": 2},
    {"name": "h10", "dpid": 9, "port": 3},
    {"name": "h11", "dpid": 9, "port": 4},
    {"name": "h12", "dpid": 9, "port": 5},
    {"name": "h13", "dpid": 10, "port": 2},
    {"name": "h14", "dpid": 10, "port": 3},
    {"name": "h15", "dpid": 10, "port": 4},
    {"name": "h16", "dpid": 10, "port": 5},
]
TEST_CLIENTS = [c["name"] for c in TEST_CLIENTS_DETAILS]

# Các Switch thuộc lớp Edge (nơi Host kết nối)
EDGE_DPIDS = {7, 8, 9, 10}

# ── PPO & SYSTEM SETTINGS ───────────────────────────────────────────
SAFETY_THRESHOLD = 0.95    # ngưỡng overload cứng
# ═════════════════════════════════════════════════════════════════════
# V3: BALANCED prior để match với balanced dataset [33%, 33%, 34%]
# Không ép AI theo capacity ratio nữa, để AI tự học từ data
CAPACITY_PRIOR = [0.33, 0.33, 0.34]  # BALANCED: 33% h5, 33% h7, 34% h8

# ═════════════════════════════════════════════════════════════════════
# EVALUATION WEIGHTS - ƯU TIÊN QoS EFFICIENCY
# ═════════════════════════════════════════════════════════════════════
# Composite score mới - công bằng hơn cho AI
EVAL_WEIGHTS = {
    "throughput": 1.0,
    "overload_penalty": 2.0,
    "fairness_penalty": 0.5,   # GIẢM từ 1.0 xuống 0.5
    "churn_penalty": 0.3,      # Giảm từ 0.5 xuống 0.3
    # New weights for fair evaluation
    "qos_efficiency": 0.30,    # Throughput / (1 + Overload_Rate)
    "burst_handling": 0.20,   # Burst handling ratio
    "stability": 0.15,        # Stability score
}
