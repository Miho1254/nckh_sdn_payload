# NCKH SDN — AI-Driven Load Balancer on Fat-Tree Topology

> Tối ưu hóa phân tải mạng SDN bằng mô hình **TFT-DQN** (Temporal Fusion Transformer + Deep Q-Network) trong môi trường Mininet mô phỏng hệ thống đăng ký tín chỉ đại học (LMS) dưới tải cao.

---

## Kiến trúc Tổng quan

```
Clients (h9–h16)
     │  HTTP → 10.0.0.100 (Virtual IP)
     ▼
[Ryu Controller — TFT-DQN]  ←→  flow_stats.csv
     │  OpenFlow NAT
     ├──► h5 (Backend 1 — Node.js :4000)  [Bandwidth Limit: 10 Mbps]
     ├──► h7 (Backend 2 — Node.js :4000)  [Bandwidth Limit: 50 Mbps]
     └──► h8 (Backend 3 — Node.js :4000)  [Bandwidth Limit: 100 Mbps]
                              │
                         h6 (PostgreSQL — 5000 users)
```

**Topology:** Fat-Tree K=4 | 10 Switches | 16 Hosts  
**Thuật toán:** RR | WRR (1:2:3) | **TFT-DQN** (Inference < 100ms)

---

## Nền tảng SDN — Software-Defined Networking

Trong mạng truyền thống, mỗi switch tự quyết định cách chuyển gói tin (Control Plane nằm trong chính thiết bị). SDN tách biệt hoàn toàn:

| Thành phần | Vai trò | Trong dự án này |
|-----------|---------|----------------|
| **Control Plane** | Não bộ — quyết định luồng | Ryu Controller (`controller_stats.py`) |
| **Data Plane** | Tay chân — thực thi chuyển mạch | OVS Switches (s1–s10) |
| **Giao thức** | Ngôn ngữ giao tiếp | OpenFlow 1.3 |

### Cách hoạt động của Load Balancer SDN (NAT 5-tuple)

```
1. Client gửi gói tin HTTP đến VIP 10.0.0.100
2. Switch chưa có flow rule → gửi PacketIn lên Controller
3. Controller (TFT-DQN) chọn backend tốt nhất (h5/h7/h8)
4. Controller cài bộ đôi FlowMod (NAT rules) có độ ưu tiên cao:
   - Match: {ipv4_src, ipv4_dst=100, ip_proto, tcp_src, tcp_dst}
   - Lượt đi: Đổi IP đích 100 → IP backend, sửa MAC đích tương ứng.
   - Lượt về: Sửa IP nguồn backend → 100, sửa MAC nguồn → VIP MAC.
5. Giao thức hỗ trợ: TCP (HTTP) và UDP.
   (idle_timeout=30s để duy trì session và dọn dẹp rule cũ)
```

> [!IMPORTANT]
> **Persistence**: Các luật VIP Redirect được cài đặt tự động bởi script khởi tạo SAU khi cây Spanning Tree (STP) đã hội tụ hoàn toàn, đảm bảo không bị xoá bởi sự kiện Topology Change.


### Fat-Tree Topology K=4

```
[Core Switches]         s9  s10
                       / \ / \
[Aggr. Switches]     s5  s6  s7  s8
                    / \ / \ / \ / \
[Edge Switches]    s1  s2  s3  s4
                  /\  /\  /\  /\
[Hosts]         h1 h2 h3 h4 ... h16
```
- **16 hosts** chia thành 4 pod, mỗi pod có 4 host
- **Máy chủ bất đối xứng (Heterogeneous Servers):**
  - Khác biệt hóa khả năng xử lý mạng thông qua thắt nút cổ chai băng thông (để tránh lỗi tương thích môi trường Cgroups). Điều này đủ sức chứng minh sự ưu việt của AI so với các thuật toán truyền thống (RR/WRR).
  - `h5`: Băng thông nhánh 10 Mbps (Mạng yếu, dễ ngẽn)
  - `h7`: Băng thông nhánh 50 Mbps (Mạng trung bình)
  - `h8`: Băng thông nhánh 100 Mbps (Siêu mạng, thoải mái tài nguyên)
- **Fault tolerance:** Nhiều đường đi giữa bất kỳ 2 host nào

---

## Phương hướng AI — Huấn luyện TFT-DQN

### Tại sao TFT + DQN?

| Thành phần | Mục đích |
|-----------|----------|
| **TFT (Temporal Fusion Transformer)** | Nhìn vào chuỗi 5 timestep quá khứ để **dự đoán** trạng thái mạng tiếp theo. Cảm nhận "bão traffic" trước khi nó ập đến. |
| **DQN (Deep Q-Network)** | Dựa trên dự đoán của TFT để **ra quyết định** chuyển luồng sang server nào. Học qua thưởng/phạt từ lịch sử. |
| **Kết hợp** | TFT dự báo → DQN hành động → Hệ thống tự tiến hóa |

### Pipeline Huấn luyện (Offline RL)

```
Bước 1: THU THẬP DỮ LIỆU
  └── Chạy 4 kịch bản với LB_ALGO=COLLECT
  └── Ryu ghi flow_stats.csv (packet_count, byte_count, label HIGH/NORMAL)

Bước 2: TIỀN XỬ LÝ (data_processor.py)
  ├── Tính byte_rate, packet_rate từ cumulative counters
  ├── Làm tròn timestamp → gộp dữ liệu từ 10 switch
  ├── Min-Max Normalize về [0, 1]
  └── Sliding Window (seq_len=5) → X.npy, y.npy

Bước 3: HUẤN LUYỆN (train.py)
  ├── Môi trường Offline: SDN_Offline_Env phát lại dữ liệu (V3 Clean Static)
  ├── Agent dùng Boltzmann Selection (Softmax) để chọn action
  ├── Reward V3: Dựa trên chênh lệch tải thực tế trong state
  ├── Replay Buffer (50,000 samples) → chống overfitting
  ├── Epsilon: 1.0 → 0.15 (decay=0.985/epoch) — Duy trì 15% exploration
  ├── Temperature (Tau): 2.0 → 0.5 — Kiểm soát độ mượt của phân phối tải
  └── Mỗi epoch đồng bộ Target Network → ổn định Q-learning

Bước 4: INFERENCE (controller_stats.py)
  ├── Ring Buffer 5 bước trạng thái mạng real-time
  ├── TFT-DQN forward pass → Q-values cho 3 server
  └── Chọn action bằng Boltzmann softmax (τ=0.5) → bẻ luồng OpenFlow NAT
```

### Hàm Phần thưởng (Reward Function V3 - Clean Static)

```python
# Base Reward: Đảm bảo nền dương để Agent không bị nản lòng
reward = 3.0 + throughput * 0.5

# Balance Bonus: So sánh với trạng thái TĨNH của network (Offline RL)
if action == min_load_backend:
    reward += 4.0 * load_spread  # Thưởng lớn khi chọn đúng server nhẹ
elif action == max_load_backend:
    reward -= 3.0 * load_spread  # Phạt khi dồn thêm vào server nặng nhất

# Traffic-Type Modifier:
if label == HIGH:
    reward_load *= 2.0  # Ưu tiên load balancing gấp đôi khi nghẽn
```

### Thông số mô hình

| Tham số | Giá trị |
|---------|--------|
| Input features | 5 (byte_rate, packet_rate, load_h5, load_h7, load_h8) |
| Sequence length | 5 timesteps |
| Hidden size | 32 |
| Attention heads | 4 |
| Actions | 3 (h5, h7, h8) |
| Action Selection | Boltzmann Softmax (τ=0.5) |
| Inference latency | < 100ms |

---

## 4 Kịch bản Stress Test

Mỗi kịch bản mô phỏng một tình huống thực tế của hệ thống đăng ký tín chỉ, với 8 Artillery nodes (h9–h16) bắn đồng thời về VIP `10.0.0.100:4000`.

### Kịch bản 1 — Flash Crowd: Cơn lốc Đăng ký
`flash_crowd.yml`

```
Tải trọng:  ████████████████████ Burst cao đột ngột
Pattern:    0→1000 users chỉ trong 2 phút
Thời gian:  ~10 phút
```
**Mục đích:** Kiểm tra phản ứng của AI khi tải tăng **đột ngột** (Spike traffic). Đây là tình huống ngày mở đăng ký: toàn bộ sinh viên cùng đổ vào 1 thời điểm.  
**Yếu điểm của RR/WRR:** Không phản ứng kịp, tiếp tục phân phối vòng tròn dù 1 server đã quá tải.  
**Ưu điểm của AI:** TFT nhận diện spike pattern trước 5 timestep, DQN bẻ luồng sang server còn rảnh.

---

### Kịch bản 2 — Predictable Ramping: Thi trực tuyến
`predictable_ramping.yml`

```
Tải trọng:  ___/‾‾‾‾\___/‾‾‾‾\ Tăng đều, có thể dự báo
Pattern:    Sinh viên vào dần dần theo giờ thi
Thời gian:  ~15 phút
```
**Mục đích:** Kiểm tra khả năng **dự báo sớm** của TFT. Tải tăng có quy luật, AI phải nhận ra xu hướng trước khi đạt đỉnh.  
**Yếu điểm của RR/WRR:** Không dự báo được, chỉ phản ứng khi nghẽn đã xảy ra.  
**Ưu điểm của AI:** TFT nhìn ra quy luật tăng dần ngay từ sớm → cân tải chủ động.

---

### Kịch bản 3 — Targeted Congestion: Server tê liệt
`targeted_congestion.yml`

```
Tải trọng:  ████░░░░███░░░░███  h5 bị bóp nghẽn
Pattern:    Toàn bộ traffic nhắm thẳng vào h5
Thời gian:  ~10 phút
```
**Mục đích:** Kiểm tra **failover** — khả năng phát hiện và né 1 server đang quá tải cục bộ.  
**Yếu điểm của RR/WRR:** RR vẫn tiếp tục gửi 1/3 request về h5 dù nó đang kẹt.  
**Ưu điểm của AI:** DQN học được rằng action=0 (h5) cho reward âm liên tục → giảm xác suất chọn h5 xuống gần 0%.

---

### Kịch bản 4 — Gradual Shift: Biến đổi xu hướng
`gradual_shift.yml`

```
Tải trọng:  _____/‾‾‾‾‾‾‾‾‾‾‾‾‾ Tăng dần không bao giờ giảm
Pattern:    NORMAL → HIGH → VERY HIGH liên tục
Thời gian:  ~15 phút
```
**Mục đích:** Kiểm tra **độ bền** của AI trong thời gian dài. Đây là kịch bản thu thập data chính vì nó bao gồm cả 2 nhãn NORMAL và HIGH theo tỉ lệ tốt.  
**Yếu điểm của RR/WRR:** Không thể điều chỉnh weight theo thời gian thực.  
**Ưu điểm của AI:** Epsilon đã giảm về mức thấp → AI tự tin ra quyết định dựa trên pattern đã học.

---

## Yêu cầu Hệ thống

| Thành phần | Phiên bản |
|-----------|-----------|
| Docker + Docker Compose | >= 24 |
| Python | >= 3.10 |
| NVIDIA GPU (tùy chọn, cho Host Training) | CUDA >= 11.8 |
| RAM | >= 8 GB |

---

## Bắt đầu Nhanh

### 1. Khởi động môi trường

```bash
# Clone repo và vào thư mục
cd nckh_sdn

# Khởi động Docker container Mininet
docker compose up -d --build

# Cấp quyền thư mục kết quả
sudo chown -R $USER:$USER stats/
mkdir -p stats/results
```

### 2. Chạy toàn bộ Pipeline (Khuyến nghị)

Lệnh duy nhất thu thập 4 kịch bản → gộp data → train AI → xuất biểu đồ:

```bash
./scripts/full_pipeline.sh
```

> **Thời gian ước tính:** ~1 giờ (4 kịch bản x ~15 phút/kịch bản).  
> Script tự động dừng khi mỗi kịch bản hoàn tất — không cần can thiệp thủ công.

### 3. Chạy từng bước (Tùy chọn)

| Mục đích | Lệnh |
|---------|------|
| Đánh giá 1 kịch bản (chọn thuật toán) | `./scripts/evaluate_sdn.sh` |
| Báo cáo Traffic CLI (đẹp, có màu) | `python3 scripts/analyze_stats.py --all` |
| Train AI trong Docker | `./scripts/train_ai.sh` |
| Train AI với GPU Host | `./scripts/train_host.sh` |
| Vào Mininet CLI | `./scripts/enter_env.sh` |

---

## Quy trình Nghiên cứu Đầy đủ

```
Phase 1: Thu thập data (4 kịch bản, chế độ COLLECT)
  └── ./scripts/full_pipeline.sh COLLECT
       └── flow_stats_merged.csv → train AI → charts/

Phase 2: So sánh thuật toán (chạy lần lượt)
  ├── ./scripts/full_pipeline.sh RR
  ├── ./scripts/full_pipeline.sh WRR
  └── ./scripts/full_pipeline.sh AI
       └── Tự tổng hợp biểu đồ so sánh AI vs RR vs WRR
```

---

## Biểu đồ Nghiệm thu (Tự động xuất)

### Nhóm A — Biểu đồ Training
Xuất tại: `ai_model/processed_data/charts/`

| File | Nội dung |
|------|---------|
| `00_training_dashboard.png` | Dashboard tổng hợp |
| `01_loss_curve.png` | Q-Loss + TFT Auxiliary Loss |
| `02_reward_curve.png` | Total Reward per Epoch |
| `03_tft_prediction.png` | TFT Forecast vs Actual Traffic |
| `04_decay_schedule.png` | Epsilon & Temperature Decay |
| `05_action_distribution.png` | Phân phối chọn Server |

### Nhóm B — Biểu đồ So sánh
Xuất tại: `stats/results/charts/`

| File | Nội dung |
|------|---------|
| `00_comparison_dashboard_{scene}.png` | Dashboard tổng hợp so sánh |
| `06_throughput_{scene}.png` | Throughput Stability |
| `07_packet_loss_{scene}.png` | Packet Loss Rate |
| `08_heatmap_{scene}.png` | Server Load Heatmap |
| `09_inference_{scene}.png` | AI Inference Overhead |
| `10_latency_{scene}.png` | Latency Comparison |

### Nhóm C — Báo cáo Terminal (CLI)
Dữ liệu tường minh ngay sau khi chạy `evaluate_sdn.sh` hoặc `full_pipeline.sh`:
- **CV% (Hệ số biến thiên)**: Đánh giá độ lệch tải giữa các server (Càng thấp = càng cân bằng).
- **Phân bổ Traffic (MB)**: Tổng lượng dữ liệu thực tế h5, h7, h8 đã xử lý.
- **Trạng thái HIGH/NORMAL**: Thống kê tỉ lệ các luồng bị dán nhãn nghẽn tải.

---

## Kịch bản Stress Test (Artillery)

| # | File | Mô tả |
|---|------|-------|
| 1 | `flash_crowd.yml` | Cơn lốc truy cập đột ngột |
| 2 | `predictable_ramping.yml` | Tăng tải dự đoán được (Thi online) |
| 3 | `targeted_congestion.yml` | Kẹt nghẽn cục bộ tại h5 |
| 4 | `gradual_shift.yml` | Biến đổi xu hướng dần dần |

**Clients:** h9–h16 (8 Artillery nodes) → VIP `10.0.0.100:4000`  
**Labeling:** `NORMAL` / `HIGH` tự động gán vào `flow_stats.csv`

---

## Cấu trúc Dự án

```
nckh_sdn/
├── controller_stats.py       # Ryu controller: L2 Switch + Stats + AI Load Balancer
├── run_lms_mininet.py        # Orchestrator: khởi động mạng, LMS, Artillery
├── topo_fattree.py           # Fat-Tree K=4 topology definition
├── docker-compose.yml
│
├── ai_model/
│   ├── tft_dqn_net.py        # Kiến trúc mô hình (VSN + LSTM + Attention + DQN)
│   ├── dqn_agent.py          # DQN Agent (Epsilon-greedy, Replay Buffer)
│   ├── sdn_env.py            # Môi trường Offline cho RL training
│   ├── data_processor.py     # Pipeline tiền xử lý: CSV → NPY (Sliding Window)
│   ├── train.py              # Training loop + xuất 6 biểu đồ Training
│   ├── generate_comparison_charts.py  # Script vẽ biểu đồ so sánh
│   ├── processed_data/       # NPY sequences + charts Training
│   └── checkpoints/          # Model checkpoint (.pth)
│
├── lms/
│   ├── backend/              # Node.js Express + PostgreSQL
│   ├── frontend/             # React + Vite
│   └── stress-test/          # Artillery scenarios + run_labeled_test.py
│
├── scripts/
│   ├── full_pipeline.sh      # CHAY TOAN BO: Thu thap → Train → Bieu do
│   ├── evaluate_sdn.sh       # Chay 1 kich ban + 1 thuat toan
│   ├── train_host.sh         # Train AI tren GPU Host
│   ├── train_ai.sh           # Train AI trong Docker
│   └── enter_env.sh          # Vao Mininet CLI
│
└── stats/
    ├── flow_stats.csv         # Du lieu thu thap tu Ryu (real-time)
    ├── port_stats.csv         # Port-level metrics
    ├── flow_stats_merged.csv  # Du lieu gop tu tat ca kich ban
    └── results/
        ├── RR_flash_crowd/       # Ket qua tu tung thuat toan x kich ban
        ├── WRR_flash_crowd/
        ├── AI_flash_crowd/
        └── charts/               # Bieu do so sanh tong hop
```

---

## Biến môi trường

| Biến | Mô tả | Mặc định |
|------|-------|---------|
| `SCENARIO` | File kịch bản Artillery | `flash_crowd.yml` |
| `LB_ALGO` | Thuật toán load balancing | `RR` |
| `VIP` | Virtual IP của Load Balancer | `10.0.0.100` |
| `DB_HOST` | IP máy chủ PostgreSQL | `10.0.0.6` |

---

## Train AI trên GPU Host (Arch Linux / NVIDIA)

```bash
# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate    # bash/zsh
# hoặc
source venv/bin/activate.fish  # fish shell

# Cài dependencies
pip install -r ai_model/requirements.txt

# Chạy (tự nhận diện CUDA)
./scripts/train_host.sh
```

---

## License

MIT
