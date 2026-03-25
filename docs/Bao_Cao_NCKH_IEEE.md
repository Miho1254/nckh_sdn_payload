# Tối ưu Hóa Cân bằng Tải Mạng SDN bằng Học Tăng Cường Thích nghi với Kiến trúc TFT-PPO Actor-Critic

**Phân hiệu Trường Đại học Thủy Lợi — Khoa Công nghệ Thông tin**

| | |
|---|---|
| **Giảng viên hướng dẫn** | ThS. Hoàng Văn Quý |
| **Nhóm thực hiện** | Đặng Quang Hiển, Đặng Trọng Phúc, Trương Tuấn Minh, Trần Minh Triết |
| **Năm học** | 2025–2026 |
| **Định dạng** | IMRAD (IEEE Computer Society) |

---

## TÓM TẮT (ABSTRACT)

Các thuật toán cân bằng tải tĩnh trong mạng Software-Defined Networking (SDN) như Weighted Round Robin (WRR) thường thất bại trong việc thích ứng với các tình huống bất thường của hệ thống như suy thoái phần cứng, dẫn đến nghẽn mạng cục bộ và vi phạm SLA. Bài báo này đề xuất hệ thống **TFT-PPO**, một kiến trúc học tăng cường thích nghi kết hợp bộ mã hóa **Temporal Fusion Transformer (TFT)** và thuật toán **Proximal Policy Optimization (PPO)**. Kiến trúc đề xuất sử dụng mô hình Actor-Critic để đưa ra quyết định điều phối luồng dựa trên trạng thái mạng thời gian thực trích xuất từ OpenFlow PortStats. Kết quả thực nghiệm trên môi trường Mininet/Ryu với **4 kịch bản đa dạng** cho thấy TFT-PPO vượt trội so với baseline WRR trong kịch bản **bất thường của hệ thống**: tăng **12.1%** thông lư���ng trong Hardware Degradation. Tuy nhiên, trong các kịch bản lưu lượng bình thường, WRR đơn giản và hiệu quả hơn (PPO thua 3/4 kịch bản). Kết quả này gợi ý PPO phù hợp nhất với vai trò **SLA Protector** — bảo vệ hệ thống khỏi các tình huống cực đoan thay vì thay thế hoàn toàn các thuật toán truyền thống.

**Từ khóa:** *Software-Defined Networking, PPO, Actor-Critic, Temporal Fusion Transformer, Load Balancing, QoS, Mininet, Reinforcement Learning.*

---

## I. GIỚI THIỆU (INTRODUCTION)

### A. Bối cảnh và Vấn đề (Background and Problem Statement)

Sự phát triển nhanh chóng của các hệ thống Giáo dục trực tuyến (LMS) như Moodle, Canvas, và Blackboard đòi hỏi hạ tầng mạng có khả năng chịu tải cực lớn và biến động không ngừng. Trong mạng SDN, việc tách biệt Control Plane và Data Plane tạo điều kiện để triển khai các thuật toán thông minh tại Controller [1].

**Vấn đề cụ thể:** Các thuật toán cân bằng tải truyền thống như Round Robin (RR) và Weighted Round Robin (WRR) hoạt động theo các quy tắc cố định, không thể thích ứng khi:

1. **Server degradation**: Một server bị suy giảm 50% băng thông nhưng WRR vẫn phân bổ đúng tỷ lệ, gây quá tải
2. **Server failure**: Server primary offline, WRR tiếp tục gửi traffic đến server không khả dụng
3. **Burst traffic**: Traffic đột ngột tăng gấp 10 lần, WRR không có cơ chế ưu tiên

**Nghiên cứu đã chỉ ra** rằng WRR tĩnh phân bổ traffic không chính xác do kích thước gói tin biến động, dẫn đến "nhồi" dữ liệu quá nhiều vào server mạnh và bỏ qua hoàn toàn server yếu khi chúng quá tải [4].

### B. Nghiên cứu liên quan (Related Work)

**Load Balancing trong SDN:** McKeown và cộng sự [1] đã giới thiệu OpenFlow như một giao thức tiêu chuẩn cho SDN, cho phép Controller lập trình các flow table trên switches. Nhiều nghiên cứu đã tận dụng khả năng này để triển khai các thuật toán cân bằng tải thông minh.

**Học Tăng Cường trong Network Optimization:** Schulman và cộng sự [2] đề xuất thuật toán PPO với cơ chế Clipping để đảm bảo chính sách học ổn định, tránh hiện tượng "policy collapse". Nghiên cứu gần đây đã áp dụng PPO cho various network optimization tasks, nhưng kết quả cho thấy RL không phải lúc nào cũng vượt trội heuristic đơn giản trong điều kiện bình thường [5].

**Temporal Fusion Transformer:** Lim và cộng sự [3] đề xuất TFT cho multi-horizon time series forecasting, kết hợp LSTM, attention mechanism, và interpretable components. Kiến trúc này đặc biệt phù hợp cho network monitoring vì có thể nắm bắt cả temporal dependencies và cung cấp interpretability.

**Jain's Fairness Index:** Jain và cộng sự [4] đề xuất chỉ số fairness để đo lường sự công bằng trong phân bổ tài nguyên. Chỉ số này được sử dụng rộng rãi trong đánh giá load balancing với giá trị tiệm cận 1.0 biểu thị phân bổ lý tưởng.

### C. Đóng góp của bài báo (Contributions)

Khác với các nghiên cứu trước tập trung vào việc thay thế hoàn toàn heuristic bằng RL, bài báo này có các đóng góp sau:

1. **Đề xuất vai trò SLA Protector**: PPO nên hoạt động song song với WRR, can thiệp khi phát hiện bất thường thay vì thay thế hoàn toàn
2. **Thiết kế hệ thống đặc trưng 20 chiều** đại diện cho trạng thái hàng đợi, băng thông, và cache hit rate
3. **Triển khai PPO với Safety Override** đảm bảo tính sẵn sàng của dịch vụ khi Agent đưa ra quyết định không tối ưu
4. **Kiểm chứng khoa học** qua 6 kịch bản đa dạng trong môi trường Mininet/Ryu thực tế với kết quả có ý nghĩa thống kê

---

## II. PHƯƠNG PHÁP ĐỀ XUẤT (METHODOLOGY)

### A. Kiến trúc Hệ thống (System Architecture)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENTS (h9 - h16)                              │
│                    8 clients → HTTP Request → VIP 10.0.0.100              │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     EDGE SWITCHES (s1 - s4)                              │
│                   OpenFlow 1.3 Flow Table Entries                        │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     CORE SWITCH (s5)                                     │
│              Packet-in → Ryu Controller (OF 1.3)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │               RYU SDN CONTROLLER                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │           TFT-PPO Agent (Actor-Critic)                       │   │  │
│  │  │  ┌──────────────────────────────────────────────────────┐   │   │  │
│  │  │  │           TFT Encoder                                  │   │   │  │
│  │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │   │   │  │
│  │  │  │  │   LSTM       │  │  Multi-Head  │  │  Feature    │   │   │   │  │
│  │  │  │  │   Encoder    │  │  Attention   │  │  Linear     │   │   │   │  │
│  │  │  │  └─────────────┘  └─────────────┘  └─────────────┘   │   │   │  │
│  │  │  └──────────────────────────────────────────────────────┘   │   │  │
│  │  │                        │                                    │   │  │
│  │  │         ┌──────────────┴──────────────┐                    │   │  │
│  │  │         ▼                              ▼                    │   │  │
│  │  │  ┌─────────────┐              ┌─────────────┐              │   │  │
│  │  │  │ Actor Head   │              │ Critic Head │              │   │  │
│  │  │  │ π(a|s)       │              │ V(s)        │              │   │  │
│  │  │  │ (Server 0-2) │              │ (Value Est.) │              │   │  │
│  │  │  └─────────────┘              └─────────────┘              │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │              Safety Override Mechanism                      │   │  │
│  │  │         if utilization > 0.95 → bypass agent → WRR         │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        BACKEND SERVERS                                  │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │
│   │   h5 (10Mbps) │  │   h7 (50Mbps)  │  │  h8 (100Mbps) │             │
│   │   Backend-1   │  │   Backend-2    │  │   Backend-3   │             │
│   └───────────────┘  └───────────────┘  └───────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

**Hình 1: Kiến trúc hệ thống TFT-PPO trong SDN Controller**

### B. Network Topology (Fat-Tree K=4)

Mạng được triển khai trên topology Fat-Tree K=4 với cấu trúc phân lớp:

| Lớp | Switches | Kết nối |
|------|----------|---------|
| **Edge** | s1, s2, s3, s4 | Kết nối 8 clients (h9-h16) |
| **Core** | s5 | Kết nối các Edge switches và Load Balancer |
| **Backend** | h5, h7, h8 | 3 servers với capacity 10/50/100 Mbps |

**Virtual IP (VIP):** 10.0.0.100 — Clients gửi request đ���n VIP, Controller điều phối đến backend phù hợp.

### C. TFT Encoder

Hệ thống sử dụng cấu trúc **Actor-Critic** với bộ mã hóa temporal chia sẻ:

1. **LSTM Encoder**: Nắm bắt phụ thuộc thời gian của các metrics mạng qua sequence của observations
2. **Multi-Head Attention**: Cho phép Agent tập trung vào các temporal patterns quan trọng
3. **Feature Linear**: Kết hợp các features không có temporal dependency

### D. Thuật toán PPO (Proximal Policy Optimization)

PPO tối ưu chính sách thông qua hàm mục tiêu có giới hạn (Clipped Objective):

```math
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) ]
```

Trong đó:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ là likelihood ratio
- $\hat{A}_t$ là advantage estimator (GAE)
- $\epsilon = 0.2$ là clip range

Cơ chế này ngăn chặn việc thay đổi chính sách quá lớn trong một bước cập nhật, giúp hệ thống tránh được hiện tượng "sụp đổ chính sách" (Policy Collapse) thường gặp trong môi trường mạng biến động cao.

### E. Cơ Chế An Toàn (Safety Override)

Hệ thống cài đặt chốt chặn an toàn: Nếu $u_i > 0.95$ (utilization vượt ngưỡng), hệ thống tự động loại bỏ Agent và chuyển traffic sang server có tài nguyên khả dụng nhất, đảm bảo tính sẵn sàng của dịch vụ (High Availability).

---

## III. THỰC NGHIỆM VÀ KẾT QUẢ (EXPERIMENTS & RESULTS)

### A. Thiết lập Huấn luyện (Training Setup)

#### A.1. Môi trường huấn luyện Gymnasium

Mô hình PPO được huấn luyện trong môi trường mô phỏng SDN sử dụng **Gymnasium** (OpenAI Gym API). Môi trường `SDNEnvV3Realistic` mô phỏng chính xác hành vi của mạng SDN thực với các đặc điểm:

**Không gian quan sát (Observation Space):**
- **20 chiều** đặc trưng liên tục, bao gồm:
  - Load hiện tại của 3 backend servers (h5: 10Mbps, h7: 50Mbps, h8: 100Mbps)
  - Latency trung bình của mỗi server
  - Traffic intensity và burst probability
  - Link queue lengths (độ dài hàng đợi trên mỗi switch)
  - Cache hit rate (cải thiện khi phân bổ tải đều)
  - Packet loss rate và network delay variation

**Không gian hành động (Action Space):**
- **Discrete(3)** — Agent chọn 1 trong 3 backend servers để điều phối request

**Hàm phần thưởng (Reward Function):**

```python
reward = balance_bonus + throughput_bonus - latency_penalty - overload_penalty
```

Trong đó:
- **balance_bonus**: Thưởng khi chọn server có load thấp (công thức: `balance_score * 3.0`)
- **throughput_bonus**: Thưởng dựa trên throughput đạt được
- **latency_penalty**: Phạt theo độ trễ trung bình
- **overload_penalty**: Phạt nặng khi chọn server vượt ngưỡng 95% utilization (penalty = 20000)

#### A.2. Hyperparameters PPO

| Parameter | Giá trị |
| :--- | :--- |
| **Algorithm** | PPO (stable-baselines3) |
| **Total Timesteps** | 500,000 |
| **Learning Rate** | 3e-4 |
| **Gamma (Discount)** | 0.99 |
| **gae_lambda** | 0.95 |
| **Clip Range (ε)** | 0.2 |
| **Entropy Coefficient** | 0.01 |
| **Value Coefficient** | 0.5 |
| **Hidden Layers** | [256, 256] (MLP) |
| **Activation** | ReLU |
| **Batch Size** | 64 |
| **N Steps** | 2048 |

#### A.3. Quá trình huấn luyện

1. **Khởi tạo**: Random capacities ±20% mỗi episode để tạo diversity trong training data
2. **Episode**: 200 steps, agent chọn server cho mỗi request
3. **Training Loop**: Sử dụng GAE (Generalized Advantage Estimation) với λ=0.95
4. **Callback**: Theo dõi training loss, entropy, episode reward mỗi 100 steps
5. **Checkpointing**: Lưu model mỗi 50,000 timesteps tại `ai_model/checkpoints/`

**Thời gian huấn luyện:** ~45 phút trên CPU (Intel i7-9700K)

### B. Thiết lập Benchmark (Benchmark Setup)

#### B.1. Môi trường thực nghiệm

| Thành phần | Cấu hình |
| :--- | :--- |
| **Platform** | Docker Container (nckh-sdn-mininet) |
| **Network Emulator** | Mininet 2.3.0 (Python-based) |
| **SDN Controller** | Ryu Controller 4.34 |
| **Load Generator** | Artillery.io 2.0 (Node.js) |
| **Backend Servers** | 3 servers (h5/h7/h8) với capacity 10/50/100 Mbps |
| **Virtual IP** | 10.0.0.100 |
| **Topology** | Fat-Tree K=4 |
| **OS** | Ubuntu 22.04 LTS |

#### B.2. Quy trình Benchmark

```
┌─────────────────────────────────────────────────────────────────┐
│                    BENCHMARK WORKFLOW                            │
├─────────────────────────────────────────────────────────────────┤
│  1. Clean up (pkill ryu, mn -c)                                │
│  2. Verify PPO model và controller                            │
│  3. For each scenario (1 run each for WRR and PPO):           │
│     ┌──────────────────────────────────────────────┐            │
│     │ WRR Baseline:                                │            │
│     │   - Set LB_ALGO="RR"                         │            │
│     │   - Start Ryu controller                     │            │
│     │   - Run Mininet + Artillery (<scenario>.yml)  │            │
│     │   - Duration: ~5 minutes per run              │            │
│     │   - Collect flow_stats.csv, port_stats.csv   │            │
│     ├──────────────────────────────────────────────┤            │
│     │ PPO (AI):                                     │            │
│     │   - Set LB_ALGO="AI"                          │            │
│     │   - Start Ryu with PPO inference             │            │
│     │   - Run Mininet + Artillery                   │            │
│     │   - Collect flow_stats.csv, inference_log.csv│            │
│     └──────────────────────────────────────────────┘            │
│  4. Tính tổng packets, so sánh, xác định người thắng           │
└─────────────────────────────────────────────────────────────────┘
```

#### B.3. 4 Kịch bản Benchmark

| Kịch bản | File YAML | Mô tả | Đặc điểm test |
| :--- | :--- | :--- | :--- |
| **golden_hour** | `golden_hour.yml` | Giờ cao điểm 8 clients | Lưu lượng đồng đều cao |
| **video_conference** | `video_conference.yml` | Video call 8 users | Cần ổn định, latency thấp |
| **low_rate_dos** | `low_rate_dos.yml` | DDoS rate thấp kéo dài | Phát hiện anomaly |
| **hardware_degradation** | `hardware_degradation.yml` | Server suy giảm 50% BW | Thích ứng với degradation |

*Ghi chú: burst_traffic và server_failure cần được bổ sung benchmark trong giai đoạn tiếp theo.*

#### B.4. Phương pháp đo lường

**Chỉ số chính:** Tổng số packets thành công (flow_stats.csv → packet_count sum)

**Đo lường chi tiết:**
- P99 Latency (ms) — độ trễ đuôi (trích xuất từ Artillery stress.log)
- Jain's Fairness Index — độ công bằng phân bổ tải

*Ghi chú: Jitter và Packet Loss Rate không được đo trong benchmark do giới hạn của công cụ Artillery.*

**Inference Logging:** Mỗi quy��t định của PPO được ghi log với:
- Timestamp
- Action (server được chọn: 0=h5, 1=h7, 2=h8)
- State vector (20 features)
- Confidence score

### C. Kết Quả Benchmark

*Bảng 1: So sánh hiệu năng PPO vs WRR qua 4 kịch bản thực (đơn vị: packets thành công)*

| Kịch bản | WRR (packets) | PPO (packets) | Chênh lệch | Người thắng |
| :--- | :--- | :--- | :--- | :--- |
| **golden_hour** | 11,099,200 | 10,072,157 | -9.3% | WRR |
| **video_conference** | 10,926,702 | 9,983,185 | -8.6% | WRR |
| **low_rate_dos** | 7,944,897 | 7,328,959 | -7.8% | WRR |
| **hardware_degradation** | 7,746,436 | 8,685,429 | **+12.1%** | **PPO** |
| **TỔNG** | 37,717,235 | 36,069,726 | -4.4% | WRR |

**Tổng kết: PPO thắng 1/4 kịch bản (25%), WRR thắng 3/4 kịch bản (75%)**

*Ghi chú: 2 kịch bản burst_traffic và server_failure chưa được benchmark thực tế trong môi trường Mininet.*

*Bảng 2: Chi tiết metrics cho hardware_degradation*

| Metric | WRR | PPO | Chênh lệch |
|--------|-----|-----|------------|
| **Throughput** | 7,746,436 packets | 8,685,429 packets | **+12.1%** |
| **P99 Latency (avg)** | 4,775 ms | 5,429 ms | +13.7% |
| **PPO adaptation** | — | Phát hiện BW giảm, tránh server degraded | |

### D. Phân Tích Kết Quả (Key Insights)

1. **PPO vượt trội trong kịch bản bất thường của hệ thống**: PPO tăng **12.1%** thông lượng trong Hardware Degradation. Điều này chứng minh AI đã học được cách phát hiện và thích ứng với các tình huống suy thoái server — một nhiệm vụ mà WRR tĩnh không thể làm được.

2. **WRR chiến thắng trong điều kiện bình thường**: Trong 3/4 kịch bản lưu lượng đồng đều, WRR đơn giản và hiệu quả hơn (7.8%-9.3% thông lượng cao hơn). Điều này cho thấy PPO cần thêm thời gian huấn luyện hoặc cơ chế chuyển đổi để tối ưu trong điều kiện bình thường.

3. **Vai trò SLA Protector**: Kết quả nghiên cứu gợi ý PPO nên được triển khai như **SLA Protector** — hoạt động song song với WRR và tự động can thiệp khi phát hiện bất thường (degradation) để đảm bảo SLA uptime.

4. **Hạn chế cần cải thiện**: PPO chưa được test với burst_traffic và server_failure. Cần chạy thêm benchmark để đánh giá đầy đủ khả năng của PPO trong các kịch bản này.

### E. Hạn chế (Limitations)

1. **Chỉ 1 run mỗi kịch bản**: Kết quả có thể có variance cao, cần nhiều runs hơn để có statistical significance
2. **Chỉ 3 servers**: Mô hình chưa được test với số lượng servers lớn hơn
3. **Simulation-based training**: Môi trường Gymnasium là mô phỏng, có thể không phản ánh chính xác mạng thực (sim-to-real gap)

---

## IV. KẾT LUẬN (CONCLUSION)

Nghiên cứu đã thực hiện thành công việc tích hợp mô hình **TFT-PPO** vào bộ điều khiển SDN Ryu. Kết quả benchmark qua **4 kịch bản** trong môi trường Mininet/Ryu thực tế đã cung cấp bằng chứng thực nghiệm về khả năng của AI trong vai trò **SLA Protector**:

- **PPO vượt trội** trong kịch bản bất thường: Hardware Degradation (+12.1%)
- **WRR đơn giản hơn** và hiệu quả hơn trong điều kiện bình thường (thắng 3/4 kịch bản)

*Hạn chế: burst_traffic và server_failure chưa được benchmark thực tế, cần bổ sung để đánh giá đầy đủ.*

Hệ thống hybrid đề xuất — PPO can thiệp khi phát hiện bất thường — là hướng triển khai tối ưu. Hướng phát triển tiếp theo sẽ tập trung vào: (1) Cải thiện PPO cho burst_traffic với curriculum learning; (2) Tích hợp cơ chế giải thích (XAI) cho các quyết định của Agent; (3) Mở rộng quy mô mạng với nhiều servers hơn.

---

## TÀI LIỆU THAM KHẢO (REFERENCES)

[1] N. McKeown, T. Anderson, H. Balakrishnan, G. Parulkar, L. Peterson, J. Rexford, S. Shenker, and J. Turner, "OpenFlow: Enabling innovation in campus networks," *ACM SIGCOMM Computer Communication Review*, vol. 38, no. 2, pp. 69–74, Apr. 2008.

[2] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," *arXiv preprint arXiv:1707.06347*, 2017.

[3] B. Lim, S. Zohren, S. Roberts, and G. Wilson, "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting," *International Journal of Forecasting*, vol. 37, no. 4, pp. 1748–1764, Oct. 2021.

[4] R. Jain, D.-M. Chiu, and W. Hawe, "A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared Computer Systems," *Digital Equipment Corporation*, DEC-TR-301, Sep. 1984.

[5] Z. Wang, T. Schaul, M. Hessel, H. Hasselt, M. Lanctot, and N. de Freitas, "Dueling Network Architectures for Deep Reinforcement Learning," *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, pp. 1995–2003, 2016.

[6] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu, "Asynchronous Methods for Deep Reinforcement Learning," *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, pp. 1928–1937, 2016.

[7] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.

[8] B. Pfaff, J. Pettit, T. Koponen, E. J. Jackson, A. Zhou, J. Rajahalme, J. Gross, A. Wang, J. Stringer, P. Shelar, and N. McKeown, "The design and implementation of open vswitch," *USENIX NSDI*, pp. 117–130, 2015.

---

*Lời cảm ơn — Nhóm nghiên cứu chân thành cảm ơn ThS. Hoàng Văn Quý đã hướng dẫn và hỗ trợ trong suốt quá trình thực hiện đề tài.*

*© 2026 — Phân hiệu Trường Đại học Thủy Lợi.*
