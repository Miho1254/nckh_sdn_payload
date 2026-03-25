# 🎯 MASTER PLAN: AI-Driven SDN Load Balancing với TFT-CQL
## Brainstorm & Architectural Blueprint — 2026

---

## MỤC LỤC

1. [Tổng quan Kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Phân tích Chiến lược Tarpit & Draining](#2-phân-tích-chiến-lược-tarpit--draining)
3. [Reward Function Chính thức](#3-reward-function-chính-thức)
4. [4 Kịch bản Chi tiết](#4-4-kịch-bản-chi-tiết)
5. [Metrics & Tools](#5-metrics--tools)
6. [TODO List Hoàn chỉnh](#6-todo-list-hoàn-chỉnh)

---

## 1. Tổng quan Kiến trúc

### 1.1 Vị trí AI trong SDN Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONTROL PLANE (Ryu Controller)               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              TFT-CQL Agent (AI Brain)                    │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │ Temporal    │  │ Actor Head  │  │ Critic Head │      │    │
│  │  │ Encoder     │  │ π(a|s)      │  │ Q(s,a)      │      │    │
│  │  │ (VSN+LSTM   │  │             │  │             │      │    │
│  │  │ +Attn)      │  │             │  │             │      │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │
│  │         ↑                ↓                              │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │           Reward Function R(s,a,s')            │    │    │
│  │  │  = W₁·Throughput - W₂·TailLatency - W₃·Crash   │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                 OpenFlow Stats/Mod Messages                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PLANE (OVS Switches)                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │   h5     │    │   h7     │    │   h8     │                   │
│  │ (10Mbps) │    │ (50Mbps) │    │ (100Mbps)│                   │
│  │ "Hố bùn" │    │ "Trung   │    │ "VIP"    │                   │
│  │ Tarpit   │    │  gian"   │    │ Elephants│                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
│                                                              │
│  Traffic Patterns:                                           │
│  - Elephant Flows → h8 (100Mbps)                            │
│  - Mice Flows → h7/h8 (50/100Mbps)                         │
│  - DoS/Tarpit → h5 (10Mbps)                                 │
│  - Degrading Node → DRAINING → h7/h5                         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Phân biệt Expert System vs RL

| Khía cạnh | Expert System (if/else) | Reinforcement Learning (RL) |
|-----------|------------------------|-----------------------------|
| **Rule** | Hardcoded: "Cấm video vào h5" | Emergent: AI tự học qua hậu quả |
| **Adaptation** | Static, cần manual update | Tự động qua Reward feedback |
| **Tarpit Strategy** | Được viết sẵn | AI tự "ngộ" ra khi thấy penalty |
| **Edge Cases** | Cần programmer foresight | Khám phá qua exploration |

**Nguyên tắc VÀNG:** Không viết rule, viết Reward Function. Để AI tự ra quyết định.

---

## 2. Phân tích Chiến lược Tarpit & Draining

### 2.1 Tarpit (Hố bùn) — Low Rate DoS

**Bản chất:** Kẻ tấn công mở nhiều TCP connection nhưng gửi data rất chậm (1 byte/giây), ngâm server.

**AI nhận diện:**
```
Pattern: TCP connections cao + Byte rate cực thấp + Connection duration dài
L3/L4 signature: packet_count >> byte_count
```

**WRR chết thế nào:**
```
WRR chia đều: h5=33%, h7=33%, h8=33%
→ h8 (100Mbps) bị ngâm bởi 33% "rác"
→ Goodput của h8 giảm 33%
→ Khách thật không vào được
```

**AI Tarpit Strategy (Emergent Behavior):**
```
AI Observe: h5 đang nhàn rỗi, h8 bị ngâm
AI Action: Redirect tất cả "rác" vào h5
AI Reward: +High(Throughput) vì h8 giải phóng, -Low(PacketLoss) vì h5 "nuốt" được
AI Learn: "Cứ thấy TCP bé tí đều đều → h5 là bạn của tao"
```

**Hành vi Emergent:** Không hardcode "if DoS then h5". AI tự phát hiện pattern qua reward signal.

### 2.2 Draining (Thoát tải) — Hardware Degradation

**Bản chất:** Node bắt đầu suy giảm (CPU>95%, IOPS tăng, latency tăng 20ms→500ms) nhưng chưa chết hẳn.

**AI nhận diện:**
```
Pattern: Latency tăng + Queue length tăng + CPU/IOPS warning
L3/L4 signature: response_time EWMA tăng đều qua các timestep
```

**WRR chết thế nào:**
```
WRR không biết h8 đang "hấp hơi"
WRR tiếp tục nhồi 60% traffic vào h8
→ h8 ngộp thở → 5xx errors tăng vọt
→ TTM = ∞ (đợi DevOps 2h sáng)
```

**AI Draining Strategy (Emergent Behavior):**
```
AI Observe: h8 latency tăng 20ms→200ms trong 3 timestep liên tiếp
AI Action: 
  - Step N: Giảm weight h8 từ 60% → 30%
  - Step N+1: Giảm tiếp 30% → 10%
  - Step N+2: Đặt h8 = DRAINING (chỉ xử lý request cũ, không nhận mới)
AI Reward: -High(Crash Penalty) + High(Throughput vẫn giữ)
AI Learn: "Cứ thấy latency tăng đều → bóp vòi trước khi nó chết"
```

### 2.3 Least Outstanding Requests (LOR) — Golden Hour

**Bản chất:** Traffic tăng x5-x10 đột ngột, cần đếm active requests không phải băng thông.

**AI nhận diện:**
```
Pattern: arrivalRate tăng đột ngột + byte_rate tăng + mixed (elephant+mice)
L3/L4 signature: packet_count cao + diverse flow sizes
```

**WRR chết thế nào:**
```
WRR: Cứ đến lượt thì ném, không biết h8 đang có 50 requests dở dang
→ h8 overload → h7 nhàn rỗi → throughput không tối ưu
```

**AI LOR Strategy (Emergent Behavior):**
```
AI Observe: h8.active_requests = 50, h7.active_requests = 10, h5.active_requests = 5
AI Action: Route request mới vào h7 (active_requests = 10, thấp nhất)
AI Reward: +High(Throughput) vì giảm queuing, +High(QoS) vì giảm latency
AI Learn: "Đừng đếm băng thông, hãy đếm requests đang xử lý"
```

### 2.4 Consistent Hashing + Dedicated Lane — Video Conference

**Bản chất:** Video stream cần stability, không thể nhảy node liên tục.

**AI nhận diện:**
```
Pattern: UDP traffic + packet_rate ổn định + duration dài + latency < 50ms
L3/L4 signature: UDP flows với stable byte_rate
```

**WRR chết thế nào:**
```
WRR: Round robin các request vào các node
→ Video stream nhảy h8→h7→h8→h5
→ Jitter cao, packet loss, video giật lag
```

**AI Consistent Hashing Strategy (Emergent Behavior):**
```
AI Observe: Flow ID X đang trên h8, latency stable 30ms
AI Action: 
  - Hash(Flow_ID) → luôn map vào same node (h8)
  - VIP rooms → h8 only
  - Small rooms → h7 only
  - Text/heartbeat → h5 (nếu cần)
AI Reward: +High(QoS) vì jitter=0, +High(Throughput) vì không re-route
AI Learn: "Video stream cần stability, đừng có nhảy lung tung"
```

---

## 3. Reward Function Chính thức

### 3.1 Công thức Tổng quát

```
R(s_t, a_t, s_{t+1}) = 
    W₁ · Throughput(s_{t+1})                           // Thưởng throughput cao
  - W₂ · Tail_Latency(s_{t+1})                         // Phạt latency đuôi
  - W₃ · Overload_Penalty(s_{t+1}, a_t)               // Phạt overload nặng
  - W₄ · Node_Crash_Penalty(s_{t+1}, a_t)             // Phạt CHÁY NODE (tai họa)
  - W₅ · Action_Churn(a_t, a_{t-1})                    // Phạt nhảy đổi action liên tục
  - W₆ · Fairness_Deviation(s_{t+1})                   // Phạt lệch fairness
```

### 3.2 Chi tiết từng thành phần

#### W₁ · Throughput
```python
# Normalized goodput — chỉ đếm request thật, không đếm "rác"
throughput_score = sum(successful_requests) / total_time_window
# Đạt điểm cao khi: high throughput + low packet loss
```

#### W₂ · Tail_Latency
```python
# p99 latency — đo lường "đuôi" trễ
# WRR để lọt 10% user vào h5 → p99 cao ngất ngưởng
# AI điều phối đúng → p99 phẳng lì
tail_latency_score = -alpha * p99_latency
```

#### W₃ · Overload_Penalty
```python
# Khi node vượt ngưỡng 85% capacity
if node_utilization > 0.85:
    overload_penalty = (node_utilization - 0.85) * W3
    # Càng quá ngưỡng nhiều → phạt càng nặng
```

#### W₄ · Node_Crash_Penalty (CỰC NẶNG)
```python
# Nếu AI đẩy quá nhiều request vào node yếu → node chết
# Phạt theo cấp số nhân để AI không bao giờ để chuyện này xảy ra
if node_crashed:
    crash_penalty = W4 * (capacity_weight ** 2)  # h8 crash = phạt 100 lần
    # W4 >> W1 để AI ưu tiên tránh crash hơn là đạt throughput
```

#### W₅ · Action_Churn
```python
# Phạt nếu action thay đổi liên tục (instability)
churn_penalty = W5 * (a_t != a_{t-1}) * churn_weight
# Video conference cần stability → churn penalty cao
# DoS detection cần flexibility → churn penalty thấp
```

#### W₆ · Fairness_Deviation
```python
# Độ lệch so với capacity-weighted optimal
# Target: h5=6.25%, h7=31.25%, h8=62.5%
fairness_dev = sum(|action_dist[i] - capacity_ratio[i]|) / 2
```

### 3.3 Trọng số Khuyến nghị

```python
WEIGHTS = {
    "W1_throughput": 1.0,         # Base throughput reward
    "W2_tail_latency": 2.0,       # Ưu tiên latency thấp
    "W3_overload": 5.0,           # Tránh overload nặng
    "W4_crash": 100.0,            # NGUY HIỂM: Không bao giờ được crash node
    "W5_churn": 0.3,             # Stability (giảm cho DoS scenario)
    "W6_fairness": 0.5,          # Không quá strict — để AI tự do
}
```

### 3.4 Scenario-Specific Reward Shaping

```python
REWARD_SHAPING = {
    "golden_hour": {
        "W1": 1.5,   # Tăng throughput bonus
        "W2": 2.0,   # Tăng latency penalty
        "W5": 0.1,   # Giảm churn penalty (cần linh hoạt)
    },
    "video_conference": {
        "W2": 3.0,   # CỰC nặng latency penalty
        "W5": 0.8,   # CỰC nặng churn penalty (cần stability)
    },
    "hardware_degradation": {
        "W3": 8.0,   # Nặng overload penalty
        "W4": 150.0, # Nặng hơn crash penalty (node đã yếu)
    },
    "low_rate_dos": {
        "W2": 1.0,   # Giảm latency penalty (chấp nhận h5 chậm)
        "W4": 50.0,  # Giảm crash penalty (h5 không quan trọng)
        "W5": 0.05,  # Gần như không phạt churn (cần flexibility)
    },
}
```

---

## 4. 4 Kịch bản Chi tiết

### 4.1 Kịch bản 1: Golden Hour (Burst Traffic)

**Mô tả:** Đồng thời cả Elephant flows (file upload nặng) + Mice flows (login/register đông).

**AI Observability:**
```
- arrivalRate: 5 → 150 → 300 (tăng x60)
- Mixed flow sizes: 10KB (Mice) + 10MB (Elephant)
- Target: h8 (100Mbps) + h7 (50Mbps) + h5 (10Mbps)
```

**AI Decision thay vì WRR:**
```
WRR: Chia theo tỷ lệ 1:5:10 bất chấp active requests
AI:  Least Outstanding Requests (LOR)
     → Đếm active requests trên mỗi node
     → Ném vào node có active_requests thấp nhất
```

**Luật giao chiến cho AI:**
```
Rule: Khi arrivalRate > 200
  → Chuyển sang "Burst Mode"
  → Ưu tiên h8 cho Elephant flows (capacity cao)
  → h7 cho Mice flows (cân bằng)
  → h5 tuyệt đối không nhận Elephant
  → Khi h8/h7 > 85% capacity → Kích hoạt Load Shedding
     → Trả về 503 cho request ưu tiên thấp
```

**Emergent Behaviors kỳ vọng:**
- AI tự học phân luồng Elephant/Mice
- AI tự kích hoạt Load Shedding khi cần
- AI giảm p99 latency so với WRR

**Metrics đo lường:**
| Metric | WRR kỳ vọng | AI kỳ vọng |
|--------|-------------|------------|
| p99 Latency | Cao (10% lọt h5) | Thấp (AI tránh h5) |
| Throughput | 0.7 | 0.85 |
| Queue Length | 50 avg | 20 avg |
| Overload Rate | 15% | <5% |

### 4.2 Kịch bản 2: Video Conference (Low Latency)

**Mô tả:** UDP streams cần ổn định, latency < 50ms, jitter ≈ 0.

**AI Observability:**
```
- Protocol: UDP (không phải TCP)
- Packet rate: ổn định 60fps
- Duration: Long-lived connections
- Target: h8 (VIP) + h7 (small rooms)
```

**AI Decision thay vì WRR:**
```
WRR: Nhảy node liên tục → Jitter cao, video giật
AI:  Consistent Hashing + Dedicated Lane
     → Hash(Room_ID) → cố định vào 1 node
     → Không re-route trong suốt session
```

**Luật giao chiến cho AI:**
```
Rule: Khi protocol = UDP AND packet_rate > threshold
  → Dedicated Lane Mode
  → VIP rooms (participants > 10) → h8 only
  → Small rooms (participants <= 10) → h7
  → Text/heartbeat → h5
  → Tuyệt đối không nhảy node trong session
```

**Emergent Behaviors kỳ vọng:**
- AI tự học "sticky routing" cho video
- Jitter ≈ 0 (không re-route)
- Packet loss < 1%

**Metrics đo lường:**
| Metric | WRR kỳ vọng | AI kỳ vọng |
|--------|-------------|------------|
| Jitter | Cao (nhảy node) | ≈ 0 |
| p99 Latency | 150ms | <50ms |
| Packet Loss | 5% | <1% |
| Video Quality | Giật lag | Mượt mà |

### 4.3 Kịch bản 3: Hardware Degradation (Suy giảm từ từ)

**Mô tả:** Node h8 bắt đầu throttling (CPU>95%, IOPS tăng, latency tăng 20ms→500ms).

**AI Observability:**
```
- CPU: 60% → 95% (tăng dần)
- IOPS: Baseline → 2x
- Latency EWMA: 20ms → 200ms → 500ms
- Chưa crash nhưng suy giảm
```

**AI Decision thay vì WRR:**
```
WRR: Không biết h8 đang "hấp hơi", vẫn nhồi 60% vào h8
AI:  EWMA Latency Routing + Draining
     → Nhìn vào latency trend, không phải instant value
     → Bóp vòi từ từ, không ngắt rụp
```

**Luật giao chiến cho AI:**
```
Rule: Khi latency_trend > 0.1 (EWMA increase)
  → Phase 1: Giảm h8 weight 60% → 40%
  → Phase 2: Giảm tiếp 40% → 20%
  → Phase 3: DRAINING mode
     → Chỉ xử lý request cũ
     → Không nhận request mới
  → Backup: h7 gánh tạm
  → Alert: Gọi DevOps
```

**Emergent Behaviors kỳ vọng:**
- AI phát hiện degradation sớm (trước crash)
- Smooth transition (không drop requests)
- TTM (Time To Mitigate) < 100ms

**Metrics đo lường:**
| Metric | WRR kỳ vọng | AI kỳ vọng |
|--------|-------------|------------|
| Error Rate (5xx) | 20% | <1% |
| TTM | ∞ (DevOps manual) | <100ms |
| Requests Dropped | 50 | <5 |
| Recovery Time | Manual | Auto <30s |

### 4.4 Kịch bản 4: Low Rate DoS (Slowloris)

**Mô tả:** Botnet mở nhiều TCP connection nhưng gửi 1 byte/giây, ngâm server.

**AI Observability:**
```
- TCP connections: 1000+ (cao bất thường)
- Byte rate: 1 byte/s per connection (cực thấp)
- Packet count: Cao (vì nhiều tiny packets)
- Pattern: connections >> bytes
```

**AI Decision thay vì WRR:**
```
WRR: Chia đều rác cho h8/h7/h5
     → h8 bị ngâm, Goodput giảm 33%
     → Khách thật không vào được
AI:  Tarpit Strategy
     → Redirect tất cả "rác" vào h5
     → h5 trở thành "Nhà tù"
     → h8/h7 sạch bóng, Goodput = 100%
```

**Luật giao chiến cho AI:**
```
Rule: Khi detect_pattern = "slowloris"
  → Tarpit Mode
  → IP có behavior: connections_high AND bytes_low
  → Redirect ALL to h5 (hố bùn)
  → h5 set timeout dài (300s)
  → h5 response: placeholder/loading
  → h8/h7: Tuyệt đối sạch sẽ
```

**Emergent Behaviors kỳ vọng:**
- AI tự phát hiện anomalous pattern
- Auto-redirect to h5
- Goodput on h8/h7 ≈ 100%

**Metrics đo lường:**
| Metric | WRR kỳ vọng | AI kỳ vọng |
|--------|-------------|------------|
| Goodput (h8/h7) | 67% (33% bị ngâm) | 100% |
| Attack Traffic on h8 | 33% | 0% |
| Legitimate User Success | 70% | 99% |
| Attack Mitigation | Manual | Auto <50ms |

---

## 5. Metrics & Tools

### 5.1 Golden Signals (SRE)

```yaml
User Experience Metrics:
  - Tail Latency: p95, p99, p999
  - Jitter: (đặc biệt cho video)
  - Packet Loss Rate: %
  - Success Rate: %

Performance Metrics:
  - Goodput: (throughput - garbage)
  - Queue Length: avg, max
  - Connection Queuing Time: ms

Reliability Metrics:
  - Error Rate: 5xx count
  - Time To Mitigate (TTM): ms
  - Node Uptime: %
```

### 5.2 Tools cho Lab Environment

| Mục đích | Tool | Ghi chú |
|----------|------|---------|
| Load Generator | Artillery | Đã có config trong `lms/evaluation/` |
| Network Stats | Ryu Controller | StatsCollection |
| Latency/Errors | Artillery JSON logs | Parse p95/p99 |
| Visualization | Matplotlib/Seaborn | Charts cho báo cáo |
| Statistical Test | SciPy (Wilcoxon) | Significance testing |

### 5.3 Artiller Config Parameters cho 4 Kịch bản

```yaml
# Golden Hour - burst_traffic.yml
phases:
  - arrivalRate: 5      # Baseline
  - arrivalRate: 150    # Golden Hour start
  - arrivalRate: 300    # Peak flash crowd
  - rampTo: 50          # Sudden drop
  - rampTo: 200         # Ramp up
scenarios:
  - weight: 70  # Mice flows
  - weight: 30  # Elephant flows

# Video Conference - video_conference.yml  
phases:
  - arrivalRate: 50     # Video active
  - arrivalRate: 150    # Conflict: backup + video
scenarios:
  - weight: 30  # Video (high priority)
  - weight: 70  # Backup (low priority)

# Hardware Degradation - hardware_degradation.yml
phases:
  - arrivalRate: 20     # Normal
  - arrivalRate: 100    # h8 degrading
  - arrivalRate: 250    # Critical throttling
scenarios:
  - weight: 40  # Registration (CPU heavy)
  - weight: 30  # Browse
  - weight: 30  # Analytics (CPU intensive)

# Low Rate DoS - low_rate_dos.yml
phases:
  - arrivalRate: 10     # Normal
  - arrivalRate: 100    # Botnet attack
  - arrivalRate: 50     # Mixed: attack + legitimate
scenarios:
  - weight: 40  # Legitimate browse
  - weight: 30  # Legitimate register
  - weight: 30  # Malicious (botnet)
```

---

## 6. TODO List Hoàn chỉnh

### Phase 1: Architecture & Reward Design
- [ ] Cập nhật Reward Function trong `sdn_env_v2.py` với 6 thành phần
- [ ] Thêm scenario-specific reward shaping
- [ ] Verify W4 crash penalty >> W1 throughput (đảm bảo AI không bao giờ crash node)
- [ ] Viết unit test cho Reward Function

### Phase 2: Environment Enhancements
- [ ] Thêm LOR (Least Outstanding Requests) logic vào state
- [ ] Thêm EWMA latency tracking cho degradation detection
- [ ] Thêm Consistent Hashing support cho video scenario
- [ ] Thêm Tarpit detection (connections/bytes ratio)

### Phase 3: Training với Reward Mới
- [ ] Generate synthetic data cho 4 scenarios (đủ diverse)
- [ ] Train với new reward shaping
- [ ] Verify emergent behaviors xuất hiện:
  - [ ] Tarpit behavior (DoS → h5)
  - [ ] Draining behavior (degradation → reduce h8)
  - [ ] LOR behavior (burst → active_requests aware)
  - [ ] Sticky routing (video → consistent node)

### Phase 4: Benchmarking & Validation
- [ ] Chạy Artillery cho 4 scenarios
- [ ] So sánh AI vs WRR:
  - [ ] p99 Latency
  - [ ] Goodput
  - [ ] Error Rate
  - [ ] Jain's Fairness Index
- [ ] Statistical significance test (Wilcoxon)
- [ ] Tạo comparison charts

### Phase 5: Documentation & IEEE Submission
- [ ] Viết Technical Report mô tả Reward Design
- [ ] Tạo benchmark results table (IEEE format)
- [ ] Generate all charts với professional styling
- [ ] Finalize Báo cáo NCKH

### Milestone Checklist

```
[ ] Milestone 1: Reward Function Hoàn chỉnh
[ ] Milestone 2: Training với Emergent Behaviors  
[ ] Milestone 3: 4 Scenarios Benchmark完成
[ ] Milestone 4: IEEE-Compliant Report
[ ] Milestone 5: Final Submission
```

---

## 7. Key Insights từ Cuộc Thảo Luận

### 7.1 Expert System vs RL — Ranh giới mong manh

**ĐIỂM Mấu chốt:**
- Hardcode rule = Expert System, không phải RL
- Reward Function = Chìa khóa để AI tự học
- Emergent Behavior = AI tự "ngộ" ra chiến thuật, không do lập trình viên

### 7.2 AI Position in Architecture

```
AI = Control Plane, không phải Data Plane
├── Hoạt động ở L3/L4 (không nhìn L7 payload)
├── Thu thập Stats qua OpenFlow
├── Điều phối qua Flow Mod messages
└── Không can thiệp vào packet content
```

### 7.3 Metrics Collection Strategy

```
Lab Environment (Mininet + Docker):
├── Network Metrics: Ryu Controller (có sẵn)
├── L7 Metrics: Artillery logs (parse JSON)
└── Visualization: Matplotlib + Grafana (optional)

Production Environment:
├── APM: Datadog/Dynatrace
├── Metrics: Prometheus + Node Exporter
└── Trending: Grafana dashboards
```

### 7.4 So sánh WRR vs AI — Kỳ vọng

| Scenario | WRR Problem | AI Solution | AI Advantage |
|----------|-------------|-------------|--------------|
| Golden Hour | 10% user vào h5 → p99 cao | LOR, tránh h5 cho heavy | p99 giảm 50% |
| Video | Nhảy node → jitter cao | Sticky routing | Jitter ≈ 0 |
| Degradation | Crash h8 | EWMA + Draining | 0 crashes |
| DoS | 33% bandwidth mất | Tarpit → h5 | Goodput 100% |

---

## 8. Artiller Configuration Checklist

Bạn hỏi về Artillery config — Đây là checklist:

### Golden Hour (`golden_hour.yml`)
- [x] Phase 1: baseline (arrivalRate: 5)
- [x] Phase 2: Mice flash (arrivalRate: 150)
- [x] Phase 3: Elephant + Mice (arrivalRate: 300)
- [x] Phase 4: Flash crowd peak
- [x] Phase 5: Sudden drop (test adaptation)
- [x] Phase 6: Ramp up (test prediction)
- [x] Scenarios: 70% Mice + 30% Elephant

### Video Conference (`video_conference.yml`)
- [x] Video session với heartbeat
- [x] Priority traffic (weight: 30)
- [x] Backup service (weight: 70)
- [x] Conflict phase: backup + video đồng thời

### Hardware Degradation (`hardware_degradation.yml`)
- [x] h10 degradation simulation
- [x] Critical throttling phase
- [x] Sudden spike test
- [x] Recovery phase

### Low Rate DoS (`low_rate_dos.yml`)
- [x] Botnet reconnaissance
- [x] Brute force phase
- [x] Mixed: attack + legitimate
- [x] Malicious scenarios với wrong passwords

**Đánh giá:** Config đã đủ "đô" để ép AI bung sức. Không cần tăng intensity.

---

## Kết luận

Plan này tổng hợp toàn bộ kiến thức từ cuộc thảo luận về:
1. **Architecture:** AI là Control Plane, L3/L4, tách biệt khỏi Data Plane
2. **Reward Design:** 6 thành phần với W4 (crash) >> W1 (throughput)
3. **4 Scenarios:** Mỗi scenario cóEmergent Behavior riêng
4. **Metrics:** Golden Signals + Jain's Fairness + Artillery logs
5. **TODO:** Rõ ràng, có milestone, có deadline potential

**Điểm cốt lõi:** Không viết if/else cho AI. Viết Reward Function để AI tự học chiến thuật Tarpit, Draining, LOR, Consistent Hashing qua trial-and-error.

---

*Lần cuối cập nhật: 2026-03-24*
*Status: Ready for Implementation*
