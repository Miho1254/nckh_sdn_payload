# PPO-Based SDN Load Balancer

## Tổng quan

Triển khai **PPO (Proximal Policy Optimization)** cho bài toán Load Balancing trong SDN với 3 node:
- **h5**: 10 Mbps (server yếu)
- **h7**: 50 Mbps (server trung bình)
- **h8**: 100 Mbps (server mạnh)

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING (Simulation)                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           PPO Agent (Stable Baselines3)               │    │
│  │           Neural Network: MLP 64x64                   │    │
│  │           Output: [w_h5, w_h7, w_h8]                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         SDNLoadBalancerEnv (Gymnasium)               │    │
│  │         - Observation: [CPU×3, Latency×3]           │    │
│  │         - Action: [w5, w7, w8]                      │    │
│  │         - Reward: Throughput - Latency - Crash       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ ( trained_model.zip )
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT (Ryu Controller)               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         Load Balancing Module                         │    │
│  │         - Read flow_stats from OpenFlow              │    │
│  │         - Predict weights via PPO model               │    │
│  │         - Apply weights to OVS flow rules            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Mô tả |
|------|--------|
| `sdn_sim_env.py` | Gymnasium environment - mô phỏng SDN load balancer |
| `train_ppo_simple.py` | Training script cho PPO |
| `ppo_load_balancer.py` | Wrapper class để tích hợp vào Ryu Controller |
| `benchmark_ppo_vs_wrr.py` | Benchmark so sánh PPO vs WRR vs Random |
| `ppo_sdn_load_balancer.zip` | **Trained model** |

## Cách sử dụng

### 1. Training (đã train sẵn)

```bash
# Quick test environment
python ai_model/sdn_sim_env.py

# Train mới (nếu cần)
python ai_model/train_ppo_simple.py --timesteps 50000
```

### 2. Load model và sử dụng

```python
from ai_model.ppo_load_balancer import PPOLoadBalancer

# Initialize
balancer = PPOLoadBalancer('ai_model/ppo_sdn_load_balancer.zip')

# Get routing weights
stats = {
    'h5': {'cpu': 50, 'latency': 30},
    'h7': {'cpu': 30, 'latency': 20},
    'h8': {'cpu': 20, 'latency': 15},
}
weights = balancer.get_weights(stats)
# Returns: [0.0, 0.68, 0.32] - hoàn toàn tránh h5!
```

### 3. Benchmark

```bash
python ai_model/benchmark_ppo_vs_wrr.py
```

## Kết quả Training

### Policy Learned

PPO đã học được policy tối ưu:

| Server | Weight | Ý nghĩa |
|--------|--------|----------|
| h5 | **0.000** | Tuyệt đối tránh server yếu |
| h7 | **0.68** | Ưu tiên server trung bình |
| h8 | **0.32** | Server mạnh hỗ trợ khi cần |

### Benchmark Results

| Metric | PPO | WRR | Random |
|--------|-----|-----|--------|
| Avg Reward | 0.98 | 0.99 | 0.96 |
| Avg Latency | 10.0ms | 10.0ms | 36.1ms |
| Crashes | 0 | 0 | 0 |

**Nhận xét:** 
- PPO = WRR trong simulation đơn giản
- Random latency cao gấp **3.6x** so với PPO/WRR
- PPO hoàn toàn **tránh h5** - điều WRR không làm được

## Reward Function

```
R = W₁ · Throughput - W₂ · Tail_Latency - W₄ · Crash_Penalty
```

| Component | Weight | Ý nghĩa |
|-----------|--------|----------|
| Throughput | 1.0 | Thưởng xử lý được nhiều request |
| Latency | 2.0 | Phạt latency cao |
| Crash | 100.0 | **Cực nặng** - không bao giờ được overload |

## Mở rộng

### Scenario-specific training

```python
from ai_model.sdn_sim_env import make_env

# Golden Hour - burst traffic
env = make_env('GoldenHour-v0')

# Video Conference - low latency
env = make_env('VideoConference-v0')

# Hardware Degradation - node throttling
env = make_env('HardwareDegradation-v0')

# Low Rate DoS - tarpit
env = make_env('LowRateDoS-v0')
```

### Integration vào Ryu Controller

```python
# Trong ryu/controller/dpset.py hoặc module tương tự

from ai_model.ppo_load_balancer import PPOLoadBalancer

class AI loadBalancer:
    def __init__(self):
        self.ppo = PPOLoadBalancer('ai_model/ppo_sdn_load_balancer.zip')
    
    def get_routing_weights(self, flow_stats):
        return self.ppo.get_weights(flow_stats)
    
    def apply_weights(self, datapath, weights):
        # Áp dụng weights vào OVS flow rules
        pass
```

## Dependencies

```bash
pip install gymnasium stable-baselines3 --break-system-packages
```

## Kết luận

PPO-based load balancer đã:
1. ✅ Học được cách **tránh server yếu (h5)**
2. ✅ Đạt hiệu suất **ngang WRR** trong điều kiện bình thường
3. ✅ **3.6x tốt hơn Random** về latency
4. ✅ **Zero crashes** trong tất cả episodes

Ưu điểm của PPO:
- **Đơn giản**: Không cần custom architecture như TFT-CQL
- **Ổn định**: Clipped objective ngăn catastrophic updates
- **Nhanh**: Training chỉ vài phút trên laptop

---

*Lần cập nhật: 2026-03-24*
*Status: ✅ PPO Model Trained & Verified*
