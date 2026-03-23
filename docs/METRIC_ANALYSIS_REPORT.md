# Phân Tích Sâu Về Metric Đánh Giá TFT-CQL

## Ngày: 2026-03-22

## 1. Vấn Đề Phát Hiện

### 1.1 Throughput và Overload Rate Bằng Nhau

**Kết quả benchmark:**
- Throughput: AI = WRR = 0.0098 (tất cả scenarios)
- Overload Rate: AI = WRR = 0.0000 (tất cả scenarios)

**Nguyên nhân:**
```python
# Trong sdn_env_v2.py, line 105
throughput = byte_rate + packet_rate  # Từ data, không phụ thuộc action
```

**Phân tích:**
- `byte_rate` và `packet_rate` được extract từ data features (X[:, -1, 0] và X[:, -1, 1])
- Giá trị này **không thay đổi** dựa trên action được chọn
- Do đó, throughput **không phân biệt được** AI vs WRR

### 1.2 Overload Rate = 0%

**Phân tích data:**
```
Data shape: X=(24, 5, 44), y=(24,)
Server 0 (h5): Util > 0.95: 3 samples (12.5%)
Server 1 (h7): Util > 0.95: 0 samples
Server 2 (h8): Util > 0.95: 0 samples
```

**Vấn đề:**
- Chỉ có 3/24 samples có util > 0.95 trên h5
- AI và WRR đều tránh chọn h5 khi util cao
- Kết quả: Overload Rate = 0% cho cả hai

### 1.3 WRR Distribution Không Đúng Capacity Ratio

**Kết quả:**
```
WRR Action Distribution: h5=8.3%, h7=41.7%, h8=50.0%
Target Capacity Ratio:   h5=6.25%, h7=31.25%, h8=62.5%
```

**Nguyên nhân:**
- WRR cycle: [0] * 1 + [1] * 5 + [2] * 10 = 16 actions
- 24 samples / 16 actions = 1.5 cycles
- Cycle 1: h5=1, h7=5, h8=10
- Cycle 2 (8 samples): h5=1, h7=5, h8=2
- Total: h5=2, h7=10, h8=12 → 8.33%, 41.67%, 50%

## 2. Metric Mới Đề Xuất

### 2.1 Capacity Weighted Score

**Công thức:**
```python
capacities = [1, 5, 10]  # h5=1, h7=5, h8=10
capacity_weighted = sum(action_dist[i] * capacities[i])
target = 7.875  # 0.0625*1 + 0.3125*5 + 0.625*10
```

**Ý nghĩa:**
- Đo lường mức độ "hiệu quả" của action distribution
- Cao hơn = chọn server mạnh hơn
- Target = 7.875 (optimal theo capacity ratio)

### 2.2 Utilization Variance

**Công thức:**
```python
util_variance = mean(var(server_utils, axis=1))
```

**Ý nghĩa:**
- Đo lường độ lệch utilization giữa các servers
- Thấp hơn = cân bằng tải tốt hơn

### 2.3 Average Chosen Utilization

**Công thức:**
```python
avg_chosen_util = mean(chosen_utils)
```

**Ý nghĩa:**
- Utilization trung bình của server được chọn
- Thấp hơn = chọn server ít tải hơn
- Phản ánh khả năng tránh server quá tải

## 3. Kết Quả Benchmark Mới

### 3.1 Bảng Kết Quả

| Scenario | AI Fairness Dev | WRR Fairness Dev | AI Cap Weighted | WRR Cap Weighted | AI Avg Util | WRR Avg Util | Winner |
|----------|-----------------|------------------|-----------------|------------------|-------------|--------------|--------|
| Golden Hour | 0.1667 | 0.1250 | 6.9583 | 7.1667 | 0.0593 | 0.0475 | WRR |
| Video Conference | 0.1667 | 0.1250 | 6.9583 | 7.1667 | 0.0444 | 0.0475 | WRR |
| Hardware Degradation | 0.1042 | 0.1250 | 7.5417 | 7.1667 | 0.0318 | 0.0475 | **AI** |
| Low Rate DoS | 0.0208 | 0.1250 | 7.9583 | 7.1667 | 0.0302 | 0.0475 | **AI** |

### 3.2 Phân Tích Kết Quả

**AI thắng ở 2/4 scenarios:**
- **Hardware Degradation**: AI có Fairness Dev thấp hơn (0.1042 vs 0.1250), Capacity Weighted cao hơn (7.5417 vs 7.1667), Avg Util thấp hơn (0.0318 vs 0.0475)
- **Low Rate DoS**: AI có Fairness Dev thấp hơn (0.0208 vs 0.1250), Capacity Weighted cao hơn (7.9583 vs 7.1667), Avg Util thấp hơn (0.0302 vs 0.0475)

**AI thua ở 2/4 scenarios:**
- **Golden Hour & Video Conference**: AI có Fairness Dev cao hơn (0.1667 vs 0.1250)
- Lý do: Stochastic sampling gây variance trong action distribution

### 3.3 Action Distribution

| Server | Target | WRR | AI (Golden Hour) | AI (Low Rate DoS) |
|--------|--------|-----|------------------|-------------------|
| h5 | 6.25% | 8.3% | 8.3% | 4.2% |
| h7 | 31.25% | 41.7% | 45.8% | 33.3% |
| h8 | 62.5% | 50.0% | 45.8% | 62.5% |

**Nhận xét:**
- AI trong Low Rate DoS scenario rất gần target: h5=4.2%, h7=33.3%, h8=62.5%
- AI trong Golden Hour scenario lệch xa target: h5=8.3%, h7=45.8%, h8=45.8%

## 4. Hạn Chế Của Data Hiện Tại

### 4.1 Số Lượng Samples

- **Hiện tại**: 24 samples
- **Vấn đề**: Quá ít để đánh giá đúng
- **Giải pháp**: Cần ít nhất 1000+ samples

### 4.2 Phân Bổ Utilization

- **h5**: 3/24 samples có util > 0.95
- **h7, h8**: 0 samples có util > 0.95
- **Vấn đề**: Không có tình huống overload thực sự

### 4.3 Throughput Không Phụ Thuộc Action

- **Hiện tại**: Throughput = byte_rate + packet_rate từ data
- **Vấn đề**: Không phản ánh quyết định action
- **Giải pháp**: Cần mô phỏng throughput thực tế dựa trên server capacity

## 5. Đề Xuất Cải Thiện

### 5.1 Thêm Metric Phản Ánh Action

```python
# Đề xuất: Throughput thực tế
def compute_real_throughput(action, server_utils, byte_rate):
    """Throughput thực tế dựa trên server capacity và load."""
    capacities = [1, 5, 10]  # h5=1, h7=5, h8=10
    headroom = 1 - server_utils[action]
    effective_capacity = capacities[action] * headroom
    return byte_rate * effective_capacity
```

### 5.2 Mở Rộng Data

- Thu thập thêm data với các tình huống:
  - Server overload
  - Traffic burst
  - Hardware degradation
  - DoS attack

### 5.3 Metric Tổng Hợp Mới

```python
composite_score = (
    + capacity_weighted_score  # Cao hơn = tốt
    - fairness_deviation        # Thấp hơn = tốt
    - avg_chosen_util          # Thấp hơn = tốt
    - overload_rate * 2        # Penalty cho overload
)
```

## 6. Kết Luận

1. **Throughput và Overload Rate hiện tại không phân biệt được AI vs WRR** vì:
   - Throughput từ data, không phụ thuộc action
   - Overload Rate = 0% vì data không có tình huống overload thực sự

2. **Metric mới đề xuất** (Capacity Weighted, Util Variance, Avg Chosen Util) phản ánh tốt hơn sự khác biệt giữa AI và WRR

3. **AI thắng ở 2/4 scenarios** với metric mới, đặc biệt tốt trong Low Rate DoS scenario

4. **Cần mở rộng data** để đánh giá chính xác hơn

5. **Cần sửa cách tính throughput** để phản ánh quyết định action