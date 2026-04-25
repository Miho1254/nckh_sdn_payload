# Phân tích Packet Distribution cho Hardware Degradation Scenario

## Mục tiêu
Xác định packet distribution tới các hosts h5, h7, h8 cho cả PPO và WRR algorithms trong `hardware_degradation` scenario để tính Jain's Fairness Index.

## Thông tin đã thu thập

### 1. Ánh xạ Server
| Host | IP | Action | Weight | Capacity |
|------|-----|--------|--------|----------|
| h5 | 10.0.0.5 | 0 | 1 | 10 Mbps |
| h7 | 10.0.0.7 | 1 | 5 | 50 Mbps |
| h8 | 10.0.0.8 | 2 | 10 | 100 Mbps |

### 2. Cấu trúc dữ liệu
- **PPO**: `benchmark_results_quick/hardware_degradation/run_1/ppo/inference_log.csv` chứa action logs
- **WRR**: `benchmark_results_quick/hardware_degradation/run_1/wrr/` - chỉ có flow_stats.csv và port_stats.csv (không có inference_log)

### 3. WRR Action Distribution (deterministic)
WRR cycle: `[0]*1 + [1]*5 + [2]*10` = 16 actions/chu kỳ
- h5 (action 0): 1/16 = 6.25%
- h7 (action 1): 5/16 = 31.25%
- h8 (action 2): 10/16 = 62.5%

## Phân tích PPO Actions
Từ inference_log.csv, đếm số lần mỗi action được chọn:
- Action 0 (h5): count
- Action 1 (h7): count
- Action 2 (h8): count

## Jain's Fairness Index Formula
```
J = (Σx_i)² / (n × Σx_i²)
```
Trong đó x_i = throughput của server i

## Execute Plan
1. Đọc và phân tích PPO inference_log.csv
2. Xác định WRR action distribution (theo cycle)
3. Tính Jain's Fairness Index cho cả hai
4. Tạo báo cáo so sánh chi tiết

## Mermaid: Analysis Flow
```mermaid
flowchart TD
    A[hardware_degradation scenario] --> B[PPO: inference_log.csv]
    A --> C[WRR: deterministic cycle]
    B --> D[Count actions 0, 1, 2]
    C --> E[Calculate expected distribution]
    D --> F[Compute Jain's Index PPO]
    E --> G[Compute Jain's Index WRR]
    F --> H[Compare & Report]
    G --> H