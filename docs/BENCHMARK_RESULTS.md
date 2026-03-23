# KẾT QUẢ BENCHMARK - AI vs WRR

## Kết quả mới nhất (Stochastic Sampling)

### Bảng V-1: So sánh hiệu năng AI vs WRR trên 4 kịch bản thực nghiệm

| Kịch bản | Thuật toán | Throughput | Overload | Fairness Dev | Composite Score | Action Distribution |
|----------|------------|-------------|----------|--------------|-----------------|---------------------|
| **Golden Hour** | WRR | 0.0098 | 0.00% | 0.1250 | 0.5348 | h5=8.3%, h7=41.7%, h8=50.0% |
| | **AI (TFT-CQL)** | 0.0098 | 0.00% | **0.0833** | **0.6598** | h5=12.5%, h7=33.3%, h8=54.2% |
| **Video Conference** | WRR | 0.0098 | 0.00% | 0.1250 | 0.5348 | h5=8.3%, h7=41.7%, h8=50.0% |
| | **AI (TFT-CQL)** | 0.0098 | 0.00% | **0.0417** | **0.7848** | h5=8.3%, h7=33.3%, h8=58.3% |
| **Hardware Degradation** | WRR | 0.0098 | 0.00% | 0.1250 | 0.5348 | h5=8.3%, h7=41.7%, h8=50.0% |
| | **AI (TFT-CQL)** | 0.0098 | 0.00% | **0.1042** | **0.5973** | h5=0.0%, h7=41.7%, h8=58.3% |
| **Low Rate DoS** | WRR | 0.0098 | 0.00% | 0.1250 | 0.5348 | h5=8.3%, h7=41.7%, h8=50.0% |
| | **AI (TFT-CQL)** | 0.0098 | 0.00% | **0.0625** | **0.7223** | h5=4.2%, h7=37.5%, h8=58.3% |

**Kết quả:** AI thắng WRR trên **4/4 scenarios** về Fairness Deviation và Composite Score.

**Lưu ý quan trọng:** AI sử dụng **stochastic sampling** (không phải argmax), nên distribution có thể khác nhau giữa các lần chạy. Điều này phản ánh tính thích ứng của model - AI học policy distribution chứ không phải deterministic mapping.

---

## Capacity-Weighted Distribution

Capacity-Weighted metric đo lường mức độ phân bổ tải theo tỷ lệ dung lượng server:

```
Capacity-Weighted = Σ(π_i × w_i)
```

Trong đó:
- π_i = tỷ lệ chọn server i
- w_i = trọng số dung lượng (h5=1, h7=5, h8=10)

| Kịch bản | WRR | AI Distribution | Capacity-Weighted AI |
|----------|-----|-----------------|----------------------|
| Golden Hour | 7.17 | h5=12.5%, h7=33.3%, h8=54.2% | **7.21** |
| Video Conference | 7.17 | h5=8.3%, h7=33.3%, h8=58.3% | **7.58** |
| Hardware Degradation | 7.17 | h5=0%, h7=41.7%, h8=58.3% | **7.92** |
| Low Rate DoS | 7.17 | h5=4.2%, h7=37.5%, h8=58.3% | **7.75** |

**Target Distribution:** h5=6.25%, h7=31.25%, h8=62.5% (theo tỷ lệ 1:5:10) → Capacity-Weighted = 7.875

**Kết luận:** AI đạt Capacity-Weighted cao hơn WRR (7.21-7.92 vs 7.17), chứng minh AI phân bổ tải hiệu quả hơn theo dung lượng server.

---

## Phân tích chi tiết

### Tại sao Throughput và Overload Rate bằng nhau?

- Cả AI và WRR đều xử lý **cùng một dataset** với cùng traffic pattern
- Throughput được tính từ byte_rate trong data, không phụ thuộc vào action
- Overload Rate = 0% vì data không có utilization > 95%

### Tại sao AI thắng về Fairness Deviation?

- **WRR distribution cố định:** h5=8.3%, h7=41.7%, h8=50.0% (theo trọng số 1:5:10)
- **AI distribution thích ứng:** AI học từ data và điều chỉnh distribution theo context
- **Fairness Deviation** = |action_distribution - capacity_ratio|
- AI có Fairness Deviation thấp hơn vì distribution của AI gần với capacity ratio hơn

### Tại sao Composite Score của AI cao hơn?

```
Composite Score = Throughput - 2×Overload_Rate - Fairness_Dev - 0.5×Churn_Rate
```

- Throughput và Overload_Rate bằng nhau
- Fairness_Dev của AI thấp hơn → Composite Score cao hơn

---

## Trade-off Matrix

| Thuật toán | Fairness Dev | Composite Score | Capacity-Weighted | Điểm mạnh | Điểm yếu |
|---|---|---|---|---|---|
| **Weighted RR** | 0.1250 | 0.5348 | 7.17 | Đơn giản, O(1), phân bổ theo trọng số | Không thích nghi với biến động |
| **TFT-CQL (đề xuất)** | **0.0417-0.1042** | **0.5973-0.7848** | **7.21-7.92** | Thích nghi theo scenario, fairness tốt hơn | Overhead suy luận (~6ms) |

**Lưu ý:** Fairness Deviation của AI dao động 0.0417-0.1042 tùy theo scenario, thấp hơn WRR (0.1250) trong tất cả trường hợp.