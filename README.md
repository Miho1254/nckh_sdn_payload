<div align="center">
  <img src="https://cdn.haitrieu.com/wp-content/uploads/2021/10/Logo-DH-Thuy-Loi.png" alt="Logo Đại học Thủy lợi" width="120" />
  <p><b>PHÂN HIỆU TRƯỜNG ĐẠI HỌC THỦY LỢI</b></p>
</div>

---

# ĐÁNH GIÁ SỰ ĐÁNH ĐỔI GIỮA HIỆU NĂNG VÀ TÍNH THÍCH NGHI TRONG CÂN BẰNG TẢI SDN SỬ DỤNG HỌC TĂNG CƯỜNG ACTOR-CRITIC

**Phân hiệu Trường Đại học Thủy lợi — Khoa Công nghệ Thông tin**

| | |
|---|---|
| **Giảng viên hướng dẫn** | ThS. Hoàng Văn Quý |
| **Nhóm thực hiện** | Đặng Quang Hiển, Đặng Trọng Phúc, Trương Tuấn Minh, Trần Minh Triết |
| **Năm học** | 2025–2026 |
| **Định dạng** | [IEEE Paper (PDF)](ICT_TienGiang_Hien_SDN_3.pdf) |

---

## TÓM TẮT NGHIÊN CỨU

### 1. Vấn đề nghiên cứu
Các thuật toán cân bằng tải tĩnh (Weighted Round Robin - WRR) không thích ứng được với các tình huống bất thường của hệ thống (server degradation, failure, burst traffic), dẫn đến vi phạm SLA trong mạng SDN.

### 2. Giải pháp đề xuất
Hệ thống **TFT-PPO** với kiến trúc lai:
- **Temporal Fusion Transformer (TFT)**: Mã hóa state 20 chiều bằng LSTM + Multi-Head Attention để trích xuất đặc trưng không gian-thờigian từ OpenFlow PortStats
- **Proximal Policy Optimization (PPO)**: Mô hình Actor-Critic với Clipped Objective (ε=0.2) và advantage estimator GAE (λ=0.95)
- **Hybrid Controller**: WRR xử lý 95% traffic bình thường, PPO chỉ can thiệp khi phát hiện bất thường hoặc utilization vượt ngưỡng 0.95

### 3. Kết quả thực nghiệm
Thực nghiệm trên Mininet/Ryu với topology Fat-Tree K=4 (8 clients → 3 servers 10/50/100 Mbps), 4 kịch bản, n=5 paired runs:

| Kịch bản | So sánh PPO vs WRR | Kết luận |
|----------|-------------------|----------|
| **Hardware Degradation** | +8.6% throughput | PPO vượt trội - thích ứng được sự cố |
| **Golden Hour** | -18.6% throughput | WRR tốt hơn trong điều kiện bình thường |
| **Video Conference** | -16.4% throughput | WRR tốt hơn trong điều kiện bình thường |
| **Low-rate DoS** | -14.7% throughput | WRR tốt hơn trong điều kiện bình thường |

**Tỷ lệ thắng**: PPO 1/4 (25%), WRR 3/4 (75%)

### 4. Luận điểm chính
**"Chi phí của sự thông minh"**: PPO mất 14.7%-18.6% throughput trong điều kiện bình thường, nhưng tăng 8.6% khi server bị suy thoái. Phát hiện này xác định vai trò **SLA Protector** cho PPO - nên hoạt động song song với WRR, chỉ can thiệp khi phát hiện bất thường.

---

## ĐÓNG GÓP CHÍNH

1. **Vai trò SLA Protector**: PPO nên là bảo vệ SLA chứ không thay thế hoàn toàn WRR trong cân bằng tải SDN
2. **Feature engineering**: Thiết kế state space 20 chiều từ OpenFlow PortStats cho môi trường SDN thực
3. **Safety mechanism**: Cơ chế bypass PPO khi utilization vượt ngưỡng 0.95, đảm bảo tính sẵn sàng
4. **Benchmark mở rộng**: Đề xuất 2 kịch bản bổ sung (burst_traffic, server_failure) cho nghiên cứu tiếp theo

---

## THÔNG SỐ KỸ THUẬT

**Môi trường mạng**: Fat-Tree K=4, 8 clients → 3 backend servers (10/50/100 Mbps)  
**Platform**: Docker, Mininet 2.3.0, Ryu Controller 4.34, Artillery.io 2.0  
**Training**: Gymnasium SDNEnv, stable-baselines3 PPO, 500K timesteps (~45 phút trên CPU)  
**Hyperparameters**: lr=3e-4, γ=0.99, ε=0.2, hidden layers [256,256], batch size=64, n_steps=2048  
**Metrics**: Total packets (OpenFlow flow_stats), P99 Latency (Artillery), Jain's Fairness Index  
**Statistical analysis**: Mean ± std, 95% CI (Student-t, n=5, t=2.776)

---

## HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN

**Hạn chế**:
- Variance cao trong một số kịch bản (cần n≥10 runs để kết quả ổn định)
- Sim-to-real gap giữa môi trường Gymnasium và mạng thực
- Inference overhead (PPO có P99 latency cao hơn WRR 16.2% do chi phí tính toán mạng Neural)
- Chỉ đánh giá với 3 server, cần mở rộng quy mô lớn hơn

**Hướng phát triển**:
1. **Knowledge Distillation**: Nén mô hình PPO để giảm độ trễ xuống mức tương đương WRR
2. **Hybrid Controller**: WRR xử lý 95% traffic "sạch", PPO chỉ can thiệp khi cần thiết
3. **Curriculum Learning**: Huấn luyện PPO với các kịch bản từ đơn giản đến phức tạp
4. **XAI (Explainable AI)**: Tích hợp cơ chế giải thích cho các quyết định của Agent
5. **Domain Randomization**: Giảm sim-to-real gap bằng cách tăng đa dạng môi trường huấn luyện

---

## TÀI LIỆU THAM KHẢO

Các kết quả chi tiết, phân tích thống kê, và bằng chứng thực nghiệm đầy đủ trong file gốc: **[ICT_TienGiang_Hien_SDN_3.pdf](ICT_TienGiang_Hien_SDN_3.pdf)** (IEEE Computer Society Format, 2026)

Tài liệu tham khảo chính: [1] McKeown et al. (OpenFlow), [2] Schulman et al. (PPO), [3] Lim et al. (TFT), [4] Jain et al. (Fairness), [5] Wang et al. (Dueling Network), [6] Mnih et al. (Async RL), [7] Sutton & Barto (RL Book), [8] Pfaff et al. (Open vSwitch), [9] Sharma et al. (Deep RL Load Balancing)

*Lưu ý: README này trích dẫn file gốc IEEE, xem PDF để có đầy đủ chi tiết kỹ thuật, phương pháp luận, và phân tích thống kê chi tiết.*

---

> **Trích dẫn**: Nếu sử dụng nội dung này, vui lòng trích dẫn file PDF gốc: `ICT_TienGiang_Hien_SDN_3.pdf` (IEEE Format, 2026)
