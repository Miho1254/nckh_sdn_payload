<div align="center">
  <img src="https://cdn.haitrieu.com/wp-content/uploads/2021/10/Logo-DH-Thuy-Loi.png" alt="Logo Đại học Thủy lợi" width="120" />
  <p><b>PHÂN HIỆU TRƯỜNG ĐẠI HỌC THỦY LỢI</b></p>
</div>

---

# Đánh giá sự đánh đổi giữa hiệu năng và tính thích nghi trong cân bằng tải SDN sử dụng Học tăng cường Actor-Critic

**Phân hiệu Trường Đại học Thủy lợi — Khoa Công nghệ Thông tin**

| | |
|---|---|
| **Giảng viên hướng dẫn** | ThS. Hoàng Văn Quý |
| **Nhóm thực hiện** | Đặng Quang Hiển, Đặng Trọng Phúc, Trương Tuấn Minh, Trần Minh Triết |
| **Năm học** | 2025–2026 |
| **Định dạng** | [IEEE Paper (PDF)](ICT_TienGiang_Hien_SDN_3.pdf) |

---

## 📄 Tóm Tắt Nghiên Cứu

### Vấn Đề
Các thuật toán cân bằng tải tĩnh (WRR) thất bại khi gặp sự cố server (degradation, failure, burst traffic), gây vi phạm SLA trong các hệ thống SDN.

### Giải Pháp Đề Xuất
Hệ thống **TFT-PPO** kết hợp:
- **Temporal Fusion Transformer (TFT)**: Mã hóa state 20 chiều bằng LSTM + Multi-Head Attention
- **Proximal Policy Optimization (PPO)**: Actor-Critic với Clipped Objective, safety override threshold 0.95
- **Hybrid Architecture**: WRR (baseline) + PPO Override (khi phát hiện bất thường)

### Kết Quả Thực Nghiệm (Mininet/Ryu, 4 kịch bản, n=5 runs)

| Kịch bản | PPO vs WRR | Kết luận |
|----------|-----------|----------|
| Hardware Degradation | **+8.6%** throughput | ✅ **PPO vượt trội** - thích ứng được sự cố |
| Golden Hour | **-18.6%** throughput | ❌ WRR tốt hơn (bình thường) |
| Video Conference | **-16.4%** throughput | ❌ WRR tốt hơn (bình thường) |
| Low-rate DoS | **-14.7%** throughput | ❌ WRR tốt hơn (bình thường) |

**Tỷ lệ thắng**: PPO 1/4 (25%), WRR 3/4 (75%)

### Luận Điểm Chính
**"Chi phí của sự thông minh"**: PPO mất 14.7%-18.6% throughput trong điều kiện bình thường, nhưng tăng 8.6% khi server bị suy thoái. Điều này xác định **vai trò SLA Protector** cho PPO - hoạt động song song với WRR, chỉ can thiệp khi phát hiện bất thường.

---

## 🎯 Đóng Góp Chính

1. **Vai trò SLA Protector**: PPO nên là bảo vệ SLA chứ không thay thế WRR hoàn toàn
2. **Feature engineering**: State space 20 chiều từ OpenFlow PortStats
3. **Safety mechanism**: Bypass PPO khi utilization > 0.95
4. **Benchmark mở rộng**: 2 kịch bản mới (burst_traffic, server_failure) đề xuất

---

## 📊 Thông Số Kỹ Thuật

**Network**: Fat-Tree K=4, 8 clients → 3 servers (10/50/100 Mbps)
**Training**: 500K timesteps, Gymnasium SDNEnv, stable-baselines3 PPO
**Hyperparameters**: lr=3e-4, γ=0.99, ε=0.2, 256×256 hidden, batch=64
**Metrics**: Packets (OpenFlow), P99 Latency (Artillery), Jain's Fairness Index

---

## 🔬 Hạn Chế & Hướng Phát Triển

**Hạn chế**:
- Variance cao (cần n≥10 runs)
- Sim-to-real gap (mô phỏng → thực tế)
- Inference overhead (P99 latency +16.2%)

**Hướng phát triển**:
1. Knowledge Distillation → giảm latency
2. Hybrid Controller (WRR 95% + PPO override)
3. Curriculum Learning cho training
4. XAI cho interpretability

---

## 📚 Tài Liệu Tham Khảo

Các kết quả chi tiết và bằng chứng thực nghiệm đầy đủ trong file gốc: **[ICT_TienGiang_Hien_SDN_3.pdf](ICT_TienGiang_Hien_SDN_3.pdf)**  
(Format: IEEE Computer Society, 2026)

**Key references**: [1] McKeown et al. (OpenFlow), [2] Schulman et al. (PPO), [3] Lim et al. (TFT), [4] Jain et al. (Fairness)

*Lưu ý: README này trích dẫn file gốc IEEE, xem PDF để có đầy đủ chi tiết kỹ thuật và phân tích thống kê.*

---

> **Trích dẫn**: Nếu sử dụng nội dung này, vui lòng trích dẫn file PDF gốc: `ICT_TienGiang_Hien_SDN_3.pdf` (IEEE Format, 2026)
