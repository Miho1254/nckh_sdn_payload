<div align="center">
  <img src="https://cdn.haitrieu.com/wp-content/uploads/2021/10/Logo-DH-Thuy-Loi.png" alt="Logo Đại học Thủy lợi" width="120" />

  <p><b>PHÂN HIỆU TRƯỜNG ĐẠI HỌC THỦY LỢI</b></p>
  <hr width="30%">

  # ĐỀ TÀI NGHIÊN CỨU KHOA HỌC
  ## Tối ưu hóa Cân bằng Tải Mạng SDN bằng Học Tăng Cường Thích nghi<br>với Kiến trúc TFT-PPO Actor-Critic

  <p>
    <b>Giảng viên hướng dẫn:</b> ThS. Hoàng Văn Quý <br>
    <b>Nhóm sinh viên thực hiện:</b> <br>
    Đặng Quang Hiển (2551067129) | Đặng Trọng Phúc (2551267312) <br>
    Trương Tuấn Minh (2551067144) | Trần Minh Triết (2551067170)
  </p>

  > Hệ thống tối ưu hóa cân bằng tải trong mạng **Software-Defined Networking (SDN)** thế hệ mới,
  > sử dụng kiến trúc **TFT-PPO** (Temporal Fusion Transformer + Proximal Policy Optimization)
  > kết hợp mô hình **Actor-Critic** và cơ chế **Safety Override** nhằm tối ưu hóa QoS
  > trong các kịch bản bùng nổ lưu lượng (Flash Crowd).
</div>

---

## 1. GIỚI THIỆU (ABSTRACT)

Nghiên cứu này đề xuất hệ thống cân bằng tải thích nghi cho mạng SDN dựa trên thuật toán **Proximal Policy Optimization (PPO)** tích hợp kiến trúc **Temporal Fusion Transformer (TFT)**. Khác với các thuật toán tĩnh truyền thống, TFT-PPO có khả năng quan sát trạng thái mạng theo thời gian thực để đưa ra các quyết định điều phối luồng thông minh.

Hệ thống sử dụng bộ đặc trưng 22 chiều trích xuất từ OpenFlow PortStats để huấn luyện chính sách Actor-Critic. Kết quả thực nghiệm trên môi trường **Mininet/Ryu** thực tế với kịch bản **Flash Crowd** (N=5 lần chạy độc lập) chứng minh TFT-PPO vượt trội so với baseline Weighted Round Robin (WRR) ở hai khía cạnh cốt lõi: giảm **12.3%** độ trễ trung bình và tăng đột phá tính công bằng phân bổ tải (Fairness MAE giảm từ **19.77% xuống 6.59%**, Jain's Index cải thiện **23.3%**). Dù có sự đánh đổi về thông lượng cực đại (-20.2%), hệ thống chứng minh được tính ổn định và khả năng tối ưu hóa Chất lượng Dịch vụ (QoS) ưu việt.

> **Tài liệu Báo Cáo:**
> - [Báo cáo Nghiên cứu Đầy đủ (IMRAD/IEEE)](docs/Bao_Cao_NCKH_IEEE.md)

---

## 2. KIẾN TRÚC HỆ THỐNG (TFT-PPO)

### 2.1. Kiến trúc Mạng (Fat-Tree K=4)

```text
Clients (h9–h16) — HTTP Request → 10.0.0.100 (Virtual IP)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│        SDN Controller (Ryu) — TFT-PPO Model             │
│    ┌───────────────────────────────────────────────┐    │
│    │  Sequence Encoder (Temporal Integration)       │    │
│    │           ↙               ↘                  │    │
│    │  Actor Head (π)         Critic Head (V)       │    │
│    │  (Action Sampling)     (Value Estimation)     │    │
│    └───────────────────────────────────────────────┘    │
│    Real-time State: 22 Features (Utilization, RTT)      │
│    Safety Mechanism: Threshold-based Server Protection  │
└─────────────────────────────────────────────────────────┘
        │  OpenFlow 1.3 Flow Table Entry
        ├──► h5  (Server Low: ~10 Mbps)
        ├──► h7  (Server Mid: ~50 Mbps)
        └──► h8  (Server High: ~100 Mbps)
                 └── h6 (PostgreSQL Backend)
```

### 2.2. Đặc điểm nổi bật
- **TFT Encoder**: Nắm bắt xu hướng tải mạng thông qua cơ chế Attention và LSTM.
- **PPO Clipping**: Đảm bảo chính sách thay đổi mượt mà, tránh tình trạng sụp đổ chính sách (Policy Collapse).
- **Safety Override**: Tự động chuyển hướng traffic nếu phát hiện server quá tải (>95% CPU/Băng thông).

---

## 3. KẾT QUẢ THỰC NGHIỆM (FINAL BENCHMARK N=5)

Hệ thống được đánh giá qua 5 lượt chạy độc lập (Isolated Runs) để lấy giá trị trung bình ± độ lệch chuẩn (Mean ± Std Dev), đảm bảo tính khách quan khoa học theo chuẩn IEEE.

| Chỉ số | Đơn vị | WRR (Baseline) | PPO (AI Policy) | Cải thiện |
| :--- | :--- | :--- | :--- | :--- |
| **Trễ Trung Bình (Mean)** | ms | 1813.47 ± 156.58 | **1590.69 ± 129.17** | ✅ **12.3%** |
| **Trễ Đuôi (P99)** | ms | 8604.58 ± 383.95 | 8919.52 ± 236.35 | ❌ -3.7% |
| **Thông Lượng (Throughput)** | MB | 35.98 ± 2.63 | 28.73 ± 2.32 | ❌ -20.2% |
| **Jain's Fairness Index** | - | 0.54 ± 0.02 | **0.66 ± 0.00** | ✅ **23.3%** |
| **Sai số Phân bổ (Fairness MAE)**| % | 19.77 ± 0.29 | **6.59 ± 0.24** | ✅ **66.6%** |

*Ghi chú: Giá trị Jain's Fairness tiệm cận 1.0 biểu thị mạng phân bổ luồng lý tưởng theo đúng tỷ lệ năng lực của 3 server (10Mbps, 50Mbps, 100Mbps).*

### C. Phân Tích Kết Quả (Key Insights)

1. **Đột phá về Độ trễ Trung bình (Mean Latency)**: PPO giảm thành công **12.3%** độ trễ trung bình. Điều này chứng minh AI đã học được cách tránh các hàng đợi (queues) đang phình to tại các switch biên để định tuyến request HTTP nhanh hơn.
2. **Sự đánh đổi Băng thông - Công bằng (The Fairness-Throughput Trade-off)**: Điểm sáng khoa học nhất của báo cáo nằm ở chỉ số MAE (Mean Absolute Error giữa tỷ lệ luồng phân bổ thực tế và năng lực lý thuyết của máy chủ). PPO kéo giảm MAE từ 19.77% xuống chỉ còn **6.59%**, đẩy chỉ số Jain's Fairness lên 0.66. WRR tĩnh (tỷ lệ 1-5-10) thực tế phân bổ traffic rất thiếu chính xác do kích thước gói tin biến động, dẫn đến "nhồi" dữ liệu quá nhiều vào node 100Mbps tạo ra thông lượng giả tạo cao (35.98MB). PPO chấp nhận hy sinh thông lượng rác (-20.2%) để duy trì **tính công bằng phân bổ tải (Fairness)**, một biểu hiện của sự thông minh hệ thống (Intelligence).
3. **Giảm độ rủi ro (Jitter)**: Mặc dù P99 latency của PPO cao hơn một chút (3.7%), nhưng độ lệch chuẩn (Std Dev) của P99 giảm từ ±383.95 xuống ±236.35. Điều này cho thấy hệ thống AI hoạt động **ổn định và dễ dự báo hơn** nhiều so với thuật toán mù (blind algorithm) như WRR.

---

## 4. HƯỚNG DẪN TRIỂN KHAI

### Yêu cầu hệ thống
- Môi trường Docker/Linux.
- SDN Container: `lms-sdn-env`.

### Chạy thử nghiệm tự động (N=5)
```bash
# Khởi tạo container
docker compose up -d

# Thực hiện benchmark 5 lượt chuẩn IEEE
docker exec nckh-sdn-mininet /work/scripts/run_real_mininet_benchmark_multi.sh 5

# Phân tích kết quả
python3 scripts/analyze_multi_run_results.py benchmark_results_multi
```

---

<p align="center">
    <b>Nghiên cứu Khoa học 2026</b> <br>
    <i>© Bản quyền dành cho Phiên bản Nghiên cứu & Học thuật — Đại Học Thủy Lợi.</i>
</p>
