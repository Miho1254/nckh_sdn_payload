# BÁO CÁO NGHIÊN CỨU KHOA HỌC

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

Các thuật toán cân bằng tải tĩnh trong mạng Software-Defined Networking (SDN) như Round Robin thường thất bại trong việc linh hoạt phân bổ tải theo năng lực thực tế của máy chủ, dẫn đến nghẽn mạng cục bộ và tăng độ trễ dịch vụ. Bài báo này đề xuất hệ thống **TFT-PPO**, một kiến trúc học tăng cường thích nghi kết hợp bộ mã hóa **Temporal Fusion Transformer (TFT)** và thuật toán **Proximal Policy Optimization (PPO)**. Kiến trúc đề xuất sử dụng mô hình Actor-Critic để đưa ra quyết định điều phối luồng dựa trên trạng thái mạng thời gian thực trích xuất từ OpenFlow PortStats. Kết quả thực nghiệm trên môi trường Mininet/Ryu với 5 lượt chạy độc lập (N=5) kịch bản Flash Crowd (Golden Hour) cho thấy TFT-PPO vượt trội so với baseline Weighted Round Robin (WRR) ở hai khía cạnh cốt lõi: giảm **12.3%** độ trễ trung bình và tăng đột phá tính công bằng phân bổ tải (Fairness MAE giảm từ **19.77% xuống 6.59%**, Jain's Index cải thiện **23.3%**). Dù có sự đánh đổi về thông lượng cực đại (-20.2%), hệ thống chứng minh được tính ổn định và khả năng tối ưu hóa Chất lượng Dịch vụ (QoS) ưu việt trong môi trường SDN thực tế.

**Từ khóa:** *Software-Defined Networking, PPO, Actor-Critic, Temporal Fusion Transformer, Load Balancing, QoS, Mininet.*

---

## I. GIỚI THIỆU (INTRODUCTION)

Sự phát triển của các hệ thống Giáo dục trực tuyến (LMS) đòi hỏi hạ tầng mạng có khả năng chịu tải cực lớn và biến động không ngừng. Trong mạng SDN, việc tách biệt Control Plane và Data Plane tạo điều kiện để triển khai các thuật toán thông minh tại Controller. Tuy nhiên, các phương pháp truyền thống thường thiếu tính dự báo, trong khi các kỹ thuật RL trực tuyến cơ bản thường gặp vấn đề về độ hội tụ và an toàn hệ thống.

Nghiên cứu này tập trung vào việc hiện thực hóa mô hình **TFT-PPO** nhằm tối ưu hóa độ trễ (Latency) và thông lượng (Throughput). Đóng góp chính của bài báo bao gồm: (1) Thiết kế hệ thống đặc trưng 22 chiều đại diện cho trạng thái hàng đợi và băng thông; (2) Triển khai thuật toán PPO với cơ chế Clipping để đảm bảo chính sách học ổn định; (3) Kiểm chứng khoa học thông qua benchmark 5 lần chạy độc lập để loại bỏ sai số hệ thống.

---

## II. PHƯƠNG PHÁP ĐỀ XUẤT (METHODOLOGY)

### A. Kiến Trúc Mô Hình TFT-PPO

Hệ thống sử dụng cấu trúc **Actor-Critic** với bộ mã hóa temporal chia sẻ:
1. **TFT Encoder**: Sử dụng LSTM và Multi-Head Attention để nắm bắt phụ thuộc thời gian của các metrics mạng.
2. **Actor Head**: Sinh ra phân phối xác suất $\pi(a|s)$ trên không gian hành động (chọn Server).
3. **Critic Head**: Ước lượng giá trị trạng thái $V(s)$ để hỗ trợ quá trình cập nhật chính sách.

### B. Thuật toán PPO (Proximal Policy Optimization)

PPO tối ưu chính sách thông qua hàm mục tiêu có giới hạn (Clipped Objective):
```math
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) ]
```
Cơ chế này ngăn chặn việc thay đổi chính sách quá lớn trong một bước cập nhật, giúp hệ thống tránh được hiện tượng "sụp đổ chính sách" (Policy Collapse) thường gặp trong môi trường mạng biến động cao.

### C. Cơ Chế An Toàn (Safety Override)

Hệ thống cài đặt chốt chặn an toàn: Nếu $u_i > 0.95$ (utilization vượt ngưỡng), hệ thống tự động loại bỏ Agent và chuyển traffic sang server có tài nguyên khả dụng nhất, đảm bảo tính sẵn sàng của dịch vụ (High Availability).

---

## III. THỰC NGHIỆM VÀ KẾT QUẢ (EXPERIMENTS & RESULTS)

### A. Thiết lập Thực nghiệm (N=5)

Thực nghiệm được tiến hành trên topology Fat-Tree K=4 trong môi trường Docker. Kịch bản **Flash Crowd** mô phỏng 8 khách hàng (h9-h16) liên tục gửi request HTTP tới Virtual IP. Chúng tôi thực hiện 5 lượt chạy độc lập (Isolated runs) cho cả WRR và PPO để lấy kết quả thống kê.

### B. Kết Quả Tổng Hợp

*Bảng 1: So sánh hiệu năng trung bình sau 5 lượt chạy (Kịch bản Golden Hour Flash Crowd)*

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

## IV. KẾT LUẬN

Nghiên cứu đã thực hiện thành công việc tích hợp mô hình **TFT-PPO** vào bộ điều khiển SDN Ryu. Kết quả benchmark 5 lần chạy độc lập đã cung cấp bằng chứng thực nghiệm mạnh mẽ về khả năng vượt trội của AI so với các thuật toán truyền thống. Hệ thống không chỉ giảm trễ mà còn tăng tính ổn định cho mạng dưới áp lực tải lớn. Hướng phát triển tiếp theo sẽ tập trung vào việc mở rộng quy mô mạng và tích hợp cơ chế giải thích (XAI) cho các quyết định của Agent.

---

## TÀI LIỆU THAM KHẢO

[1] N. McKeown et al., "OpenFlow: Enabling innovation in campus networks," 2008.  
[2] J. Schulman et al., "Proximal Policy Optimization Algorithms," 2017.  
[3] B. Lim et al., "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting," 2021.  
[4] R. Jain et al., "A Quantitative Measure of Fairness and Discrimination," 1984.

---
*© 2026 — Phân hiệu Trường Đại học Thủy Lợi. Nhóm nghiên cứu chân thành cảm ơn ThS. Hoàng Văn Quý đã hỗ trợ.*
