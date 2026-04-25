<div align="center">
  <img src="https://cdn.haitrieu.com/wp-content/uploads/2021/10/Logo-DH-Thuy-Loi.png" alt="Logo Đại học Thủy lợi" width="120" />
  <p><b>PHÂN HIỆU TRƯỜNG ĐẠI HỌC THỦY LỢI</b></p>
  <p><a href="ICT_TienGiang_Hien_SDN_3.pdf">📑 Xem bản gốc IEEE (PDF)</a></p>
</div>

---

# ĐÁNH GIÁ SỰ ĐÁNH ĐỔI GIỮA HIỆU NĂNG VÀ TÍNH THÍCH NGHI TRONG CÂN BẰNG TẢI SDN SỬ DỤNG HỌC TĂNG CƯỜNG ACTOR-CRITIC

<div align="center">

**Hội thảo khoa học Quốc gia về Công nghệ thông tin và Truyền thông (ICT) – Đồng Tháp, 22/5/2026**

**Phân hiệu Trường Đại học Thủy lợi — Khoa Công nghệ Thông tin**

---

## BAN CHỦ NHIỆM

| | |
|:---:|:---:|
| **Giảng viên hướng dẫn** | **Nhóm sinh viên thực hiện** |
| ThS. Hoàng Văn Quý<br>*Trường Đại học Nông lâm Bắc Giang*<br>📧 quyhv@bafu.edu.vn | Đặng Quang Hiển<br>*BM CNTT – Phân hiệu ĐH Thủy Lợi*<br>📧 wanghien.miho.dev@gmail.com |
| | Trương Tuấn Minh<br>*BM CNTT – Phân hiệu ĐH Thủy Lợi*<br>📧 truongtanminhbh2022@gmail.com |
| | Bùi Danh Hường<br>*Khoa CNTT – Trường ĐH Công nghệ TP.HCM*<br>📧 bd.huong@hutech.edu.vn |
| | Giáp Thị Yến<br>*Trường Đại học Nông lâm Bắc Giang*<br>📧 yengt@bafu.edu.vn |

---

| | |
|---|---|
| **Năm học** | 2025–2026 |
| **File gốc** | [ICT_TienGiang_Hien_SDN_3.pdf](ICT_TienGiang_Hien_SDN_3.pdf) |

</div>

---

## TÓM TẮT

Bài báo này khảo sát một cách thực nghiệm sự đánh đổi giữa hiệu năng và khả năng thích nghi khi áp dụng học tăng cường cho cân bằng tải trong mạng định nghĩa bằng phần mềm (SDN). Cụ thể, nghiên cứu đối sánh một tác tử PPO dùng bộ mã hóa TFT với đường cơ sở WRR trong môi trường Mininet/Ryu qua bốn kịch bản lưu lượng. Kết quả cho thấy PPO đạt mức tăng thông lượng **8.6%** ở kịch bản suy thoái phần cứng, thể hiện năng lực thích nghi tốt khi hệ thống rơi vào trạng thái bất thường. Tuy nhiên, ở các kịch bản ổn định, PPO kém WRR từ **14.7% đến 18.6%** do chi phí suy luận và độ trễ ra quyết định. Phát hiện này chỉ ra một giới hạn có tính cấu trúc: học tăng cường có thể linh hoạt, nhưng không hiệu quả nếu được dùng như cơ chế cân bằng tải chính trong SDN. Từ đó, nghiên cứu đề xuất định hướng kiến trúc lai ở mức thiết kế: PPO nên đóng vai trò **bảo vệ SLA theo điều kiện bất thường**, thay vì thay thế hoàn toàn các heuristic truyền thống.

**Từ khóa**— Software-Defined Networking, PPO, Actor-Critic, Temporal Fusion Transformer, Load Balancing, SLA Protection, Resilience, Mininet, Reinforcement Learning.

---

## I. GIỚI THIỆU

### A. Bối cảnh và Vấn đề

Sự phát triển nhanh chóng của các hệ thống Giáo dục trực tuyến (LMS) như Moodle, Canvas, và Blackboard đòi hỏi hạ tầng mạng có khả năng chịu tải cực lớn và biến động không ngừng. Trong mạng SDN, việc tách biệt Control Plane và Data Plane tạo điều kiện để triển khai các thuật toán thông minh tại Controller [1].

**Vấn đề cụ thể**: Các thuật toán cân bằng tải truyền thống như Round Robin (RR) và Weighted Round Robin (WRR) hoạt động theo các quy tắc cố định, không thể thích ứng khi:
1. **Server degradation**: Một server bị suy giảm 50% băng thông nhưng WRR vẫn phân bổ đúng tỷ lệ, gây quá tải
2. **Server failure**: Máy chủ chính sụp đổ, WRR tiếp tục gửi traffic đến server không khả dụng  
3. **Burst traffic**: Traffic đột ngột tăng gấp 10 lần, WRR không có cơ chế ưu tiên

Nghiên cứu đã chỉ ra rằng WRR tĩnh phân bổ traffic không chính xác do kích thước gói tin biến động, dẫn đến quá tải dữ liệu quá nhiều vào server mạnh và bỏ qua hoàn toàn server yếu khi chúng quá tải [4].

### B. Nghiên cứu liên quan

**Load Balancing trong SDN**: McKeown và cộng sự [1] đã giới thiệu OpenFlow như một giao thức tiêu chuẩn cho SDN, cho phép Controller lập trình các flow table trên switches. Nhiều nghiên cứu đã tận dụng khả năng này để triển khai các thuật toán cân bằng tải thông minh.

**Học Tăng Cường trong Network Optimization**: Schulman và cộng sự [2] đề xuất thuật toán PPO với cơ chế Clipping để đảm bảo chính sách học ổn định, tránh hiện tượng "policy collapse". Nghiên cứu gần đây đã áp dụng PPO cho various network optimization tasks, nhưng kết quả cho thấy RL không phải lúc nào cũng vượt trội heuristic đơn giản trong điều kiện bình thường [5].

**Temporal Fusion Transformer**: Lim và cộng sự [3] đề xuất TFT cho multi-horizon time series forecasting, kết hợp LSTM, attention mechanism, và interpretable components. Kiến trúc này đặc biệt phù hợp cho network monitoring vì có thể nắm bắt cả temporal dependencies và cung cấp interpretability.

**Jain's Fairness Index**: Jain và cộng sự [4] đề xuất chỉ số fairness để đo lường sự công bằng trong phân bổ tài nguyên.

### C. Đóng góp của bài báo

Khác với các nghiên cứu trước tập trung vào việc thay thế hoàn toàn heuristic bằng RL, bài báo này có các đóng góp sau:

1. **Đề xuất vai trò SLA Protector**: PPO nên hoạt động song song với WRR, can thiệp khi phát hiện bất thường thay vì thay thế hoàn toàn
2. **Thiết kế hệ thống đặc trưng 20 chiều** đại diện cho trạng thái hàng đợi, băng thông, và cache hit rate
3. **Triển khai PPO với Safety Override** đảm bảo tính sẵn sàng khi Agent đưa ra quyết định không tối ưu
4. **Kiểm chứng khoa học** qua 4 kịch bản benchmark thực trong môi trường Mininet/Ryu

---

## TÀI LIỆU THAM KHẢO

Các kết quả chi tiết, phân tích thống kê, và bằng chứng thực nghiệm đầy đủ trong file gốc: **[ICT_TienGiang_Hien_SDN_3.pdf](ICT_TienGiang_Hien_SDN_3.pdf)** (IEEE Computer Society Format, 2026)

*Lưu ý: README này trích dẫn file gốc IEEE, xem PDF để có đầy đủ chi tiết kỹ thuật, phương pháp luận, và phân tích thống kê.*
