# BÁO CÁO NGHIÊN CỨU KHOA HỌC: Ứng dụng Học máy trong Phát hiện Truy cập Bất thường trên Dữ liệu Luồng

Tối ưu hóa phân tải mạng SDN bằng mô hình Temporal Fusion Transformer + Deep Q-Network (TFT-DQN) trong môi trường Mininet mô phỏng hệ thống đăng ký tín chỉ đại học (LMS).

**Giảng viên hướng dẫn:** ThS. Hoàng Văn Quý
**Nhóm sinh viên thực hiện:** 
- Đặng Quang Hiển (MSSV: 2551067129)
- Đặng Trọng Phúc (MSSV: 2551267312)
- Trương Tuấn Minh (MSSV: 2551067144)
- Trần Minh Triết (MSSV: 2551067170)
**Cơ quan chủ quản:** Phân hiệu Trường Đại học Thủy Lợi

---

## TÓM TẮT (ABSTRACT)
Sự phát triển mạnh mẽ của các hệ thống đăng ký trực tuyến tại các trường đại học thường kéo theo hiện tượng quá tải đột ngột (Flash Crowd) và nguy cơ tắc nghẽn mạng nghiêm trọng. Các thuật toán cân bằng tải tĩnh truyền thống như Round Robin (RR) hay Weighted Round Robin (WRR) thiếu khả năng dự báo và không linh hoạt thích ứng với các biến động thời gian thực. Bài báo này đề xuất một kiến trúc tối ưu hóa phân tải mạng dựa trên công nghệ Mạng số liệu điều khiển bằng phần mềm (SDN) kết hợp với học tăng cường sâu đa biến thời gian, cụ thể là mô hình Temporal Fusion Transformer kết hợp Deep Q-Network (TFT-DQN). Các mô phỏng thực nghiệm trên môi trường Mininet với kiến trúc mạng Fat-Tree và hệ thống Backend đa tải cho thấy sự ưu việt của AI trong việc dự đoán điểm đỉnh luồng, tự động định tuyến lại để giảm thiểu độ trễ và tránh điểm nghẽn. Kết quả đóng góp vào lĩnh vực bảo mật và quản trị mạng thế hệ mới cho các dịch vụ đám mây công cộng và mạng trường học.

**Từ khóa:** Cân bằng tải, SDN, Học tăng cường sâu, Temporal Fusion Transformer, Mininet.

---

## 1. GIỚI THIỆU (INTRODUCTION)

Hệ thống mạng hiện đại đang phải đối mặt với các lưu lượng dữ liệu tăng đột biến (spiky traffic), điển hình như vào thời điểm mở cổng đăng ký tín chỉ tại các trường đại học. Khi lưu lượng vượt qua ngưỡng chịu đựng của một máy chủ, nếu bộ cân bằng tải không đủ thông minh để nhận biết kịp thời, hiện tượng sập cục bộ (bottleneck) sẽ diễn ra, kéo theo hàng loạt hệ lụy về trải nghiệm người dùng và an toàn dữ liệu.

Phương pháp cân bằng tải SDN cho phép tách biệt mặt phẳng điều khiển (Control Plane) ra khỏi mặt phẳng dữ liệu (Data Plane), mang lại lợi thế tập trung hóa góc nhìn toàn mạng. Tại bộ điều khiển trung tâm (SDN Controller), chúng ta có thể can thiệp sâu vào các quyết định định tuyến bằng cách ứng dụng AI. 

Trong nghiên cứu này, thay vì dựa dẫm vào các chỉ số quá khứ gần để thay đổi quy tắc, chúng tôi ứng dụng mô hình **Temporal Fusion Transformer (TFT)** đi liền với **Deep Q-Network (DQN)**. TFT đóng vai trò thu thập đa biến không - thời gian để "nhìn" trước xu hướng bùng nổ traffic, từ đó làm cơ sở vững chắc cho Agent DQN ra quyết định lựa chọn máy chủ tương lai chuẩn xác.

---

## 2. CƠ SỞ LÝ THUYẾT & KIẾN TRÚC HỆ THỐNG (ARCHITECTURE)

### 2.1. Cấu trúc Mạng vật lý Fat-Tree K=4
Môi trường thử nghiệm áp dụng cấu trúc liên kết Fat-Tree chuẩn với $k=4$, bao gồm 10 switches (OVS) thuộc 3 lớp (Core, Aggregation, Edge), hỗ trợ 16 node (h1-h16). Nhóm Edge (h9-h16) đóng vai trò là Client bắn traffic, trong đó IP `10.0.0.100` là Virtual IP (VIP) điều phối.

Backend xử lý gồm 3 máy chủ Web (Port 4000) được giới hạn vật lý bằng công nghệ TC (Traffic Control) nhằm tạo cấu trúc bất đối xứng:
- **h5 (Backend 1):** Băng thông thắt cổ chai 10 Mbps (Mạng yếu).
- **h7 (Backend 2):** Băng thông 50 Mbps.
- **h8 (Backend 3):** Băng thông 100 Mbps (Khả năng xử lý cao nhất).
Toàn bộ Backend sử dụng chung cơ sở dữ liệu PostgreSQL (h6) chứa 5,000 bản ghi sinh viên.

### 2.2. Lớp Điều Khiển SDN và Kỹ thuật NAT
Bộ điều khiển Ryu (Ryu Controller) thực hiện công việc kép: Dẫn đường gói tin L2 (Learning Switch) và Ghi nhận/Điều hướng dồn dịch NAT (LB Logic) tại lớp L3/L4.
1. Match: Nhắm vào các gói tin HTTP gửi đến VIP.
2. Action: Đổi IP/MAC đích thành thông số của Backend, đồng thời lắp luật chiều ngược lại để biến IP nguồn thành VIP. 
3. Controller định kỳ mỗi 10 giây sẽ truy vấn Switch (FlowStats, PortStats) để lấy Metrics thực gửi về cho Model Agent.

---

## 3. PHƯƠNG PHÁP ĐỀ XUẤT (METHODOLOGY: TFT-DQN)

### 3.1. Hợp nhất TFT và DQN
TFT nổi bật với Gated Residual Network (GRN) và Variable Selection Network (VSN) giúp lọc bỏ yếu tố nhiễu trong dữ liệu đa biến. Kết hợp cùng khối Temporal Self-Attention nhiều "đầu" (Multi-Head), mô hình này chiết xuất một véctơ ngữ cảnh (context vector).
Thay vì tạo ra một mô hình dự báo thuần túy, véctơ ngữ cảnh này được đưa sang mạng Q-Net đa tầng (Deep Q-Network) với 3 Node Output (tương ứng với 3 Action chọn h5, h7, h8). Cụm này trả ra Q-values và thông qua cơ chế phân bổ Boltzmann (Softmax) kết hợp với **Capacity-Aware Probability Scaling** (Nhân Xác Suất với Công Suất Phần Cứng). Bằng cách kết hợp xác suất softmax động dẻo dai từ AI với trọng số công suất thực tế của máy chủ cơ sở ($1:5:10$), AI đóng vai trò như một **Dynamic WRR**, giúp phân bổ cấu hình máy chủ thông minh dựa theo trạng thái nghẽn thực tại hạn chế việc đánh sập cục bộ server yếu.

### 3.2. Hàm Phần Thưởng Không - Thời Gian (Spatio-Temporal Reward)
Tâm điểm của phương pháp học tăng cường nằm ở Hàm khen thưởng $R$. Chúng tôi kiến trúc hàm $R$ mở rộng, không chỉ lấy Thông lượng (Throughput) mà còn kết hợp phân tích Không Gian (Balancer Load) và Thời Gian (Trend).

$R = \text{Base} + R_{\text{throughput}} + R_{\text{balance}} + R_{\text{congestion}} + R_{\text{temporal}}$

- **Tối ưu Thông lượng:** $R_{\text{throughput}} = 0.5 \times \text{throughput rate}$
- **Cân bằng Phương sai Tải:** Thưởng khi máy chủ có tải trọng thấp hơn mức trung bình toàn cụm.
- **Phạt Nghẽn Cổ Chai Phi Tuyến:** Penalty hàm mũ nếu máy chủ được chọn có tải $\geq 70\%$, mô phỏng Queueing Latency.
- **Phạt Xu hướng Thời Gian (Temporal trend):** Chặn đánh giá cực đoan nếu Server ghi nhận đà tăng (Derivative $>$ 0).
- **Phản ứng Bão (Flash Crowd Modification):** Trong trường hợp Switch gắn nhãn `HIGH-TRAFFIC`, trọng số phạt điểm nghẽn lên đến $1.5$ để ép mạng tản tải khẩn cấp.

---

## 4. KỊCH BẢN THỰC NGHIỆM (EXPERIMENTAL SETUP)

Để làm nổi trội những giới hạn của cơ chế tĩnh (Static Load Balancing), chúng tôi thay vì chỉ spam gói ICMP đã sử dụng công cụ Artillery bắn HTTP thực vào Backend. Kịch bản seeding Database trước đó đã tạo 5,000 users nhằm tạo độ trễ truy vấn sâu và thắt nút tại DB lock. 

Chúng tôi đưa vào 4 kịch bản Stress-test chính:
1. **Flash Crowd (`flash_crowd.yml`):** Bùng nổ tải tức thì từ 0 lên 1000 users. Kiểm tra khả năng vỡ kế hoạch của thuật toán tĩnh cứng.
2. **Predictable Ramping (`predictable_ramping.yml`):** Tải tăng đều, kiểm tra khả năng dự báo xu hướng từ sớm do TFT cung cấp.
3. **Targeted Congestion (`targeted_congestion.yml`):** Nhắm thẳng nút cổ chai `h5`. Kiểm tra năng lực failover độc lập của mạng.
4. **Gradual Shift (`gradual_shift.yml`):** Từ Normal sang High liên tục mượt mà để đo độ bền theo thời gian.

Các cuộc đánh giá được chạy offline trên Database để train trước khoảng 50,000 Sequence.

---

## 5. KẾT QUẢ VÀ BÀN LUẬN (RESULTS & DISCUSSION)

Hệ thống cung cấp một cơ sở so sánh đầy đủ 3 cơ chế: RR, WRR (Tỷ lệ 1:2:3), và TFT-DQN. Dữ liệu đánh giá tập trung vào:

- **Hệ số Biến thiên (CV) và Phân bổ tải:** Sử dụng "Capacity-Aware Probability Scaling", AI TFT-DQN chia tải rất sát với sức chứa phần cứng thực tế, đạt Hệ số CV% ưu việt ~50.2%. AI đánh bại hoàn toàn Round Robin (94.5%) và bám sát mức phân bổ tải lý tưởng của WRR, nhưng vượt trội hơn nhờ khả năng chuyển trạng thái nắn dòng linh hoạt.
- **Thông lượng (Throughput) & Tối thiểu hóa Độ trễ:** Trong kịch bản sốc tải Flash Crowd, AI giải quyết triệt để điểm nghẽn bằng cách mở thông lượng mạng đạt 0.10 MB/s, tăng xấp xỉ gấp đôi so với WRR (0.06 MB/s). Quan trọng hơn, lượng nghẽn cổ chai của AI được kiểm soát cực thấp ở mức 1.7%, trong khi các phương pháp tĩnh bộc lộ tình trạng nghẽn lên tới 5.2%.
- **Inference Overhead:** Đạt ngưỡng cực nhanh $< 10\text{ms}$ một lần đánh giá (thực tế đo được trung bình ~4.5ms) ở môi trường Docker thuần CPU, chứng minh tốc độ phản ứng thời gian thực ưu việt của AI.

---

## 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (CONCLUSION)

Nghiên cứu đã phác họa thành công phương pháp tổ chức hạ tầng phân phối tải dựa trên sức mạnh của SDN và Deep Reinforcement Learning. Bằng việc kết nối trực tiếp Transformer dự báo xu hướng và mạng Q-Learning điều phối hành động, hệ thống không chỉ giải quyết điểm nghẽn của dịch vụ đại học dưới tải bùng phát, mà còn chỉ ra lối đi mới cho Edge Computing và Cloud. 

Tương lai nghiên cứu có thể tập trung vào Federated Learning Multi-Controller đối với không gian liên mạng khổng lồ, và tối ưu số lượng Node Layer Data Plane.
