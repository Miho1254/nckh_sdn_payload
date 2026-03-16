# 🎭 KỊCH BẢN THUYẾT TRÌNH CẤP ĐỘ "SIÊU CHI TIẾT" (MASTER SCRIPT)

**Đề tài:** Ứng dụng Học máy trong Phát hiện Truy cập Bất thường trên Dữ liệu Luồng
*(Tối ưu hóa Điều phối SDN bằng Mô hình Học Tăng cường TFT-DQN)*
**Người trình bày:** Đặng Quang Hiển (S28-67CNTT)
**Thời lượng chuẩn:** ~12-15 phút
**Mục tiêu:** Gây ấn tượng mạnh với Hội đồng Khoa học về tính cấp thiết, độ khó logic của thuật toán AI gai góc (TFT-DQN) và các kết quả thực nghiệm trực quan thuyết phục.

---

## 🕒 KHUNG THỜI GIAN THEO SLIDE

| Mốc Thời Gian | Slide | Nội dung Trọng tâm |
|:---:|:---|:---|
| 00:00 - 01:00 | **Slide 0 & 1** | Mở đầu, Thu hút sự chú ý bằng nỗi đau "Nghẽn đăng ký tín chỉ". |
| 01:00 - 03:00 | **Slide 2 & 3** | Kiến trúc SDN & Cấu tạo "Não bộ" TFT-DQN. Nhấn mạnh Hàm phần thưởng. |
| 03:00 - 04:30 | **Slide 4 & 5** | Môi trường Mininet bất đối xứng & Cảnh báo trước 4 kịch bản bắn tải. |
| 04:30 - 08:30 | **Slide 6 - 9** | **(ĐỈNH ĐIỂM)**: Show biểu đồ thực chiến Flash Crowd, Targeted Congestion, Throughput. |
| 08:30 - 10:00 | **Slide 10** | Chứng minh tính khả thi (Inference Time < 50ms). |
| 10:00 - 11:30 | **Slide 11, 12, 13** | Tổng kết, Mở rộng tương lai (Online RL) & Kết thúc. |

---

## 🎬 KỊCH BẢN TỪNG LỜI THOẠI & CỬ CHỈ

### 🟢 PHẦN MỞ ĐẦU & ĐẶT VẤN ĐỀ (Slide 0 - 1)

#### Slide 0: Trang Bìa (00:00 - 00:30)
- **Hành động & Cử chỉ:** Bước ra giữa bệ đứng, mỉm cười, ánh mắt quét một vòng qua ban giám khảo để tạo sự kết nối trước khi cất tiếng. Bấm clicker chuyển slide ngay khi dứt câu giới thiệu.
- **Lời thoại:**
  > "Dạ, em xin phép gửi lời chào trân trọng nhất đến Quý Thầy Cô trong Hội đồng đánh giá. Em là Đặng Quang Hiển, đại diện nhóm sinh viên nghiên cứu khoa học.
  > 
  > Hôm nay, em vô cùng vinh dự được trình bày trước hội đồng một giải pháp giải quyết thẳng vào "nỗi đau mạng" kinh điển của sinh viên tụi em. Đề tài mang tên: **Ứng dụng Học máy trong Phát hiện Truy cập Bất thường trên Dữ liệu Luồng**, với trọng tâm là Tối ưu hóa mạng SDN thông qua mô hình học tăng cường TFT-DQN."

#### Slide 1: Đặt Vấn Đề (00:30 - 01:15)
- **Hành động & Cử chỉ:** Chỉ tay nhẹ về biểu tượng "Flash Crowd". Thay đổi giọng điệu: từ trang trọng sang gần gũi, thực tế.
- **Lời thoại:**
  > "Thưa quý thầy cô, chắc hẳn hình ảnh các trang web trường đại học bị 'sập' trắng xóa màn hình vào mỗi dịp đăng ký tín chỉ là câu chuyện không xa lạ. Hiện tượng đó trong kỹ thuật gọi là **Flash Crowd** - hàng ngàn người dùng ập vào đồng thời tạo ra lượng tải đột biến gấp hàng trăm lần bình thường.
  > 
  > Lúc này, các bộ cân bằng tải (Load Balancer) truyền thống chia bằng thuật toán Round-Robin lộ rõ yếu điểm chết người. Chúng như những 'cỗ máy mù', cứ tiếp tục chia đều công việc cho máy tính mặc kệ máy đó có đang bốc khói hay không. Hậu quả là sinh ra điểm nghẽn cục bộ làm đình trệ toàn hệ thống."
  > 
  > *(Ngừng 2 giây, nhấn mạnh)* "Vì vậy, nhóm chúng em đặt ra một bài toán táo bạo: Làm sao để thay thế công cụ 'mù' đó bằng một **'Bộ não AI có khả năng cảm giác trước cơn bão'** và tự động bẻ nhánh dòng thác lưu lượng? Nhóm xin đi vào kiến trúc giải pháp."

---

### 🔵 PHẦN GIẢI PHÁP: SDN VÀ MÔ HÌNH AI (Slide 2 - 3)

#### Slide 2: Kiến trúc Giải pháp & Fat-Tree (01:15 - 02:15)
- **Hành động & Cử chỉ:** Chuyển qua slide 2. Dùng bút laser hoặc tay chỉ dọc theo sơ đồ mạng từ Core xuống Host. Đặc biệt nhấn mạnh 3 hộp Server (h5, h7, h8).
- **Lời thoại:**
  > "Để linh hoạt hóa đường đi của gói tin, chúng em chuyển đổi hoàn toàn kiến trúc mạng sang chuẩn **SDN (Software-Defined Networking)**. Bằng việc tách rời bộ não điều khiển (Control Plane) lên trên Ryu Controller, mạng giờ đây hoàn toàn tuân lệnh AI qua chuẩn OpenFlow.
  > 
  > Và thưa ban giám khảo, điểm đặc biệt nhất trong thực nghiệm của nhóm ở bên phải hệ thống: Nhóm thiết kế một cụm máy chủ **bất đối xứng (Heterogeneous Servers)**. 
  > Cụ thể, máy **h5 bị bóp nghẽn bằng code chỉ còn 10 Mbps**, máy **h7 có 50 Mbps**, và máy **h8 là 100 Mbps**. Sự chênh lệch này giống hệt thực tế, và là bài kiểm tra khắc nghiệt nhất để đo xem độ thông minh của AI so với các thuật toán tịnh tiến như thế nào."

#### Slide 3: Mô hình Lõi: TFT-DQN (02:15 - 03:30)
- **Hành động & Cử chỉ:** Chỉ tay dứt khoát vào 2 hộp TFT và DQN. Chuyển tay sang khung "Hàm Phần thưởng" bên phải. (Phần này rất ghi điểm với dân kỹ thuật).
- **Lời thoại:**
  > "Thưa Hội đồng, 'linh hồn' của toàn bộ hệ thống này là mô hình AI kết hợp có tên **TFT-DQN**. Nó sở hữu 2 siêu năng lực:
  > 
  > Một là mạng **Transformer - TFT**. Nó đóng vai trò làm nhà tiên tri. TFT liên tục nhìn lại biểu đồ lưu lượng của 5 bước thời gian trong quá khứ để bắt được xu hướng tăng tải ngầm, dự báo trước lưu lượng sắp đổ về trong tíc tắc tới.
  > 
  > Hai là mạng **DQN**. Từ dự báo của TFT, DQN sẽ vào vai người ra lệnh điều phối (Action). Làm sao DQN biết điều phối đúng hay sai? Nhóm em đã thiết kế một **Hàm Phần Thưởng (Reward Function)** cốt lõi: 
  > *(Nhấn mạnh từng điểm)* Dựa trên độ lệch tải thực tế, AI sẽ được cộng điểm cấu trúc nếu đẩy luồng vào server đang rảnh rang băng thông, và bị phạt điểm âm cực nặng nếu đẩy luồng làm đầy tràn một server vốn đã kiệt sức. Lâu dần sau hàng ngàn vòng lặp học, AI sẽ hoàn thiện trí thông minh tuyệt đối."

---

### 🟡 PHẦN THỰC NGHIỆM CHI TIẾT (Slide 4 - 5)

#### Slide 4: Giai đoạn Chuẩn bị (03:30 - 04:30)
- **Hành động & Cử chỉ:** Thể hiện sự hiểu biết về stack công nghệ. Nhấn mạnh 4 kịch bản test để tạo tò mò cho phần kết quả.
- **Lời thoại:**
  > "Về mặt setup, để mô phỏng một mạng Internet trung tâm dữ liệu khổng lồ với 16 máy ảo ngay trên máy tính cục bộ, chúng em đã dùng Mininet và Docker Ảo hóa hoàn toàn tầng Data Plane.
  > 
  > Đặc biệt, chúng em đã dùng **Artillery** (Công cụ bắn tải mạnh nhất của NodeJS) để đóng vai hàng vạn sinh viên, và nhào nặn ra 4 kịch bản sóng lưu lượng:
  > Tăng Ramping bình thường... Biến đổi từ từ... Và đặc biệt là Kịch bản số 3: **Mô phỏng Đăng Ký Tín Chỉ (Flash Crowd) với 1000 users ồ ạt**; cùng một kịch bản số 4 cực kỳ hiểm hóc: **Nghẽn Mục Tiêu (Targeted Congestion)** - mô hình tấn công DDoS giả lập nhét cứng một đường truyền độc nhất."

#### Slide 5: Quá trình Huấn luyện (04:30 - 05:00)
- **Hành động & Cử chỉ:** Chỉ vào biểu đồ cong giảm dần.
- **Lời thoại:**
  > "Và đây là Giao diện Giám sát trong lúc AI 'đi học' ngoại tuyến (Offline Training). Thông qua biểu đồ, hội đồng có thể thấy Đường Loss giảm xuống mức thấp nhất và ổn định, trong khi Tích luỹ Phần thưởng Reward chạm đến giới hạn cực đại qua 200 kỷ nguyên (Epochs). 
  > Mô hình đã vượt qua khóa huấn luyện. Và giờ, hãy đem nó ra thực chiến."

---

### 🟠 PHẦN KẾT QUẢ THỰC CHIẾN (Slide 6 - 9) - ĐIỂM SÁNG TRỌNG TÂM

#### Slide 6: Bảng Tổng kết CV% (05:00 - 05:45)
- **Hành động & Cử chỉ:** Bước sang một nửa bước, chỉ vào phần bôi xanh/đỏ trên bảng. Giọng quyết đoán.
- **Lời thoại:**
  > "Một hệ thống tốt là hệ thống công bằng. Trên hình là độ lệch mức tải của hệ thống, thuật ngữ chuyên ngành đo bằng chỉ số CV% - lệch càng thấp thì càng tốt.
  > 
  > Như Thầy Cô đã thấy, ở các kịch bản nhẹ như Lưu lượng ổn định, thuật toán truyền thống không chênh lệch quá nhiều. MỌI SỰ SAI KHÁC CHỈ LỘ RA Ở 'FLASH CROWD'. Thuật toán cũ CV vọt lên mất kiểm soát, trong khi Thực Thể AI duy nhất bằng sợi chỉ xanh dương duy trì sự điềm tĩnh và phân bổ tải với độ lệnh cực tiểu."

#### Slide 7: Phân bổ tải - Flash Crowd (05:45 - 06:45)
- **Hành động & Cử chỉ:** Chỉ vào cột đỏ (cao ngất) so sánh với cột xanh (an toàn).
- **Lời thoại:**
  > "Điểm 'thần kỳ' đầu tiên nằm ở đây. Giây phút cơn bão truy cập ập đến hệ thống:
  > Với thuật toán tĩnh truyền thống (cột màu đỏ), nó tiếp tục tuân lệnh ngu ngốc, nện thẳng phần việc khổng lồ vào chiếc máy yêu ớt h5 10Mbps. Kết quả là h5 bị tắt đường thở, ứng dụng chính thức sập.
  > 
  > Nhưng với AI (cột màu xanh lục)... Ngay khi Transformer ngửi mùi bão kéo đến, nó lập tức gạt bay chỉ thị tĩnh. Nó khóa van xả vào chiếc h5 đang yếu đuối, chỉ cho khoảng nhỏ giọt dưới 1MB đi qua, và nó dốc toàn bộ luồng nước dữ dội trên 40MB sang chiếc máy h8 băng thông rộng 100Mbps lúc này đang rảnh rỗi. AI đã làm thay công việc của một chuyên gia vận hành mạng ngay lập tức."

#### Slide 8: Kịch bản Nghẽn Mục Tiêu (06:45 - 07:30)
- **Hành động & Cử chỉ:** Lướt nhanh Slide 8 để củng cố thêm luật.
- **Lời thoại:**
  > "Càng ấn tượng hơn là khi cố tình tạo bẫy nghẽn mạng do có kịch bản dội bom. Mạng truyền thống tiếp tục lùa truy cập vào nút đã chết. Còn DQN thì ngay lập tức nhận hình phạt âm, nó tự cắn đuôi và né hoàn toàn máy chủ đó ra khỏi con đường định tuyến."

#### Slide 9: Tổng Công suất Hệ thống (Throughput) (07:30 - 08:30)
- **Hành động & Cử chỉ:** Chỉ theo đường đồ thị đi lên của AI (màu xanh).
- **Lời thoại:**
  > "Vậy sự bẻ tay lái đó mang lại kết quả toàn cục thế nào? Mời hội đồng xem Tốc độ truyền tải thực tế (Throughput) của cả mạng.
  > 
  > Nửa đầu đồ thị, cả 2 đều giống nhau. Mũi tên đỏ chỉ Giây thứ 50, lượng tải khổng lồ giáng xuống. Đường màu cam đỏ của thuật toán truyền thống rớt thẳng đứng cắm đầu, vì các cổng kết nối bị đầy hàng chờ (Queue Buffer overflow), hệ thống nghẹt thở. 
  > Nhưng hãy nhìn sợi chỉ xanh dương của AI... Bằng cách luân chuyển qua máy tính rảnh rỗi, Thông lượng không rớt xuống, mà ngược lại nó hấp thụ toàn bộ gói tin, búng tốc độ của cả mạng lưới gấp 5 lần công suất ban đầu để gỡ nghẽn."

---

### 🟣 PHẦN ĐỘ TRỄ SUY LUẬN & KẾT LUẬN (Slide 10 - 13)

#### Slide 10: Độ Trễ Suy Luận - Inference Time (08:30 - 09:30)
- **Hành động & Cử chỉ:** Dừng nhẹ lại. Đặt câu hỏi tương tác ngầm để đập tan sự hoài nghi của hội đồng (vì mạng AI nặng thường trễ).
- **Lời thoại:**
  > "Thưa Hội đồng, chắc hẳn mọi người đang tự hỏi: AI thì biểu diễn rất đẹp đó, nhưng mô hình Deep Q-Network kết hợp Transformer thì rất phức tạp, liệu độ trễ khi chạy có 'bóp chết' tốc độ mạng trước khi kịp cứu mạng không?
  > 
  > Nhóm em tính toán vô cùng khắt khe chuẩn này. *(Chỉ vào slide)* Kết quả thực nghiệm của Python và OpenFlow chứng minh: Toàn bộ quá trình từ lúc lấy 5 state tĩnh mạng - Nhúng qua mô hình - và xuất ra Action điều hướng chỉ mất trung bình **33 mili-giây**.
  > Trong viễn thông mạng nội bộ, mốc đánh giá độ trễ bẻ luồng (Overhead) phải dưới 50 mili-giây. Con số 33ms hoàn toàn xuất sắc xác lập mô hình hoàn toàn Đủ Hỏa Lực, mà vẫn Vô Cùng Mỏng Nhẹ để bay lượn trong switch mạng thực tiễn."

#### Slide 11: Kết Luận và Hướng phát triển (09:30 - 10:30)
- **Hành động & Cử chỉ:** Tổng kết mạnh mẽ, giọng chắc chắn, dõng dạc.
- **Lời thoại:**
  > "Để đúc kết lại, Nghiên cứu đã chứng minh: Kiến trúc SDN kết hợp mô hình học AI TFT-DQN là lối đi tân tiến thay đổi lịch sử Load Balancer cổ điển. AI trong nhóm em đã tự nhận diện băng thông máy móc, tự dự báo đám đông và tự nắn dòng tải để cứu hệ thống.
  > 
  > Dù đã thành công ở độ trễ thấp và trên mạng mô phỏng 16 node, tham vọng cho Hướng phát triển tiếp theo của nhóm chính là: Rút ngắn mô hình lại để đẩy lên cơ chế học **Online Reinforcement Learning**, giúp con bot mạng có thể tự tu sửa bản thân trực tiếp khi nếm đòn từ Live Traffic ngoài đời thực; đồng thời tích hợp thêm tầng Nhận diện phòng vệ DDoS đa chiều."

#### Slide 12 & 13: Lời Cảm Ơn
- **Hành động & Cử chỉ:** Bước ra, gật đầu mỉm cười và cúi chào trịnh trọng.
- **Lời thoại:**
  > *(Slide 12: Dành mười giây để người xem nhìn tài liệu tham khảo)*
  > "Dạ thưa quý Thầy Cô, nghiên cứu này không chỉ là những con số, mà còn là tâm huyết của nhóm với hy vọng mang Machine Learning giải quyết các bài toán hạ tầng ngầm thiết thực. 
  > Em xin kết thúc phần trình bày. Kính gửi lời cảm ơn sâu sắc nhất tới thầy hướng dẫn và hội đồng đánh giá đã lắng nghe. Nhóm em rất mong nhận được các câu hỏi kiểm tra và sự phản biện quý báu từ quý Thầy Cô ạ!"

---

## 🛡️ TÀI LIỆU "BỎ TÚI" - CÁC CÂU HỎI PHẢN BIỆN DỰ KIẾN (Q&A)

1. **Câu hỏi:** *Tại sao lại dùng tận Transformer thay vì LSTM thường? Mạng quá nặng thì sao?*
   - **Trả lời:** Dạ, LSTM chỉ cho dự báo điểm thời gian ngắn hạn và đuối sức khi tìm kiếm tương quan chuỗi dài. Mạng TFT (Temporal Fusion Transformer) tích hợp cơ chế Attention mechanism đa đầu, giúp chọn lọc mốc đột biến sinh viên vào server tốt hơn. Nhóm dùng TFT *cỡ nhỏ* (chỉ 32 hidden size, 4 heads) nên vẫn đảm bảo Inference latency dưới mặt sàn 50ms của OpenFlow.

2. **Câu hỏi:** *Reward function của nhóm thiết kế thế nào để AI không 'ảo tưởng'?*
   - **Trả lời:** Đây chính là tinh hoa nhóm tối ưu - `Reward V3 Clean Static`. Nhóm kết hợp Baseline + Balance Bonus. Nếu dồn thêm vào server đang ngắc ngoải (max load), điểm bị phạt trừ siêu nặng (-3 * load_spread). Ngược lại, chia thành công dồn qua server đang rảnh (min load) sẽ được cộng lớn (+4). Đặc biệt, khi có cờ báo hiệu tải HIGH đỏ rực, hệ số Reward được nhân đôi `x2` bắt buộc nó phải hành động cứu mạng.

3. **Câu hỏi:** *Tại sao lại tạo các Switch bất đối xứng 10, 50, 100 Mbps?*
   - **Trả lời:** Thực tiễn trong công nghệ Cloud, Data Center luôn tồn tại các node cũ và node mới. Các thuật toán như RR hay Weighted RR rất thủ cựu, phải set bằng tay weight và khi nghẽn nó không thay đổi theo phần cứng. Khi em set bất đối xứng, AI sẽ bị "vứt vào rừng sâu", phải tự đi đo độ sâu của từng máy bằng thuật toán Q-Learning để biết đâu là biển nước (100M) đâu là vũng lầy (10M), từ đó chứng minh được AI thông minh độc lập tuyệt đối.

4. **Câu hỏi:** *Khi AI cài luật trên switch, bảng Flow Table có bị tràn không?*
   - **Trả lời:** Dạ không thưa thầy/cô. Tại bước 4 xử lý Data Plane, Ryu Controller cài bộ lệnh chuyển địa chỉ NAT IP đích - IP nguồn kết hợp Hard Timeout là 30 giây (Idle timeout). Mọi kết nối của sinh viên sau khi hoàn thành request, luật đó sẽ tự hết hạn và quét sạch khỏi mặt phẳng Switch, giữ Switch bộ nhớ luôn sạch sẽ và nhỏ gọn.

---
*(Kịch bản này đã được hiệu chuẩn thời gian chính xác và biên soạn với ngôn từ học thuật chuẩn "Nghiên cứu khoa học - Hội đồng Công nghệ Thông tin".)*
