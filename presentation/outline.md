# KỊCH BẢN THUYẾT TRÌNH CHI TIẾT ĐỀ TÀI NCKH

**Tên đề tài:** Ứng dụng Học máy trong Phát hiện Truy cập Bất thường trên Dữ liệu Luồng
**Sinh viên trình bày:** Đặng Quang Hiển
**Lớp:** S28-67CNTT

---

## TỔNG QUAN BỐ CỤC
⏱ **Thời lượng dự kiến:** 10 - 15 phút
🎯 **Mục tiêu:** Thuyết phục Hội đồng về tính ưu việt của mô hình AI (TFT-CQL Actor-Critic) trong việc tự động điều phối lưu lượng mạng SDN thay vì các thuật toán truyền thống.

| Phần | Nội dung | Số phút |
|---|---|---|
| **P1** | Giới thiệu & Đặt vấn đề (Tại sao cần AI?) | 2 phút |
| **P2** | Giải pháp đề xuất (SDN + TFT-CQL + Topology) | 2 phút |
| **P3** | Thực nghiệm: Môi trường & Thiết lập kịch bản | 2 phút |
| **P4** | **Huấn luyện Mô hình AI (Training Dashboard)** | 1.5 phút |
| **P5** | **Thực chiến & Kết quả (Trọng tâm - Điểm nhấn)** | 4 phút |
| **P6** | Kết luận & Hướng phát triển | 1.5 phút |

---

## KỊCH BẢN CHI TIẾT (LỜI THUYẾT TRÌNH & HÀNH ĐỘNG)

> **💡 Mẹo:** Các phần `[HÀNH ĐỘNG]` là ghi chú thao tác click slide hoặc chỉ tay. Phần in nghiêng là lời thoại. Hãy nói với tốc độ vừa phải, tự tin và ngắt nghỉ đúng chỗ.

### PHẦN 1: GIỚI THIỆU & ĐẶT VẤN ĐỀ

**[Slide 0: Trang bìa (Tên đề tài & Sinh viên)]**
Dạ, em chào quý thầy cô trong hội đồng. Em là Đặng Quang Hiển, sinh viên lớp S28-67CNTT. Hôm nay em xin phép được trình bày đề tài Nghiên cứu Khoa học: *"Ứng dụng học máy trong phát hiện truy cập bất thường trên dữ liệu luồng"*. 
Trọng tâm nghiên cứu của nhóm là thiết kế một Cơ chế điều phối Controller SDN kết hợp Mô hình AI thông minh (TFT-DQN) có khả năng tự động bẻ luồng dữ liệu, đặc biệt là trong các tình huống đột biến lưu lượng như ngày đăng ký tín chỉ khiến hệ thống bị quá tải hoặc bị tấn công đột ngột.

**[Slide 1: Đặt vấn đề - Bối cảnh thực tế]**
Quý thầy cô chắc hẳn rất quen thuộc với bài toán nghẽn mạng "huyền thoại" mỗi dịp đăng ký tín chỉ: Hàng ngàn sinh viên cùng ập vào một lúc khiến hệ thống server quá tải và sập hoàn toàn.
Hiện nay, đa số các Load Balancer (Bộ cân bằng tải) truyền thống đang sử dụng các thuật toán như Round-Robin (chia đều) hoặc Weighted Round-Robin (chia theo trọng số cố định). Chúng hoạt động như các "cỗ máy mù", không phản ứng kịp khi một server bất ngờ đuối sức hay xảy ra hiện tượng "cổ chai mạng".

Do đó, nhóm chúng em đặt ra câu hỏi: *Liệu có thể tạo ra một "bộ não" AI có khả năng cảm nhận độ trễ thời gian thực để dồn traffic khỏi máy bị nghẽn và cứu toàn hệ thống không?* Đó là lý do mô hình của chúng em ra đời.

---

### PHẦN 2: CƠ SỞ LÝ THUYẾT & GIẢI PHÁP ĐỀ XUẤT

**[Slide 2: Mạng SDN và Sơ đồ Thực nghiệm Fat-Tree]**
Để hiện thực hóa điều đó, chúng em triển khai kiến trúc SDN (Software-Defined Networking), nơi "não bộ" là Controller (bộ điều khiển trung tâm) được tách rời khỏi phần cứng thiết bị mạng. 

**[Slide 3: Mô hình AI TFT-CQL Actor-Critic (Lõi hệ thống)]**
Và "trái tim" của hệ thống này chính là mô hình Học Tăng Cường Offline (Offline Reinforcement Learning) tên là **TFT-CQL Actor-Critic**. Mô hình kết hợp 2 thành tố chính:
1. Thứ nhất, mạng **Transformer (TFT)** liên tục nhìn lại 5 bước thời gian trong quá khứ để "đoán trước bão" – dự báo lưu lượng tiếp theo đổ về.
2. Thứ hai, mạng **Actor-Critic với Conservative Q-Learning (CQL)** dùng chính dự báo đó để ra quyết định chia cục traffic hiện tại vào các server dựa trên **Hàm Phần Thưởng (Reward Function)**.
   *(Hơi dừng lại một nhịp)* Cụ thể, hệ thống sẽ "thưởng điểm" tính theo độ lệch tải (Load Spread) nếu mô hình chọn đẩy dữ liệu vào server đang vắng khách, và "phạt điểm âm" nếu cố tình đẩy vào server đã đầy tải. AI qua hàng ngàn lần thử sai sẽ tự động tiến hóa để thu gom phần thưởng cao nhất, tức là hệ thống được cân bằng tải hoàn hảo.

Để chứng minh tính thực tiễn, em xin đưa hội đồng đi thẳng vào phần Thử nghiệm và đánh giá thực tế mà nhóm đã chuẩn bị.

---

### PHẦN 3: GIAI ĐOẠN CHUẨN BỊ (THIẾT LẬP THỰC NGHIỆM)

**[Slide 4: Môi trường thử nghiệm & Tech Stack]**
Để xây dựng một môi trường mạng hoàn chỉnh mà không tốn chi phí phần cứng đắt đỏ, chúng em đã thiết lập toàn bộ hệ thống mô phỏng thông qua mạng máy tính ảo.
*`[Chỉ tay vào danh sách công nghệ trên slide]`* Toàn bộ hệ thống chạy trên nền tảng **Docker**. Mạng SDN vật lý K=4 được ảo hóa bởi **Mininet** và **OVS Switches**. Não bộ trung tâm được lập trình bằng Python qua framework **Ryu Controller**. Phía thiết bị đầu cuối, chúng em xây dựng một hệ thống thi / đăng ký tín chỉ tự code có Backend viết bằng **Node.js** và Cơ sở dữ liệu **PostgreSQL** có sức chứa 5.000 users.

**[Slide 5: Cấu trúc Topology Bất đối xứng]**
*`[Chỉ tay vào sơ đồ 3 server]`* Để thử thách khả năng "học và hiểu phần cứng" của AI, chúng em không dùng các server giống nhau. Nhóm thiết kế một "nút thắt cổ chai" với 3 Backend Server có sức mạnh khác biệt hoàn toàn:
- Máy số **h5** là máy "yếu ớt" (băng thông bị giới hạn ở 10 Mbps)
- Máy số **h7** có sức mạnh trung bình (50 Mbps)
- Máy số **h8** là siêu máy tính nội bộ (100 Mbps)

**[Slide 6: Bắn tải & 4 Kịch bản Thử nghiệm bằng Artillery]**
Dạ thưa quý thầy cô, để nhào nặn ra "sức ép", chúng em đã thiết lập cụm 8 máy khách liên tục "bắn" request vào mạng bằng công cụ test tải **Artillery**. Nhóm đã thiết kế 4 kịch bản tạo hình thái lưu lượng khác nhau:
1. **Kịch bản 1 (Golden Hour):** Cơn lốc truy cập (Ngày mở đăng ký) – 1.000 user ập thẳng vào server trong 2 phút.
2. **Kịch bản 2 (Video Conference):** Mô phỏng họp trực tuyến, băng thông ổn định nhưng độ trễ yêu cầu cao.
3. **Kịch bản 3 (Hardware Degradation):** Mô phỏng suy giảm phần cứng – một server dần mất khả năng xử lý.
4. **Kịch bản 4 (Low-Rate DoS):** Tấn công từ chối dịch vụ tốc độ thấp – nhắm thẳng vào làm tê liệt một máy duy nhất.

Và ngay sau đây là cách chúng em tiến hành huấn luyện AI và bức tranh tổng thể về kết quả đạt được.

---

### PHẦN 4: QUÁ TRÌNH HUẤN LUYỆN (TRAINING PHASE)

**[Slide 5: Quá trình Huấn luyện AI] - (Tương ứng: 00_training_dashboard.png)**
*`[Chỉ vào biểu đồ huấn luyện]`* Trước khi đưa AI vào đương đầu với các kịch bản khắc nghiệt, mô hình phải trải qua giai đoạn "đi học" (Offline Training). Màn hình quý thầy cô đang xem chính là Bảng điều khiển (Training Dashboard) giám sát quá trình học của AI.
Trải qua hàng chục ngàn bước thử và sai (Steps/Epochs), đường biểu diễn hàm Lỗi (Loss) giảm dần và hội tụ ổn định, trong khi Tích lũy Phần thưởng (Reward) tăng vọt lên mức tối đa. Điều này minh chứng cho việc AI đã thực sự "hiểu" được không gian mạng, học được quy luật phần cứng của từng server và biết cách tự đưa ra quyết định phân bổ luồng tối ưu nhất.
Và thành quả của quá trình học tập này được thể hiện rõ nhất khi bước vào thực chiến.

---

### PHẦN 5: KẾT QUẢ THỰC CHIẾN (TRỌNG TÂM - ĐIỂM NHẤN)

**[Slide 6: Bảng tổng kết 4 kịch bản] - (Tương ứng: P4_bang_tongket.png)**
*`[Chỉ vào Bảng CV%]`* Ở đây, chúng em dùng chỉ số độ lệch CV% để đo độ công bằng tải (chỉ số càng nhỏ thì hệ thống càng trơn tru). 
Như quý thầy cô thấy, ở 2 kịch bản mạng dễ đoán (*Video Conference* và *Hardware Degradation*), thuật toán chia theo trọng số tĩnh thông thường (WRR) vẫn hoạt động rất tốt.
Tuy nhiên, khi đối diện với 2 kịch bản "sốc hông" là *Golden Hour* và *Low-Rate DoS*, thì công nghệ cũ vỡ trận. Chỉ duy nhất mô hình AI của hệ thống xuất sắc vượt quá sự mong đợi, duy trì độ công bằng tốt nhất. Cụ thể nó vận hành ra sao? Em xin mời hội đồng xem kịch bản thứ 1: Golden Hour.

**[Slide 7: Phân bổ tải - Golden Hour] - (Tương ứng: P1_phan_bo_tai_golden_hour.png)**
Khi một lượng lớn truy cập bất ngờ đổ về, thuật toán chia đều cổ điển (cột màu đỏ) vẫn tiếp tục "mù quáng" đẩy lượng công việc y hệt nhau vào cả 3 máy chủ, hậu quả là máy h5 (chỉ có 10Mbps băng thông) bị quá tải hoàn toàn và tê liệt cục bộ.
Ngược lại, hãy nhìn vào hành vi *màu xanh của AI*. Nó tự động nhận diện phần cứng máy 5 đang đuối sức nên đã bẻ luồng thông minh: lập tức dồn lượng traffic khổng lồ (hơn 40MB) sang máy h8 mạnh nhất, và chỉ giao cho máy h5 một lượng yêu cầu rất nhỏ (dưới 1MB). Như một lá chắn thép, AI đã bảo vệ thành công máy chủ yếu nhất trước cơn bão truy cập.

**[Slide 8: Phân bổ tải - Low-Rate DoS] - (Tương ứng: P1_phan_bo_tai_low_rate_dos.png)**
*`[Chuyển slide]`* Sự thông minh của mô hình càng minh chứng rõ hơn ở kịch bản rủi ro: "Tấn công từ chối dịch vụ tốc độ thấp" nhắm thẳng vào máy số 7.
Trong khi các thuật toán cũ tiếp tục dẩy lưu lượng vào tâm điểm gây nghẽn (cột biểu đồ màu vàng), thì AI của bọn em đã nhạy bén phát hiện và lập tức "né" hoàn toàn máy bị nghẽn đó. Lưu lượng đổ vào máy số 7 thấp kỷ lục, nhường không gian cho máy tính này phục hồi và gánh vác tải cho 2 máy còn lại, đảm bảo toàn mạng lưới không gãy đổ.

**[Slide 9: Tốc độ xử lý (Throughput) - Golden Hour] - (Tương ứng: P3_toc_do_golden_hour.png)**
Vậy sự phân bổ thông minh đó mang lại kết quả tổng thể gì thưa quý thầy cô? Dạ, đây chính là Tốc độ xử lý đồ thị của toàn hệ thống theo thời gian thực.
Ở giây thứ 50 trên màn hình, khi đám đông bắt đầu ập tới, đường màu đỏ của thuật toán chia đều bị sụt giảm cực mạnh và không thể phục hồi do có máy bị crash. Trong khi đó, đường màu xanh của AI đã vọt lên như một cái lò xo, tận dụng tới 99% phần cứng rảnh rỗi của máy 100Mbps để cày nát lượng request này; duy trì một cái nền tảng tốc độ vô cùng ổn định.

**[Slide 10: Thời gian suy luận AI] - (Tương ứng: P5_suy_luan_golden_hour.png)**
*`[Mỉm cười tự tự tin]`* Tới đây, có thể sẽ có một câu hỏi đặt ra: *Mô hình Deep Learning thường cồng kềnh, liệu nó có quá chậm chạp cho việc bẻ nhánh dữ liệu mạng vốn phải "tính bằng mili-giây" hay không?*
Biểu đồ cuối cùng chứng minh tốc độ tư duy của AI. Mô hình của chúng em chỉ mất trung bình **~33 mili-giây** để hoàn thành một chu trình lấy tham số - dự báo - và xuất lệnh. 
Xin lưu ý, thời gian Suy luận Thực tế (Inference Time) này hoàn toàn nằm sâu dưới ngưỡng an toàn độ trễ của SDN (50 mili-giây). Điều này chứng tỏ AI của chúng em đủ hỏa lực mà vẫn vô cùng "mỏng nhẹ" để chạy thực tế mà không gây ra điểm nghẽn (bottleneck).

---

### PHẦN 6: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

**[Slide 11: Kết luận & Hướng phát triển]**
Tóm lại, thông qua kiến trúc mạng ảo hóa SDN, thực nghiệm chứng minh rằng mô hình TFT-CQL Actor-Critic của chúng em đã thay đổi quy luật điều phối truyền thống. Bằng cơ chế học tăng cường offline có phần thưởng, hệ thống tự định hình được quy mô tài nguyên và lập tức né dòng tải khỏi máy chết.
Dù đã đạt kết quả xuất sắc, nhóm vẫn định hướng bước tiếp theo là chuyển sang cơ chế Online Reinforcement Learning - tự sửa mình trực tuyến trên Live Traffic, cũng như mở rộng kịch bản phòng bị tấn công DDos phức tạp hơn.

**[Slide 12: Cảm ơn]**
Dạ, vừa rồi là toàn bộ nội dung của đề tài. Nhóm nghiên cứu xin gửi lời cảm ơn tới sự lắng nghe và đóng góp của hội đồng. Em xin trân trọng cảm ơn!

---
*Kịch bản này được tối ưu nhằm mục đích đẩy mạnh khả năng thuyết phục, kết hợp ngữ điệu tự tin, chặt chẽ của một đề tài ứng dụng AI chuyên sâu.*