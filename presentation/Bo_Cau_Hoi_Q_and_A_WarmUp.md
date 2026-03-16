# 🌟 BỘ TÀI LIỆU "WARM-UP": 30 CÂU HỎI Q&A DỄ MỨC ĐỘ CƠ BẢN - TRUNG BÌNH

> **Mục đích:** Document này được chuẩn bị dành cho các câu hỏi mang tính "khởi động" (Warm-up), thường đến từ các thầy cô muốn kiểm tra xem sinh viên có **thực sự hiểu bản chất đề tài của mình không**, hay chỉ là đi copy code trên mạng.
> **Chiến thuật trả lời:** Tự tin, tươi tắn, dùng từ ngữ tượng hình dễ hiểu, không cần đi quá sâu vào thuật toán trừ khi được hỏi xoáy thêm.

---

## 🟢 NHÓM 1: MỤC TIÊU & TỔNG QUAN ĐỀ TÀI (6 Câu)

**1. Tóm tắt lại trong 1-2 câu, mục tiêu lõi của đề tài các em là gì?**
> **Trả lời:** Dạ, mục tiêu lõi của nhóm em là dùng một mô hình AI thông minh (TFT-DQN) để thay thế các thuật toán chia tải tự động cứng nhắc cũ. Nó làm nhiệm vụ làm "người gác cổng" chia đều lượng sinh viên truy cập vào các máy chủ theo năng lực thời gian thực để chống sập mạng thưa thầy/cô.

**2. Tại sao lại lấy ý tưởng là "Đăng ký tín chỉ", có phải vì trường mình hay nghẽn không?**
> **Trả lời:** *(Cười nhẹ)* Dạ đây là một "nỗi đau" rất điển hình của tụi em ạ. Đăng ký tín chỉ là ví dụ hoàn hảo nhất cho hiện tượng mạng gọi là "Flash Crowd" - Đám đông bùng phát. Tụi em muốn giải quyết một bài toán thực tế sinh viên hay gặp thay vì một bài toán vĩ mô xa xôi ạ.

**3. Công sức của nhóm trong dự án này tập trung nhiều nhất ở đâu? Có phải code từ con số 0 hết không?**
> **Trả lời:** Dạ, phần khung Mininet ảo hóa và kiến trúc nền DQN thì nhóm dùng các thư viện mã nguồn mở có sẵn. Khối lượng công sức cực lớn của nhóm nằm ở 3 điểm: Thiết kế môi trường bất đối xứng (10-50-100 Mbps), Tự code hàm Phần thưởng V3 để AI thực sự khôn ra, và tự giả lập kịch bản mạng bắn bằng Artillery ạ.

**4. Dữ liệu luồng (Streaming Data) trong tên đề tài của em nghĩa là gì? Tại sao không dùng dữ liệu tĩnh?**
> **Trả lời:** Dạ, dữ liệu luồng ở đây là các con số "Lượng gói tin (Packet_count)" và "Dung lượng (Byte_count)" được Switch báo cáo liên tục về Controller mỗi vài giây. Mạng thì luôn biến động từng giây, nên phải thu thập và tính toán dữ liệu đang chảy (dữ liệu luồng) thì AI mới bẻ lái kịp thời ạ. 

**5. SDN là gì? Giải thích ngắn gọn cho người ngoài ngành hiểu.**
> **Trả lời:** SDN (Mạng định nghĩa bằng phần mềm) là kiến trúc lôi bộ não (Control Plane) của tất cả các cục Switch vứt lên tập trung ở một cái Controller trung tâm. Giống như cảnh sát giao thông đứng trên cao nhìn được toàn thành phố, thay vì tự ai nấy lo phân luồng ở từng ngã tư ạ.

**6. AI của em tên là TFT-DQN. Viết tắt của chữ gì?**
> **Trả lời:** Dạ, TFT là *Temporal Fusion Transformer* - mạng chuyên dự báo tương lai. Còn DQN là *Deep Q-Network* - mạng chuyên bẻ lái ra quyết định (Action) dựa trên thuật toán Học tăng cường ạ.

---

## 🟡 NHÓM 2: CÁCH HOẠT ĐỘNG CỦA HỆ THỐNG (7 Câu)

**7. Hệ thống của em phân biệt thế nào giữa một luồng mạng bình thường và luồng "Flash Crowd"?**
> **Trả lời:** Dạ, trong giao diện nhãn dữ liệu, nếu tổng lưu lượng đổ về vượt qua một ngưỡng chịu đựng (Threshold) của các máy chủ mà tụi em thiết lập sẵn, hệ thống sẽ tự động dán nhãn là 'HIGH' (Quá tải/Flash Crowd), còn dưới ngưỡng đó là 'NORMAL' (Bình thường).

**8. Kịch bản "Gradual Shift" (Chuyển biến chậm) có ý nghĩa gì trong số 4 kịch bản?**
> **Trả lời:** Dạ, kịch bản này lưu lượng cứ 5 phút lại tăng lên một nấc, không tăng sốc. Mục đích chính của nó là để sinh ra bộ dữ liệu huấn luyện (Training Data) "đẹp" nhất cho AI đi học, vì nó rải đều từ lúc rảnh rỗi đến lúc sập mạng để AI đủ góc nhìn ạ.

**9. Round-Robin (RR) và Weighted Round-Robin (WRR) khác nhau thế nào?**
> **Trả lời:** Dạ, Round-Robin là chia bài chia đều: Sinh viên 1 vô máy A, sinh viên 2 vô máy B, xoay vòng. Còn *Weighted* Round-Robin là có trọng số tĩnh: Em set máy B mạnh gấp đôi máy A, thì em chia 2 sinh viên vô máy B, 1 sinh viên vô máy A. 

**10. Nếu WRR (chia theo trọng số) đã nhận ra rẳng máy B mạnh gấp đôi máy A, thế sao không dùng luôn đi cần gì AI?**
> **Trả lời:** Dạ, câu hỏi rất hay ạ! WRR là một hằng số tính trước. Nếu hôm đó máy B (dù béo) đột nhiên hỏng ổ cứng nghẽn bất thình lình, cái thông số WRR nó không tự thay đổi được, nó vẫn cố đấm ăn xôi nén tải vào máy B cho chết hẳn. AI của em thì "thấy nghẽn là lập tức né" theo thời gian thực ạ.

**11. Node.js và PostgreSQL có vai trò gì trong hệ thống mạng này? Sao không test mô phỏng chay (Ping loop)?**
> **Trả lời:** Dạ, nếu chỉ dùng `Ping` gửi gói tin trống rỗng thì các switch mạng hầu như chẳng tốn sức xíu nào, không xi-nhê gì để AI học cả. Nên nhóm cố tình dựng một Web App Node.js và DB Postgres y chang trang web đăng ký môn thật, để tạo ra lượng Traffic và độ trễ rớt mạng nặng độ, thực tế nhất ạ.

**12. VIP là gì? User truy cập mạng thì gõ IP nào?**
> **Trả lời:** VIP là Virtual IP (IP Ảo). User sinh viên chỉ gõ một IP duy nhất là IP Ảo mặt tiền (`10.0.0.100`), không hề biết phía sau có máy h5, h7, h8. Cái IP ảo này AI sẽ tự động "hô biến" thành IP thật để bẻ luồng ngầm mượt mà.

**13. Bảng Flow Table trong Switch là gì?**
> **Trả lời:** Dạ thưa nó giống như "bảng chỉ đường" cắm ở ngã tư. Ví dụ: Sinh viên IP A đi tới đây, bắt buộc quẹo phải vào Máy Chủ h8. Controller AI là người đứng từ xa viết nội dung lên cái bảng này ạ.

---

## 🟢 NHÓM 3: KHÁI NIỆM AI & HUẤN LUYỆN (7 Câu)

**14. Học Tăng Cường (Reinforcement Learning) khác gì với AI Nhận diện khuôn mặt (Supervised Learning)?**
> **Trả lời:** Nhận diện khuôn mặt là em đưa AI 10 vạn bức ảnh mèo, bắt nó học thuộc chó/mèo (Học có giám sát). Còn Mạng của em là Học tăng cường - Mất bò mới lo làm chuồng. Em vứt nó vô vòng chạy ngẫu nhiên, làm đúng em cho Kẹo (Cộng điểm), làm sai em Gõ đầu (Phạt âm điểm), nó tự sợ mà thông minh lên ạ.

**15. AI Của nhóm có chạy trên Internet thực tế không hay chỉ chạy offline?**
> **Trả lời:** Dạ, phần *Học (Training)* nhóm em nhốt nó chạy Offline để nó không làm hỏng mạng thật. Nhưng sau khi nó "tốt nghiệp", file AI `.pth` nó đẻ ra sẽ được bê lên Bộ điều khiển chạy Online thời gian thực để dập nghẽn luôn ạ.

**16. Quá trình tiền xử lý dữ liệu (Min-Max Normalize) để làm gì?**
> **Trả lời:** Dạ, số `byte_count` mạng thì to dài thòng lọng (ví dụ hàng tỷ byte), nhưng cái trọng số tải có khi chỉ nằm từ 0 tới 1. Để AI không bị "sốc tỉ lệ chữ số", em phải nén tât cả các con số khổng lồ đó về thang đo từ 0.0 đến 1.0 (chuẩn hóa) để AI dễ tiêu hóa ạ.

**17. "Sliding Window - Cửa sổ trượt 5 timestep" dễ hiểu là sao em?**
> **Trả lời:** Giống như mình coi 5 frame ảnh trên video đoán người ta chuẩn bị đi thế nào. Tại thời điểm T, em gom dữ liệu của T, T-1, T-2, T-3, T-4 đưa một cục cho AI xem. Cứ thế trượt dần đi 1 giây. Nó nhìn 5 giây quá khứ liên tục để tiên tri 1 giây tương lai tới ạ.

**18. Bảng số liệu Loss đi xuống nghĩa là gì trong quá trình Train?**
> **Trả lời:** Loss là độ sai số của mạng AI sau mỗi vòng học. Biểu đồ cắm đầu đi xuống sát mức 0 nghĩa là mức dự đoán của AI so với thực tế sai số ngày càng nhỏ dần, mô hình đã "hiểu bài" ạ.

**19. Bảng Reward (Phần thưởng) đi lên kịch trần có nghĩa là gì?**
> **Trả lời:** Tức là thay vì ban đầu đi lung tung bị phạt liên tục, AI nay đã tìm ra chân lý luôn luôn lách luồng vào máy mạnh nhất để gom tối đa túi tiền thưởng. Lên kịch trần tức là nó đã hoàn thiện kĩ năng ạ.

**20. Batch size (Sample) trong AI là gì? Em setup bao nhiêu?**
> **Trả lời:** Dạ là mỗi lần học, em bốc một cụm bao nhiêu bài tập cho nó làm đồng loạt cho nhanh. Nhóm setup Minibatches bốc ngẫu nhiên (chống học vẹt) từ một kho nhớ Memory Buffer lưu 50,000 sự kiện lịch sử qua khứ ạ.

---

## 🟡 NHÓM 4: KẾT QUẢ ĐÁNH GIÁ (7 Câu)

**21. Trong đồ thị Phân bổ tải, Cột màu Đỏ và Cột màu Xanh lục đại diện cho gì?**
> **Trả lời:** Dạ, màu đỏ đại diện cho con số cứng nhắc của thuật toán cũ (Round Robin/WRR) - phân phối sai trái gây quá tải máy h5. Màu Xanh là sự thông minh bẻ lái của mô hình em, né tránh máy yếu h5 mà đập luồng mạnh qua máy xịn h8.

**22. Nhìn đồ thị, tại sao Tốc độ (Throughput) lúc Đăng ký tín chỉ tụt thê thảm đối với Mạng cũ?**
> **Trả lời:** Dạ vì khi lượng lớn sinh viên nghẽn cứng ở máy tính số 1, thì đường truyền đứt. Gói tin (Request) cứ phải liên tục gửi lại (Re-transmit). Càng gửi lại càng chất đống tắt đường cục bộ, nên năng lực phản hồi của toàn mạng đứt gánh đi xuống bằng 0 ạ.

**23. Inference Time (Thời gian suy luận) của AI nhóm em là 33ms. Nhanh hay chậm?**
> **Trả lời:** Rất nhanh và đạt chuẩn quốc tế thưa thầy cô! Các chuẩn mạng SDN đều yêu cầu Overhead để xử lý luật phải ép dưới 50 mili-giây thì sinh viên bấm Web mới không thấy bị Delay. 33ms hoàn toàn nằm trọn trong ngưỡng siêu an toàn ạ.

**24. CV% (Độ biến thiên) hiểu dễ nhất là gì?**
> **Trả lời:** Giống như điểm thi đều hay lệch ạ. Hệ thống lý tưởng là khi cả 3 server đều hoạt động với công suất 60% 61% 59%. Lúc đó độ lệch là 0%. CV% cao nghĩa là có 1 máy 99% (ngộp thở), còn máy khác thì 10% (ngáp ruồi).

**25. Kịch bản số 2 "Ramping" (Tăng tải đều) kết quả ra sao?**
> **Trả lời:** Dạ, ở kịch bản "sóng yên biển lặng" này thì AI và các công thức cổ điển chênh lệch nhau không nhiều (do chưa đụng mức tới hạn). Tuy nhiên nó chứng minh rằng AI của tụi em làm tốt cả những việc nhỏ nhặt hằng ngày chứ không làm rườm rà hệ thống ạ.

**26. Trong thực tế, AI của nhóm em mất bao lâu để phát hiện ra 'có đột biến'?**
> **Trả lời:** Tính theo thời gian thực thu thập số liệu (Polling interval) là 3 giây cho mỗi bước, thêm 33ms Model suy luận. AI chỉ mất khoảng một cái chớp mắt để tung ra quyết định bẻ bão rồi thưa hội đồng.

**27. Có khi nào AI của nhóm phân quá nhiều qua máy mạnh h8, làm cho máy h8 lăn ra chết luôn không?**
> **Trả lời:** Dạ không ạ. Nhóm có thiết lập con số Temperature (Tau = 0.5) ghim dưới Hàm kích hoạt. Thay vì AI tham lam nện 100% vào h8, Tau sẽ tán đều tải một cách mượt mà (Smoothing) bắt máy h7 cũng phải gánh chia bớt xíu để h8 không ngộp thở ngay cả khi nó rảnh ạ.

---

## 🔵 NHÓM 5: MỞ RỘNG & TÍNH THỰC TIỄN (3 Câu)

**28. Chi phí lớn nhất nếu mang giải pháp của em bán cho trường áp dụng là gì?**
> **Trả lời:** Chi phí lớn nhất là quá trình Thu thập thông số tĩnh Offline ban đầu để Huấn luyện AI sinh ra Model. Còn khi đã có file Model (vài MB) nhét lên Server chạy Inference để điều phối hằng ngày thì tốn cực ít vi xử lý (chạy CPU thường vẫn kịp). Chi phí dài hạn là siêu rẻ ạ.

**29. Nếu sinh viên có đường truyền mạng ở nhà yếu (lag gốc), thì AI em xử lý sao?**
> **Trả lời:** Dạ AI của em chỉ can thiệp điều phối trong biên giới các Server của Trường Đại học. Nếu mạng nhà sinh viên bị cá mập cắn cáp thì AI đằng trước của Data Center không can thiệp được khúc đó ạ.

**30. Nhóm e 4 thành viên (Hiển, Phúc, Minh, Triết), có gặp bất đồng lớn nào khi ráp model AI lên mạng không?**
> **Trả lời:** Dạ có ạ *(cười)*. Giai đoạn lớn nhất là khi team Mạng viết script chạy bắn tải, còn team AI ráp code vào Train thì bị "nổ" lỗi vì tốc độ hai bên không khớp nhau. Nhóm phải ngồi chụm đầu rã file CSV lại, chia Window step và thiết kế ra cái Hàm Reward V3 để máy học mượt mà như hôm trình bày đây ạ. 

---
> 💡 **Tip cho nhóm:** Nếu một thầy cô hỏi câu quá căn bản, anh hãy nở một nụ cười tươi, giữ phong thái tự nhiên như sinh viên trao đổi học thuật bình thường, tránh việc có thái độ "câu này dễ ẹt". Giám khảo thường dùng nhóm câu hỏi này để kiểm tra xem team work có nắm đều kỹ thuật hay là rớt hẳn việc làm AI cho một bạn duy nhất "gánh".
