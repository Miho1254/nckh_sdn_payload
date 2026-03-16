# 🛡️ BỘ TÀI LIỆU "SỐNG CÒN": 30 CÂU HỎI PHẢN BIỆN TỪ HỘI ĐỒNG KHÓ TÍNH

> **Mục đích:** Document này đóng vai trò như một "Chiếc phao cứu sinh" dành cho các câu hỏi xoáy, câu hỏi hóc búa, hoặc các câu hỏi đánh thẳng vào nhược điểm hệ thống. Bộ câu hỏi được biên soạn với tư duy của một giám khảo chuyên môn sâu về Mạng và Trí tuệ Khôn nhân (AI/SDN).
> **Chiến thuật trả lời:** Nhanh gọn – Đi thẳng vào cốt lõi (Khoa học) – Tự tin – Nêu bật được công sức nghiên cứu (Tránh trả lời kiểu học thuộc).

---

## 🏗️ NHÓM 1: KIẾN TRÚC MẠNG SDN & ĐIỀU PHỐI (5 Câu)

**1. Tại sao nhóm lại đưa bài toán này lên kiến trúc SDN mà không cấu hình Load Balancer thẳng trên Router/Switch truyền thống?**
> **Trả lời:** Dạ thưa thầy/cô, hệ thống mạng truyền thống thì Control Plane bị gắn chết vào từng thiết bị. Nếu đổi luật, ta phải đi cấu hình lại từng Switch vô cùng phức tạp và không có cái nhìn toàn cảnh (Global View). Kiến trúc SDN tập trung bộ não lên Ryu Controller, cho phép AI có thể thu thập dữ liệu của *cả mạng* và lập tức ra lệnh xuống các thiết bị Switch qua OpenFlow.

**2. Quá trình bẻ luồng (Flow Rule) hoạt động như thế nào? Bảng Flow Table của Switch có bị tràn bộ nhớ khi có hàng ngàn user (Flash Crowd) không?**
> **Trả lời:** Em sử dụng cơ chế NAT 5-tuple (Đổi IP/MAC). Để chống tràn bảng luồng, Controller gửi lệnh đi kèm với thông số `idle_timeout = 30s`. Nghĩa là nếu Sinh viên xong phiên duyệt web (hoặc đóng tab), luật đó trong 30 giây không xài tới sẽ bị vứt khỏi Switch, trả lại bộ nhớ. Bảng luồng luôn sạch và nhỏ nhắn.

**3. Tại sao cấu hình mô phỏng mạng lại chọn Fat-Tree K=4?**
> **Trả lời:** Kiến trúc Fat-Tree là cấu trúc chuẩn của Data Center hiện đại (Google, Facebook đang xài). Vì nó tổ chức dưới dạng nhiều đường đi song song (Multipath) đa lớp (Edge-Agg-Core). Điều này cho phép thuật toán AI tha hồ tìm đường khác khi một nhánh nào đó bị nghẽn (Fault tolerance).

**4. Dùng duy nhất một cái Controller (Ryu) để xử lý mọi thứ. Nếu Controller này sập vì quá tải thì sao? (Single Point of Failure)**
> **Trả lời:** Dạ, ở phạm vi thực nghiệm chứng minh thuật toán, nhóm xài 1 Controller. Nhưng khi đưa ra thực tế Doanh nghiệp, người ta luôn cài cắm "Cluster Controller" (VD: mô hình ONOS hoặc OpenDaylight ghép 3 mạng Controller chạy song song). Khi Master chết, Slave sẽ tự động lên thay. Bài toán phân tán này đã có sẵn chuẩn công nghiệp lo ạ.

**5. Điểm bất lợi lớn nhất của SDN là giao tiếp giữa Data Plane (Switch) và Control Plane (Controller) rất chậm. Có bị nghẽn ở bước này không?**
> **Trả lời:** Dạ không ạ. Cơ chế OpenFlow rất thông minh: Switch *chỉ gửi duy nhất* gói tin số 1 (gói mồi) lên cho Controller nhờ phân tích. Khi Controller gửi luật về, thì từ gói tin số 2 đến n của phiên đăng nhập đó sẽ đi lướt trực tiếp qua phần cứng Switch với tốc độ ánh sáng (line-rate) chứ không gọi Controller nữa.

---

## 🧠 NHÓM 2: CƠ CHE AI VÀ MÔ HÌNH TFT-DQN (8 Câu)

**6. Tại sao kết hợp Transformer (TFT) với DQN? Sao không dùng PPO hay DDPG vốn là các siêu mạng nổi rầm rộ hiện nay?**
> **Trả lời:** Dạ, bài toán của nhóm là chọn 1 máy chủ cụ thể (VD: h5, h7 hoặc h8) để quăng tải. Đây là không gian hành động rời rạc (Discrete Action). PPO hay DDPG sinh ra để cho không gian liên tục (Continuous - kiểu di chuyển vô lăng xe tự lái). Nên với bài toán rời rạc, DQN (Deep Q-Network) là con át chủ bài mạnh nhất.

**7. Mạng Transformer rất nặng nề (thực tế như ChatGPT dở dang). Làm sao nhét nó vào cái độ trễ nhạy cảm của SDN được?**
> **Trả lời:** Dạ nhóm em nhận thức rõ được "Tử huyệt" này. Nên model TFT của nhóm là bản *micro-TFT* cực nhẹ. Chỉ dùng 32 hidden size, 4 attention heads, và chiều dài chuỗi dự báo chỉ là 5 timestep. Nhờ ép cân thành công, nên mô hình chạy suy luận (Inference time) chỉ mất ~33 mili-giây, cực kì phù hợp nằm dưới đáy quy chuẩn mạng viễn thông.

**8. Sao không dùng mạng hồi quy LSTM để dự đoán Time-Series cho nó nhàn, mà lại phức tạp hóa thành TFT?**
> **Trả lời:** LSTM xử lý chuỗi tốt, nhưng nó mắc bệnh "quên dần" dữ liệu cũ khi chuỗi dài ra, đồng thời không biết nhấn mạnh điểm nào. Chữ T đầu tiên trong TFT là *Temporal Fusion Transformer* - Nó có bộ Attention. Nhờ vậy nó biết phải "nhìn chằm chằm" vào 1 hạt đột biến nhỏ từ 3 timestep trước để mường tượng ra cơn lốc đăng ký tín chỉ sắp tới, LSTM không nhạy bằng.

**9. Cách nhóm viết Hàm Phần Thưởng (Reward) V3 như thế nào để AI không rơi vào "cân bằng giả"?**
> **Trả lời:** Tinh túy của RL cắm ở Hàm Reward. Code của bọn em có phạt và thưởng kết hợp:
> Thưởng điểm dương cực lớn = nếu chọn đúng máy chủ đang thảnh thơi (Low Load).
> Phạt điểm âm (-3x) = nếu chọn nhầm máy chủ đang hấp hối (Overload). 
> Nhờ phạt âm nặng, DQN bị đập một gậy rất nhớ đời để nó không bao giờ đẩy vào máy chủ chết.

**10. Có chắc là AI thực sự thông minh hơn hay chỉ do may mắn lấn át thuật toán truyền thống?**
> **Trả lời:** Bảng số liệu Q-learning nói lên tất cả ạ. Điểm mấu chốt là RR (chia đều) không biết cái gì trong mạng, máy 10Mbps hay máy 100Mbps nó quăng 1/3 luồng y hệt nhau. Còn AI, thông qua Epsilon Decay khám phá, tự DQN nhận ra máy 10Mbps (h5) "như cái cốc nước nhỏ" rót xíu là tràn (Phạt điểm), máy 100Mbps "như cái lu" rót thoải mái (Cộng điểm). AI thật sự "đo lường" sức chứa qua thưởng - phạt.

**11. Nêu rõ định nghĩa "Truy cập bất thường" trong phạm vi nghiên cứu của nhóm?**
> **Trả lời:** "Truy cập bất thường" ở đây được giới hạn là **Bất thường về Khối Lượng và Tần Suất (Volume-based anomaly)**, cụ thể là Flash Crowd (Tăng tải hợp lệ nhưng ồ ạt) và Targeted Congestion (Có chủ đích bóp nghẹt 1 node). Nghiên cứu *không tập trung vào* bất thường mã độc phần mềm sâu (như chèn SQL Injection).

**12. Epsilon và Temperature (Tau) trong thuật toán có nhiệm vụ gì? Bỏ được không?**
> **Trả lời:** Phải có ạ. Epsilon decay là để AI "thử nghiệm ngu ngốc" (Explore) ban đầu để tìm ra quy luật, sau đó nó khôn lên thì giảm dần thử nghiệm để "tận dụng" quy luật (Exploit). Còn Temperature trong Softmax làm cho AI "chia đều tay" các luồng chứ không đập cục bộ 100% traffic ngốc nghếch vào 1 máy rảnh, tránh rảnh hóa nghẽn liền tập tức.

**13. Q-values là gì trong mạng DQN?**
> **Trả lời:** Q-values là "Giá trị điểm số Tương Lai". Nó thể hiện kỳ vọng rằng nếu tại tình trạng mạng hiện giờ (State), AI chọn quăng Data vào máy 1 (Action), thì bao nhiêu phần thưởng ròng sẽ rớt xuống trong dài hạn. Cái action nào Q-value to nhất = Xài cái đó.

---

## 🛠️ NHÓM 3: MÔI TRƯỜNG THỰC NGHIỆM VÀ DỮ LIỆU (5 Câu)

**14. Chơi mininet mà làm ra 3 server bất đối xứng: 10Mbps, 50Mbps, 100 Mbps để làm gì? Bắt bẻ thuật toán truyền thống à?**
> **Trả lời:** *(Mỉm cười gật đầu)* Dạ thưa thầy, chính xác là thế ạ. Trong thực tế các máy chủ hạ tầng đều khác biệt sức mạnh, cái mua năm 2010 cái mua 2024. Nếu môi trường bằng phẳng 100-100-100, thì thuật toán Round Robin cũ chạy vẫn tốt chán. Phải có băng thông cọc cạch thì cái sự thông minh linh hoạt của AI mới tỏa sáng rực rỡ được. Đội nghèo (10M) AI cứu, tài phiệt (100M) AI chất đồ đè thêm.

**15. Mạng lưới thực tế rất nhiễu. Artillery giả lập được thế nào?**
> **Trả lời:** Artillery là Node.js framework xịn nhất. Nhóm dùng cluster 8 máy ảo Artillery bắn JSON requests, code định nghĩa Ramping (Leo dốc từng phần), Flash Crowd (nổ tung 0-1000 user trong vài chục giây). Dòng traffic không phải hằng số đều đặn, mà bắn giật cục liên hồi y như ngón tay người bấm F5 ạ.

**16. Giám khảo thấy nhóm xài "Kịch bản Targeted Congestion" (Tập trung nghẽn), mục đích sâu xa là gì?**
> **Trả lời:** Đây chính là kịch bản giả lập Failover (Phục hổi sau sự cố) hoặc rớt vô trận mạc DDoS đánh úp 1 server cấp cứng. Nhóm chứng minh khả năng AI "lánh nạn": AI tự cảm giác được 1 vùng chết, tự tước hoàn toàn traffic bẻ nhánh đi nơi khác mà không cần IT Manager vô can thiệp rút dây cáp.

**17. Nếu Model mất tận 200 epochs để huấn luyện (Training), thì làm sao khi đưa ra đời ứng dụng cho mạng thay đổi liên tục kịp?**
> **Trả lời:** 200 epochs đó là "Offline Training" sinh ra con AI ban đầu (Pre-trained model). Trên thực tế đưa ra mạng, con AI này đã có "Trực giác" rồi, xài luôn (Zero-shot) vẫn tốt hơn Round-Robin. Tương lai ta gắn thêm Online Reinforcement Learning, nó vừa chạy điều hướng vừa dùng máy song song tự mài dũa thêm.

**18. Khi load balancer phân luồng (NAT 5-tuple), có sợ làm rớt session của sinh viên đang nộp bài thi không?**
> **Trả lời:** Dạ không ạ. Nhờ định lý bám dính phiên. Khi TCP Handshake đầu tiên đi qua, NAT lập tức nối SRC_IP sinh viên và DST_IP Backend đích. Từ đó mọi gói tin của bài thi đều đi đúng một luồng cố định duy nhất cho hết bài, chứ AI không bẻ luồng lung tung giữa đường đi của một sinh viên gây đứt quãng ạ.

---

## 📊 NHÓM 4: ĐÁNH GIÁ & KẾT QUẢ (5 Câu)

**19. Đánh giá tính Công Bằng của phân phối. Nhóm dùng tiêu chuẩn kỹ thuật nào?**
> **Trả lời:** Dạ nhóm dùng **CV% (Hệ số biến thiên - Coefficient of Variation)**. Công thức là Độ lệch chuẩn tính trên phần trăm tải của 3 máy, chia cho tải trung bình. Số phần trăm CV% càng tụt về 0 thì hệ thống phân tải bằng phẳng mượt mà như tấm lụa. 

**20. Tại sao phải đánh nhãn (Label HIGH/NORMAL) cho data trong lúc thu thập?**
> **Trả lời:** Để làm modifier (Cấp số nhân) cho Hàm Phần Thưởng. Khi luồng có chữ HIGH, AI hiểu "Trời ơi mạng sắp sập rồi", nó biết phải tối ưu phân luồng cẩn trọng x2 bình thường. Nó là chất xúc tác để AI phân biệt giữa rảnh rỗi và báo động đỏ.

**21. Có sợ Overfitting (Học vẹt mạng, lôi ra mạng khác chạy là sập) không? Khắc phục thế nào trên RL?**
> **Trả lời:** Sợ nhất là cái này ạ. Giải quyết bằng Memory Replay Buffer chứa 50,000 sự kiện lịch sử qua khứ, nhưng khi lấy ra train em bắt bốc ngẫu nhiên (Mini-batches Sampling). Nó khử tính liên tục, làm vỡ trí nhớ học vẹt của AI để nó phải học bản chất tương quan (Quy luật) chứ không phải học ảnh của đồ thị.

**22. Trong biểu đồ Throughput (slide 9), tại sao đường đỏ của RR rớt không phanh vậy? Cụ thể là cái gì rớt?**
> **Trả lời:** Rớt túi Drop Packet ạ. Do cái máy 10Mbps năng lực CPU và bộ nhớ đệm Buffer của port bị cạn. Khi quá sức chứa nó bắt đầu xả láng hủy gói (TCP Drop). Gói mất, TCP lặp lại gửi lại --> càng gửi lại càng nghẽn sập sâu hơn. Đồ thị cắm đầu thẳng đứng do thảm họa Retransmission timeout thưa thầy/cô.

**23. Thời điểm Flash Crowd, AI làm giảm Packet Loss rate đi bao nhiêu? Cụm từ "Lá chắn thép" nghe hơi kêu quá không?**
> **Trả lời:** Chắn chắn là kêu nhưng có cơ sở ạ! So với gần 50% packet drop cực hạn của Round-Robin, AI giữ Packet Loss Rate chỉ loanh quanh 2-5% (gần như hoàn hảo trong môi trường quá tải giới hạn). "Lá chắn thép" ở đây là ý nói AI làm tấm đệm đàn hồi bẻ qua luồng khác, giúp server yếu đứng vững không bị treo máy.

**23b. (❗ CÂU HỎI ĐIỂM 10) Đồ thị Reward (Phần thưởng) của nhóm ở những Epoch cuối tại sao lại đi theo chiều ngang tuyến tính chứ không tiếp tục cắm đầu đi lên như lúc đầu?**
> **Trả lời:** Dạ thưa Hội đồng, đây chính là đặc tính "Hội tụ (Convergence)" trong học máy ạ. Khi đồ thị Reward đi ngang (đi vào đường tiệm cận ngang) nghĩa là mô hình đã ***chạm đến giới hạn tối ưu tuyệt đối của môi trường đó***. Giống như chơi game Mario đã đạt max 10,000 điểm không thể ăn thêm nấm được nữa. Nếu đồ thị Reward mà cứ cắm đầu đi lên mãi vô tận, thì đó mới là lúc chứng tỏ Hàm Reward của tụi em bị lỗi (Reward Hacking) khiến AI "ăn gian" điểm số ạ. Đi ngang chứng tỏ AI đã trưởng thành và ổn định.

**23c. (❗ CÂU HỎI ĐIỂM 10) Tại sao ở 2 kịch bản "Tăng dần đều" (Ramping / Gradual Shift), chỉ số của mô hình AI đôi khi lại THUA (hoặc chỉ bằng) thuật toán tĩnh WRR, trong khi các bạn đề cao AI là siêu việt?**
> **Trả lời:** *(Cười tự tin)* Dạ câu hỏi này rất xoáy vào bản chất của RL ạ. 
> Thực chất, WRR là một công thức "Hard-code tĩnh": Nó đã biết trước server là 10-50-100 và chia một dòng nước êm đềm rất máy móc theo tỷ lệ cố định. 
> Trong khi đó, con AI của tụi em là một "Thực thể sống động": Kể cả khi sóng yên biển lặng, AI vẫn phải duy trì 1 lượng Epsilon (khoảng 10-15%) để thỉnh thoảng cố tình **"Nghịch ngu" (Exploration)** thử quăng data vào máy yếu xem máy đó có còn sống không. Chính cái 15% sự cố tình tò mò đó làm cho điểm CV% của AI có chút gợn sóng và ***đôi khi nhỉnh hơn (tệ hơn chút xíu) so với sự hoàn hảo máy móc của WRR*** ở lúc bình thường.
> Nhưng sự "hy sinh một chút hoàn hảo lúc bình thái" này được đền đáp xứng đáng lúc *Bão Flash Crowd* tới: WRR lăn đùng ra chết cứng vì sự ngây thơ của nó, còn AI nhờ luôn giữ thói quen "tò mò" thăm dò mạng nên lập tức né được hố băng. Tóm lại: AI chấp nhận thua 1% ở lúc dễ, để đảm bảo sống sót 100% tài nguyên ở lúc khó thưa hội đồng.

---

## 🚀 NHÓM 5: TRIỂN KHAI THỰC TẾ & MỞ RỘNG (7 Câu)

**24. Model học quá khứ, nếu đụng Hacker dùng chiêu Adversarial Attack (Tấn công mô hình AI bằng cách làm nhiễu data) thì mô hình có trở nên vô dụng?**
> **Trả lời:** Đây rủi ro cấp cao của Deep Learning. Hacker gửi packet giả để làm nhiễu byte_rate đầu vào, AI sẽ tính toán sai Q-value. Hiện nay em chưa giới hạn được phạm vi này, nhưng có thể bọc lót bằng module lọc Filter thông dụng chặn tần suất nhiễu hoặc kẹp chuẩn đoán Isolation Forest trước khi làm data cho AI.

**25. Các bạn thiết lập Postgres DB 5000 user. Có sợ thắt cổ chai ở CSDL không?**
> **Trả lời:** CSDL có là nút thắt. Mạng ngon mà DB chậm thì người dùng vẫn kêu ca. Em biết nên hệ Nodejs Backend của bọn em đã setup Connection Pooling giữ cứng các rễ kết nối. Và em nhắm chủ đề vào "Nghẽn băng thông tầng mạng ứng dụng". Tuy nhiên nếu test nặng hơn 5000, DB chịu hết nổi, AI đằng trước có điều phối hay thế nào cũng vô dụng (Nút Cổ Chai nằm ngoài luồng SDN).

**26. Đóng góp nổi bật nhất của Đề tài đối với cộng đồng NCKH là gì? So với những người khác thì làm gì mới?**
> **Trả lời:** Dạ có 2 điểm nhấn cốt lõi: 1- Mang TFT (kẻ hủy diệt Time-Series dự đoán kinh tế) áp vào không gian mạng 5 steps. Và 2- Setup một mô phỏng *Bất đối xứng phần cứng (10-50-100)* thay vì một môi trường sạch đẹp để bóc trần năng lực chống chọi thực tế của AI mà các bài NCKH trên thế giới hay "giấu lướt" qua. 

**27. Triển khai vào một doanh nghiệp mạng đang sống (Live), làm thế nào để Rollback (Lùi lại) nếu con AI của bạn bị ngáo và đánh sập mạng ngay ngày đầu cài đặt?**
> **Trả lời:** Phải có cơ chế ngắt cầu dao tự động (Fallback rule). Bên trong Controller, em sẽ code 1 vòng bắt Exception, hoặc giám sát Flow Rules. Nếu AI ném ra 1 mớ lệnh rác khiến Delay Ping tăng vọt >200ms, bộ Fallback đá bay DQN và kích hoạt hard-code thuật toán Weighted-RR khẩn cấp để đảm mạng an toàn sinh tồn. 

**28. Code toàn bộ là của nhóm làm từ đầu à? Mất bao nhiêu lâu để hoàn thành hệ thống ngần này?**
> **Trả lời:** Core Backend, Topology, Kịch bản Artillery và Hàm Reward V3 là công xưởng nhóm đổ máu đập đi xây lại 3 lần rưỡi. Framework Mininet, DQN base architecture thì bọn em fork kế thừa thư viện mã nguồn mở. Mất khoảng gần 3 tháng code, train, và test liên hoàn. 

**29. Vừa chạy AI tốn GPU, lại phải xử lý packet, có bài chi phí nào rẻ hơn không? Doanh nghiệp nhỏ thì cài làm gì tốn kém quá!**
> **Trả lời:** Nhóm Train AI tốn GPU, nhưng lúc Đưa ra xài (Inference) thì mạng chỉ là mớ Vector Ma trận tính bằng MByte chạy vèo vèo trên CPU Intel Core i3 thôi ạ. Doanh nghiệp đào tạo sẵn một cái Model nhỏ cầm file .pth quăng lên cái PC thường chạy Controller là hoạt động ngon lành, vốn siêu nhẹ.

**30. Sau khi làm đề tài này, bài học "nhớ đời" nào các bạn nhận ra về AI trong viễn thông?**
> **Trả lời:** AI không phải viên thần hoàn quăng vào cái gì cũng thành vàng. Bọn em hì hục train nó, nhưng nó ra quyết định siêu dốt nát. Mất hơn tháng trời tụi em mới nhận ra: Thần chú nằm ở Reward Function. Cho nó cái củ cà rốt và cây gậy chuẩn (địa lấp kẽ hở Reward V3), nó mới cắn mồi thành vị thần. Con người điều khiển AI bằng thiết kế Luật thi đua.

---

> 🎯 **Tip cho người thuyết trình:** Có thể in một vài câu in đậm ra tờ note. Khi nghe giám khảo hỏi, bình tĩnh gật đầu lấy nhịp khoảng 2 giây, và đưa những keywork "*Mili-giây*", "*Lá chắn thép*", "*Hàm Reward*", "*Fat-Tree Multipath*" vào câu trả lời để xoáy thẳng vào thính giác các chuyên gia. Chúc Báo cáo NCKH Thành Công Rực Rỡ!
