# Plan: AI Inference & OpenFlow Deployment

Mục tiêu cốt lõi: Nâng cấp `controller_stats.py` từ một Switch và Collector thụ động thành một **AI-Driven Load Balancer** chủ động. Nó sẽ sử dụng mô hình TFT-DQN (`.pth`) đã train để bẻ luồng các gói tin HTTP gửi tới API Server.

## Vấn đề hiện tại
- `controller_stats.py` chỉ thu thập dữ liệu mỗi 10s và đẩy vào CSV.
- Tính năng L2 Switch chỉ Forward/Flood gói tin dựa trên địa chỉ MAC học được một cách bị động.

## Giải pháp: Hệ thống Controller 2 luồng (Dual-thread)
Sẽ chia Ryu Controller thành 2 Thread hoạt động song song.

### Thread 1: AI Inference Loop (Não bộ)
1. Cứ mỗi `T` giây (ví dụ 5s), tiến trình này sẽ gọi một hàm đọc 5 dòng cuối cùng của `flow_stats` và `port_stats` (thông qua Buffer trên RAM thay vì đọc File CSV để giảm I/O).
2. Biến đổi dữ liệu thô thành Tensor State (Giống với file `data_processor.py`).
3. Đẩy State vào Model `TFT-DQN` (`agent.policy_net(state)`) để dự đoán hàm Q-value.
4. Chọn Action (Chọn IP backend tốt nhất trong 3 máy `10.0.0.5`, `10.0.0.7`, `10.0.0.8`).
5. Ghi Action này vào một biến toàn cục `current_best_backend`.

### Thread 2: Packet-In & OpenFlow Rule (Cơ bắp)
1. Khi có bất kỳ một luồng gói tin mới nào đi tới cổng Load Balancer ảo (Virtual IP `10.0.0.100`), Controller sẽ chặn gói TCP SYN (`PacketIn`).
2. Controller đọc `current_best_backend` từ Thread 1.
3. Controller tạo ra 2 rule `FlowMod`:
   - Rule 1 (Luồng đi vào): Chuyển đổi Đích (NAT) từ `10.0.0.100` sang IP Backend được AI chỉ định.
   - Rule 2 (Luồng quay về): Chuyển đổi Nguồn (NAT) từ IP Backend trở lại `10.0.0.100` để giấu Backend khỏi Client.
4. Đẩy 2 Rule này trực tiếp xuống OVS Switch `s1` (Switch rìa nối máy chủ) thông qua giao thức OpenFlow 1.3.

## Các bước triển khai (Implementation Steps)
1. [ ] Wrap mô hình PyTorch thành class `AI_Predictor` chuyên xử lý Inference (load weights từ `checkpoints/tft_dqn_policy.pth`).
2. [ ] Viết chức năng đệm State 5 nhịp thời gian vào RAM của Ryu Controller.
3. [ ] Bổ sung code `packet_in_handler` để can thiệp (intercept) IP/ARP của luồng Load Balancer.
4. [ ] Khởi chạy Mininet và chạy Artillery Kịch bản 1 (Flash Crowd) đễ xem khả năng nhảy số của Controller.
