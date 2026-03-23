# Plan: Clean Architecture & 1-Click Deployment

## Nhận Định Vấn Đề Hiện Tại
Sau nhiều vòng thử nghiệm (Training, Evaluation, Controller Tuning), cấu trúc dự án đang dính nhiều "rác" không cần thiết nếu mang đi chấm điểm hay bàn giao cho một máy tính mới:
1. **Rác ở Host OS**: Các thư mục `ai_env/` (7GB), `ryu/` (chứa source code bị build lỗi clone từ github), ổ cứng sinh ra `__pycache__`.
2. **Khó dùng (UX kém)**: Để khởi chạy test, người dùng đang phải mở tận 2-3 tab Terminal (`ryu-manager` riêng, `run_lms_mininet.py` riêng), truyền biến môi trường thủ công rườm rà.
3. **Quản lý Docker thủ công**: Câu lệnh `docker build` dài dòng và `docker run` chứa hàng tá Argument (`--gpus all --network host --privileged`). Dễ gõ sai.

## Giải Pháp Đề Xuất (1-Click Docker Compose)

Mục tiêu cốt lõi: **Biến siêu dự án này thành hệ thống "Ăn Liền" cho bất kỳ Laptop nào (chỉ cần có Docker).**

### 1. Dọn dẹp Thư mục (Clean-up)
- Xóa sổ bộ thư viện tĩnh cứng ngắc trên Host: `rm -rf ai_env/ ryu/ __pycache__/`
- Chỉ giữ lại Source Code thực sự (Thư mục `ai_model`, `lms`, `stats`, `docs`, `*.py`).

### 2. Định hình lại Dockerfile
- Đảm bảo trong `Dockerfile` chốt cứng `CMD /bin/bash` để làm môi trường chứa tương tác CLI chuẩn.

### 3. Sức mạnh của `docker-compose.yml`
Tạo file cấu hình vòng đời dịch vụ duy nhất. Người mới kéo code về chỉ cần gõ `docker compose up -d` là toàn bộ thiết lập `--gpus all` và `--privileged` tự động apply vào 컨테이너.

### 4. Bơm "Script Automation" (Trải nghiệm hoàn hảo)
Tạo 1 file duy nhất bọc toàn bộ tinh hoa lại: `run_demo.sh`
File này khi được kích hoạt ở máy mới (hoặc máy hội đồng bảo vệ) sẽ tự làm các trò diễn ảo thuật sau:
1. Hiện ra cái ASCII Art chữ "NCKH SDN AI-LOAD-BALANCER" cực kỳ hoành tráng.
2. Tự build docker (nếu chưa có).
3. Đẩy Menu cho chọn 4 Kịch Bản: Flash Crowd, Tsunami, Target Congestion, hay Gradual.
4. Tự chui vào Container gọi Ryu-Manager làm nền, và gọi Script Mininet lên Console.
Người dùng từ đầu tới cuối chỉ cần xem Menu và ấn phím số 1, 2, 3, 4.

## Các Bước Thực Hiện
- [ ] Xóa rác, clean repo Git.
- [ ] Viết `docker-compose.yml`
- [ ] Viết script `run_demo.sh` chứa Menu tương tác (Interactive).
- [ ] Cập nhật lại `README.md` với thần chú "Cài code trong 1 nốt nhạc".
