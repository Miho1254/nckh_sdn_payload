# Plan: Stress Testing & Database Seeding

Mục tiêu cốt lõi: Tạo ra một kịch bản đánh giá (Evaluation) sát với thực tế nhất cho hệ thống LMS để kiểm chứng năng lực của AI TFT-DQN so với các giải pháp tĩnh (RR/WRR). Quá trình đánh giá không chỉ dựa trên mức độ gửi gói tin rác (spam) mà phải mô phỏng chính xác "User Journey" của sinh viên.

## Giai đoạn khởi tạo (Phase -1): Khảo sát Context
- **Bài toán:** Các kịch bản Artillery Yaml hiện tại đang gửi dữ liệu khá ngẫu nhiên.
- **Giải pháp:** Cần xây dựng Database Postgres chứa từ 5,000 đến 10,000 bản ghi sinh viên và khoá học. Điều này gây áp lực I/O Database thật, buộc hệ thống phát sinh độ trễ thật sự thay vì chỉ là độ trễ mạng ảo.
- **Ràng buộc:** Cần chèn dữ liệu (Insert) theo lô (Batch) để tránh sập container Postgres khi khởi tạo.

## Giai đoạn 1 (Giai đoạn Hậu Cần): Database Seeding
Tạo một script Node.js độc lập: `lms/backend/seed_massive.js`
- [ ] Sử dụng thư viện `@faker-js/faker` để tạo data (Tên, MSSV, Ngành học).
- [ ] Tạo 100 Course (Mỗi course có giới hạn `slots` ~ 50-100).
- [ ] Tạo 5000 Student, hash passwords bằng `bcrypt`.
- [ ] INSERT dữ liệu vào Postgres theo batch 500 dòng một lần dùng `pg-promise` hoặc transaction thuần tuý.

## Giai đoạn 2 (Thiết kế User Journey): Nâng cấp Artillery Yaml
Chỉnh sửa file `artillery.yml` và bộ hàm `functions.js`.
- [ ] **Giai đoạn Login (POST)**: Script rút ngẫu nhiên user từ DB (hoặc danh sách file JSON đã chuẩn bị), đánh API `auth/login` -> Bơm tải CPU bằng bcrypt.
- [ ] **Giai đoạn Browse (GET)**: User lướt xem môn học. Lệnh Backend gọi `SELECT JOIN` làm phình bộ nhớ RAM Postgres. Sinh viên dừng lại 2-3s (Random think time).
- [ ] **Giai đoạn Đăng ký (POST)**: Tấn công đồng loạt vào các môn học khan hiếm slot. Gây ra DB Row Locking.

## Giai đoạn 3 (Kịch Bản Thực Thi Bùng Nổ): The Attack Plan
Cấu trúc phase cho Artillery `stress_burst.yml`:
1. **Warm-up** (60s, 50 Users): Khởi động hệ thống nền.
2. **Ramping Up** (120s, 100 -> 500 Users): Mô phỏng sát giờ G.
3. **The Burst** (180s, 1000+ Users): Giai đoạn High Traffic, gọi cơ chế gán nhãn HIGH cho Ryd.
4. **Cool down** (60s, 10 Users): Giảm nhiệt phục hồi.

## Giai đoạn 4: Tích hợp AI
- Chuyển quyền định tuyến `Ryu Controller` sang Mạng TFT-DQN.
- Đo các thông số KPI (Throughput throughput, Latency, Error 429/503) để vẽ biểu đồ báo cáo.
