<div align="center">
  <img src="https://cdn.haitrieu.com/wp-content/uploads/2021/10/Logo-DH-Thuy-Loi.png" alt="Logo Đại học Thủy lợi" width="120" />

  <p><b>PHÂN HIỆU TRƯỜNG ĐẠI HỌC THỦY LỢI</b></p>
  <hr width="30%">
  
  # ĐỀ TÀI NGHIÊN CỨU KHOA HỌC
  ## Ứng dụng Học máy trong Phát hiện Truy cập Bất thường trên dữ liệu luồng
  
  <p>
    <b>Sinh viên thực hiện:</b> Đặng Quang Hiển <br>
    <b>Lớp:</b> S28-67CNTT | <b>MSSV:</b> 2551067129
  </p>

  > Dự án Tối ưu hóa phân tải mạng Mạng Điều khiển bằng Phần mềm (Software-Defined Networking - SDN) sử dụng mô hình học tăng cường sâu **TFT-DQN** (Temporal Fusion Transformer + Deep Q-Network) nhằm ứng phó với hiện tượng thắt nút cổ chai (bottleneck) trong hệ thống quản lý học tập (LMS).
</div>

---

## 1. GIỚI THIỆU (ABSTRACT)

Dự án này là minh chứng thực nghiệm cho khả năng áp dụng trí tuệ nhân tạo vào cân bằng tải (Load Balancing) lớp mạng L3/L4 thay cho các hệ thống phần cứng phân phối tĩnh. Trong mạng SDN, cấu trúc điều khiển tập trung cho phép thu thập nhanh số liệu tổng quan toàn bộ sơ đồ mạng (Topology Fat-Tree). 

Bộ mô hình **TFT-DQN** tích hợp bên trong trình điều khiển Ryu Controller tiến hành khai thác chuỗi dữ liệu thời gian thực (Time-series) bằng Transformers. Phương pháp này có năng lực xử lý vượt trội các đợt lưu lượng bùng phát đột ngột (Flash Crowd) – điển hình như trong dịch vụ đăng ký tín chỉ trường đại học, mà các thuật toán như Round-Robin không thể đáp ứng.

> **Tài liệu Báo Cáo:** 
> Vui lòng tham khảo chi tiết [Báo cáo NCKH chuẩn IEEE (IMRAD)](docs/Bao_Cao_NCKH_IEEE.md).

---

## 2. KIẾN TRÚC MẠNG (NETWORK ARCHITECTURE)

### Sơ đồ Topology (Fat-Tree K=4)

```text
Clients (h9–h16)
     │  HTTP → 10.0.0.100 (Virtual IP)
     ▼
[Ryu Controller — TFT-DQN]  ←→  flow_stats.csv
     │  OpenFlow NAT
     ├──► h5 (Backend 1 — Node.js :4000)  [Bandwidth Limit: 10 Mbps]
     ├──► h7 (Backend 2 — Node.js :4000)  [Bandwidth Limit: 50 Mbps]
     └──► h8 (Backend 3 — Node.js :4000)  [Bandwidth Limit: 100 Mbps]
                              │
                         h6 (PostgreSQL — 5000 users)
```

**Đặc tả:**
- **Mặt phẳng điều khiển (Control Plane):** Ryu Controller (`controller_stats.py`) thực thi logic định tuyến.
- **Protocol:** OpenFlow 1.3.
- **Cơ chế NAT phi trạng thái:** Dùng OpenFlow Match Fields để che giấu danh tính Backend Server.

---

## 3. THIẾT KẾ CÂN BẰNG TẢI TRÍ TUỆ NHÂN TẠO (PROPOSED METHODOLOGY)

### Cơ chế Hoạt động TFT + DQN
- **Tính toán Không-Thời gian (Spatio-Temporal):** Model sẽ quan sát tốc độ truyền bytes, tốc độ gói tin và nhịp độ giao tiếp (load\_trend) của 5 khoảnh khắc (Timesteps) trước đó.
- Khối **Temporal Fusion Transformer** lọc bỏ nhiễu động và cung cấp vectơ phán đoán đỉnh Flash Crowd. Khối **Deep Q-Network** kết hợp với hàm **Spatio-Temporal Reward (V4)** (xem chi tiết trong mục 3.2 của Báo Cáo) thưởng điểm cho hành vi chọn Backend ổn định và trừng phạt điểm nghẽn cổ chai (như Server `h5` với 10 Mbps).

### Cấu trúc Mô Phỏng Các Kịch Bản (Experimental Setup)
Dưới sự hỗ trợ của các công cụ mô phỏng tải Artillery:
1. `flash_crowd.yml`: Bài toán Tsunami khi hàng nghìn sinh viên ồ ạt truy cập.
2. `predictable_ramping.yml`: Thi trực tuyến, với tải dự báo tăng đều.
3. `targeted_congestion.yml`: Bóp nghẽn 1 Server cục bộ, kiểm thử fail-over cực hạn.
4. `gradual_shift.yml`: Đo đạc độ bền vững và hội tụ AI.

---

## 4. HƯỚNG DẪN TRIỂN KHAI NHANH DÀNH CHO HỘI ĐỒNG ĐÁNH GIÁ (DEPLOYMENT)

Để đáp ứng tiêu chuẩn tái lập được cấu trúc của chuẩn nghiên cứu quốc tế, hệ thống đã được cô đọng dưới dạng ứng dụng Container.

### Môi trường Yêu cầu:
- **Hệ điều hành:** Linux distribution (Ubuntu/Arch/Debian).
- **Phần mềm cốt lõi:** Docker, Docker Compose, Python 3.10+.

### 1-Click Bootstrap

1. Tải bộ mã nguồn và thiết lập phân quyền:
```bash
git clone <your-github-repo>
cd nckh_sdn
sudo chown -R $USER:$USER stats/
mkdir -p stats/results
```

2. Khởi tạo môi trường ảo Mininet cách ly:
```bash
docker compose up -d --build
```

3. Khởi động Quá trình Kiểm nghiệm Cân bằng bằng AI:
Chạy script tự động bao gồm: (1) Khởi tạo cấu trúc môi trường ảo, (2) Bắn tải Artillery, (3) Thống kê AI quyết định tải, và (4) Trích xuất kết quả tự động.
```bash
./scripts/full_pipeline.sh
```

---

## 5. KẾT QUẢ ĐÁNH GIÁ (RESULTS & CHARTS)

Dữ liệu so sánh năng lực của mạng TFT-DQN so với Round Robin (RR) và Weighted Round Robin (WRR) được lưu trực tiếp bằng CSV và Biểu đồ vào thư mục kết quả. Kết quả đánh giá được tự động xuất sau lệnh pipeline.
- Training Curves and Overfitting Analysis: `ai_model/processed_data/charts/`
- Heatmap, Throughput Simulation: `stats/results/charts/`

---

## 6. CẤU TRÚC MÃ NGUỒN CỐT LÕI (DIRECTORY STRUCTURE)

Dự án được ứng dụng chuẩn bố cục Micro-modules (Clean Architecture).

```text
nckh_sdn/
├── ai_model/                # Model Pytorch (TFT_DQN_Model), Training Loop & RL Environments.
├── docs/                    # Tài liệu đề tài chuẩn IEEE, Thư mục Archive.
├── lms/                     # Mã nguồn Backend, Web Dashboard và Script stress-test Artillery.
├── scripts/                 # Automation Bash script cho mô hình thực thi 1-Click.
├── stats/                   # Dataset thống kê Port và Điểm nghẽn từ Ryu.
├── controller_stats.py      # SDN Brains: Module Ryu Switch & AI Load Balancer Inference
├── run_lms_mininet.py       # Orchestrator khởi tạo mạng và Container hóa.
├── topo_fattree.py          # Xây dựng cây sơ đồ Fat-Tree.
└── docker-compose.yml       # Môi trường chạy cách ly 1-click.
```

---

<p align="center">
    <b>Nghiên cứu Khoa học 2026</b> <br>
    <i>© Bản quyền dành cho Phiên bản Nghiên cứu & Học thuật Đại Học Thủy Lợi.</i>
</p>
