# NCKH SDN - AI-Driven Load Balancer on Fat-Tree Topology

> Tối ưu hóa phân tải mạng SDN thông qua mô hình Deep Learning (TFT-DQN) với độ trễ suy luận tức thời trong hệ thống Mininet mô phỏng Lưu lượng Đăng ký Tín chỉ (LMS).

## Quick Start

Hệ thống được đóng gói hoàn toàn trong một **Docker Container duy nhất (Single-Container)** để đảm bảo tính nhất quán môi trường. Nhờ kế thừa từ `nvidia/cuda` image, Docker container này vừa có thể chạy trơn tru mạng ảo Mininet, vừa truy xuất trực tiếp Card Rời đồ họa (GPU) để gia tốc bộ não AI.

**Yêu cầu:** Máy trạm Linux cài sẵn Docker và NVIDIA Container Toolkit.

### Bước 1: Build & Truy cập Môi trường Ảo
Mở một Terminal và khởi tạo vùng không gian mô phỏng. Container này cần cờ `--gpus all` để AI Pytorch bung sức mạnh và `--privileged` để Mininet được quyền tạo Mạng ảo.

```bash
# 1. Build File Docker (Cài đặt Mininet, NodeJS, Ryu, PyTorch CUDA 12.4...)
sudo docker build -t lms-sdn-env .

# 2. Khởi chạy và chui vào Không gian Mô phỏng
sudo docker run -it --rm --privileged --gpus all --network host -v $(pwd):/work lms-sdn-env
```

### Bước 2: Kích hoạt Load Balancer AI (Từ bên trong Docker)
Ngay tại Terminal vừa mở ở Bước 1 (đang ở trong Docker), hãy kêu gọi bộ não Ryu Controller thức tỉnh:

```bash
# Khởi động Load Balancer AI
ryu-manager controller_stats.py
```

### Bước 3: Nã Pháo (Từ Terminal 2)
Mở một cửa sổ Terminal THỨ HAI trên máy tính của bạn. Xuyên qua lớp vỏ Docker để đánh thức hệ thống mô phỏng Traffic.

```bash
# 1. Chèn thêm một Tab mới vào Không gian Mô phỏng đang chạy
sudo docker exec -it $(sudo docker ps -q --filter ancestor=lms-sdn-env) /bin/bash

# 2. Kích hoạt Kịch bản Tấn công (Bên trong Docker)
SCENARIO=flash_crowd.yml python3 run_lms_mininet.py
```

## Features

- **Fat-Tree Topology (Mininet):** Khởi tạo mạng SDN chia theo 3 lớp Core, Aggregation, Edge.
- **Node.js LMS Backend/Frontend:** Máy chủ đăng ký tín chỉ và xem thời khóa biểu (ReactJS Vite) cùng PostgreSQL kết nối thông qua Sequelize.
- **Seeding Script (`seed_massive.js`):** Script tự động đổ 5.000 hồ sơ sinh viên với ID, chuyên ngành, số điểm tín chỉ đã tích lũy vào Host Database `h6`.
- **Artillery Stress Scenarios:** 4 kịch bản tạo nút thắt cổ chai chân thực:
  1. `flash_crowd.yml`: Cơn lốc (Gia tăng cực đại).
  2. `predictable_ramping.yml`: Ramping Thi trực tuyến (Sinh viên vào đều đặn).
  3. `targeted_congestion.yml`: Bóp nghẽn tĩnh tại 1 switch sập `h5`.
  4. `gradual_shift.yml`: Phân phối mòn dần theo ca.
- **TFT-DQN Dual-Thread Load Balancer:**
  - *Inference Loop*: Controller phân tích Window 5 trạng thái mạng mới nhất từ Data Buffer, chỉ định Backend (`10.0.0.5`, `10.0.0.7` hoặc `10.0.0.8`) trong dưới 100ms.
  - *Arp Spoofing & Openflow NAT*: Tự động chặn IP ảo `10.0.0.100` từ các Client (Host `h9`-`h16`) sau đó cài rule FlowMod (Timeout: 15s) bẻ luồng dữ liệu về thẳng IP do AI đánh giá.

## Configuration

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `SCENARIO`           | Kịch bản Artillery nào sẽ được gắp để tấn công mạng | `flash_crowd.yml` |
| `TARGET`             | IP Backend Loadbalancer (Virtual IP) | `http://10.0.0.100:4000` |
| `VIP`                | Địa chỉ IP Ảo (Sử dụng bởi SDN Ryu NAT) | `10.0.0.100` |
| `VMAC`               | Địa chỉ MAC Ảo cho VIP | `00:00:00:00:01:00` |

## Documentation

Tài liệu chi tiết về hệ thống cùng kế hoạch tư duy giải quyết vấn đề bằng Socratic Brainstorm:
- [Kế hoạch Xây dựng Stress-Scenarios và Seeding Database](./docs/PLAN-stress-testing.md)
- [Kế hoạch Áp dụng Dual-thread cho Controller Ryu AI](./docs/PLAN-deployment.md)

## Changelog

### Phase 4 - [Hoàn Tất] Deployment AI Controller & Đánh Giá
- **Added:** Thêm script `seed_massive.js` faker 5000 users.
- **Added:** Sinh ra 4 file kịch bản `.yml` cho Artillery với thời lượng dài và luồng requests phức tạp.
- **Added:** `controller_stats.py` tái cấu trúc, trở thành Load Balancer Động nhận chỉ thị từ `tft_dqn_policy.pth`.
- **Changed:** `run_lms_mininet.py` điều phối toàn bộ Artillery bắn về điểm đứt gãy ảo `10.0.0.100`.
- **Changed:** `run_labeled_test.py` dán nhãn label Log theo cách đọc luồng Console `Burst`/`Peak`.

### Phase 3 - Training AI Model
- **Added:** 3 thuật toán huấn luyện Deep Learning (`tft_dqn_net.py`, `train.py`)
- **Fixed:** Vấn đề Window Length không đồng đều với dữ liệu nhãn "HIGH/NORMAL". Dùng kỹ thuật Min-Max Scaler và Oversampling/Dropout để cân đối Label Data.

## License

MIT
