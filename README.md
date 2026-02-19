# NCKH SDN — Mạng Định Nghĩa Bằng Phần Mềm & Cân Bằng Tải AI

> Nghiên cứu ứng dụng **TFT-DQN** (Temporal Fusion Transformer + Deep Q-Network) để cân bằng tải trên mạng SDN sử dụng Fat-Tree topology.

## Tổng Quan

Hệ thống mô phỏng mạng SDN trên **Mininet** với controller **Ryu**, triển khai ứng dụng LMS (Hệ thống Đăng ký Học phần) để tạo traffic thực tế phục vụ thu thập dataset cho mô hình AI.

### Kiến Trúc

```
┌─────────────────────────────────────────────────┐
│                  Ryu Controller                  │
│         (L2 Switch + Stats Collector)            │
└──────────────────────┬──────────────────────────┘
                       │ OpenFlow 1.3
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────┴────┐    ┌────┴────┐   ┌────┴────┐
   │ Core s1 │    │ Core s2 │   │  ...    │
   └────┬────┘    └────┬────┘   └─────────┘
        │              │
   ┌────┴────┐    ┌────┴────┐
   │ Agg s3  │    │ Agg s4  │  ...
   └────┬────┘    └────┬────┘
        │              │
   ┌────┴────┐    ┌────┴────┐
   │ Edge s7 │    │ Edge s8 │  ...
   └──┬──┬───┘    └──┬──┬───┘
      │  │           │  │
     h1  h2         h5  h6
   (FE) (...)    (API) (DB)
```

**Fat-Tree (k=4):** 2 Core + 4 Aggregation + 4 Edge switches, 16 hosts.

## Cấu Trúc Thư Mục

```
nckh_sdn/
├── controller_stats.py     # Ryu controller (L2 learning + stats)
├── topo_fattree.py          # Fat-Tree topology Mininet
├── Dockerfile               # Build environment đầy đủ
├── entrypoint.sh            # Auto-start OVS + PostgreSQL
├── .dockerignore
├── stats/                   # CSV output (flow + port stats)
│   ├── flow_stats.csv
│   └── port_stats.csv
└── lms/                     # Hệ thống Đăng ký Học phần
    ├── README.md
    ├── backend/             # Express + PostgreSQL
    ├── frontend/            # React + Tailwind
    └── stress-test/         # Artillery scenarios
```

## Yêu Cầu

- **Docker** (với `--privileged` flag)
- **Host OS** cần có kernel module `openvswitch`:
  ```bash
  # Arch/CachyOS
  sudo pacman -S openvswitch
  sudo modprobe openvswitch
  ```

## Hướng Dẫn

### 1. Build & Chạy Container

```bash
docker build -t sdn-research .
docker run -it --privileged \
  --name nckh_box \
  --dns 8.8.8.8 \
  sdn-research
```

### 2. Khởi Động Hệ Thống (trong container)

```bash
# Terminal 1 — Ryu Controller
ryu-manager controller_stats.py --ofp-tcp-listen-port 6633

# Terminal 2 — Mininet Fat-Tree
docker exec -it nckh_box bash
python3 topo_fattree.py
```

### 3. Triển Khai LMS trên Mininet

Sau khi vào Mininet CLI:

```bash
# Start PostgreSQL trên h6
h6 pg_ctlcluster 14 main start &

# Seed database + Start backend trên h5
h5 cd /work/lms/backend && DB_HOST=10.0.0.6 node seed.js
h5 cd /work/lms/backend && DB_HOST=10.0.0.6 node server.js &

# Chạy Stress Test từ h9
h9 cd /work/lms/stress-test && artillery run artillery.yml
```

### 4. Thu Thập Dataset

Ryu controller tự động ghi CSV mỗi 10 giây:

| File | Nội dung |
|---|---|
| `stats/flow_stats.csv` | Flow statistics (packet/byte count, duration) |
| `stats/port_stats.csv` | Port statistics (rx/tx packets, errors, drops) |

## Công Nghệ

| Thành phần | Công nghệ |
|---|---|
| SDN Controller | Ryu (OpenFlow 1.3) |
| Network Emulator | Mininet |
| Topology | Fat-Tree (k=4) |
| Backend | Node.js + Express |
| Database | PostgreSQL |
| Frontend | React + Tailwind CSS |
| Stress Test | Artillery |
| AI Models | TFT + DQN (PyTorch) |

## Tài Liệu Tham Khảo

- "A Transformer-Based Deep Q-Learning Approach for SDN Load Balancing"
- Ryu SDN Framework: https://ryu-sdn.org
- Mininet: http://mininet.org
