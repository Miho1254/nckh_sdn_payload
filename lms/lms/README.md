# LMS — Hệ Thống Đăng Ký Học Phần

Ứng dụng web tạo traffic thực tế trên mạng SDN Fat-Tree phục vụ thu thập dataset stress test.

## Stack

- **Backend:** Node.js + Express + PostgreSQL
- **Frontend:** React + Tailwind CSS (Vite)
- **Stress Test:** Artillery

## Cấu Trúc

```
lms/
├── backend/
│   ├── server.js              # Express :4000 entry point
│   ├── db.js                  # PostgreSQL pool (20 connections)
│   ├── seed.js                # Tạo 500 SV + 20 môn học
│   ├── middleware/auth.js     # JWT verification
│   └── routes/
│       ├── auth.js            # POST /api/auth/login
│       ├── courses.js         # GET /api/courses
│       └── register.js        # POST/DELETE /api/register
├── frontend/
│   ├── vite.config.js         # Proxy API → :4000
│   └── src/
│       ├── App.jsx            # Auth state + layout
│       ├── pages/Login.jsx    # Đăng nhập
│       ├── pages/Courses.jsx  # Browse + đăng ký + hủy
│       └── components/CourseCard.jsx
└── stress-test/
    ├── artillery.yml          # 3 phases config
    └── functions.js           # Random generators
```

## API Endpoints

| Method | Path | Auth | Mô tả |
|---|---|---|---|
| `POST` | `/api/auth/login` | — | `{username, password}` → JWT |
| `GET` | `/api/courses` | — | Danh sách môn + available slots |
| `GET` | `/api/courses/:id` | — | Chi tiết 1 môn |
| `POST` | `/api/register` | JWT | `{course_id}` → đăng ký (stress point) |
| `DELETE` | `/api/register/:id` | JWT | Hủy đăng ký |
| `GET` | `/api/register/my` | JWT | Xem môn đã đăng ký |
| `GET` | `/api/health` | — | Health check |

## Database

**PostgreSQL** — chạy trên host riêng (h6 / 10.0.0.6) trong Mininet.

```
students    → 500 accounts (sv0001 - sv0500 / password123)
courses     → 20 môn học (25-50 slots/môn)
registrations → student ↔ course (UNIQUE constraint)
```

`POST /api/register` sử dụng **`SELECT ... FOR UPDATE`** row-level lock ngăn race condition khi burst traffic.

## Stress Test (Artillery)

3 phases mô phỏng đợt đăng ký học phần thực tế:

| Phase | Duration | Rate | Mô tả |
|---|---|---|---|
| Baseline | 2 phút | 2 req/s | Browse courses |
| Burst | 30s | 50 req/s | POST /register (write-heavy) |
| Recovery | 2 phút | 50 → 2 req/s | Check results |

Scenario weights: Browse 30% / Register 60% / Cancel 10%

### Chạy

```bash
# Cài deps
cd /work/lms/backend && npm install

# Seed database
DB_HOST=10.0.0.6 node seed.js

# Start server
DB_HOST=10.0.0.6 node server.js

# Stress test (từ terminal khác)
cd /work/lms/stress-test
artillery run artillery.yml
```

## Kiến Trúc Triển Khai trên Mininet

```
Pod 1 (Edge s1, s2)          Pod 2 (Edge s3, s4)
┌─────────────┐              ┌──────────────┐
│ h1: Frontend│  ──→         │ h5: Backend A│ (10.0.0.5)
│     :3000   │  API         │ h7: Backend B│ (10.0.0.7)
└─────────────┘              │ h8: Backend C│ (10.0.0.8)
                             │ h6: Postgres │ (10.0.0.6)
Pod 3-4 (Edge s9, s10)       └──────────────┘
┌───────────────┐
│ h9-h16:       │  ──→  Stress Clients
│ Artillery     │        (8 nodes, burst traffic)
└───────────────┘
```

Traffic được điều phối bởi **AI Load Balancer** (Ryu Controller) phân bổ đều giữa 3 Backend (h5, h7, h8) dựa trên trạng thái mạng real-time.
