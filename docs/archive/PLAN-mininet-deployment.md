# Project Plan: LMS Mininet Deployment

## Phase -1: Context Check
- **User Request**: Lên kế hoạch và brainstorm chiến lược triển khai hệ thống LMS phân tán trên mạng SDN Fat-Tree nhằm tạo dataset (Flow Stats, Port Stats) phục vụ train mô hình AI tải phân bằng TFT-DQN.
- **Goal**: Hoạch định phương án phân bổ 16 hosts tối ưu nhất để sinh ra network traffic phức tạp, nghẽn cổ chai cục bộ, và phản ánh đúng thực tế.

## Phase 0: Socratic Gate
(Đã vượt qua - code đã build xong, cần orchestration).

## Task Breakdown

### 1. Chuẩn bị Môi trường (Environment Prep)
- [ ] Đảm bảo Docker image `sdn-research` build thành công với Node.js 20.x và PostgreSQL.
- [ ] Kiểm tra `entrypoint.sh` đã cấp quyền và tạo tài khoản DB `lms` thành công khi start container.

### 2. Chiến lược Phân bổ Hosts (Host Allocation)
- [ ] **Pod 1 (h1-h4)**: 
  - `h1`: React Frontend (dành cho người thao tác thủ công để demo).
- [ ] **Pod 2 (h5-h8)**: 
  - `h6`: PostgreSQL Database Server (Tâm điểm ghi dữ liệu/Transaction).
  - `h5`, `h7`, `h8`: Node.js Backend API Servers (Load balanced endpoints).
- [ ] **Pod 3 & 4 (h9-h16)**: 
  - Chạy `run_labeled_test.py` (Artillery dán nhãn AI) mô phỏng người dùng tạo Burst traffic.

### 3. Bash/Python Orchestration Script
- [ ] Tạo script tự động `deploy_lms.sh` (chạy trên không gian Mininet) để tự động hóa:
  - Khởi động Backend trên h5, h7, h8 (trỏ DB_HOST=10.0.0.6).
  - Tự động trigger stress test trên h9-h16 cùng lúc.
- [ ] Đồng bộ hóa cơ chế ghi Label (`stats/current_label.txt`) để Controller Ryu thu thập chuẩn xác.

### 4. Thu thập và Xác thực Dataset
- [ ] Chạy Controller Ryu (`controller_stats.py`) song song.
- [ ] Thu thập `flow_stats.csv` và `port_stats.csv`.
- [ ] Visualize nhanh biểu đồ packet_count để xác thực dataset có thực sự phản ánh giai đoạn **BURST** (Cao điểm) không.

## Agent Assignments
- `backend-specialist`: Kiểm tra lại kết nối từ nhiều backend instances (h5, h7) vào 1 DB (h6).
- `bash-linux`: Viết script orchestration tự động hóa các lệnh CLI của Mininet (vốn đánh bằng tay rất mỏi).

## Verification Checklist
- [ ] Backend ở `h5` ping thấy `h6`.
- [ ] Test sinh viên đăng ký học phần từ `h9` thành công.
- [ ] `flow_stats.csv` có chứa dòng ghi nhãn `HIGH` và `NORMAL`.
