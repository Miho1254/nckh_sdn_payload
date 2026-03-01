#!/usr/bin/env bash
# Tự động hóa quá trình deploy LMS và chạy Stress Test trên mạng Fat-Tree
# Yêu cầu: Đang chạy script này bên trong Mininet CLI (hoặc container).
# Cách dùng tốt nhất: m <hostname> <command>
# Tuy nhiên, trong Mininet shell, có thể m không khả dụng.
# Nếu chạy ở ngoài, dùng `m` (mininet util). Tốt nhất là gọi script python qua Mininet API.
