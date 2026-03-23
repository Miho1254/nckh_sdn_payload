#!/bin/bash
# Script để fix permission issues với checkpoints
# Chạy script này trước khi train nếu gặp lỗi "Permission denied"

echo "Fixing permissions for checkpoints directory..."

# Xóa thư mục checkpoints cũ (cần sudo)
if [ -d "checkpoints" ]; then
    echo "Removing old checkpoints directory..."
    rm -rf checkpoints 2>/dev/null || sudo rm -rf checkpoints
fi

# Tạo thư mục mới với quyền của user hiện tại
mkdir -p checkpoints
chmod 755 checkpoints

echo "Done! Checkpoints directory created."
echo "You can now run: python3 train_actor_critic.py --phase all --epochs 200"