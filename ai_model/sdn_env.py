import os
import sys
import numpy as np

# Dynamically import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAPACITIES, NUM_ACTIONS, BACKENDS

class SDN_Offline_Env:
    """
    Môi trường RL Offline cho TFT-DQN Load Balancer (V3 - Clean Static).
    State: [byte_rate, packet_rate, load_hX...] x 5 timesteps
    Action: 0=h5, 1=h7, 2=h8...
    
    THIẾT KẾ V3:
    - Reward DETERMINISTIC: cùng state + cùng action = cùng reward (không random, không feedback loop)
    - Dựa 100% trên dữ liệu tải TĨNH từ features (load_hX...)
    - Không dùng simulated_load hay deque (offline env không thể simulate)
    - Agent học: "nhìn tải các server trong state → route vào server nhẹ nhất"
    """
    def __init__(self, data_x_path, data_y_path):
        self.X_data = np.load(data_x_path)
        self.y_labels = np.load(data_y_path)
        
        self.num_samples = len(self.X_data)
        self.current_step = 0
        self.num_actions = NUM_ACTIONS
        
        num_features = self.X_data.shape[2] if self.X_data.ndim == 3 else 2
        print(f"[*] SDN Env: {self.num_samples} states | {num_features} features | {self.num_actions} actions")

    def reset(self):
        self.current_step = 0
        return self.X_data[self.current_step]
    
    def step(self, action):
        current_label = self.y_labels[self.current_step]
        current_state = self.X_data[self.current_step]
        last_step = current_state[-1]  # Features từ timestep cuối
        prev_step = current_state[-2] if len(current_state) > 1 else last_step
        
        # ── BASE THROUGHPUT ──
        # Tương đương với thành phần alpha * sum(Throughput) trong IEEE paper
        throughput = last_step[0] + last_step[1]  # byte_rate + packet_rate (0-2)
        
        # ── HÀM REWARD V4 (TIÊU CHUẨN NCKH IEEE) ──
        # Khắc phục hạn chế của bản gốc bằng cách tích hợp Temporal trend và Congestion penalty
        # Giúp Agent thông minh hơn trong 4 kịch bản: Flash Crowd, Predictable Ramping, Targeted Congestion, Gradual Shift.
        load_reward = 0.0
        if len(last_step) >= (2 + self.num_actions):
            server_loads = last_step[2 : 2 + self.num_actions]  # Lấy dynamic length
            prev_loads = prev_step[2 : 2 + self.num_actions]
            
            # Khắc phục lỗi: Load phải được chuẩn hóa theo Capacity (trọng số 1:5:10 v.v.)
            # để AI không "sợ" server mạnh (h8) do absolute load cao
            utils = server_loads / CAPACITIES
            prev_utils = prev_loads / CAPACITIES
            
            chosen_util = utils[action]
            
            # 1. Thành phần Tối ưu Thông lượng (Throughput) - V7 ASSASSIN MODE
            # Thưởng cực đậm cho việc sử dụng đúng năng lực server.
            # h8 (10) sẽ có điểm cao gấp 10 lần h5 (1) cho cùng 1 lượng throughput.
            relative_capacity = CAPACITIES[action] / np.max(CAPACITIES)
            r_throughput = throughput * 10.0 * relative_capacity
            
            # 2. Thành phần Tránh nghẽn (Congestion Avoidance)
            # Phạt trực tiếp dựa trên Utilization của server được chọn trong dữ liệu thô.
            # Nếu server đó đã quá tải trong quá khứ, AI sẽ bị phạt nặng.
            r_congestion = -5.0 * chosen_util
            
            # 3. Thưởng Khả dụng (Headroom Reward)
            # Khuyến khích chọn server còn dư nhiều năng lực (Capacity - Load)
            # Headroom tính theo đơn vị Mbps (quy đổi từ Weight)
            headroom = CAPACITIES[action] - (server_loads[action] / 1e6) 
            r_headroom = 0.5 * headroom
                
            # Tổng hợp Reward V7 (Capacity-Master)
            # Baseline 10.0 để giữ gradient luôn dương, dễ học.
            reward = 10.0 + r_throughput + r_congestion + r_headroom
            
            # ── TRAFFIC-TYPE MODIFIER (Xử lý Flash Crowd / Burst) ──
            if current_label == 1:  # HIGH traffic
                # Khi bão về, chỉ có server mạnh mới cứu được hệ thống.
                # Ép AI phải ignore hoàn toàn server yếu (h5) nếu traffic cao.
                reward += r_throughput * 5.0
        else:
            reward = 3.0 + throughput * 0.5
        
        # ── NEXT STATE ──
        self.current_step += 1
        done = self.current_step >= self.num_samples - 1
        next_state = self.X_data[min(self.current_step, self.num_samples - 1)]
            
        return next_state, reward, done
    
    def get_state_shape(self):
        return self.X_data[0].shape
