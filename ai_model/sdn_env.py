import numpy as np

class SDN_Offline_Env:
    """
    Môi trường RL Offline cho TFT-DQN Load Balancer (V3 - Clean Static).
    State: [byte_rate, packet_rate, load_h5, load_h7, load_h8] x 5 timesteps
    Action: 0=h5, 1=h7, 2=h8
    
    THIẾT KẾ V3:
    - Reward DETERMINISTIC: cùng state + cùng action = cùng reward (không random, không feedback loop)
    - Dựa 100% trên dữ liệu tải TĨNH từ features (load_h5, load_h7, load_h8)
    - Không dùng simulated_load hay deque (offline env không thể simulate)
    - Agent học: "nhìn tải các server trong state → route vào server nhẹ nhất"
    """
    def __init__(self, data_x_path, data_y_path):
        self.X_data = np.load(data_x_path)
        self.y_labels = np.load(data_y_path)
        
        self.num_samples = len(self.X_data)
        self.current_step = 0
        self.num_actions = 3
        
        num_features = self.X_data.shape[2] if self.X_data.ndim == 3 else 2
        print(f"[*] SDN Env: {self.num_samples} states | {num_features} features")

    def reset(self):
        self.current_step = 0
        return self.X_data[self.current_step]
    
    def step(self, action):
        current_label = self.y_labels[self.current_step]
        current_state = self.X_data[self.current_step]
        last_step = current_state[-1]  # Features từ timestep cuối
        
        # ── BASE THROUGHPUT ──
        throughput = last_step[0] + last_step[1]  # byte_rate + packet_rate (0-2)
        
        # ── LOAD-AWARE ROUTING REWARD ──
        # Dùng dữ liệu tải TĨNH từ features để hướng dẫn Agent
        # Agent học: "khi load_h7 cao → tránh h7, chọn server nhẹ hơn"
        load_reward = 0.0
        if len(last_step) >= 5:
            server_loads = last_step[2:5]  # [load_h5, load_h7, load_h8] (đã normalize 0-1)
            min_idx = np.argmin(server_loads)
            max_idx = np.argmax(server_loads)
            load_spread = server_loads[max_idx] - server_loads[min_idx]
            
            if load_spread > 0.03:  # Chỉ khi tải thực sự khác nhau
                if action == min_idx:
                    # THƯỞNG: Chọn server nhẹ nhất — scale theo mức chênh lệch
                    load_reward = 4.0 * load_spread
                elif action == max_idx:
                    # PHẠT: Chọn server nặng nhất
                    load_reward = -3.0 * load_spread
                else:
                    # TRUNG LẬP: Chọn server giữa
                    load_reward = 1.0 * load_spread
        
        # ── TRAFFIC-TYPE MODIFIER ──
        if current_label == 0:  # NORMAL traffic
            reward = 3.0 + throughput * 0.5 + load_reward
        else:  # HIGH traffic — load routing quan trọng GẤP ĐÔI
            reward = 2.5 + throughput * 0.5 + load_reward * 2.0
        
        # ── NEXT STATE ──
        self.current_step += 1
        done = self.current_step >= self.num_samples - 1
        next_state = self.X_data[min(self.current_step, self.num_samples - 1)]
            
        return next_state, reward, done
    
    def get_state_shape(self):
        return self.X_data[0].shape
