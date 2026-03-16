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
        prev_step = current_state[-2] if len(current_state) > 1 else last_step
        
        # ── BASE THROUGHPUT ──
        # Tương đương với thành phần alpha * sum(Throughput) trong IEEE paper
        throughput = last_step[0] + last_step[1]  # byte_rate + packet_rate (0-2)
        
        # ── HÀM REWARD V4 (TIÊU CHUẨN NCKH IEEE) ──
        # Khắc phục hạn chế của bản gốc bằng cách tích hợp Temporal trend và Congestion penalty
        # Giúp Agent thông minh hơn trong 4 kịch bản: Flash Crowd, Predictable Ramping, Targeted Congestion, Gradual Shift.
        load_reward = 0.0
        if len(last_step) >= 5:
            server_loads = last_step[2:5]  # [load_h5, load_h7, load_h8]
            prev_loads = prev_step[2:5]
            chosen_load = server_loads[action]
            
            # 1. Thành phần Tối ưu Thông lượng (Throughput Maximization)
            r_throughput = throughput * 0.5
            
            # 2. Thành phần Cân bằng Tải (Load Balancing / Variance Reduction)
            # Khuyến khích hệ thống san phẳng tải (trọng số của reward lấy khoảng cách đến trung bình)
            mean_load = np.mean(server_loads)
            r_balance = 3.0 * (mean_load - chosen_load)
            
            # 3. Phạt Nghẽn Cổ Chai (Non-linear Congestion Penalty)
            # Mô phỏng Latency và Packet Loss của IEEE: Khi tải > 70%, trễ sẽ tăng theo hàm mũ.
            # Rất quan trọng cho kịch bản Targeted Congestion.
            r_congestion = 0.0
            if chosen_load > 0.7:
                r_congestion = -10.0 * ((chosen_load - 0.7) ** 2)
                
            # 4. Phạt Xu Hướng Thời Gian (Temporal Trend Penalty)
            # Khai thác sức mạnh của Transformer: tránh các server đang có đạo hàm tải tăng quá nhanh (Predictable Ramping)
            load_trend = chosen_load - prev_loads[action]
            r_temporal = 0.0
            if load_trend > 0:
                r_temporal = -2.0 * load_trend
                
            # Tổng hợp Reward (Có baseline 3.0 để tránh phân kì Gradient)
            reward = 3.0 + r_throughput + r_balance + r_congestion + r_temporal
            
            # ── TRAFFIC-TYPE MODIFIER (Xử lý Flash Crowd / Burst) ──
            if current_label == 1:  # HIGH traffic
                # Khi có bão truy cập, nguy cơ sập mạng (collapses) rất lớn, hệ số phạt nghẽn và thưởng cân bằng tăng vọt
                reward += (r_balance + r_congestion) * 1.5
        else:
            reward = 3.0 + throughput * 0.5
        
        # ── NEXT STATE ──
        self.current_step += 1
        done = self.current_step >= self.num_samples - 1
        next_state = self.X_data[min(self.current_step, self.num_samples - 1)]
            
        return next_state, reward, done
    
    def get_state_shape(self):
        return self.X_data[0].shape
