import numpy as np

class SDN_Offline_Env:
    """
    Môi trường Giả lập RL (OpenAI Gym-like) chạy Offline dựa trên Dataset Numpy.
    Mục tiêu: Dạy cho DQN Agent biết định tuyến (chọn Action) né các Server đang quá tải.
    """
    def __init__(self, data_x_path, data_y_path):
        # Data format: X shape [537, 5, 2] | y shape [537,]
        self.X_data = np.load(data_x_path)
        self.y_labels = np.load(data_y_path) # 0 = NORMAL, 1 = HIGH
        
        self.num_samples = len(self.X_data)
        self.current_step = 0
        
        # Có 3 nodes API để chuyển hướng tải: h5 (0), h7 (1), h8 (2)
        self.num_actions = 3 
        
        print(f"[*] SDN Env Khởi tạo: Sẵn sàng phát lại {self.num_samples} trạng thái mạng.")

    def reset(self):
        """ Reset về Sequence đầu tiên """
        self.current_step = 0
        return self.X_data[self.current_step]
    
    def step(self, action):
        """
        Nhận Action từ DQN Agent -> Chuyển Next State -> Trả về Reward
        """
        # Trạng thái hiện tại của mạng (Có nhãn HIGH hay NORMAL)
        current_label = self.y_labels[self.current_step]
        
        # ---------------- R E W A R D   L O G I C ----------------
        # Đây là Hàm Phần thưởng mô phỏng (Có thể tùy chỉnh lại theo logic thực tế hơn).
        # Giả định:
        # - Mạng đang NORMAL (0): Chuyển traffic đi đâu cũng được (+1 điểm)
        # - Mạng đang HIGH (1): Nếu dồn hết vào 1 Server cố định thì sẽ chết nghẽn. 
        # Cần luân chuyển Action liên tục (Load Balancing).
        
        reward = 0.0
        
        if current_label == 0: # Mạng Rảnh rỗi
            reward = 1.0
        else: # Mạng Quá Tải!
            # Nếu Agent chọn bừa 1 server khác với mặc định (VD mặc định là h5=0)
            if action != 0:
                reward = 5.0 # Mày thông minh! Đã biết san tải sang Server 1 và 2.
            else:
                reward = -5.0 # Bị phạt sấp mặt vì dồn vào Server 0 vốn đang quá tải nghiêm trọng.
                
        # -------------------------------------------------------------
        
        self.current_step += 1
        done = False
        
        # Hết file Data thì dừng 1 Episode
        if self.current_step >= self.num_samples - 1:
            done = True
            next_state = self.X_data[self.current_step]
        else:
            next_state = self.X_data[self.current_step]
            
        return next_state, reward, done
    
    def get_state_shape(self):
        # Kết quả: shape của 1 sample (e.g., (5, 2))
        return self.X_data[0].shape
