import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from tft_dqn_net import TFT_DQN_Model

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Lưu trữ cặp (S, A, R, S_next) dưới dạng Numpy array để đỡ hao RAM
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*samples)
        
        # Convert qua Tensor
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(done))
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_size=2, seq_len=5, hidden_size=32, num_actions=3,
                 lr=3e-4, gamma=0.95, epsilon_start=1.0, epsilon_end=0.15, epsilon_decay=0.985,
                 tau=0.005, temperature_start=2.0, temperature_end=0.5, temperature_decay=0.99):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau  # Soft update coefficient
        
        # Boltzmann temperature: kiểm soát mức độ "mạnh dạn" của action selection
        # Cao = phân bố đều (explore), Thấp = tập trung vào Q cao nhất (exploit)
        self.temperature = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Khởi tạo Agent sử dụng đồ họa: {self.device}")
        
        # 1. Main Network (Nghe nhìn + Dự đoán)
        self.policy_net = TFT_DQN_Model(input_size, seq_len, hidden_size, num_actions).to(self.device)
        
        # 2. Target Network (Đóng băng update chậm để ổn định hóa Q-Learning)
        self.target_net = TFT_DQN_Model(input_size, seq_len, hidden_size, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Không train Target net
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Huber Loss (SmoothL1) cho Q-value: giới hạn gradient khi TD error lớn
        self.loss_fn = nn.SmoothL1Loss()
        self.tft_loss_fn = nn.MSELoss()
        
        self.memory = ReplayBuffer()

    def select_action(self, state):
        """Chọn hành động đơn lẻ (dùng khi inference thực tế)."""
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values, _ = self.policy_net(state_t)
        
        # Boltzmann softmax selection thay vì argmax
        return self._boltzmann_select(q_values.cpu().numpy()[0])
    
    def _boltzmann_select(self, q_values):
        """Chọn action theo phân bố Boltzmann (softmax với temperature)."""
        q = q_values / max(self.temperature, 0.01)  # Scale bởi temperature
        q = q - np.max(q)  # Stability: trừ max để tránh overflow exp()
        exp_q = np.exp(q)
        probs = exp_q / exp_q.sum()
        return np.random.choice(self.num_actions, p=probs)
    
    def boltzmann_select_batch(self, all_q_values):
        """Tính Boltzmann actions cho toàn bộ dataset (dùng cho Snapshot Forward)."""
        q = all_q_values / max(self.temperature, 0.01)
        q = q - np.max(q, axis=1, keepdims=True)  # Stability per row
        exp_q = np.exp(q)
        probs = exp_q / exp_q.sum(axis=1, keepdims=True)  # [N, 3]
        
        # Sample 1 action cho mỗi state
        actions = np.array([np.random.choice(self.num_actions, p=p) for p in probs])
        return actions

    def get_q_values_batch(self, states_array):
        """Tính toán Q-values cho toàn bộ dataset một lần duy nhất (Snapshot Forward)."""
        states_t = torch.FloatTensor(states_array).to(self.device)
        # Đảm bảo policy_net ở train mode để LSTM hoạt động đúng
        self.policy_net.train()
        with torch.no_grad():
            q_values, _ = self.policy_net(states_t)
        return q_values.cpu().numpy()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_temperature(self):
        """Giảm temperature mỗi epoch (tương tự epsilon decay)."""
        self.temperature = max(self.temperature_end, self.temperature * self.temperature_decay)
        
    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return None, None # Chưa đủ data thì chưa học
        
        # 1. Lấy memory từ Buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 2. Tính Q Target
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            # Bellman equation (KHÔNG scale reward — giữ nguyên tín hiệu gradient)
            target_q = rewards + self.gamma * max_next_q * (1.0 - dones)
            
        # 3. Tính Q Hiện tại lấy từ mạng chính
        curr_q_values, future_pred = self.policy_net(states)
        # Chỉ lấy Q của cái hành động thực tế đã chọn trong Buffer
        curr_q = curr_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 4. Tính toán hàm Suy hao (Loss)
        # 4.1. Loss của thuật toán Q-Learning
        dqn_loss = self.loss_fn(curr_q, target_q)
        
        # 4.2. Loss phụ của TFT (Bắt mạng phải học luôn cách dự đoán state t+1)
        # Ở đây ta lấy label tương lai chính là state thực ở timstep cuối của next_state
        actual_future = next_states[:, -1, :] 
        tft_loss = self.tft_loss_fn(future_pred, actual_future)
        
        # ================= LÕI TRÁI TIM PAPER =================
        # Cộng dồn 2 Loss (Tối ưu đồng thời vừa Q-Value vừa Dự báo traffic)
        total_loss = dqn_loss + tft_loss
        
        # Backpropagation (Truyền ngược củng cố nơ-ron)
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (standard approach)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                
        self.optimizer.step()
        
        return dqn_loss.item(), tft_loss.item()
    
    def update_target_network(self):
        """ Soft update: θ_target = τ*θ_policy + (1-τ)*θ_target """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
