import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from tft_dqn_net import TFT_DQN_Model

class ReplayBuffer:
    def __init__(self, capacity=10000):
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
    def __init__(self, input_size=2, seq_len=5, hidden_size=64, num_actions=3, 
                 lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Khởi tạo Agent sử dụng đồ họa: {self.device}")
        
        # 1. Main Network (Nghe nhìn + Dự đoán)
        self.policy_net = TFT_DQN_Model(input_size, seq_len, hidden_size, num_actions).to(self.device)
        
        # 2. Target Network (Đóng băng update chậm để ổn định hóa Q-Learning)
        self.target_net = TFT_DQN_Model(input_size, seq_len, hidden_size, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Không train Target net
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 2 dạng Loss: MSE cho Q-value và Aux-Loss (MSE) cho khối dự đoán tương lai TFT
        self.loss_fn = nn.MSELoss()
        self.tft_loss_fn = nn.MSELoss()
        
        self.memory = ReplayBuffer()

    def select_action(self, state):
        """
        Epsilon-Greedy: Khám phá vs Tận dụng
        """
        # Nếu random number nhỏ hơn epsilon -> Khám phá ngu ngốc
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        # Ngược lại: Trí tuệ AI ra quyết định (Argmax Q-Value)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Add batch_dim
            q_values, _ = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update_epsilon(self):
        # Mài giuòn từ từ độ random sau mỗi bước
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
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
            # Công thức Bellman (Q_target = R + Gamma * max(Q_next)) nếu state chưa chết (done=0)
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
        
        # Trick: Cắt gradient tránh phát nổ Parameter
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
                
        self.optimizer.step()
        
        return dqn_loss.item(), tft_loss.item()
    
    def update_target_network(self):
        """ Hard update: Copy 100% tủy não của policy qua target """
        self.target_net.load_state_dict(self.policy_net.state_dict())
