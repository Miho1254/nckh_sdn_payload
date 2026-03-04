import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Các thành phần con của TFT
# ==========================================
class GatedResidualNetwork(nn.Module):
    """
    GRN - Lọc và xử lý các non-linear relationships.
    Sử dụng GLU (Gated Linear Unit) để kiểm soát luồng thông tin.
    """
    def __init__(self, input_size, hidden_size):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2) # *2 cho GLU
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Skip connection projection nếu input != hidden
        self.skip_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        res = self.skip_proj(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        # Bẻ đôi cho Gated Linear Unit
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x = x1 * torch.sigmoid(x2)
        return self.layer_norm(res + x)

class VariableSelectionNetwork(nn.Module):
    """
    VSN - Lựa chọn tự động feature nào là quan trọng nhất ở mỗi timestep.
    Đã được VECTOR HÓA để chạy nhanh trên CPU.
    """
    def __init__(self, num_features, hidden_size):
        super(VariableSelectionNetwork, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.weight_grn = GatedResidualNetwork(num_features, num_features)
        
        # Một GRN duy nhất cho tất cả features (vectorized)
        self.feature_grn = GatedResidualNetwork(1, hidden_size)

    def forward(self, x):
        # x shape: [batch, seq_len, num_features]
        batch, seq, feat = x.shape
        
        # 1. Tính trọng số [batch, seq_len, num_features]
        weights = F.softmax(self.weight_grn(x), dim=-1)
        
        # 2. Xử lý toàn bộ features cùng lúc (vectorized)
        # Flatten x sang [batch*seq*feat, 1]
        x_flat = x.view(-1, 1)
        x_processed = self.feature_grn(x_flat) # [batch*seq*feat, hidden]
        
        # Reshape lại [batch, seq, feat, hidden]
        x_processed = x_processed.view(batch, seq, feat, self.hidden_size)
        
        # 3. Nhân trọng số và tổng hợp
        # weights: [batch, seq, feat] -> [batch, seq, feat, 1]
        weights = weights.unsqueeze(-1)
        x_out = (weights * x_processed).sum(dim=2) # [batch, seq, hidden]
        
        return x_out

class TemporalSelfAttention(nn.Module):
    """
    Multi-Head Attention học ngữ cảnh giữa các timestep. 
    """
    def __init__(self, hidden_size, num_heads=2):
        super(TemporalSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_out)

# ==========================================
# 2. Kiến trúc Tổng hợp TFT + DQN
# ==========================================
class TFT_DQN_Model(nn.Module):
    """
    Mô hình End-to-End tối ưu tốc độ (hidden=32, heads=2).
    """
    def __init__(self, input_size=2, seq_len=5, hidden_size=32, num_actions=3):
        super(TFT_DQN_Model, self).__init__()
        
        self.vsn = VariableSelectionNetwork(num_features=input_size, hidden_size=hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.temporal_attn = TemporalSelfAttention(hidden_size, num_heads=2)
        self.tft_output = nn.Linear(hidden_size, input_size)
        
        self.q_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
    def forward(self, state):
        x = self.vsn(state)
        x, (hn, cn) = self.lstm(x)
        x = self.temporal_attn(x)
        context_vector = x[:, -1, :] 
        future_traffic_pred = self.tft_output(context_vector)
        q_values = self.q_net(context_vector)
        return q_values, future_traffic_pred

if __name__ == '__main__':
    # Test thử kích thước ma trận
    print("Mô phỏng 1 Batch gồm 32 Samples, 5 timesteps, 2 features (Byte_rate, Packet_rate)")
    dummy_input = torch.randn(32, 5, 2)
    
    # Init Model (Actions = 3: server h5, h7, h8)
    model = TFT_DQN_Model(input_size=2, seq_len=5, hidden_size=64, num_actions=3)
    
    q_vals, future_pred = model(dummy_input)
    
    print("Shape của Q-Values:", q_vals.shape) # Dự kiến: [32, 3]
    print("Shape của Dự đoán Traffic (Bước t+1):", future_pred.shape) # Dự kiến: [32, 2]
    print("\n✅ Mạng PyTorch TFT-DQN khởi tạo thành công!")
