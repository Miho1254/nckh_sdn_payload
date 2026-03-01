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
    """
    def __init__(self, num_features, hidden_size):
        super(VariableSelectionNetwork, self).__init__()
        self.num_features = num_features
        # Tính trọng số cho từng feature
        self.weight_grn = GatedResidualNetwork(num_features, num_features)
        # Pass từng feature qua GRN độc lập (để làm embedding phẳng)
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size) for _ in range(num_features)
        ])

    def forward(self, x):
        # x shape: [batch, seq_len, num_features]
        # Trọng số mềm qua Softmax
        weights = F.softmax(self.weight_grn(x), dim=-1) # [batch, seq_len, num_features]
        
        # Áp dụng trọng số vào từng feature mượt mà
        processed_features = []
        for i in range(self.num_features):
            feat_i = x[:, :, i:i+1] # Lấy 1 cột giữ nguyên dim 3
            feat_processed = self.feature_grns[i](feat_i) # [batch, seq_len, hidden]
            
            # Nhân với trọng số của biến đó
            w_i = weights[:, :, i:i+1]
            processed_features.append(w_i * feat_processed)
            
        # Tổng hợp các feature đã chọn
        x_out = torch.stack(processed_features, dim=-1).sum(dim=-1) # [batch, seq_len, hidden]
        return x_out

class TemporalSelfAttention(nn.Module):
    """
    Multi-Head Attention học ngữ cảnh giữa các timestep. 
    (Ví dụ timestep t-5 tương quan thế nào với t-1 để biết đang bùng nổ traffic)
    """
    def __init__(self, hidden_size, num_heads=4):
        super(TemporalSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x shape: [batch, seq_len, hidden]
        # Query, Key, Value đều là x (Self-Attention)
        attn_out, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_out)

# ==========================================
# 2. Kiến trúc Tổng hợp TFT + DQN
# ==========================================
class TFT_DQN_Model(nn.Module):
    """
    Mô hình End-to-End:
    Data -> TFT Encoder (VSN + Self-Attention) -> TFT Output (Dự đoán t+1)
                                      |
                                      +-> DQN MLP (Hành động Server)
    """
    def __init__(self, input_size=2, seq_len=5, hidden_size=64, num_actions=3):
        super(TFT_DQN_Model, self).__init__()
        
        # --- Phần TFT (Tái tạo đặc trưng Traffic theo thời gian) ---
        self.vsn = VariableSelectionNetwork(num_features=input_size, hidden_size=hidden_size)
        
        # LSTM để học tính tuần tự (như cấu trúc seq2seq của TFT gốc)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        
        self.temporal_attn = TemporalSelfAttention(hidden_size)
        self.tft_output = nn.Linear(hidden_size, input_size) # Nhánh 1: Dự đoán tương lai để huấn luyện embedding
        
        # --- Phần DQN (Học tăng cường) ---
        # Agent đưa ra quyết định dựa trên Context vector học được từ TFT
        self.q_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions) # Nhánh 2: Q-Values cho 3 Actions
        )
        
    def forward(self, state):
        """
        Input state: [batch, seq_len, num_features]
        """
        # 1. Local Processing (biến)
        x = self.vsn(state)
        
        # 2. Sequential Processing
        x, (hn, cn) = self.lstm(x)
        
        # 3. Temporal Attention 
        x = self.temporal_attn(x)
        
        # 4. Lấy context vector cuối cùng (của timestep cuối cùng trong cửa sổ để ra quyết định hiện tại)
        context_vector = x[:, -1, :] # [batch, hidden_size]
        
        # --- Nhánh 1: Dự đoán traffic (Auxiliary Loss) ---
        future_traffic_pred = self.tft_output(context_vector)
        
        # --- Nhánh 2: Tính Q-Value để đưa vào DQN ---
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
