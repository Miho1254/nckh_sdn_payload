import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sdn_env import SDN_Offline_Env
from dqn_agent import DQNAgent

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X_DATA = os.path.join(BASE_DIR, 'processed_data', 'X_sequences.npy')
Y_DATA = os.path.join(BASE_DIR, 'processed_data', 'y_labels.npy')
MODEL_DIR = os.path.join(BASE_DIR, 'checkpoints')

def train_tft_dqn():
    if not os.path.exists(X_DATA):
        print("❌ Chưa có data NPY. Hãy chạy lệnh: python3 data_processor.py")
        return

    # Khởi tạo Môi trường Offline S-D-N
    env = SDN_Offline_Env(X_DATA, Y_DATA)
    state_shape = env.get_state_shape() # (seq_len=5, num_features=2)
    seq_len = state_shape[0]
    num_features = state_shape[1]
    
    # Khởi tạo Agent Béo của chúng ta
    agent = DQNAgent(input_size=num_features, seq_len=seq_len, hidden_size=64, num_actions=env.num_actions)
    
    # --- THÔNG SỐ HUẤN LUYỆN (HYPERPARAMS) ---
    NUM_EPISODES = 50   # Số vòng lặp qua lại hết file Data (Epochs)
    BATCH_SIZE = 32     # Mỗi lần lấy ngẫu nhiên 32 bước trong quá khứ để học
    TARGET_UPDATE = 5   # Cập nhật não (Target Net) sau mỗi 5 Episodes
    
    # Trackers để vẽ đồ thị
    episode_rewards = []
    dqn_losses_history = []
    
    print("\n🚀 BẮT ĐẦU TRAINING Mô Hình AI: [TFT + DQN]")
    print("=" * 50)
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        dqn_loss_eps = 0.0
        steps = 0
        
        while not done:
            # 1. DQN Ra quyết định chọn Server (Action 0, 1 hoặc 2)
            action = agent.select_action(state)
            
            # 2. Vứt quyết định vào Simulator -> Nhận Trạng thái mới & Điểm số
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 3. Quăng vào Trí nhớ (Replay Buffer)
            agent.memory.push(state, action, reward, next_state, float(done))
            
            # 4. Agent lấy 1 nắm quá khứ ra "Nhai" và tự điều chỉnh não bộ (Backprop)
            # Tối ưu CPU: Chỉ Backprop mỗi 4 steps (chống train qua mức gây chậm hệ thống)
            TRAIN_FREQ = 4
            if len(agent.memory) > BATCH_SIZE and steps % TRAIN_FREQ == 0:
                d_loss, t_loss = agent.train_step(BATCH_SIZE)
                dqn_loss_eps += d_loss
                
                # Update Epsilon (Bot bớt ngu ngơ random mà dùng IQ vô hạn)
                agent.update_epsilon()
                
            state = next_state
            steps += 1
            
        # Cập nhật não chậm (Target Net)
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
            
        # Lưu Log từng Episode
        episode_rewards.append(total_reward)
        avg_loss = dqn_loss_eps / steps if steps > 0 else 0
        dqn_losses_history.append(avg_loss)
        
        print(f"✅ Epoch {episode+1}/{NUM_EPISODES} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Tổng Thưởng (Reward): {total_reward:6.1f} | "
              f"Avg Q-Loss: {avg_loss:.4f}")
              
    print("=" * 50)
    print("🎉 Training Hoàn Tất! Đang sao lưu mô hình...")
    
    # Lưu Model Checkpoint
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'tft_dqn_master.pth')
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"💾 Đã đóng gói bộ não AI thành file `.pth` tại: {model_path}")
    
    # Vẽ Đồ thị Huấn luyện để báo cáo
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, color='green')
    plt.title('Tổng điểm Reward theo Epoch\n(Càng tăng càng giỏi)')
    plt.xlabel('Epochs')
    plt.ylabel('Cumulative Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(dqn_losses_history, color='red')
    plt.title('Độ lệnh Q-Loss theo Epoch\n(Càng giảm càng xịn)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    
    plt.tight_layout()
    chart_path = os.path.join(BASE_DIR, 'processed_data', 'training_dashboard.png')
    plt.savefig(chart_path, dpi=300)
    print(f"📊 Đã tạo Dashboard Báo Cáo tại: {chart_path}")

if __name__ == '__main__':
    train_tft_dqn()
