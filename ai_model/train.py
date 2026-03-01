import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sdn_env import SDN_Offline_Env
from dqn_agent import DQNAgent

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X_DATA = os.path.join(BASE_DIR, 'processed_data', 'X_sequences.npy')
Y_DATA = os.path.join(BASE_DIR, 'processed_data', 'y_labels.npy')
MODEL_DIR = os.path.join(BASE_DIR, 'checkpoints')
CHARTS_DIR = os.path.join(BASE_DIR, 'processed_data', 'charts')

# Màu sắc chuyên nghiệp
COLORS = {
    'primary': '#2563EB',
    'secondary': '#10B981',
    'danger': '#EF4444',
    'warning': '#F59E0B',
    'accent': '#8B5CF6',
    'bg': '#0F172A',
    'card_bg': '#1E293B',
    'text': '#E2E8F0',
    'grid': '#334155',
}

def setup_dark_style():
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['card_bg'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
    })

def moving_average(data, window=5):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def train_tft_dqn():
    if not os.path.exists(X_DATA):
        print("Chua co data NPY. Hay chay lenh: python3 data_processor.py")
        return

    # Khoi tao Moi truong Offline S-D-N
    env = SDN_Offline_Env(X_DATA, Y_DATA)
    state_shape = env.get_state_shape()
    seq_len = state_shape[0]
    num_features = state_shape[1]
    
    # Khoi tao Agent
    agent = DQNAgent(input_size=num_features, seq_len=seq_len, hidden_size=64, num_actions=env.num_actions)
    
    # --- THONG SO HUAN LUYEN (HYPERPARAMS) ---
    NUM_EPISODES = 50
    BATCH_SIZE = 32
    TARGET_UPDATE = 5
    
    # ================================================================
    # TRACKERS cho 5 bieu do
    # ================================================================
    episode_rewards = []
    dqn_losses_history = []
    tft_losses_history = []
    epsilon_history = []
    
    # TFT Prediction vs Actual (thu thap o epoch cuoi)
    tft_predictions = []
    tft_actuals = []
    
    # Action Distribution (dem so lan chon tung server moi epoch)
    action_counts_history = []  # List of [count_h5, count_h7, count_h8]
    
    print("\n🚀 BAT DAU TRAINING Mo Hinh AI: [TFT + DQN]")
    print("=" * 60)
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        dqn_loss_eps = 0.0
        tft_loss_eps = 0.0
        steps = 0
        train_steps = 0
        
        # Dem action trong episode nay
        action_counts = [0, 0, 0]
        
        while not done:
            action = agent.select_action(state)
            action_counts[action] += 1
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            agent.memory.push(state, action, reward, next_state, float(done))
            
            TRAIN_FREQ = 4
            if len(agent.memory) > BATCH_SIZE and steps % TRAIN_FREQ == 0:
                d_loss, t_loss = agent.train_step(BATCH_SIZE)
                if d_loss is not None:
                    dqn_loss_eps += d_loss
                    tft_loss_eps += (t_loss if t_loss else 0)
                    train_steps += 1
                
                agent.update_epsilon()
            
            # Thu thap TFT Pred vs Actual (chi o 5 epoch cuoi)
            if episode >= NUM_EPISODES - 5:
                with torch.no_grad():
                    s_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    _, future_pred = agent.policy_net(s_tensor)
                    actual_future = next_state[-1, :]
                    tft_predictions.append(future_pred.cpu().numpy().flatten())
                    tft_actuals.append(actual_future)
                
            state = next_state
            steps += 1
            
        # Cap nhat Target Net
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
            
        # Luu Log
        episode_rewards.append(total_reward)
        avg_dqn_loss = dqn_loss_eps / train_steps if train_steps > 0 else 0
        avg_tft_loss = tft_loss_eps / train_steps if train_steps > 0 else 0
        dqn_losses_history.append(avg_dqn_loss)
        tft_losses_history.append(avg_tft_loss)
        epsilon_history.append(agent.epsilon)
        action_counts_history.append(action_counts)
        
        print(f"  Epoch {episode+1:2d}/{NUM_EPISODES} | "
              f"Eps: {agent.epsilon:.3f} | "
              f"Reward: {total_reward:6.1f} | "
              f"Q-Loss: {avg_dqn_loss:.4f} | "
              f"Actions: h5={action_counts[0]} h7={action_counts[1]} h8={action_counts[2]}")
              
    print("=" * 60)
    print("Training Hoan Tat! Dang sao luu mo hinh...")
    
    # Luu Model Checkpoint
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'tft_dqn_master.pth')
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"  Da luu model tai: {model_path}")
    
    # ================================================================
    # LUU RAW DATA (JSON) de co the ve lai bieu do bat ky luc nao
    # ================================================================
    raw_data = {
        'episode_rewards': episode_rewards,
        'dqn_losses': dqn_losses_history,
        'tft_losses': tft_losses_history,
        'epsilon_history': epsilon_history,
        'action_counts': action_counts_history,
        'num_episodes': NUM_EPISODES,
    }
    raw_path = os.path.join(BASE_DIR, 'processed_data', 'training_metrics.json')
    with open(raw_path, 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  Da luu raw metrics tai: {raw_path}")
    
    # Luu TFT pred/actual rieng (numpy)
    if tft_predictions:
        np.save(os.path.join(BASE_DIR, 'processed_data', 'tft_predictions.npy'), np.array(tft_predictions))
        np.save(os.path.join(BASE_DIR, 'processed_data', 'tft_actuals.npy'), np.array(tft_actuals))
    
    # ================================================================
    # VE 5 BIEU DO CHUYEN NGHIEP
    # ================================================================
    print("\n📊 Dang xuat bieu do nghiem thu...")
    os.makedirs(CHARTS_DIR, exist_ok=True)
    setup_dark_style()
    
    _plot_training_dashboard(
        episode_rewards, dqn_losses_history, tft_losses_history,
        epsilon_history, action_counts_history, 
        tft_predictions, tft_actuals, NUM_EPISODES
    )
    
    print(f"\n✅ HOAN TAT! Tat ca bieu do da nam tai: {CHARTS_DIR}/")


def _plot_training_dashboard(rewards, dqn_losses, tft_losses, epsilons,
                              action_counts, tft_preds, tft_actuals, num_eps):
    """Ve Dashboard tong hop 5 bieu do training."""
    
    epochs = range(1, num_eps + 1)
    
    # ── 1. Q-Loss Curve ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, dqn_losses, color=COLORS['danger'], alpha=0.35, linewidth=1, label='Q-Loss (Raw)')
    if len(dqn_losses) >= 5:
        ma = moving_average(dqn_losses, 5)
        ax.plot(range(3, 3 + len(ma)), ma, color=COLORS['danger'], linewidth=2.5, label='Q-Loss (MA-5)')
    ax.plot(epochs, tft_losses, color=COLORS['warning'], alpha=0.35, linewidth=1, label='TFT-Loss (Raw)')
    if len(tft_losses) >= 5:
        ma_tft = moving_average(tft_losses, 5)
        ax.plot(range(3, 3 + len(ma_tft)), ma_tft, color=COLORS['warning'], linewidth=2.5, label='TFT-Loss (MA-5)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Q-Loss & TFT Auxiliary Loss Curve')
    ax.legend(loc='upper right', framealpha=0.8)
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '01_loss_curve.png'), dpi=200)
    plt.close()
    print("  [1/5] 01_loss_curve.png")
    
    # ── 2. Total Reward ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(epochs, rewards, alpha=0.15, color=COLORS['secondary'])
    ax.plot(epochs, rewards, color=COLORS['secondary'], linewidth=1.5, marker='o', markersize=4, label='Total Reward')
    if len(rewards) >= 5:
        ma_r = moving_average(rewards, 5)
        ax.plot(range(3, 3 + len(ma_r)), ma_r, color='#34D399', linewidth=2.5, linestyle='--', label='Trend (MA-5)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Total Reward per Epoch (Higher = Smarter)')
    ax.legend(loc='lower right', framealpha=0.8)
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '02_reward_curve.png'), dpi=200)
    plt.close()
    print("  [2/5] 02_reward_curve.png")
    
    # ── 3. TFT Prediction vs Actual ───────────────────────────
    if len(tft_preds) > 0:
        preds_arr = np.array(tft_preds)
        actuals_arr = np.array(tft_actuals)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        feature_names = ['Byte Rate (Normalized)', 'Packet Rate (Normalized)']
        colors_feat = [COLORS['primary'], COLORS['accent']]
        
        # Chi lay 200 diem cuoi de bieu do doc duoc
        n_show = min(200, len(preds_arr))
        steps_range = range(n_show)
        
        for i, (ax, fname) in enumerate(zip(axes, feature_names)):
            ax.plot(steps_range, actuals_arr[-n_show:, i], color=colors_feat[i], 
                    linewidth=1.8, label='Actual', alpha=0.9)
            ax.plot(steps_range, preds_arr[-n_show:, i], color=COLORS['warning'], 
                    linewidth=1.5, linestyle='--', label='TFT Predicted', alpha=0.8)
            ax.set_xlabel('Timestep')
            ax.set_ylabel(fname)
            ax.set_title(f'TFT Prediction: {fname}')
            ax.legend(framealpha=0.8)
            ax.grid(True, linestyle='--')
        
        plt.suptitle('TFT Forecast vs Actual Traffic (Last 5 Epochs)', 
                     fontsize=14, fontweight='bold', color=COLORS['text'])
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, '03_tft_prediction.png'), dpi=200)
        plt.close()
        print("  [3/5] 03_tft_prediction.png")
    else:
        print("  [3/5] SKIPPED (khong du du lieu TFT)")
    
    # ── 4. Epsilon Decay ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(epochs, epsilons, alpha=0.15, color=COLORS['accent'])
    ax.plot(epochs, epsilons, color=COLORS['accent'], linewidth=2.5, marker='s', markersize=4)
    
    # Ve duong phan chia Exploration / Exploitation
    mid_eps = 0.5
    ax.axhline(y=mid_eps, color=COLORS['warning'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(num_eps * 0.85, mid_eps + 0.03, 'Exploration Zone', color=COLORS['warning'], fontsize=9, alpha=0.8)
    ax.text(num_eps * 0.85, mid_eps - 0.07, 'Exploitation Zone', color=COLORS['secondary'], fontsize=9, alpha=0.8)
    ax.axhline(y=0.05, color=COLORS['danger'], linestyle=':', linewidth=1, alpha=0.5)
    ax.text(1, 0.07, 'Min Epsilon (0.05)', color=COLORS['danger'], fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Epsilon')
    ax.set_title('Epsilon Decay: Exploration vs Exploitation')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '04_epsilon_decay.png'), dpi=200)
    plt.close()
    print("  [4/5] 04_epsilon_decay.png")
    
    # ── 5. Action Distribution ────────────────────────────────
    action_arr = np.array(action_counts)
    totals = action_arr.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    action_pct = (action_arr / totals) * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    server_labels = ['h5 (Server 1)', 'h7 (Server 2)', 'h8 (Server 3)']
    server_colors = [COLORS['primary'], COLORS['secondary'], COLORS['warning']]
    
    bottom = np.zeros(num_eps)
    for i in range(3):
        ax.bar(epochs, action_pct[:, i], bottom=bottom, label=server_labels[i],
               color=server_colors[i], alpha=0.85, width=0.8)
        bottom += action_pct[:, i]
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Action Ratio (%)')
    ax.set_title('Server Selection Distribution Over Training')
    ax.legend(loc='upper right', framealpha=0.8)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '05_action_distribution.png'), dpi=200)
    plt.close()
    print("  [5/5] 05_action_distribution.png")
    
    # ── DASHBOARD TONG HOP ────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('NCKH SDN: TFT-DQN Training Dashboard', fontsize=18, fontweight='bold', 
                 color=COLORS['text'], y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Q-Loss (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, dqn_losses, color=COLORS['danger'], alpha=0.4, linewidth=1)
    if len(dqn_losses) >= 5:
        ma = moving_average(dqn_losses, 5)
        ax1.plot(range(3, 3+len(ma)), ma, color=COLORS['danger'], linewidth=2.5)
    ax1.set_title('Q-Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.grid(True, linestyle='--')
    
    # 2. Reward (top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(epochs, rewards, alpha=0.15, color=COLORS['secondary'])
    ax2.plot(epochs, rewards, color=COLORS['secondary'], linewidth=2, marker='o', markersize=3)
    ax2.set_title('Total Reward')
    ax2.set_xlabel('Epoch')
    ax2.grid(True, linestyle='--')
    
    # 3. Epsilon (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.fill_between(epochs, epsilons, alpha=0.15, color=COLORS['accent'])
    ax3.plot(epochs, epsilons, color=COLORS['accent'], linewidth=2.5)
    ax3.axhline(y=0.5, color=COLORS['warning'], linestyle=':', linewidth=1, alpha=0.5)
    ax3.set_title('Epsilon Decay')
    ax3.set_xlabel('Epoch')
    ax3.set_ylim(-0.02, 1.05)
    ax3.grid(True, linestyle='--')
    
    # 4. Action Distribution (bottom-left)
    ax4 = fig.add_subplot(gs[1, 0])
    bottom_d = np.zeros(num_eps)
    for i in range(3):
        ax4.bar(epochs, action_pct[:, i], bottom=bottom_d, label=server_labels[i],
                color=server_colors[i], alpha=0.85, width=0.8)
        bottom_d += action_pct[:, i]
    ax4.set_title('Action Distribution')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('%')
    ax4.legend(fontsize=7)
    ax4.grid(True, linestyle='--', axis='y')
    
    # 5. TFT Prediction (bottom-center & right)
    if len(tft_preds) > 0:
        preds_arr = np.array(tft_preds)
        actuals_arr = np.array(tft_actuals)
        n_show = min(150, len(preds_arr))
        sr = range(n_show)
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(sr, actuals_arr[-n_show:, 0], color=COLORS['primary'], linewidth=1.5, label='Actual')
        ax5.plot(sr, preds_arr[-n_show:, 0], color=COLORS['warning'], linewidth=1.2, linestyle='--', label='Predicted')
        ax5.set_title('TFT: Byte Rate')
        ax5.legend(fontsize=7)
        ax5.grid(True, linestyle='--')
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(sr, actuals_arr[-n_show:, 1], color=COLORS['accent'], linewidth=1.5, label='Actual')
        ax6.plot(sr, preds_arr[-n_show:, 1], color=COLORS['warning'], linewidth=1.2, linestyle='--', label='Predicted')
        ax6.set_title('TFT: Packet Rate')
        ax6.legend(fontsize=7)
        ax6.grid(True, linestyle='--')
    
    plt.savefig(os.path.join(CHARTS_DIR, '00_training_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("\n  >> 00_training_dashboard.png (TONG HOP)")


if __name__ == '__main__':
    train_tft_dqn()
