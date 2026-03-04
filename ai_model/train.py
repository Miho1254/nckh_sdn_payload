import os
import json
import torch
import random
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
    agent = DQNAgent(input_size=num_features, seq_len=seq_len, hidden_size=32, num_actions=env.num_actions)
    
    # --- THONG SO HUAN LUYEN (HYPERPARAMS) ---
    NUM_EPISODES = 200  # Tăng lên 200 để epsilon giảm sâu hơn (0.9999^200 = 0.819)
    BATCH_SIZE = 32
    TARGET_UPDATE = 5
    
    # ================================================================
    # TRACKERS cho 5 bieu do
    # ================================================================
    episode_rewards = []
    dqn_losses_history = []
    tft_losses_history = []
    epsilon_history = []
    temperature_history = []
    
    # TFT Prediction vs Actual (thu thap o epoch cuoi)
    tft_predictions = []
    tft_actuals = []
    
    # Action Distribution (dem so lan chon tung server moi epoch)
    action_counts_history = []  # List of [count_h5, count_h7, count_h8]
    
    print("\n🚀 BAT DAU TRAINING Mo Hinh AI: [TFT + DQN] (V3 - Clean + Boltzmann)")
    print("=" * 60)
    
    # Best model tracking
    best_reward = -float('inf')
    best_epoch = 0
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        dqn_loss_eps = 0.0
        tft_loss_eps = 0.0
        steps = 0
        
        # 1. SNAPSHOT FORWARD: Tính Q-values cho toàn bộ states 1 lần
        all_q_values = agent.get_q_values_batch(env.X_data)
        
        # Pre-compute Boltzmann actions cho toàn bộ dataset
        boltzmann_actions = agent.boltzmann_select_batch(all_q_values)
        
        action_counts = [0, 0, 0]
        
        # 2. FAST SIMULATION: Tra cứu pre-computed actions
        while not done:
            # Epsilon-greedy + Boltzmann
            if random.random() < agent.epsilon:
                action = random.randrange(agent.num_actions)
            else:
                action = boltzmann_actions[steps]  # Boltzmann thay vì argmax
                
            action_counts[action] += 1
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.memory.push(state, action, reward, next_state, float(done))
            
            # Thu thap TFT Pred (chỉ ở epoch cuối - dùng snapshot lúc nãy)
            if episode >= NUM_EPISODES - 1:
                tft_actuals.append(next_state[-1, :])
                
            state = next_state
            steps += 1

        # 3. BATCH TRAINING: Train bù vào cuối episode (ổn định & cực nhanh)
        # 724 steps / TRAIN_FREQ = ~72 lượt train
        TRAIN_FREQ = 10
        num_train_passes = steps // TRAIN_FREQ
        train_steps = 0
        
        if len(agent.memory) > BATCH_SIZE:
            for _ in range(num_train_passes):
                d_loss, t_loss = agent.train_step(BATCH_SIZE)
                if d_loss is not None:
                    dqn_loss_eps += d_loss
                    tft_loss_eps += (t_loss if t_loss else 0)
                    train_steps += 1
        
        # Epsilon decay + Temperature decay: 1 lần mỗi epoch
        agent.update_epsilon()
        agent.update_temperature()
        
        # Soft update Target Net: 1 lần mỗi episode (ổn định)
        agent.update_target_network()
            
        # Luu Log
        episode_rewards.append(total_reward)
        avg_dqn_loss = dqn_loss_eps / train_steps if train_steps > 0 else 0
        avg_tft_loss = tft_loss_eps / train_steps if train_steps > 0 else 0
        dqn_losses_history.append(avg_dqn_loss)
        tft_losses_history.append(avg_tft_loss)
        epsilon_history.append(agent.epsilon)
        temperature_history.append(agent.temperature)
        action_counts_history.append(action_counts)
        
        # Print progress
        is_best = ''
        if total_reward > best_reward:
            best_reward = total_reward
            best_epoch = episode + 1
            is_best = ' \u2b50 BEST'
        
        print(f"  Epoch {episode+1:3d}/{NUM_EPISODES} | "
              f"Eps: {agent.epsilon:.3f} | "
              f"Tau: {agent.temperature:.2f} | "
              f"Reward: {total_reward:7.1f} | "
              f"Q-Loss: {avg_dqn_loss:.6f} | "
              f"Actions: h5={action_counts[0]} h7={action_counts[1]} h8={action_counts[2]}{is_best}")
              
    print("=" * 60)
    print(f"Training Hoan Tat! Best Reward: {best_reward:.1f} tai Epoch {best_epoch}")
    print("Dang sao luu mo hinh...")
    
    # 4. FINAL SNAPSHOT: Thu thập dự báo cuối cùng cho toàn bộ dataset
    print("[*] Dang thu thap du bao cuoi cung (Final Snapshot)...")
    with torch.no_grad():
        all_q, all_future = agent.policy_net(torch.FloatTensor(env.X_data).to(agent.device))
        tft_predictions = all_future.cpu().numpy()

    # Luu Model Checkpoint
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'tft_dqn_master.pth')
    try:
        torch.save(agent.policy_net.state_dict(), model_path)
        print(f"  Da luu model tai: {model_path}")
    except Exception as e:
        print(f"  Khong the luu model: {e}")
        # Luu vao thu muc hien tai neu that bai
        fallback_path = os.path.join(BASE_DIR, 'tft_dqn_master_fallback.pth')
        torch.save(agent.policy_net.state_dict(), fallback_path)
        print(f"  Da luu fallback tai: {fallback_path}")
    
    # (Raw data saving logic...)
    raw_data = {
        'episode_rewards': episode_rewards,
        'dqn_losses': dqn_losses_history,
        'tft_losses': tft_losses_history,
        'epsilon_history': epsilon_history,
        'action_counts': action_counts_history,
        'num_episodes': NUM_EPISODES,
        'tft_predictions': [p.tolist() for p in tft_predictions] if len(tft_predictions)>0 else [],
        'tft_actuals': [a.tolist() for a in tft_actuals] if len(tft_actuals)>0 else [],
    }
    raw_path = os.path.join(BASE_DIR, 'processed_data', 'training_metrics.json')
    with open(raw_path, 'w') as f:
        json.dump(raw_data, f, indent=2)
    
    # VE 5 BIEU DO CHUYEN NGHIEP
    print("\n📊 Dang xuat bieu do nghiem thu...")
    os.makedirs(CHARTS_DIR, exist_ok=True)
    setup_dark_style()
    _plot_training_dashboard(
        episode_rewards, dqn_losses_history, tft_losses_history,
        epsilon_history, temperature_history, action_counts_history, 
        tft_predictions, tft_actuals, NUM_EPISODES
    )
    print(f"\n✅ HOAN TAT! Tat ca bieu do da nam tai: {CHARTS_DIR}/")


def _plot_training_dashboard(rewards, dqn_losses, tft_losses, epsilons, temperatures,
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
    ax.set_title('Total Reward per Epoch')
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
        
        n_feat = preds_arr.shape[1] if preds_arr.ndim > 1 else 1
        n_cols = min(n_feat, 3)
        n_rows = (n_feat + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        feature_names = ['Byte Rate', 'Packet Rate', 'Load h5', 'Load h7', 'Load h8']
        colors_feat = [COLORS['primary'], COLORS['accent'], COLORS['secondary'], COLORS['warning'], COLORS['danger']]
        
        # Chi lay 200 diem cuoi de bieu do doc duoc
        n_show = min(200, len(preds_arr))
        steps_range = range(n_show)
        
        if n_rows == 1 and n_cols == 1:
            axes_list = [axes]
        else:
            axes_list = list(np.array(axes).flatten())
        for i in range(min(n_feat, len(feature_names), len(axes_list))):
            ax = axes_list[i]
            fname = feature_names[i]
            c = colors_feat[i % len(colors_feat)]
            ax.plot(steps_range, actuals_arr[-n_show:, i], color=c, 
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
    
    # ── 4. Epsilon & Temperature Decay ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, epsilons, color=COLORS['accent'], linewidth=2.5, label='Epsilon (Randomness)')
    ax.fill_between(epochs, epsilons, alpha=0.1, color=COLORS['accent'])
    
    ax2 = ax.twinx()  # Trục phụ cho Temperature
    ax2.plot(epochs, temperatures, color=COLORS['warning'], linewidth=2, linestyle='--', label='Tau (Boltzmann Temp)')
    ax2.set_ylabel('Temperature (Tau)')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Epsilon')
    ax.set_title('Epsilon & Boltzmann Temperature Decay Schedule')
    
    # Gom legend từ cả 2 trục
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')
    
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '04_decay_schedule.png'), dpi=200)
    plt.close()
    print("  [4/5] 04_decay_schedule.png")
    
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
