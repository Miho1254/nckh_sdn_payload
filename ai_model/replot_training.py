import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =====================================================================
# Cấu hình đường dẫn
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'processed_data', 'training_metrics.json')
CHARTS_DIR = os.path.join(BASE_DIR, 'processed_data', 'charts_professional')

# =====================================================================
# Màu sắc chuyên nghiệp (nền sáng cho thuyết trình)
# =====================================================================
COLORS = {
    'primary': '#1D4ED8',    # Blue
    'secondary': '#047857',  # Green
    'danger': '#B91C1C',     # Red
    'warning': '#D97706',    # Orange
    'accent': '#6D28D9',     # Purple
    'bg': '#FFFFFF',         # White
    'card_bg': '#F8FAFC',    # Light Slate
    'text': '#1E293B',       # Dark Slate
    'grid': '#CBD5E1',       # Light Gray
}

def setup_professional_style():
    """Thiết lập matplotlib theme sáng, chuyên nghiệp."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['card_bg'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.6,
        'font.size': 12,
        'axes.titlesize': 15,
        'axes.labelsize': 13,
        'axes.titleweight': 'bold',
    })

def moving_average(data, window=5):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def replot_training():
    if not os.path.exists(DATA_FILE):
        print(f"Khong tim thay tap tin du lieu: {DATA_FILE}")
        return

    # Doc file json
    with open(DATA_FILE, 'r') as f:
        metrics = json.load(f)

    rewards = metrics.get('episode_rewards', [])
    dqn_losses = metrics.get('dqn_losses', [])
    tft_losses = metrics.get('tft_losses', [])
    epsilons = metrics.get('epsilon_history', [])
    temperatures = metrics.get('temperature_history', []) # Co the khong co trong ban cu
    action_counts = metrics.get('action_counts', [])
    tft_preds = metrics.get('tft_predictions', [])
    tft_actuals = metrics.get('tft_actuals', [])
    num_eps = metrics.get('num_episodes', len(rewards))

    os.makedirs(CHARTS_DIR, exist_ok=True)
    setup_professional_style()
    print("Dang render cac bieu do moi...")
    
    # ── 1. Loss Curve ──────────────────────────────────────
    epochs = range(1, num_eps + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, dqn_losses, color=COLORS['danger'], alpha=0.35, linewidth=1, label='Q-Loss (Gốc)')
    if len(dqn_losses) >= 5:
        ma = moving_average(dqn_losses, 5)
        ax.plot(range(3, 3 + len(ma)), ma, color=COLORS['danger'], linewidth=2.5, label='Q-Loss (MA-5)')
    ax.plot(epochs, tft_losses, color=COLORS['warning'], alpha=0.35, linewidth=1, label='TFT-Loss (Gốc)')
    if len(tft_losses) >= 5:
        ma_tft = moving_average(tft_losses, 5)
        ax.plot(range(3, 3 + len(ma_tft)), ma_tft, color=COLORS['warning'], linewidth=2.5, label='TFT-Loss (MA-5)')
    ax.set_xlabel('Bước chạy (Epoch)')
    ax.set_ylabel('Hệ số suy giảm (MSE)')
    ax.set_title('Đường cong hội tụ Q-Loss & TFT-Loss')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '01_loss_curve.png'), dpi=300)
    plt.close()

    # ── 2. Total Reward ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(epochs, rewards, alpha=0.15, color=COLORS['secondary'])
    ax.plot(epochs, rewards, color=COLORS['secondary'], linewidth=1.5, marker='o', markersize=4, label='Phần thưởng tích lũy')
    if len(rewards) >= 5:
        ma_r = moving_average(rewards, 5)
        ax.plot(range(3, 3 + len(ma_r)), ma_r, color='#059669', linewidth=2.5, linestyle='--', label='Xu hướng (MA-5)')
    ax.set_xlabel('Bước chạy (Epoch)')
    ax.set_ylabel('Điểm thưởng')
    ax.set_title('Biến thiên điểm thưởng trong quá trình huấn luyện')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '02_reward_curve.png'), dpi=300)
    plt.close()

    # ── 3. Epsilon Decay ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, epsilons, color=COLORS['accent'], linewidth=2.5, label='Epsilon (Tỷ lệ khám phá)')
    ax.fill_between(epochs, epsilons, alpha=0.1, color=COLORS['accent'])
    ax.set_xlabel('Bước chạy (Epoch)')
    ax.set_ylabel('Hệ số Epsilon')
    ax.set_title('Lịch trình giảm thiểu hệ số thăm dò (Epsilon Decay)')
    ax.grid(True, linestyle='--')
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '03_epsilon_decay.png'), dpi=300)
    plt.close()

    # ── 4. Action Distribution ────────────────────────────────
    action_arr = np.array(action_counts)
    totals = action_arr.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    action_pct = (action_arr / totals) * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    server_labels = ['Máy chủ 1 (10 Mbps)', 'Máy chủ 2 (50 Mbps)', 'Máy chủ 3 (100 Mbps)']
    server_colors = [COLORS['primary'], COLORS['warning'], COLORS['danger']]
    
    bottom = np.zeros(num_eps)
    for i in range(3):
        ax.bar(epochs, action_pct[:, i], bottom=bottom, label=server_labels[i],
               color=server_colors[i], alpha=0.85, width=0.8)
        bottom += action_pct[:, i]
    
    ax.set_xlabel('Bước chạy (Epoch)')
    ax.set_ylabel('Tỷ trọng quyết định (%)')
    ax.set_title('Phân phối quyết định điều hướng qua các Epoch')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '04_action_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ── 5. DASHBOARD TONG HOP ────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Kết quả Huấn luyện AI: Mô hình TFT-DQN', fontsize=22, fontweight='bold', 
                 color=COLORS['text'], y=0.98)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)
    
    # 1. Reward (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(epochs, rewards, alpha=0.15, color=COLORS['secondary'])
    ax1.plot(epochs, rewards, color=COLORS['secondary'], linewidth=2, marker='o', markersize=4)
    if len(rewards) >= 5:
        ma_r = moving_average(rewards, 5)
        ax1.plot(range(3, 3 + len(ma_r)), ma_r, color='#059669', linewidth=2.5, linestyle='--')
    ax1.set_title('Tích lũy Phần thưởng (Reward)')
    ax1.set_xlabel('Epoch')
    ax1.grid(True, linestyle='--')
    
    # 2. Q-Loss (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, dqn_losses, color=COLORS['danger'], alpha=0.4, linewidth=1)
    if len(dqn_losses) >= 5:
        ma = moving_average(dqn_losses, 5)
        ax2.plot(range(3, 3+len(ma)), ma, color=COLORS['danger'], linewidth=2.5)
    ax2.set_title('Suy giảm Hàm mất mát (Q-Loss)')
    ax2.set_xlabel('Epoch')
    ax2.grid(True, linestyle='--')
    
    # 3. Action Distribution (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    bottom_d = np.zeros(num_eps)
    for i in range(3):
        ax3.bar(epochs, action_pct[:, i], bottom=bottom_d, label=server_labels[i],
                color=server_colors[i], alpha=0.85, width=0.8)
        bottom_d += action_pct[:, i]
    ax3.set_title('Phân bổ Lựa chọn Máy chủ')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('%')
    ax3.legend(loc='lower left', fontsize=10, bbox_to_anchor=(0.0, -0.2), ncol=3)
    ax3.grid(True, linestyle='--', axis='y')

    # 4. Epsilon (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(epochs, epsilons, alpha=0.15, color=COLORS['accent'])
    ax4.plot(epochs, epsilons, color=COLORS['accent'], linewidth=2.5)
    ax4.axhline(y=0.5, color=COLORS['warning'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax4.set_title('Hệ số Thăm dò Epsilon')
    ax4.set_xlabel('Epoch')
    ax4.grid(True, linestyle='--')
    
    plt.savefig(os.path.join(CHARTS_DIR, '00_training_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Update ten file cu de overwrite original file neu nguoi dung run lenh thay bi truc trac
    import shutil
    shutil.copy(os.path.join(CHARTS_DIR, '00_training_dashboard.png'), 
              os.path.join(BASE_DIR, 'processed_data', 'training_dashboard.png'))

    print(f"Hoan tat! Xoa cac tu ngu khong chuyen nghiep. Bieu do moi tai: {CHARTS_DIR}")
    print(f"File training_dashboard.png da duoc ghi de kem ban moi.")

if __name__ == '__main__':
    replot_training()
