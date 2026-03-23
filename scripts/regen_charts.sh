#!/bin/bash
# ================================================================
# regen_charts.sh - Tao lai toan bo bieu do tu data co san
# Su dung:
#   ./scripts/regen_charts.sh                (dark mode, tat ca kich ban)
#   ./scripts/regen_charts.sh --light        (light mode, cho in bao cao)
#   ./scripts/regen_charts.sh golden_hour    (chi 1 kich ban)
# ================================================================

BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
BOLD='\033[1m'
NC='\033[0m'

THEME="dark"
TARGET_SCENE=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --light) THEME="light" ;;
        --dark)  THEME="dark" ;;
        *)       TARGET_SCENE="$arg" ;;
    esac
done

clear
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}    NCKH SDN - REGEN ALL CHARTS (Theme: ${THEME})                 ${NC}"
echo -e "${BLUE}=================================================================${NC}\n"

# Kich hoat venv neu co
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null || source venv/bin/activate.fish 2>/dev/null || true
fi

# ================================================================
# BUOC 1: Tao lai bieu do Training (tu training_metrics.json)
# ================================================================
echo -e "${BOLD}[1/2] Bieu do Training...${NC}"
METRICS="ai_model/processed_data/training_metrics.json"

if [ -f "$METRICS" ]; then
    python3 - <<PYEOF
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import dynamic config
sys.path.append(os.getcwd())
from config import BACKENDS, CAPACITIES

THEME = "$THEME"
THEMES = {
    'dark':  {'bg': '#0F172A', 'card_bg': '#1E293B', 'text': '#E2E8F0', 'grid': '#334155'},
    'light': {'bg': '#F8FAFC', 'card_bg': '#FFFFFF',  'text': '#1E293B', 'grid': '#CBD5E1'},
}
COLORS = THEMES[THEME]
CHARTS_DIR = "ai_model/processed_data/charts" if THEME == "dark" else "ai_model/processed_data/charts_light"
os.makedirs(CHARTS_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'], 'axes.facecolor': COLORS['card_bg'],
    'axes.edgecolor': COLORS['grid'], 'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'], 'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'], 'grid.color': COLORS['grid'],
    'grid.alpha': 0.4, 'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11,
})

with open("$METRICS") as f:
    m = json.load(f)

rewards = m.get('episode_rewards', m.get('rewards', []))
q_losses = m.get('dqn_losses', m.get('q_losses', []))
tft_losses = m.get('tft_losses', [])
epsilons = m.get('epsilon_history', m.get('epsilons', []))
action_counts = m.get('action_counts', [])
tft_predictions = m.get('tft_predictions', [])
tft_actuals = m.get('tft_actuals', [])
epochs = list(range(1, max(len(rewards), len(q_losses), 1)+1))

def moving_avg(data, window=5):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

ACCENT = ['#38BDF8','#10B981','#F59E0B','#EF4444','#A78BFA']

# BIEU DO 1: Q-Loss
fig, ax = plt.subplots(figsize=(12, 5))
if q_losses:
    x = list(range(1, len(q_losses)+1))
    ax.plot(x, q_losses, color=ACCENT[0], alpha=0.3, linewidth=0.8)
    sm = moving_avg(q_losses)
    ax.plot(list(range(1, len(sm)+1)), sm, color=ACCENT[0], linewidth=2, label='Q-Loss (smoothed)')
if tft_losses:
    x = list(range(1, len(tft_losses)+1))
    ax.plot(x, tft_losses, color=ACCENT[2], alpha=0.3, linewidth=0.8)
    sm = moving_avg(tft_losses)
    ax.plot(list(range(1, len(sm)+1)), sm, color=ACCENT[2], linewidth=2, label='TFT-Loss (smoothed)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Q-Loss & TFT Loss Curve')
if ax.get_legend_handles_labels()[1]: ax.legend(); ax.grid(True, linestyle='--')
plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/01_loss_curve.png", dpi=200); plt.close()
print("  01_loss_curve.png")

# BIEU DO 2: Reward
fig, ax = plt.subplots(figsize=(12, 5))
if rewards:
    ax.fill_between(epochs, rewards, alpha=0.15, color=ACCENT[1])
    ax.plot(epochs, rewards, color=ACCENT[1], alpha=0.4, linewidth=0.8)
    sm = moving_avg(rewards)
    ax.plot(list(range(len(sm))), sm, color=ACCENT[1], linewidth=2, label='Total Reward')
ax.axhline(max(rewards)*0.9 if rewards else 0, color=ACCENT[2], linestyle='--', linewidth=1, alpha=0.6)
ax.set_xlabel('Epoch'); ax.set_ylabel('Reward'); ax.set_title('Total Reward per Epoch')
if ax.get_legend_handles_labels()[1]: ax.legend(); ax.grid(True, linestyle='--')
plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/02_reward_curve.png", dpi=200); plt.close()
print("  02_reward_curve.png")

# BIEU DO 3: TFT Prediction vs Actual
if tft_predictions and tft_actuals:
    preds_arr = np.array(tft_predictions)
    actuals_arr = np.array(tft_actuals)
    n_feat = preds_arr.shape[1] if preds_arr.ndim > 1 else 1
    
    fig, ax = plt.subplots(figsize=(12, 5))
    n_show = min(100, len(preds_arr))
    # Plot top features (Byte Rate or first feature)
    ax.plot(range(n_show), actuals_arr[:n_show, 0], color=ACCENT[0], linewidth=1.5, label='Actual', linestyle='--')
    ax.plot(range(n_show), preds_arr[:n_show, 0], color=ACCENT[1], linewidth=1.5, label='TFT Predicted')
    ax.set_xlabel('Step'); ax.set_ylabel('Normalized Value')
    ax.set_title('TFT Prediction vs Actual Traffic (Primary Feature)')
    ax.grid(True, linestyle='--')
    ax.legend()
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/03_tft_prediction.png", dpi=200); plt.close()
    print("  03_tft_prediction.png")
else:
    print("  03_tft_prediction.png (Skipped - No data)")

# BIEU DO 4: Epsilon Decay
fig, ax = plt.subplots(figsize=(12, 5))
if epsilons:
    ax.plot(epochs, epsilons, color=ACCENT[3], linewidth=2)
    ax.fill_between(epochs, epsilons, alpha=0.15, color=ACCENT[3])
ax.set_xlabel('Epoch'); ax.set_ylabel('Epsilon')
ax.set_title('Epsilon Decay — Exploration to Exploitation'); ax.grid(True, linestyle='--')
plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/04_epsilon_decay.png", dpi=200); plt.close()
print("  04_epsilon_decay.png")

# BIEU DO 5: Action Distribution
if action_counts:
    action_arr = np.array(action_counts)
    n_epochs = len(action_arr)
    n_servers = action_arr.shape[1]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    server_names = [b['name'] for b in BACKENDS]
    colors_srv = [ACCENT[i % len(ACCENT)] for i in range(n_servers)]
    
    x = list(range(n_epochs))
    bottoms = [0] * n_epochs
    for s in range(n_servers):
        vals = action_arr[:, s]
        label = server_names[s] if s < len(server_names) else f'Server {s+1}'
        ax.bar(x, vals, bottom=bottoms, color=colors_srv[s], label=label, alpha=0.85)
        bottoms = [bottoms[i] + vals[i] for i in range(n_epochs)]
    
    ax.set_xlabel('Epoch'); ax.set_ylabel('Action Count')
    ax.set_title('Action Distribution (Server Selection per Epoch)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', axis='y')
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/05_action_distribution.png", dpi=200); plt.close()
    print("  05_action_distribution.png")

# DASHBOARD
fig = plt.figure(figsize=(20, 12))
fig.suptitle('NCKH SDN: Training Dashboard', fontsize=16, fontweight='bold', color=COLORS['text'], y=0.98)
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
axes_list = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

data_map = [
    (q_losses, 'Q-Loss', ACCENT[0]), (rewards, 'Reward', ACCENT[1]),
    (epsilons, 'Epsilon', ACCENT[3]), (tft_losses, 'TFT Loss', ACCENT[2]),
]

for i, (data, label, color) in enumerate(data_map):
    ax = axes_list[i]
    if data:
        ax.plot(range(len(data)), data, color=color, linewidth=1.5)
        ax.fill_between(range(len(data)), data, alpha=0.1, color=color)
    ax.set_title(label); ax.grid(True, linestyle='--')

# Dashboard Action Dist
ax_dist = axes_list[4]
if action_counts:
    action_arr = np.array(action_counts)
    bottoms = np.zeros(len(action_arr))
    for s in range(action_arr.shape[1]):
        ax_dist.bar(range(len(action_arr)), action_arr[:, s], bottom=bottoms, alpha=0.8)
        bottoms += action_arr[:, s]
    ax_dist.set_title("Action Mapping")

plt.savefig(f"{CHARTS_DIR}/00_training_dashboard.png", dpi=200, bbox_inches='tight'); plt.close()
print("  00_training_dashboard.png")
print(f"  Done! -> {CHARTS_DIR}/")
PYEOF

else
    echo -e "  ${YELLOW}training_metrics.json chua co. Chay train.sh truoc de co du lieu.${NC}"
fi

# ================================================================
# BUOC 2: Tao lai bieu do So sanh
# ================================================================
echo -e "\n${BOLD}[2/2] Bieu do So sanh...${NC}"

SCENARIOS=("golden_hour" "video_conference" "hardware_degradation" "low_rate_dos")

for scene in "${SCENARIOS[@]}"; do
    # Chi xu ly kich ban duoc chi dinh (hoac tat ca neu khong co arg)
    if [ -n "$TARGET_SCENE" ] && [ "$TARGET_SCENE" != "$scene" ]; then
        continue
    fi

    # Kiem tra xem co it nhat 1 thu muc ket qua cho kich ban nay khong
    COUNT=$(ls -d stats/results/*_"${scene}" 2>/dev/null | wc -l)
    if [ "$COUNT" -eq 0 ]; then
        echo -e "  ${YELLOW}Khong co data cho: ${scene} (bo qua)${NC}"
        continue
    fi

    echo -e "  ${CYAN}Dang ve: ${scene} (${COUNT} thuat toan)...${NC}"
    python3 ai_model/generate_comparison_charts.py --scenario "$scene" --theme "$THEME" 2>&1 | grep -E "^\s+\[|Hoan tat|SKIPPED"
done

echo ""
echo -e "${GREEN}=================================================================${NC}"
echo -e "${GREEN}  HOAN TAT! Bieu do da tao xong.                                 ${NC}"
echo -e "${GREEN}=================================================================${NC}"

if [ "$THEME" == "light" ]; then
    echo -e "${CYAN}Training charts (light): ai_model/processed_data/charts_light/${NC}"
    echo -e "${CYAN}Compare charts (light):  stats/results/charts_light/${NC}"
else
    echo -e "${CYAN}Training charts:  ai_model/processed_data/charts/${NC}"
    echo -e "${CYAN}Compare charts:   stats/results/charts/${NC}"
fi
echo ""
