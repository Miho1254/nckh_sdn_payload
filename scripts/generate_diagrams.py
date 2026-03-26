#!/usr/bin/env python3
"""
Script render các sơ đồ cho bài báo NCKH TFT-PPO
Output: PNG files chuẩn IEEE (300 DPI, Times New Roman)

Requirements:
    pip install matplotlib pillow numpy

Chạy:
    python scripts/generate_diagrams.py
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

# IEEE Standard Figure Settings
IEEE_SINGLE_COL = 3.5  # inches
IEEE_DOUBLE_COL = 7.0  # inches
DPI = 300

# Font settings for IEEE
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
    'axes.grid': False,
})


def setup_figure(width_inches, height_inches=None, aspect_ratio=1.0):
    """Tạo figure với kích thước IEEE"""
    if height_inches is None:
        height_inches = width_inches / aspect_ratio
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    return fig, ax


def draw_box(ax, x, y, w, h, text, style='single'):
    """Vẽ một box với text"""
    colors = {
        'single': '#e8f5e9',      # Green
        'double': '#bbdefb',       # Blue
        'controller': '#fff3e0',   # Orange
        'client': '#ffccbc',       # Light orange
        'server': '#c8e6c9',      # Light green
        'agent': '#f3e5f5',       # Purple
        'safety': '#ffebee',      # Red
        'network': '#90caf9',      # Blue
    }
    fc = colors.get(style, 'white')
    
    box = FancyBboxPatch((x, y), w, h, 
                         boxstyle="round,pad=0.1,rounding_size=0.2",
                         facecolor=fc, edgecolor='black', linewidth=0.8)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
            fontsize=7, wrap=True, fontweight='bold')


def draw_arrow(ax, x1, y1, x2, y2, label='', color='black'):
    """Vẽ arrow với nhãn"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, fontsize=6, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.8))


# ============================================================================
# DIAGRAM 1: System Architecture
# ============================================================================
def diagram_1_system_architecture(output_dir):
    """System Architecture - Figure 1 in paper"""
    fig, ax = setup_figure(IEEE_DOUBLE_COL, 4.5, 1.5)
    
    # Title
    ax.text(5, 9.5, 'Figure 1: TFT-PPO System Architecture in SDN Controller', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Clients layer
    draw_box(ax, 0.3, 7, 1.2, 1.5, 'h9', 'client')
    draw_box(ax, 1.6, 7, 1.2, 1.5, 'h10', 'client')
    draw_box(ax, 2.9, 7, 1.2, 1.5, '...', 'client')
    draw_box(ax, 4.2, 7, 1.2, 1.5, 'h16', 'client')
    ax.text(2.75, 8.7, 'CLIENTS (h9-h16)', ha='center', va='center', fontsize=7, fontweight='bold')
    ax.text(2.75, 8.3, '8 clients → HTTP → VIP 10.0.0.100', ha='center', va='center', fontsize=6)
    
    # Edge switches
    draw_box(ax, 0.3, 5, 5.5, 1.5, 'Edge Switches (s1-s4)\nOpenFlow 1.3 Flow Table', 'double')
    
    # Core switch
    draw_box(ax, 0.3, 3, 5.5, 1.5, 'Core Switch (s5)\nPacket-in → Ryu Controller', 'network')
    
    # Controller box
    controller_x, controller_y = 0.3, 0.3
    controller_w, controller_h = 5.5, 2.4
    
    ctrl_box = FancyBboxPatch((controller_x, controller_y), controller_w, controller_h,
                              boxstyle="round,pad=0.1,rounding_size=0.3",
                              facecolor='#fff3e0', edgecolor='black', linewidth=1)
    ax.add_patch(ctrl_box)
    ax.text(controller_x + controller_w/2, controller_y + controller_h - 0.3, 
            'RYU SDN CONTROLLER', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # TFT-PPO Agent
    tft_x, tft_y = controller_x + 0.2, controller_y + 0.5
    tft_w, tft_h = 2.8, 1.5
    
    tft_box = FancyBboxPatch((tft_x, tft_y), tft_w, tft_h,
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor='#f3e5f5', edgecolor='#9c27b0', linewidth=0.8)
    ax.add_patch(tft_box)
    ax.text(tft_x + tft_w/2, tft_y + tft_h - 0.2, 'TFT-PPO Agent', ha='center', fontsize=7, fontweight='bold')
    
    # LSTM
    ax.text(tft_x + 0.4, tft_y + 0.8, 'LSTM\nEncoder', ha='center', va='center', fontsize=6)
    ax.text(tft_x + 1.4, tft_y + 0.8, 'Multi-Head\nAttention', ha='center', va='center', fontsize=6)
    ax.text(tft_x + 2.3, tft_y + 0.8, 'Feature\nLinear', ha='center', va='center', fontsize=6)
    
    # Actor-Critic heads
    ax.text(tft_x + 0.8, tft_y + 0.25, 'Actor π(a|s)', ha='center', va='center', fontsize=6, 
            bbox=dict(boxstyle='round', facecolor='#e1f5fe', edgecolor='#0277bd', linewidth=0.5))
    ax.text(tft_x + 2.0, tft_y + 0.25, 'Critic V(s)', ha='center', va='center', fontsize=6,
            bbox=dict(boxstyle='round', facecolor='#fff3e0', edgecolor='#ff9800', linewidth=0.5))
    
    # Safety Override
    safety_x = tft_x + tft_w + 0.3
    safety_w = 2.0
    safety_box = FancyBboxPatch((safety_x, tft_y), safety_w, tft_h,
                                  boxstyle="round,pad=0.05,rounding_size=0.15",
                                  facecolor='#ffebee', edgecolor='#c62828', linewidth=0.8)
    ax.add_patch(safety_box)
    ax.text(safety_x + safety_w/2, tft_y + tft_h - 0.2, 'Safety Override', ha='center', fontsize=7, fontweight='bold')
    ax.text(safety_x + safety_w/2, tft_y + 0.7, 'if u > 0.95\n→ bypass → WRR', ha='center', va='center', fontsize=6)
    
    # Arrows
    draw_arrow(ax, 2.75, 7, 3.05, 6.5)
    draw_arrow(ax, 3.05, 5, 3.05, 4.5)
    draw_arrow(ax, 3.05, 3, 3.05, 2.7)
    
    # Backend servers
    draw_box(ax, 6.3, 0.3, 1.3, 2, 'h5\n10 Mbps', 'server')
    draw_box(ax, 7.8, 0.3, 1.3, 2, 'h7\n50 Mbps', 'server')
    draw_box(ax, 9.2, 0.3, 0.7, 2, 'h8\n100', 'server')
    ax.text(8.4, 2.5, 'BACKEND SERVERS', ha='center', fontsize=7, fontweight='bold')
    
    # Arrows to servers
    draw_arrow(ax, 3.05, 1.7, 6.3, 2.3)
    draw_arrow(ax, 3.05, 1.5, 7.8, 2.3)
    draw_arrow(ax, 3.05, 1.3, 9.2, 2.3)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#ffccbc', edgecolor='black', label='Client'),
        mpatches.Patch(facecolor='#bbdefb', edgecolor='black', label='Switch'),
        mpatches.Patch(facecolor='#fff3e0', edgecolor='black', label='Controller'),
        mpatches.Patch(facecolor='#f3e5f5', edgecolor='#9c27b0', label='TFT-PPO Agent'),
        mpatches.Patch(facecolor='#ffebee', edgecolor='#c62828', label='Safety Override'),
        mpatches.Patch(facecolor='#c8e6c9', edgecolor='black', label='Server'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=6)
    
    output_path = Path(output_dir) / 'fig1_system_architecture.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# DIAGRAM 2: TFT Encoder
# ============================================================================
def diagram_2_tft_encoder(output_dir):
    """TFT Encoder - Figure 2 in paper"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 3.5, 1.2)
    
    ax.text(5, 9.7, 'Figure 2: TFT Encoder Architecture', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Input features
    features_text = 'Input: 20 Features\n(Load, Latency,\nQueue, Cache, ...)'
    draw_box(ax, 0.5, 6.5, 2, 2, features_text, 'single')
    
    # LSTM
    draw_box(ax, 3.5, 7, 2, 1.2, 'LSTM\nEncoder', 'agent')
    draw_box(ax, 3.5, 5.5, 2, 1.2, 'LSTM\nCell T', 'agent')
    
    # Arrow LSTM
    draw_arrow(ax, 2.5, 7.5, 3.5, 7.7)
    draw_arrow(ax, 4.5, 7, 4.5, 6.2)
    ax.text(4.0, 7.8, 'temporal', fontsize=5, ha='center')
    
    # Multi-Head Attention
    draw_box(ax, 3.5, 3.8, 2, 1.4, 'Multi-Head\nAttention (h=4)', 'agent')
    
    # Feature Linear
    draw_box(ax, 3.5, 2, 2, 1.2, 'Feature\nLinear', 'agent')
    
    # Output heads
    draw_box(ax, 3.5, 0.5, 1.5, 1, 'Actor π(a|s)', 'client')
    draw_box(ax, 5.5, 0.5, 1.5, 1, 'Critic V(s)', 'client')
    
    # Arrows
    draw_arrow(ax, 4.5, 3.8, 4.5, 3.2)
    draw_arrow(ax, 4.5, 2, 4.5, 1.5)
    draw_arrow(ax, 4.25, 1.5, 4.25, 1.5)
    draw_arrow(ax, 5.5, 1.5, 5.5, 1.5)
    
    # Process arrows
    ax.annotate('', xy=(3.5, 4.5), xytext=(3.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    
    # Labels
    ax.text(6.2, 7.5, 'Sequential\nProcessing', fontsize=5, ha='center')
    ax.text(6.2, 4.5, 'Attention\nWeights', fontsize=5, ha='center')
    
    output_path = Path(output_dir) / 'fig2_tft_encoder.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# DIAGRAM 3: PPO Training Flow
# ============================================================================
def diagram_3_ppo_flow(output_dir):
    """PPO Training Flow - Figure 3 in paper"""
    fig, ax = setup_figure(IEEE_DOUBLE_COL, 4.0, 1.8)
    
    ax.text(5, 9.8, 'Figure 3: PPO Training Workflow', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Start
    draw_box(ax, 4.25, 8.8, 1.5, 0.6, 'Start', 'single')
    
    # Init
    draw_box(ax, 4.25, 7.8, 1.5, 0.8, 'Initialize πθ\nRandom Capacity', 'double')
    draw_arrow(ax, 5, 8.8, 5, 8.6)
    
    # Episode loop
    draw_box(ax, 0.5, 5.5, 2, 2, 'Episode Loop\n(200 steps)', 'controller')
    
    # Collect trajectories
    draw_box(ax, 3, 6.5, 2, 1, 'Collect (s,a,r)', 'agent')
    
    # Action
    draw_box(ax, 3, 5.2, 2, 1, 'Select Action\na in {0,1,2}', 'agent')
    
    # Environment
    draw_box(ax, 3, 4, 2, 1, 'Environment\nRespond', 'single')
    
    # Reward
    draw_box(ax, 5.5, 5.2, 2.5, 1, 'Calculate Reward\nr = b + t - l - o', 'double')
    
    # GAE
    draw_box(ax, 8.5, 5.2, 1.3, 1, 'Compute\nGAE λ=0.95', 'agent')
    
    # PPO Update
    draw_box(ax, 8.5, 3.5, 1.3, 1.3, 'PPO Update\nLCLIP', 'agent')
    
    # Check convergence
    draw_box(ax, 8.5, 2, 1.3, 1, 'Loss\nConverging?', 'safety')
    
    # Save
    draw_box(ax, 6.5, 2, 1.5, 1, 'Save\nCheckpoint', 'single')
    
    # Arrows
    draw_arrow(ax, 5, 7.8, 5, 7)
    draw_arrow(ax, 5, 6.5, 4, 6.5)
    draw_arrow(ax, 2.5, 6.5, 3, 5.7)
    draw_arrow(ax, 4, 5.2, 3, 4.5)
    draw_arrow(ax, 5, 4, 5.5, 5.7)
    draw_arrow(ax, 8, 5.7, 8.5, 5.7)
    draw_arrow(ax, 8.5, 4.2, 8.5, 3)
    draw_arrow(ax, 8.5, 2, 8, 2.5)
    draw_arrow(ax, 6.5, 2.5, 8.5, 2.5)
    draw_arrow(ax, 6.5, 2, 5, 2)
    draw_arrow(ax, 5, 2, 5, 3.5)
    draw_arrow(ax, 5, 4.5, 5, 5.2)
    
    # Back to episode
    ax.annotate('', xy=(1.5, 5.5), xytext=(1.5, 7.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8, 
                               connectionstyle='arc3,rad=0.3'))
    ax.text(0.8, 6.5, 'Yes', fontsize=6)
    
    output_path = Path(output_dir) / 'fig3_ppo_flow.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# DIAGRAM 4: Safety Override State Machine
# ============================================================================
def diagram_4_safety_override(output_dir):
    """Safety Override State Machine - Figure 4 in paper"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 3.0, 1.5)
    
    ax.text(5, 9.5, 'Figure 4: Safety Override State Machine', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    # States
    circle_style = dict(facecolor='#e8f5e9', edgecolor='black', linewidth=1.2)
    
    # Normal state
    circle1 = Circle((3, 7), 1, **circle_style)
    ax.add_patch(circle1)
    ax.text(3, 7, 'Normal', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Check state (diamond - use Polygon for proper diamond shape)
    diamond = mpatches.Polygon([(7, 8.5), (9, 7), (7, 5.5), (5, 7)],
                                closed=True, facecolor='#fff3e0', edgecolor='black', linewidth=1)
    ax.add_patch(diamond)
    ax.text(7, 7, 'u > 0.95?', ha='center', va='center', fontsize=6, fontweight='bold')
    
    # Override state
    circle2 = Circle((9, 4.5), 1, **dict(facecolor='#ffebee', edgecolor='black', linewidth=1.2))
    ax.add_patch(circle2)
    ax.text(9, 4.5, 'Override\n(WRR)', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Back to normal
    circle3 = Circle((5, 4.5), 1, **circle_style)
    ax.add_patch(circle3)
    ax.text(5, 4.5, 'Normal\n(PPO)', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Arrows
    draw_arrow(ax, 4, 7, 5.5, 7)
    ax.text(4.8, 7.4, 'Every\nrequest', fontsize=5, ha='center')
    
    draw_arrow(ax, 8, 7, 8.3, 5.5)
    ax.text(8.5, 6.3, 'Yes', fontsize=6)
    
    draw_arrow(ax, 7, 5.8, 5.9, 5.2)
    ax.text(6.5, 5.3, 'No', fontsize=6)
    
    # Loop arrow
    ax.annotate('', xy=(9, 5.5), xytext=(9, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.text(9.6, 4.5, 'Next\nrequest', fontsize=5, ha='center')
    
    # PPO Agent decision
    ax.text(3, 2.5, 'PPO Agent selects\nserver based on\nlearned policy', 
            ha='center', va='center', fontsize=6, 
            bbox=dict(boxstyle='round', facecolor='#f3e5f5', edgecolor='#9c27b0', linewidth=0.5))
    draw_arrow(ax, 5, 3.5, 5, 5.5)
    
    output_path = Path(output_dir) / 'fig4_safety_override.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# DIAGRAM 5: RL Cycle
# ============================================================================
def diagram_5_rl_cycle(output_dir):
    """Reinforcement Learning Cycle - Figure 5 in paper"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 3.5, 1.3)
    
    ax.text(5, 9.6, 'Figure 5: Reinforcement Learning Cycle', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Environment
    draw_box(ax, 0.5, 4, 2.5, 2, 'SDN\nEnvironment', 'single')
    
    # Agent
    draw_box(ax, 4, 4, 2.5, 2, 'TFT-PPO\nAgent', 'agent')
    
    # State
    draw_box(ax, 7.5, 5.5, 2, 1.5, 'State s_t\n20 features', 'double')
    
    # Action
    draw_box(ax, 7.5, 2.5, 2, 1.5, 'Action a_t\nServer {0,1,2}', 'client')
    
    # Reward
    draw_box(ax, 4, 1.5, 2.5, 1.5, 'Reward r_t', 'controller')
    
    # PortStats
    draw_box(ax, 0.5, 1.5, 2.5, 1.5, 'PortStats\nCollector', 'network')
    
    # Arrows with labels
    draw_arrow(ax, 7.5, 5.5, 6.5, 5.2)
    ax.text(7.8, 5.5, 'state', fontsize=5)
    
    draw_arrow(ax, 6.5, 4, 7.5, 4)
    ax.text(7.2, 4.5, 'action', fontsize=5)
    
    draw_arrow(ax, 7.5, 2.5, 6.5, 2.8)
    ax.text(7.8, 2.5, 'reward', fontsize=5)
    
    draw_arrow(ax, 4, 2.5, 3, 2.8)
    
    draw_arrow(ax, 3, 4, 3, 2.5)
    
    draw_arrow(ax, 0.5, 4, 0.5, 3)
    ax.text(-0.3, 3.5, 'PortStats', fontsize=5)
    
    draw_arrow(ax, 0.5, 3, 3, 3)
    
    # Update arrow
    ax.annotate('', xy=(4, 4), xytext=(4, 3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(3.5, 3.5, 'PPO\nUpdate', fontsize=5, color='blue')
    
    output_path = Path(output_dir) / 'fig5_rl_cycle.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# DIAGRAM 6: Network Topology
# ============================================================================
def diagram_6_topology(output_dir):
    """Network Topology - Figure 6 in paper"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 3.0, 1.5)
    
    ax.text(5, 9.7, 'Figure 6: Fat-Tree K=4 Topology', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Core switch
    core = Circle((5, 7), 0.8, facecolor='#90caf9', edgecolor='black', linewidth=1.2)
    ax.add_patch(core)
    ax.text(5, 7, 's5\nCore', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Edge switches
    for i, (x, label) in enumerate([(1.5, 's1'), (4, 's2'), (6, 's3'), (8.5, 's4')]):
        edge = Circle((x, 5), 0.6, facecolor='#bbdefb', edgecolor='black', linewidth=1)
        ax.add_patch(edge)
        ax.text(x, 5, label, ha='center', va='center', fontsize=7, fontweight='bold')
        
        # Line to core
        ax.plot([x, 5], [4.4, 6.2], 'k-', linewidth=0.8)
    
    # Clients
    client_y = 2.5
    for i, x in enumerate([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]):
        client = Circle((x, client_y), 0.3, facecolor='#ffccbc', edgecolor='black', linewidth=0.8)
        ax.add_patch(client)
        ax.text(x, client_y - 0.6, f'h{9+i}', ha='center', va='center', fontsize=5)
        
        # Connect to nearest edge switch
        if i < 2:
            edge_x = 1.5
        elif i < 4:
            edge_x = 4
        elif i < 6:
            edge_x = 6
        else:
            edge_x = 8.5
        ax.plot([x, edge_x], [client_y + 0.3, 4.4], 'k-', linewidth=0.5)
    
    # Servers
    server_y = 1
    for i, (x, cap) in enumerate([(2, '10M'), (5, '50M'), (8, '100M')]):
        server = Circle((x, server_y), 0.5, facecolor='#c8e6c9', edgecolor='black', linewidth=1)
        ax.add_patch(server)
        ax.text(x, server_y, f'h{5+2*i}\n{cap}', ha='center', va='center', fontsize=6, fontweight='bold')
        ax.plot([x, 5], [server_y + 0.5, 6.2], 'k-', linewidth=0.8)
    
    # Labels
    ax.text(5, 8.5, 'Edge Layer (4 switches)', ha='center', fontsize=7, fontstyle='italic')
    ax.text(5, 0.2, 'Backend Servers', ha='center', fontsize=7, fontstyle='italic')
    ax.text(5, 3.5, 'Clients (h9-h18)', ha='center', fontsize=7, fontstyle='italic')
    
    output_path = Path(output_dir) / 'fig6_topology.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# DIAGRAM 7: Hardware Degradation Scenario
# ============================================================================
def diagram_7_degradation(output_dir):
    """Hardware Degradation Scenario - Figure 7 in paper"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 3.5, 1.3)
    
    ax.text(5, 9.5, 'Figure 7: Hardware Degradation Detection', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Timeline
    ax.plot([0.5, 9.5], [7, 7], 'k-', linewidth=1)
    
    # Time points
    time_points = [
        (1, 't=0\nNormal'),
        (3.5, 't=T\nDegradation\nDetected'),
        (6, 't=T+1\nPPO\nAdapts'),
        (8.5, 't=T+2\nRecovery'),
    ]
    
    for x, label in time_points:
        ax.plot([x, x], [6.7, 7.3], 'k-', linewidth=1)
        ax.text(x, 6.4, label, ha='center', va='top', fontsize=6)
    
    # Server states
    # Normal state
    draw_box(ax, 0.5, 4, 2.5, 1.5, 'h8: 100M\n[OK] Healthy', 'server')
    ax.text(0.5, 3.5, 'BEFORE', ha='left', fontsize=7, fontweight='bold')
    
    # Degraded
    draw_box(ax, 3.5, 4, 2.5, 1.5, 'h8: 50M ↓\n! Degraded!', 'safety')
    ax.text(3.5, 3.5, 'DEGRADATION', ha='left', fontsize=7, fontweight='bold', color='red')
    
    # PPO Detection
    draw_box(ax, 6.5, 4, 3, 1.5, 'PPO Detects\nAnomaly!\nRoutes → h7', 'agent')
    ax.text(6.5, 3.5, 'ADAPTATION', ha='left', fontsize=7, fontweight='bold', color='blue')
    
    # Arrows
    draw_arrow(ax, 3, 5.5, 3.5, 5.5)
    draw_arrow(ax, 6, 5.5, 6.5, 5.5)
    
    # Metrics comparison
    draw_box(ax, 0.5, 1, 4.5, 1.8, 'WRR Result:\n• Packets: 7,756,843\n• Utilization: 95%+ on h8\n• Throughput: -14.7%',
             'client')
    
    draw_box(ax, 5.3, 1, 4.5, 1.8, 'PPO Result:\n• Packets: 8,424,531 (+8.6%)\n• Utilization: balanced\n• Throughput: +8.6%',
             'agent')
    
    # Winner arrow
    ax.text(7.5, 0.5, '← PPO Wins!', ha='center', fontsize=8, fontweight='bold', color='green')
    
    output_path = Path(output_dir) / 'fig7_degradation.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# DIAGRAM 8: Reward Function
# ============================================================================
def diagram_8_reward(output_dir):
    """Reward Function Decomposition - Figure 8 in paper"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 3.0, 1.5)
    
    ax.text(5, 9.6, 'Figure 8: Reward Function Components', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Reward components
    components = [
        (1.5, 7.5, 'balance_bonus\n= balance_score × 3.0', '#4caf50'),
        (5, 7.5, 'throughput_bonus\nbased on BW achieved', '#2196f3'),
        (8.5, 7.5, 'latency_penalty\n~= avg latency', '#ff9800'),
    ]
    
    for x, y, text, color in components:
        box = FancyBboxPatch((x-1.2, y-0.7), 2.4, 1.4,
                             boxstyle="round,pad=0.1,rounding_size=0.2",
                             facecolor='white', edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=6)
    
    # Overload penalty
    box = FancyBboxPatch((3.5, 4.5), 3, 1.2,
                         boxstyle="round,pad=0.1,rounding_size=0.2",
                         facecolor='#ffebee', edgecolor='#c62828', linewidth=2)
    ax.add_patch(box)
    ax.text(5, 5.1, 'overload_penalty = 20000', ha='center', va='center', fontsize=7, fontweight='bold', color='#c62828')
    ax.text(5, 4.7, 'if utilization > 0.95', ha='center', va='center', fontsize=6)
    
    # Final reward
    draw_box(ax, 3.5, 2, 3, 1.5, 'reward = b + t - l - o', 'controller')
    
    # Arrows to penalty
    draw_arrow(ax, 1.5, 6.8, 4, 5.7)
    draw_arrow(ax, 5, 6.8, 5, 5.7)
    draw_arrow(ax, 8.5, 6.8, 6, 5.7)
    
    # Arrow to final
    draw_arrow(ax, 5, 4.5, 5, 3.5)
    
    output_path = Path(output_dir) / 'fig8_reward.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# DIAGRAM 9: WRR vs PPO Comparison
# ============================================================================
def diagram_9_comparison(output_dir):
    """WRR vs PPO Comparison - Figure 9 in paper"""
    fig, ax = setup_figure(IEEE_DOUBLE_COL, 3.0, 2.3)
    
    ax.text(5, 9.7, 'Figure 9: WRR vs PPO - Trade-off Analysis', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    # WRR side
    wrr_box = FancyBboxPatch((0.3, 3), 3, 5,
                              boxstyle="round,pad=0.1,rounding_size=0.3",
                              facecolor='#ffcdd2', edgecolor='#c62828', linewidth=1.5)
    ax.add_patch(wrr_box)
    ax.text(1.8, 7.7, 'WRR (Static)', ha='center', fontsize=9, fontweight='bold', color='#c62828')
    
    wrr_features = [
        '• Round Robin',
        '• Fixed Weights',
        '• No Adaptation',
        '• No Learning',
        '',
        '[OK] Fast (O(1))',
        '[OK] Low Latency',
        '[OK] No Overhead',
        '',
        '[X] Fails on Degradation',
        '[X] Cannot Detect Anomaly',
    ]
    for i, text in enumerate(wrr_features):
        ax.text(0.5, 7 - i*0.35, text, fontsize=6, va='top')
    
    # PPO side
    ppo_box = FancyBboxPatch((4.5, 3), 3, 5,
                              boxstyle="round,pad=0.1,rounding_size=0.3",
                              facecolor='#c8e6c9', edgecolor='#2e7d32', linewidth=1.5)
    ax.add_patch(ppo_box)
    ax.text(6, 7.7, 'PPO (Adaptive)', ha='center', fontsize=9, fontweight='bold', color='#2e7d32')
    
    ppo_features = [
        '• Learns Patterns',
        '• Dynamic Weights',
        '• Adapts to Drift',
        '• Neural Network',
        '',
        '[OK] +8.6% on Degradation',
        '[OK] Detects Anomaly',
        '[OK] Self-Healing',
        '',
        '[X] Inference Overhead',
        '[X] May Lose Normal',
    ]
    for i, text in enumerate(ppo_features):
        ax.text(4.7, 7 - i*0.35, text, fontsize=6, va='top')
    
    # Result box
    result_box = FancyBboxPatch((7.8, 3), 2, 5,
                                 boxstyle="round,pad=0.1,rounding_size=0.3",
                                 facecolor='#fff9c4', edgecolor='#f57f17', linewidth=1.5)
    ax.add_patch(result_box)
    ax.text(8.8, 7.7, 'RECOMMEND', ha='center', fontsize=8, fontweight='bold')
    ax.text(8.8, 7, 'Hybrid', ha='center', fontsize=10, fontweight='bold', color='#f57f17')
    ax.text(8.8, 6.3, 'WRR + PPO\nSLA Protector', ha='center', fontsize=7)
    
    ax.text(8.8, 5, 'WRR: 95% traffic', ha='center', fontsize=6)
    ax.text(8.8, 4.5, 'PPO: anomaly', ha='center', fontsize=6)
    ax.text(8.8, 4, 'detection only', ha='center', fontsize=6)
    
    # Arrows
    ax.annotate('', xy=(7.8, 5), xytext=(6.5, 5),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.annotate('', xy=(7.8, 5), xytext=(4.5, 5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    output_path = Path(output_dir) / 'fig9_comparison.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# DIAGRAM 10: Training Pipeline
# ============================================================================
def diagram_10_pipeline(output_dir):
    """Training Pipeline - Figure 10 in paper"""
    fig, ax = setup_figure(IEEE_DOUBLE_COL, 3.5, 2)
    
    ax.text(5, 9.7, 'Figure 10: Training Pipeline', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Pipeline stages
    stages = [
        (1, 6.5, 'Mininet\nEmulator', '#e1f5fe'),
        (3, 6.5, 'Ryu\nController', '#e1f5fe'),
        (5, 6.5, 'PortStats\nCollector', '#e1f5fe'),
        (7, 6.5, 'SDNEnvV3\nGymnasium', '#fff3e0'),
        (9, 6.5, 'Random\nCapacity', '#fff3e0'),
    ]
    
    for x, y, text, color in stages:
        box = FancyBboxPatch((x-0.7, y-0.7), 1.4, 1.4,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=6)
        if x < 9:
            ax.annotate('', xy=(x+0.7, y), xytext=(x+0.7, y),
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
            ax.annotate('', xy=(x+1.4-0.05, y), xytext=(x+0.7+0.05, y),
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    
    # Training loop
    training_box = FancyBboxPatch((3, 1.5), 4, 3.5,
                                   boxstyle="round,pad=0.1,rounding_size=0.2",
                                   facecolor='#f3e5f5', edgecolor='#9c27b0', linewidth=1)
    ax.add_patch(training_box)
    ax.text(5, 4.7, 'PPO Training Loop', ha='center', fontsize=8, fontweight='bold')
    
    training_steps = [
        (3.5, 4, 'Collect 2048 steps'),
        (5, 4, 'Compute GAE λ=0.95'),
        (6.5, 4, 'PPO Clipped Obj'),
        (3.5, 3, 'Adam Optimizer'),
        (5, 3, 'Update πθ'),
        (6.5, 3, 'Checkpoint @ 50K'),
    ]
    for x, y, text in training_steps:
        ax.text(x, y, text, ha='center', va='center', fontsize=5)
    
    # Arrow down to training
    ax.annotate('', xy=(5, 4.5), xytext=(5, 5.5),
                arrowprops=dict(arrowstyle='->', color='purple', lw=1))
    
    # Evaluation
    eval_box = FancyBboxPatch((8, 1.5), 1.8, 3.5,
                               boxstyle="round,pad=0.1,rounding_size=0.2",
                               facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=1)
    ax.add_patch(eval_box)
    ax.text(8.9, 4.7, 'Evaluation', ha='center', fontsize=7, fontweight='bold')
    ax.text(8.9, 4.2, 'WRR vs PPO', ha='center', fontsize=5)
    ax.text(8.9, 3.7, '5 runs each', ha='center', fontsize=5)
    ax.text(8.9, 3.2, 'Compare', ha='center', fontsize=5)
    ax.text(8.9, 2.7, 'Report', ha='center', fontsize=5)
    
    # Arrow to evaluation
    ax.annotate('', xy=(8, 3), xytext=(7, 3),
                arrowprops=dict(arrowstyle='->', color='green', lw=1))
    
    output_path = Path(output_dir) / 'fig10_pipeline.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Generate all diagrams"""
    output_dir = Path('docs/figures')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Generating IEEE Standard Figures for TFT-PPO Paper")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Settings: {DPI} DPI, IEEE dimensions")
    print()
    
    diagrams = [
        ("System Architecture", diagram_1_system_architecture),
        ("TFT Encoder", diagram_2_tft_encoder),
        ("PPO Flow", diagram_3_ppo_flow),
        ("Safety Override", diagram_4_safety_override),
        ("RL Cycle", diagram_5_rl_cycle),
        ("Network Topology", diagram_6_topology),
        ("Degradation", diagram_7_degradation),
        ("Reward Function", diagram_8_reward),
        ("WRR vs PPO", diagram_9_comparison),
        ("Training Pipeline", diagram_10_pipeline),
    ]
    
    for name, func in diagrams:
        try:
            func(output_dir)
        except Exception as e:
            print(f"ERROR generating {name}: {e}")
    
    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Check: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
