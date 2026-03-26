#!/usr/bin/env python3
"""
Script render các sơ đồ cho bài báo NCKH TFT-PPO - VERSION 3
Output: PNG files chuẩn IEEE (300 DPI, Times New Roman)
- Tương phản cao
- Không text overlap
- Font size nhỏ, spacing rộng
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon
import numpy as np
from pathlib import Path

# IEEE Standard Figure Settings
IEEE_SINGLE_COL = 3.5  # inches
IEEE_DOUBLE_COL = 7.0  # inches
DPI = 300

# Font settings for IEEE - High Contrast
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 6,
    'axes.labelsize': 6,
    'axes.titlesize': 7,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
})


def setup_figure(width, height):
    """Tạo figure với kích thước IEEE"""
    fig, ax = plt.subplots(figsize=(width, height), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('white')
    return fig, ax


def draw_box(ax, x, y, w, h, text, bg_color='white', border_color='black', fontsize=5):
    """Vẽ box với text - tránh overlap"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.1,rounding_size=0.15",
                         facecolor=bg_color, edgecolor=border_color, linewidth=1.0)
    ax.add_patch(box)
    
    # Split text and distribute vertically
    lines = text.split('\n')
    # Calculate line height based on box height and number of lines
    total_lines = len(lines)
    line_height = h / (total_lines + 1)
    
    for i, line in enumerate(lines):
        # Center vertically
        text_y = y + h - (i + 1) * line_height + line_height * 0.3
        ax.text(x + w/2, text_y, line,
                ha='center', va='center', fontsize=fontsize, 
                fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none'))


def draw_arrow(ax, x1, y1, x2, y2, color='black', linewidth=0.8):
    """Vẽ arrow đơn giản"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth))


# ============================================================================
# FIGURE 1: System Architecture
# ============================================================================
def fig1_system_architecture(output_dir):
    """System Architecture - High contrast, no overlap"""
    fig, ax = setup_figure(IEEE_DOUBLE_COL, 4.5)
    
    ax.text(5, 9.5, 'Figure 1: TFT-PPO System Architecture', 
            ha='center', fontsize=8, fontweight='bold', color='black')
    
    # Layer 1: Clients
    draw_box(ax, 0.3, 7.5, 9.4, 1.0, 'CLIENTS (h9-h16) -> HTTP Requests -> VIP 10.0.0.100', 
             '#ffccbc', 'black')
    
    # Layer 2: Edge Switches
    draw_box(ax, 0.3, 6.2, 9.4, 1.0, 'EDGE SWITCHES (s1-s4) - OpenFlow 1.3', 
             '#bbdefb', 'black')
    
    # Layer 3: Core Switch
    draw_box(ax, 0.3, 4.9, 9.4, 1.0, 'CORE SWITCH (s5) - Packet-in to Ryu Controller', 
             '#90caf9', 'black')
    
    # Layer 4: Controller with TFT-PPO
    ctrl_box = FancyBboxPatch((0.3, 1.5), 5.7, 3.0,
                               boxstyle="round,pad=0.1", facecolor='#fff3e0', 
                               edgecolor='black', linewidth=1.2)
    ax.add_patch(ctrl_box)
    ax.text(3.15, 4.2, 'RYU SDN CONTROLLER', ha='center', fontsize=7, fontweight='bold', color='black')
    
    # TFT-PPO Agent box
    agent_box = FancyBboxPatch((0.5, 1.7), 2.5, 2.2,
                                boxstyle="round,pad=0.08", facecolor='#f3e5f5',
                                edgecolor='#9c27b0', linewidth=1.0)
    ax.add_patch(agent_box)
    ax.text(1.75, 3.6, 'TFT-PPO Agent', ha='center', fontsize=6, fontweight='bold', color='black')
    ax.text(1.75, 3.0, 'LSTM + Attention', ha='center', fontsize=5, color='black')
    ax.text(1.75, 2.4, 'Actor: pi(a|s)', ha='center', fontsize=5, color='black')
    ax.text(1.75, 1.8, 'Critic: V(s)', ha='center', fontsize=5, color='black')
    
    # Safety Override box
    safety_box = FancyBboxPatch((3.3, 1.7), 2.6, 2.2,
                                 boxstyle="round,pad=0.08", facecolor='#ffebee',
                                 edgecolor='#c62828', linewidth=1.0)
    ax.add_patch(safety_box)
    ax.text(4.6, 3.6, 'Safety Override', ha='center', fontsize=6, fontweight='bold', color='black')
    ax.text(4.6, 3.0, 'if util > 0.95', ha='center', fontsize=5, color='black')
    ax.text(4.6, 2.4, '-> Bypass Agent', ha='center', fontsize=5, color='black')
    ax.text(4.6, 1.8, '-> Use WRR', ha='center', fontsize=5, color='black')
    
    # Layer 5: Backend Servers
    draw_box(ax, 6.5, 1.7, 1.0, 2.2, 'h5\n10M', '#c8e6c9', 'black')
    draw_box(ax, 7.7, 1.7, 1.0, 2.2, 'h7\n50M', '#c8e6c9', 'black')
    draw_box(ax, 8.9, 1.7, 1.0, 2.2, 'h8\n100M', '#c8e6c9', 'black')
    ax.text(8.2, 4.2, 'BACKEND SERVERS', ha='center', fontsize=6, fontweight='bold', color='black')
    
    # Arrows
    draw_arrow(ax, 5, 7.5, 5, 6.2)
    draw_arrow(ax, 5, 6.2, 5, 4.9)
    draw_arrow(ax, 5, 4.9, 5, 4.5)
    draw_arrow(ax, 3.15, 1.5, 7, 4)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#ffccbc', edgecolor='black', label='Client'),
        mpatches.Patch(facecolor='#bbdefb', edgecolor='black', label='Switch'),
        mpatches.Patch(facecolor='#fff3e0', edgecolor='black', label='Controller'),
        mpatches.Patch(facecolor='#f3e5f5', edgecolor='#9c27b0', label='TFT-PPO'),
        mpatches.Patch(facecolor='#c8e6c9', edgecolor='black', label='Server'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=5, framealpha=0.9, facecolor='white')
    
    output_path = Path(output_dir) / 'fig1_system_architecture.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 2: TFT Encoder
# ============================================================================
def fig2_tft_encoder(output_dir):
    """TFT Encoder - Clean vertical layout"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 4.0)
    
    ax.text(5, 9.5, 'Figure 2: TFT Encoder Architecture', 
            ha='center', fontsize=8, fontweight='bold', color='black')
    
    # Input layer
    draw_box(ax, 1.5, 8.0, 7, 0.8, 'Input: 20 Features (Load, Latency, Queue, ...)', '#e8f5e9', 'black')
    
    # LSTM layer
    draw_box(ax, 1.5, 6.8, 7, 0.9, 'LSTM Encoder\n(Sequential Processing)', '#f3e5f5', 'black')
    
    # Attention layer
    draw_box(ax, 1.5, 5.6, 7, 0.9, 'Multi-Head Attention\n(h=4 heads)', '#f3e5f5', 'black')
    
    # Feature Linear
    draw_box(ax, 1.5, 4.4, 7, 0.9, 'Feature Linear\n(Dimension Reduction)', '#f3e5f5', 'black')
    
    # Output heads
    draw_box(ax, 0.5, 2.8, 3.5, 1.0, 'Actor Head\npi(a|s) - Server 0,1,2', '#e1f5fe', 'black')
    draw_box(ax, 6.0, 2.8, 3.5, 1.0, 'Critic Head\nV(s) - Value Est.', '#fff3e0', 'black')
    
    # Arrows
    draw_arrow(ax, 5, 8.0, 5, 6.8)
    draw_arrow(ax, 5, 6.8, 5, 5.6)
    draw_arrow(ax, 5, 5.6, 5, 4.4)
    draw_arrow(ax, 3.25, 4.4, 2.25, 3.8)
    draw_arrow(ax, 6.75, 4.4, 7.75, 3.8)
    
    # Labels on right
    ax.text(8.5, 6.8, 'Temporal\nProcessing', fontsize=5, ha='center', va='center', color='black')
    ax.text(8.5, 5.6, 'Attention\nWeights', fontsize=5, ha='center', va='center', color='black')
    
    output_path = Path(output_dir) / 'fig2_tft_encoder.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 3: PPO Training Flow
# ============================================================================
def fig3_ppo_flow(output_dir):
    """PPO Training Flow - Horizontal layout"""
    fig, ax = setup_figure(IEEE_DOUBLE_COL, 3.5)
    
    ax.text(5, 9.5, 'Figure 3: PPO Training Workflow', 
            ha='center', fontsize=8, fontweight='bold', color='black')
    
    # Stage 1: Initialize
    draw_box(ax, 0.3, 5.5, 2.0, 1.2, 'Initialize\nPolicy pi_theta', '#e8f5e9', 'black')
    
    # Stage 2: Collect
    draw_box(ax, 2.5, 5.5, 2.0, 1.2, 'Collect\nTrajectories', '#bbdefb', 'black')
    
    # Stage 3: Compute GAE
    draw_box(ax, 4.7, 5.5, 2.0, 1.2, 'Compute\nGAE (lambda=0.95)', '#bbdefb', 'black')
    
    # Stage 4: PPO Update
    draw_box(ax, 6.9, 5.5, 2.0, 1.2, 'PPO Update\n(Clip Objective)', '#f3e5f5', 'black')
    
    # Stage 5: Check
    draw_box(ax, 6.9, 3.0, 2.0, 1.2, 'Converged?', '#ffebee', 'black')
    
    # Stage 6: Save
    draw_box(ax, 4.7, 3.0, 2.0, 1.2, 'Save\nCheckpoint', '#c8e6c9', 'black')
    
    # Arrows
    draw_arrow(ax, 2.3, 6.1, 2.5, 6.1)
    draw_arrow(ax, 4.5, 6.1, 4.7, 6.1)
    draw_arrow(ax, 6.7, 6.1, 6.9, 6.1)
    draw_arrow(ax, 7.9, 5.5, 7.9, 4.2)
    draw_arrow(ax, 7.9, 3.0, 7.9, 1.8)
    ax.text(8.3, 4.0, 'No', fontsize=5, ha='center', color='black')
    draw_arrow(ax, 6.9, 3.6, 6.7, 3.6)
    ax.text(6.8, 3.8, 'Yes', fontsize=5, ha='center', color='black')
    draw_arrow(ax, 4.7, 3.6, 2.3, 3.6)
    ax.annotate('', xy=(1.3, 5.5), xytext=(1.3, 4.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8,
                               connectionstyle='arc3,rad=-0.3'))
    
    # Formula box
    ax.text(5, 0.8, 'L^CLIP = E[min(r*A, clip(r, 1-eps, 1+eps)*A)]', 
            ha='center', fontsize=6, 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    ax.text(5, 0.3, 'where r = pi_theta(a|s) / pi_old(a|s), eps = 0.2', 
            ha='center', fontsize=5, style='italic', color='black')
    
    output_path = Path(output_dir) / 'fig3_ppo_flow.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 4: Safety Override
# ============================================================================
def fig4_safety_override(output_dir):
    """Safety Override State Machine"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 4.0)
    
    ax.text(5, 9.5, 'Figure 4: Safety Override Mechanism', 
            ha='center', fontsize=8, fontweight='bold', color='black')
    
    # State 1: Normal
    circle1 = Circle((2.5, 6.5), 1.0, facecolor='#e8f5e9', edgecolor='black', linewidth=1.2)
    ax.add_patch(circle1)
    ax.text(2.5, 6.5, 'NORMAL\n(PPO)', ha='center', va='center', fontsize=6, fontweight='bold', color='black')
    
    # Decision diamond
    diamond = Polygon([(5, 7.5), (7, 6.5), (5, 5.5), (3, 6.5)], 
                      closed=True, facecolor='#fff3e0', edgecolor='black', linewidth=1.0)
    ax.add_patch(diamond)
    ax.text(5, 6.5, 'util >\n0.95?', ha='center', va='center', fontsize=5, fontweight='bold', color='black')
    
    # State 2: Override
    circle2 = Circle((7.5, 4.0), 1.0, facecolor='#ffebee', edgecolor='black', linewidth=1.2)
    ax.add_patch(circle2)
    ax.text(7.5, 4.0, 'OVERRIDE\n(WRR)', ha='center', va='center', fontsize=6, fontweight='bold', color='black')
    
    # Arrows
    draw_arrow(ax, 3.5, 6.5, 3, 6.5)
    ax.text(3.2, 6.8, 'Request', fontsize=5, ha='center', color='black')
    
    draw_arrow(ax, 7, 7, 7.5, 5.0)
    ax.text(7.8, 6.0, 'Yes', fontsize=5, ha='center', color='black')
    
    draw_arrow(ax, 5, 5.5, 5, 4.5)
    ax.text(5.3, 5.0, 'No', fontsize=5, ha='center', color='black')
    
    # Loop back
    ax.annotate('', xy=(7.5, 3.0), xytext=(7.5, 3.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8,
                               connectionstyle='arc3,rad=2'))
    ax.text(8.5, 3.5, 'Next\nRequest', fontsize=5, ha='center', color='black')
    
    # Description
    ax.text(5, 1.5, 'Safety Override ensures High Availability:\n'
                    'When utilization exceeds 95%, bypass PPO and use WRR fallback.',
            ha='center', va='center', fontsize=5, style='italic', color='black',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
    
    output_path = Path(output_dir) / 'fig4_safety_override.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 5: RL Cycle
# ============================================================================
def fig5_rl_cycle(output_dir):
    """Reinforcement Learning Cycle"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 4.0)
    
    ax.text(5, 9.5, 'Figure 5: Reinforcement Learning Cycle', 
            ha='center', fontsize=8, fontweight='bold', color='black')
    
    # Environment
    draw_box(ax, 0.5, 5.5, 3.0, 1.8, 'SDN\nEnvironment\n(Mininet/Ryu)', '#e8f5e9', 'black')
    
    # Agent
    draw_box(ax, 6.5, 5.5, 3.0, 1.8, 'TFT-PPO\nAgent', '#f3e5f5', 'black')
    
    # State
    draw_box(ax, 6.5, 3.0, 3.0, 1.2, 'State s_t\n(20 features)', '#bbdefb', 'black')
    
    # Action
    draw_box(ax, 0.5, 3.0, 3.0, 1.2, 'Action a_t\n(Server 0/1/2)', '#ffccbc', 'black')
    
    # Reward
    draw_box(ax, 3.5, 0.5, 3.0, 1.2, 'Reward r_t\n(bonus - penalty)', '#fff3e0', 'black')
    
    # Arrows with labels - separated vertically
    draw_arrow(ax, 6.5, 7.3, 3.5, 7.3)
    ax.text(5, 7.8, 'action', fontsize=5, ha='center', color='black')
    
    draw_arrow(ax, 3.5, 7.3, 6.5, 7.3)
    ax.text(5, 7.4, 'state', fontsize=5, ha='center', color='black')
    
    draw_arrow(ax, 3.5, 4.2, 3.5, 3.0)
    ax.text(3.8, 3.6, 'reward', fontsize=5, color='black')
    
    draw_arrow(ax, 5, 3.0, 6.5, 3.6)
    
    # Update arrow
    ax.annotate('', xy=(6.5, 5.5), xytext=(5, 3.0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=0.8,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(4.5, 4.0, 'PPO\nUpdate', fontsize=5, color='blue', ha='center')
    
    output_path = Path(output_dir) / 'fig5_rl_cycle.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 6: Network Topology
# ============================================================================
def fig6_topology(output_dir):
    """Fat-Tree K=4 Topology"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 4.5)
    
    ax.text(5, 9.5, 'Figure 6: Fat-Tree K=4 Network Topology', 
            ha='center', fontsize=8, fontweight='bold', color='black')
    
    # Core switch
    core = Circle((5, 7.5), 0.7, facecolor='#90caf9', edgecolor='black', linewidth=1.2)
    ax.add_patch(core)
    ax.text(5, 7.5, 's5\nCore', ha='center', va='center', fontsize=6, fontweight='bold', color='black')
    
    # Edge switches
    positions = [(1.5, 5.5), (4, 5.5), (6, 5.5), (8.5, 5.5)]
    labels = ['s1', 's2', 's3', 's4']
    for (x, y), label in zip(positions, labels):
        edge = Circle((x, y), 0.5, facecolor='#bbdefb', edgecolor='black', linewidth=1)
        ax.add_patch(edge)
        ax.text(x, y, label, ha='center', va='center', fontsize=5, fontweight='bold', color='black')
        ax.plot([x, 5], [y + 0.5, 6.8], 'k-', linewidth=0.8)
    
    # Clients
    for i, x in enumerate([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]):
        client = Circle((x, 3.5), 0.2, facecolor='#ffccbc', edgecolor='black', linewidth=0.6)
        ax.add_patch(client)
        ax.text(x, 3.0, f'h{9+i}', ha='center', fontsize=4, color='black')
        # Connect to nearest edge
        if i < 3:
            edge_x = 1.5
        elif i < 5:
            edge_x = 4
        elif i < 7:
            edge_x = 6
        else:
            edge_x = 8.5
        ax.plot([x, edge_x], [3.7, 5.0], 'k-', linewidth=0.5)
    
    # Servers
    server_positions = [(2, 1.5), (5, 1.5), (8, 1.5)]
    server_caps = ['h5\n10M', 'h7\n50M', 'h8\n100M']
    for (x, y), cap in zip(server_positions, server_caps):
        server = Circle((x, y), 0.4, facecolor='#c8e6c9', edgecolor='black', linewidth=1)
        ax.add_patch(server)
        ax.text(x, y, cap, ha='center', va='center', fontsize=5, fontweight='bold', color='black')
        ax.plot([x, 5], [y + 0.4, 6.8], 'k-', linewidth=0.8)
    
    # Labels
    ax.text(5, 8.5, 'Core Layer', ha='center', fontsize=5, style='italic', color='black')
    ax.text(5, 6.0, 'Edge Layer', ha='center', fontsize=5, style='italic', color='black')
    ax.text(5, 4.0, 'Clients (h9-h18)', ha='center', fontsize=5, style='italic', color='black')
    ax.text(5, 0.8, 'Backend Servers', ha='center', fontsize=5, style='italic', color='black')
    
    output_path = Path(output_dir) / 'fig6_topology.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 7: Hardware Degradation
# ============================================================================
def fig7_degradation(output_dir):
    """Hardware Degradation Detection Timeline"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 4.0)
    
    ax.text(5, 9.5, 'Figure 7: Hardware Degradation Detection', 
            ha='center', fontsize=8, fontweight='bold', color='black')
    
    # Timeline
    ax.plot([1, 9], [7, 7], 'k-', linewidth=1.5)
    
    # Time points
    times = [(1.5, 't=0\nNormal'), (4, 't=T\nDegradation'), (6.5, 't=T+1\nAdapt'), (8.5, 't=T+2\nRecover')]
    for x, label in times:
        ax.plot([x, x], [6.7, 7.3], 'k-', linewidth=1)
        ax.text(x, 6.3, label, ha='center', fontsize=5, color='black')
    
    # Server states
    draw_box(ax, 0.5, 4.0, 2.5, 1.2, 'h8: 100M\n[OK] Healthy', '#c8e6c9', 'black')
    draw_box(ax, 3.5, 4.0, 2.5, 1.2, 'h8: 50M\n[!] Degraded', '#ffebee', 'black')
    draw_box(ax, 6.5, 4.0, 2.5, 1.2, 'PPO Detects\nRoutes to h7', '#f3e5f5', 'black')
    
    # Arrows
    draw_arrow(ax, 3, 4.6, 3.5, 4.6)
    draw_arrow(ax, 6, 4.6, 6.5, 4.6)
    
    # Results
    draw_box(ax, 0.5, 1.5, 4.0, 1.8, 'WRR Result:\nPackets: 7,756,843\nUtilization: 95%+\nThroughput: -14.7%', '#ffccbc', 'black')
    draw_box(ax, 5.0, 1.5, 4.5, 1.8, 'PPO Result:\nPackets: 8,424,531\nUtilization: balanced\nThroughput: +8.6%', '#c8e6c9', 'black')
    
    # Winner
    ax.text(7.25, 0.5, 'PPO WINS!', ha='center', fontsize=7, fontweight='bold', color='green')
    
    output_path = Path(output_dir) / 'fig7_degradation.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 8: Reward Function
# ============================================================================
def fig8_reward(output_dir):
    """Reward Function Components"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 3.5)
    
    ax.text(5, 9.5, 'Figure 8: Reward Function Components', 
            ha='center', fontsize=8, fontweight='bold', color='black')
    
    # Components
    draw_box(ax, 0.5, 5.5, 2.5, 1.0, 'balance_bonus\n= score * 3.0', '#c8e6c9', 'black')
    draw_box(ax, 3.5, 5.5, 2.5, 1.0, 'throughput_bonus\n(based on BW)', '#bbdefb', 'black')
    draw_box(ax, 6.5, 5.5, 2.5, 1.0, 'latency_penalty\n(~= avg latency)', '#ffccbc', 'black')
    
    # Overload penalty
    draw_box(ax, 3, 3.5, 4, 1.0, 'overload_penalty = 20000\n(if utilization > 0.95)', '#ffebee', 'black')
    
    # Final reward
    draw_box(ax, 3, 1.5, 4, 1.0, 'reward = b + t - l - o', '#fff3e0', 'black')
    
    # Arrows
    draw_arrow(ax, 1.75, 5.5, 3.5, 4.5)
    draw_arrow(ax, 4.75, 5.5, 5, 4.5)
    draw_arrow(ax, 7.75, 5.5, 6.5, 4.5)
    draw_arrow(ax, 5, 3.5, 5, 2.5)
    
    output_path = Path(output_dir) / 'fig8_reward.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 9: WRR vs PPO Comparison
# ============================================================================
def fig9_comparison(output_dir):
    """WRR vs PPO Comparison"""
    fig, ax = setup_figure(IEEE_DOUBLE_COL, 3.5)
    
    ax.text(5, 9.5, 'Figure 9: WRR vs PPO - Trade-off Analysis', 
            ha='center', fontsize=8, fontweight='bold', color='black')
    
    # WRR box
    wrr_box = FancyBboxPatch((0.3, 1.0), 3.5, 7.0,
                              boxstyle="round,pad=0.1", facecolor='#ffcdd2',
                              edgecolor='#c62828', linewidth=1.0)
    ax.add_patch(wrr_box)
    ax.text(2.05, 7.5, 'WRR (Static)', ha='center', fontsize=7, fontweight='bold', color='#c62828')
    
    wrr_text = '''Round Robin
Fixed Weights
No Adaptation
No Learning

[+] Fast (O(1))
[+] Low Latency
[+] No Overhead

[-] Fails on Degradation
[-] Cannot Detect Anomaly'''
    ax.text(2.05, 4.0, wrr_text, ha='center', va='center', fontsize=5, color='black')
    
    # PPO box
    ppo_box = FancyBboxPatch((4.2, 1.0), 3.5, 7.0,
                              boxstyle="round,pad=0.1", facecolor='#c8e6c9',
                              edgecolor='#2e7d32', linewidth=1.0)
    ax.add_patch(ppo_box)
    ax.text(5.95, 7.5, 'PPO (Adaptive)', ha='center', fontsize=7, fontweight='bold', color='#2e7d32')
    
    ppo_text = '''Learns Patterns
Dynamic Weights
Adapts to Drift
Neural Network

[+] +8.6% on Degradation
[+] Detects Anomaly
[+] Self-Healing

[-] Inference Overhead
[-] May Lose Normal'''
    ax.text(5.95, 4.0, ppo_text, ha='center', va='center', fontsize=5, color='black')
    
    # Recommendation box
    rec_box = FancyBboxPatch((8.0, 1.0), 1.8, 7.0,
                              boxstyle="round,pad=0.1", facecolor='#fff9c4',
                              edgecolor='#f57f17', linewidth=1.0)
    ax.add_patch(rec_box)
    ax.text(8.9, 7.5, 'RECOMMEND', ha='center', fontsize=7, fontweight='bold', color='#f57f17')
    ax.text(8.9, 6.5, 'Hybrid\nApproach', ha='center', fontsize=6, fontweight='bold', color='#f57f17')
    ax.text(8.9, 5.0, 'WRR: 95%\ntraffic', ha='center', fontsize=5, color='black')
    ax.text(8.9, 4.0, 'PPO:\nanomaly\ndetection', ha='center', fontsize=5, color='black')
    
    # Arrows
    draw_arrow(ax, 3.8, 4.0, 8.0, 4.0)
    draw_arrow(ax, 7.7, 4.0, 8.0, 4.0)
    
    output_path = Path(output_dir) / 'fig9_comparison.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 10: Training Pipeline
# ============================================================================
def fig10_pipeline(output_dir):
    """Training Pipeline"""
    fig, ax = setup_figure(IEEE_DOUBLE_COL, 3.5)
    
    ax.text(5, 9.5, 'Figure 10: Training Pipeline', 
            ha='center', fontsize=8, fontweight='bold', color='black')
    
    # Data Collection
    draw_box(ax, 0.3, 5.5, 2.0, 1.0, 'Mininet\nEmulator', '#e1f5fe', 'black')
    draw_box(ax, 2.5, 5.5, 2.0, 1.0, 'Ryu\nController', '#e1f5fe', 'black')
    draw_box(ax, 4.7, 5.5, 2.0, 1.0, 'PortStats\nCollector', '#e1f5fe', 'black')
    
    # Environment
    draw_box(ax, 6.9, 5.5, 2.5, 1.0, 'SDNEnvV3\nGymnasium', '#fff3e0', 'black')
    
    # Training
    train_box = FancyBboxPatch((1.0, 2.0), 6.0, 2.5,
                                boxstyle="round,pad=0.1", facecolor='#f3e5f5',
                                edgecolor='#9c27b0', linewidth=1.0)
    ax.add_patch(train_box)
    ax.text(4.0, 4.0, 'PPO Training Loop', ha='center', fontsize=6, fontweight='bold', color='black')
    
    ax.text(2.0, 3.2, 'Collect 2048 steps', ha='center', fontsize=5, color='black')
    ax.text(4.0, 3.2, 'Compute GAE', ha='center', fontsize=5, color='black')
    ax.text(6.0, 3.2, 'PPO Update', ha='center', fontsize=5, color='black')
    ax.text(2.0, 2.5, 'Adam Optimizer', ha='center', fontsize=5, color='black')
    ax.text(4.0, 2.5, 'Update theta', ha='center', fontsize=5, color='black')
    ax.text(6.0, 2.5, 'Checkpoint', ha='center', fontsize=5, color='black')
    
    # Evaluation
    draw_box(ax, 7.5, 2.0, 2.0, 2.5, 'Evaluation\n\nWRR vs PPO\n\n5 runs\n\nReport', '#e8f5e9', 'black')
    
    # Arrows
    draw_arrow(ax, 2.3, 6.0, 2.5, 6.0)
    draw_arrow(ax, 4.5, 6.0, 4.7, 6.0)
    draw_arrow(ax, 6.7, 6.0, 6.9, 6.0)
    draw_arrow(ax, 8.15, 5.5, 8.15, 4.5)
    draw_arrow(ax, 4.0, 5.5, 4.0, 4.5)
    draw_arrow(ax, 7.0, 3.2, 7.5, 3.2)
    
    output_path = Path(output_dir) / 'fig10_pipeline.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Generate all diagrams"""
    output_dir = Path('docs/figures')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Generating IEEE Standard Figures (v3 - High Contrast)")
    print("=" * 60)
    print(f"Output: {output_dir.absolute()}")
    print()
    
    diagrams = [
        ("System Architecture", fig1_system_architecture),
        ("TFT Encoder", fig2_tft_encoder),
        ("PPO Flow", fig3_ppo_flow),
        ("Safety Override", fig4_safety_override),
        ("RL Cycle", fig5_rl_cycle),
        ("Topology", fig6_topology),
        ("Degradation", fig7_degradation),
        ("Reward", fig8_reward),
        ("Comparison", fig9_comparison),
        ("Pipeline", fig10_pipeline),
    ]
    
    for name, func in diagrams:
        try:
            func(output_dir)
        except Exception as e:
            print(f"ERROR generating {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print("Done! Check docs/figures/")
    print("=" * 60)


if __name__ == '__main__':
    main()
