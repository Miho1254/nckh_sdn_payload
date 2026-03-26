#!/usr/bin/env python3
"""
Script render các sơ đồ cho bài báo NCKH TFT-PPO - VERSION 2 (Fixed Layout)
Output: PNG files chuẩn IEEE (300 DPI, Times New Roman)

Chạy:
    python scripts/generate_diagrams_v2.py
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

# Font settings for IEEE
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
})


def setup_figure(width, height):
    """Tạo figure với kích thước IEEE"""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    return fig, ax


def draw_box(ax, x, y, w, h, text, color='white', fontsize=6):
    """Vẽ box với text - tránh overlap"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.05,rounding_size=0.1",
                         facecolor=color, edgecolor='black', linewidth=0.6)
    ax.add_patch(box)
    # Split text if too long
    lines = text.split('\n')
    line_height = h / (len(lines) + 0.5)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - (i + 0.5) * line_height, line,
                ha='center', va='center', fontsize=fontsize, fontweight='bold')


def draw_arrow(ax, x1, y1, x2, y2, color='black'):
    """Vẽ arrow đơn giản"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=0.6))


# ============================================================================
# FIGURE 1: System Architecture (Simplified)
# ============================================================================
def fig1_system_architecture(output_dir):
    """System Architecture - Simplified layout"""
    fig, ax = setup_figure(IEEE_DOUBLE_COL, 4)
    
    ax.text(5, 9.5, 'Figure 1: TFT-PPO System Architecture', 
            ha='center', fontsize=9, fontweight='bold')
    
    # Layer 1: Clients
    draw_box(ax, 0.5, 7.5, 9, 1.2, 'CLIENTS (h9-h16) -> HTTP Requests -> VIP 10.0.0.100', '#ffccbc')
    
    # Layer 2: Edge Switches
    draw_box(ax, 0.5, 6, 9, 1.2, 'EDGE SWITCHES (s1-s4) - OpenFlow 1.3', '#bbdefb')
    
    # Layer 3: Core Switch
    draw_box(ax, 0.5, 4.5, 9, 1.2, 'CORE SWITCH (s5) - Packet-in to Ryu Controller', '#90caf9')
    
    # Layer 4: Controller with TFT-PPO
    ctrl_box = FancyBboxPatch((0.5, 1.5), 5.5, 2.5,
                               boxstyle="round,pad=0.1", facecolor='#fff3e0', 
                               edgecolor='black', linewidth=1)
    ax.add_patch(ctrl_box)
    ax.text(3.25, 3.7, 'RYU SDN CONTROLLER', ha='center', fontsize=7, fontweight='bold')
    
    # TFT-PPO Agent box
    agent_box = FancyBboxPatch((0.7, 1.7), 2.5, 1.8,
                                boxstyle="round,pad=0.05", facecolor='#f3e5f5',
                                edgecolor='#9c27b0', linewidth=0.8)
    ax.add_patch(agent_box)
    ax.text(1.95, 3.2, 'TFT-PPO Agent', ha='center', fontsize=6, fontweight='bold')
    ax.text(1.95, 2.7, 'LSTM + Attention', ha='center', fontsize=5)
    ax.text(1.95, 2.3, 'Actor: pi(a|s)', ha='center', fontsize=5)
    ax.text(1.95, 1.9, 'Critic: V(s)', ha='center', fontsize=5)
    
    # Safety Override box
    safety_box = FancyBboxPatch((3.4, 1.7), 2.4, 1.8,
                                 boxstyle="round,pad=0.05", facecolor='#ffebee',
                                 edgecolor='#c62828', linewidth=0.8)
    ax.add_patch(safety_box)
    ax.text(4.6, 3.2, 'Safety Override', ha='center', fontsize=6, fontweight='bold')
    ax.text(4.6, 2.7, 'if util > 0.95', ha='center', fontsize=5)
    ax.text(4.6, 2.3, '-> Bypass Agent', ha='center', fontsize=5)
    ax.text(4.6, 1.9, '-> Use WRR', ha='center', fontsize=5)
    
    # Layer 5: Backend Servers
    draw_box(ax, 6.5, 1.5, 1, 2.5, 'h5\n10M', '#c8e6c9')
    draw_box(ax, 7.7, 1.5, 1, 2.5, 'h7\n50M', '#c8e6c9')
    draw_box(ax, 8.9, 1.5, 1, 2.5, 'h8\n100M', '#c8e6c9')
    ax.text(8.2, 4.2, 'BACKEND SERVERS', ha='center', fontsize=6, fontweight='bold')
    
    # Arrows
    draw_arrow(ax, 5, 7.5, 5, 7.2)
    draw_arrow(ax, 5, 6, 5, 5.7)
    draw_arrow(ax, 5, 4.5, 5, 4)
    draw_arrow(ax, 3.25, 1.5, 7, 4)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#ffccbc', edgecolor='black', label='Client'),
        mpatches.Patch(facecolor='#bbdefb', edgecolor='black', label='Switch'),
        mpatches.Patch(facecolor='#fff3e0', edgecolor='black', label='Controller'),
        mpatches.Patch(facecolor='#f3e5f5', edgecolor='#9c27b0', label='TFT-PPO'),
        mpatches.Patch(facecolor='#c8e6c9', edgecolor='black', label='Server'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=5, framealpha=0.9)
    
    output_path = Path(output_dir) / 'fig1_system_architecture.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 2: TFT Encoder (Clean Layout)
# ============================================================================
def fig2_tft_encoder(output_dir):
    """TFT Encoder - Clean vertical layout"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 3.5)
    
    ax.text(5, 9.5, 'Figure 2: TFT Encoder Architecture', 
            ha='center', fontsize=9, fontweight='bold')
    
    # Input layer
    draw_box(ax, 2, 7.5, 6, 1, 'Input: 20 Features (Load, Latency, Queue, ...)', '#e8f5e9')
    
    # LSTM layer
    draw_box(ax, 2, 5.5, 6, 1.2, 'LSTM Encoder\n(Sequential Processing)', '#f3e5f5')
    
    # Attention layer
    draw_box(ax, 2, 3.5, 6, 1.2, 'Multi-Head Attention\n(h=4 heads)', '#f3e5f5')
    
    # Feature Linear
    draw_box(ax, 2, 1.5, 6, 1.2, 'Feature Linear\n(Dimension Reduction)', '#f3e5f5')
    
    # Output heads
    draw_box(ax, 1, 0, 3.5, 1, 'Actor Head\npi(a|s) - Server 0,1,2', '#e1f5fe')
    draw_box(ax, 5.5, 0, 3.5, 1, 'Critic Head\nV(s) - Value Est.', '#fff3e0')
    
    # Arrows
    draw_arrow(ax, 5, 7.5, 5, 6.7)
    draw_arrow(ax, 5, 5.5, 5, 4.7)
    draw_arrow(ax, 5, 3.5, 5, 2.7)
    draw_arrow(ax, 3.5, 1.5, 2.75, 1)
    draw_arrow(ax, 6.5, 1.5, 7.25, 1)
    
    # Labels on right
    ax.text(8.5, 6.1, 'Temporal\nProcessing', fontsize=5, ha='center', va='center')
    ax.text(8.5, 4.1, 'Attention\nWeights', fontsize=5, ha='center', va='center')
    
    output_path = Path(output_dir) / 'fig2_tft_encoder.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 3: PPO Training Flow (Horizontal)
# ============================================================================
def fig3_ppo_flow(output_dir):
    """PPO Training Flow - Horizontal layout"""
    fig, ax = setup_figure(IEEE_DOUBLE_COL, 3)
    
    ax.text(5, 9.5, 'Figure 3: PPO Training Workflow', 
            ha='center', fontsize=9, fontweight='bold')
    
    # Stage 1: Initialize
    draw_box(ax, 0.3, 6, 2, 1.5, 'Initialize\nPolicy pi_theta', '#e8f5e9')
    
    # Stage 2: Collect
    draw_box(ax, 2.5, 6, 2, 1.5, 'Collect\nTrajectories', '#bbdefb')
    
    # Stage 3: Compute GAE
    draw_box(ax, 4.7, 6, 2, 1.5, 'Compute\nGAE (lambda=0.95)', '#bbdefb')
    
    # Stage 4: PPO Update
    draw_box(ax, 6.9, 6, 2, 1.5, 'PPO Update\n(Clip Objective)', '#f3e5f5')
    
    # Stage 5: Check
    draw_box(ax, 6.9, 3.5, 2, 1.5, 'Converged?', '#ffebee')
    
    # Stage 6: Save
    draw_box(ax, 4.7, 3.5, 2, 1.5, 'Save\nCheckpoint', '#c8e6c9')
    
    # Arrows
    draw_arrow(ax, 2.3, 6.75, 2.5, 6.75)
    draw_arrow(ax, 4.5, 6.75, 4.7, 6.75)
    draw_arrow(ax, 6.7, 6.75, 6.9, 6.75)
    draw_arrow(ax, 7.9, 6, 7.9, 5)
    draw_arrow(ax, 7.9, 3.5, 7.9, 2.5)
    ax.text(8.3, 4.75, 'No', fontsize=5)
    draw_arrow(ax, 6.9, 4.25, 6.7, 4.25)
    ax.text(6.8, 4.5, 'Yes', fontsize=5)
    draw_arrow(ax, 4.7, 4.25, 2.3, 4.25)
    ax.annotate('', xy=(1.3, 6), xytext=(1.3, 5),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.6,
                               connectionstyle='arc3,rad=-0.3'))
    
    # Formula box
    ax.text(5, 1.5, 'L^CLIP = E[min(r*A, clip(r, 1-eps, 1+eps)*A)]', 
            ha='center', fontsize=7, 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    ax.text(5, 0.8, 'where r = pi_theta(a|s) / pi_old(a|s), eps = 0.2', 
            ha='center', fontsize=5, style='italic')
    
    output_path = Path(output_dir) / 'fig3_ppo_flow.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 4: Safety Override (State Machine)
# ============================================================================
def fig4_safety_override(output_dir):
    """Safety Override State Machine"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 3.5)
    
    ax.text(5, 9.5, 'Figure 4: Safety Override Mechanism', 
            ha='center', fontsize=9, fontweight='bold')
    
    # State 1: Normal
    circle1 = Circle((2.5, 6.5), 1.2, facecolor='#e8f5e9', edgecolor='black', linewidth=1)
    ax.add_patch(circle1)
    ax.text(2.5, 6.5, 'NORMAL\n(PPO)', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Decision diamond
    diamond = Polygon([(5, 8), (7, 6.5), (5, 5), (3, 6.5)], 
                      closed=True, facecolor='#fff3e0', edgecolor='black', linewidth=1)
    ax.add_patch(diamond)
    ax.text(5, 6.5, 'util >\n0.95?', ha='center', va='center', fontsize=6, fontweight='bold')
    
    # State 2: Override
    circle2 = Circle((7.5, 4), 1.2, facecolor='#ffebee', edgecolor='black', linewidth=1)
    ax.add_patch(circle2)
    ax.text(7.5, 4, 'OVERRIDE\n(WRR)', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Arrows
    draw_arrow(ax, 3.7, 6.5, 3, 6.5)
    ax.text(3.3, 6.8, 'Request', fontsize=5)
    
    draw_arrow(ax, 7, 7, 7.5, 5.2)
    ax.text(7.8, 6, 'Yes', fontsize=5)
    
    draw_arrow(ax, 5, 5, 5, 4)
    ax.text(5.3, 4.5, 'No', fontsize=5)
    
    # Loop back
    ax.annotate('', xy=(7.5, 2.8), xytext=(7.5, 2.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.6,
                               connectionstyle='arc3,rad=2'))
    ax.text(8.5, 3, 'Next\nRequest', fontsize=5, ha='center')
    
    # Description
    ax.text(5, 1.5, 'Safety Override ensures High Availability:\n'
                    'When utilization exceeds 95%, bypass PPO and use WRR fallback.',
            ha='center', va='center', fontsize=5, style='italic',
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
    fig, ax = setup_figure(IEEE_SINGLE_COL, 3.5)
    
    ax.text(5, 9.5, 'Figure 5: Reinforcement Learning Cycle', 
            ha='center', fontsize=9, fontweight='bold')
    
    # Environment
    draw_box(ax, 0.5, 5, 3, 2, 'SDN\nEnvironment\n(Mininet/Ryu)', '#e8f5e9')
    
    # Agent
    draw_box(ax, 6.5, 5, 3, 2, 'TFT-PPO\nAgent', '#f3e5f5')
    
    # State
    draw_box(ax, 6.5, 2.5, 3, 1.5, 'State s_t\n(20 features)', '#bbdefb')
    
    # Action
    draw_box(ax, 0.5, 2.5, 3, 1.5, 'Action a_t\n(Server 0/1/2)', '#ffccbc')
    
    # Reward
    draw_box(ax, 3.5, 0.5, 3, 1.5, 'Reward r_t\n(bonus - penalty)', '#fff3e0')
    
    # Arrows with labels
    draw_arrow(ax, 6.5, 6, 3.5, 6)
    ax.text(5, 6.3, 'action', fontsize=5, ha='center')
    
    draw_arrow(ax, 3.5, 7, 6.5, 7)
    ax.text(5, 7.3, 'state', fontsize=5, ha='center')
    
    draw_arrow(ax, 3.5, 3.25, 3.5, 2)
    ax.text(3.8, 2.5, 'reward', fontsize=5)
    
    draw_arrow(ax, 5, 2, 6.5, 2.5)
    
    # Update arrow
    ax.annotate('', xy=(6.5, 5), xytext=(5, 2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=0.8,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(4.5, 3.5, 'PPO\nUpdate', fontsize=5, color='blue', ha='center')
    
    output_path = Path(output_dir) / 'fig5_rl_cycle.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 6: Network Topology
# ============================================================================
def fig6_topology(output_dir):
    """Fat-Tree K=4 Topology"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 4)
    
    ax.text(5, 9.5, 'Figure 6: Fat-Tree K=4 Network Topology', 
            ha='center', fontsize=9, fontweight='bold')
    
    # Core switch
    core = Circle((5, 7), 0.8, facecolor='#90caf9', edgecolor='black', linewidth=1.2)
    ax.add_patch(core)
    ax.text(5, 7, 's5\nCore', ha='center', va='center', fontsize=6, fontweight='bold')
    
    # Edge switches
    positions = [(1.5, 5), (4, 5), (6, 5), (8.5, 5)]
    labels = ['s1', 's2', 's3', 's4']
    for (x, y), label in zip(positions, labels):
        edge = Circle((x, y), 0.6, facecolor='#bbdefb', edgecolor='black', linewidth=1)
        ax.add_patch(edge)
        ax.text(x, y, label, ha='center', va='center', fontsize=6, fontweight='bold')
        ax.plot([x, 5], [y + 0.6, 6.2], 'k-', linewidth=0.8)
    
    # Clients
    for i, x in enumerate([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]):
        client = Circle((x, 3), 0.25, facecolor='#ffccbc', edgecolor='black', linewidth=0.6)
        ax.add_patch(client)
        ax.text(x, 2.5, f'h{9+i}', ha='center', fontsize=4)
        # Connect to nearest edge
        if i < 3:
            edge_x = 1.5
        elif i < 5:
            edge_x = 4
        elif i < 7:
            edge_x = 6
        else:
            edge_x = 8.5
        ax.plot([x, edge_x], [3.25, 4.4], 'k-', linewidth=0.5)
    
    # Servers
    server_positions = [(2, 1), (5, 1), (8, 1)]
    server_caps = ['h5\n10M', 'h7\n50M', 'h8\n100M']
    for (x, y), cap in zip(server_positions, server_caps):
        server = Circle((x, y), 0.5, facecolor='#c8e6c9', edgecolor='black', linewidth=1)
        ax.add_patch(server)
        ax.text(x, y, cap, ha='center', va='center', fontsize=5, fontweight='bold')
        ax.plot([x, 5], [y + 0.5, 6.2], 'k-', linewidth=0.8)
    
    # Labels
    ax.text(5, 8.2, 'Core Layer', ha='center', fontsize=6, style='italic')
    ax.text(5, 5.8, 'Edge Layer', ha='center', fontsize=6, style='italic')
    ax.text(5, 3.5, 'Clients (h9-h18)', ha='center', fontsize=6, style='italic')
    ax.text(5, 0.2, 'Backend Servers', ha='center', fontsize=6, style='italic')
    
    output_path = Path(output_dir) / 'fig6_topology.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 7: Hardware Degradation Scenario
# ============================================================================
def fig7_degradation(output_dir):
    """Hardware Degradation Detection Timeline"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 3.5)
    
    ax.text(5, 9.5, 'Figure 7: Hardware Degradation Detection', 
            ha='center', fontsize=9, fontweight='bold')
    
    # Timeline
    ax.plot([1, 9], [7, 7], 'k-', linewidth=1.5)
    
    # Time points
    times = [(1.5, 't=0\nNormal'), (4, 't=T\nDegradation'), (6.5, 't=T+1\nAdapt'), (8.5, 't=T+2\nRecover')]
    for x, label in times:
        ax.plot([x, x], [6.7, 7.3], 'k-', linewidth=1)
        ax.text(x, 6.3, label, ha='center', fontsize=5)
    
    # Server states
    draw_box(ax, 0.5, 4, 2.5, 1.5, 'h8: 100M\n[OK] Healthy', '#c8e6c9')
    draw_box(ax, 3.5, 4, 2.5, 1.5, 'h8: 50M\n[!] Degraded', '#ffebee')
    draw_box(ax, 6.5, 4, 2.5, 1.5, 'PPO Detects\nRoutes to h7', '#f3e5f5')
    
    # Arrows
    draw_arrow(ax, 3, 4.75, 3.5, 4.75)
    draw_arrow(ax, 6, 4.75, 6.5, 4.75)
    
    # Results
    draw_box(ax, 0.5, 1.5, 4, 2, 'WRR Result:\nPackets: 7,756,843\nUtilization: 95%+\nThroughput: -14.7%', '#ffccbc')
    draw_box(ax, 5, 1.5, 4.5, 2, 'PPO Result:\nPackets: 8,424,531\nUtilization: balanced\nThroughput: +8.6%', '#c8e6c9')
    
    # Winner
    ax.text(7.25, 0.8, 'PPO WINS!', ha='center', fontsize=8, fontweight='bold', color='green')
    
    output_path = Path(output_dir) / 'fig7_degradation.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 8: Reward Function
# ============================================================================
def fig8_reward(output_dir):
    """Reward Function Components"""
    fig, ax = setup_figure(IEEE_SINGLE_COL, 3)
    
    ax.text(5, 9.5, 'Figure 8: Reward Function Components', 
            ha='center', fontsize=9, fontweight='bold')
    
    # Components
    draw_box(ax, 0.5, 6, 2.5, 1.5, 'balance_bonus\n= score * 3.0', '#c8e6c9')
    draw_box(ax, 3.5, 6, 2.5, 1.5, 'throughput_bonus\n(based on BW)', '#bbdefb')
    draw_box(ax, 6.5, 6, 2.5, 1.5, 'latency_penalty\n(~= avg latency)', '#ffccbc')
    
    # Overload penalty
    draw_box(ax, 3, 3.5, 4, 1.5, 'overload_penalty = 20000\n(if utilization > 0.95)', '#ffebee')
    
    # Final reward
    draw_box(ax, 3, 1, 4, 1.5, 'reward = b + t - l - o', '#fff3e0')
    
    # Arrows
    draw_arrow(ax, 1.75, 6, 3.5, 5)
    draw_arrow(ax, 4.75, 6, 5, 5)
    draw_arrow(ax, 7.75, 6, 6.5, 5)
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
    fig, ax = setup_figure(IEEE_DOUBLE_COL, 3)
    
    ax.text(5, 9.5, 'Figure 9: WRR vs PPO - Trade-off Analysis', 
            ha='center', fontsize=9, fontweight='bold')
    
    # WRR box
    wrr_box = FancyBboxPatch((0.3, 1), 3.5, 7,
                              boxstyle="round,pad=0.1", facecolor='#ffcdd2',
                              edgecolor='#c62828', linewidth=1)
    ax.add_patch(wrr_box)
    ax.text(2.05, 7.5, 'WRR (Static)', ha='center', fontsize=8, fontweight='bold', color='#c62828')
    
    wrr_text = '''Round Robin
Fixed Weights
No Adaptation
No Learning

[+] Fast (O(1))
[+] Low Latency
[+] No Overhead

[-] Fails on Degradation
[-] Cannot Detect Anomaly'''
    ax.text(2.05, 4.5, wrr_text, ha='center', va='center', fontsize=5)
    
    # PPO box
    ppo_box = FancyBboxPatch((4.2, 1), 3.5, 7,
                              boxstyle="round,pad=0.1", facecolor='#c8e6c9',
                              edgecolor='#2e7d32', linewidth=1)
    ax.add_patch(ppo_box)
    ax.text(5.95, 7.5, 'PPO (Adaptive)', ha='center', fontsize=8, fontweight='bold', color='#2e7d32')
    
    ppo_text = '''Learns Patterns
Dynamic Weights
Adapts to Drift
Neural Network

[+] +8.6% on Degradation
[+] Detects Anomaly
[+] Self-Healing

[-] Inference Overhead
[-] May Lose Normal'''
    ax.text(5.95, 4.5, ppo_text, ha='center', va='center', fontsize=5)
    
    # Recommendation box
    rec_box = FancyBboxPatch((8, 1), 1.8, 7,
                              boxstyle="round,pad=0.1", facecolor='#fff9c4',
                              edgecolor='#f57f17', linewidth=1)
    ax.add_patch(rec_box)
    ax.text(8.9, 7.5, 'RECOMMEND', ha='center', fontsize=7, fontweight='bold')
    ax.text(8.9, 6.5, 'Hybrid\nApproach', ha='center', fontsize=7, fontweight='bold', color='#f57f17')
    ax.text(8.9, 5, 'WRR: 95%\ntraffic', ha='center', fontsize=5)
    ax.text(8.9, 4, 'PPO:\nanomaly\ndetection', ha='center', fontsize=5)
    
    # Arrows
    draw_arrow(ax, 3.8, 4, 8, 4)
    draw_arrow(ax, 7.7, 4, 8, 4)
    
    output_path = Path(output_dir) / 'fig9_comparison.png'
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 10: Training Pipeline
# ============================================================================
def fig10_pipeline(output_dir):
    """Training Pipeline"""
    fig, ax = setup_figure(IEEE_DOUBLE_COL, 3)
    
    ax.text(5, 9.5, 'Figure 10: Training Pipeline', 
            ha='center', fontsize=9, fontweight='bold')
    
    # Data Collection
    draw_box(ax, 0.3, 6, 2, 1.5, 'Mininet\nEmulator', '#e1f5fe')
    draw_box(ax, 2.5, 6, 2, 1.5, 'Ryu\nController', '#e1f5fe')
    draw_box(ax, 4.7, 6, 2, 1.5, 'PortStats\nCollector', '#e1f5fe')
    
    # Environment
    draw_box(ax, 6.9, 6, 2.5, 1.5, 'SDNEnvV3\nGymnasium', '#fff3e0')
    
    # Training
    train_box = FancyBboxPatch((1, 2), 6, 3,
                                boxstyle="round,pad=0.1", facecolor='#f3e5f5',
                                edgecolor='#9c27b0', linewidth=1)
    ax.add_patch(train_box)
    ax.text(4, 4.5, 'PPO Training Loop', ha='center', fontsize=7, fontweight='bold')
    
    ax.text(2, 3.5, 'Collect 2048 steps', ha='center', fontsize=5)
    ax.text(4, 3.5, 'Compute GAE', ha='center', fontsize=5)
    ax.text(6, 3.5, 'PPO Update', ha='center', fontsize=5)
    ax.text(2, 2.5, 'Adam Optimizer', ha='center', fontsize=5)
    ax.text(4, 2.5, 'Update theta', ha='center', fontsize=5)
    ax.text(6, 2.5, 'Checkpoint', ha='center', fontsize=5)
    
    # Evaluation
    draw_box(ax, 7.5, 2, 2, 3, 'Evaluation\n\nWRR vs PPO\n\n5 runs\n\nReport', '#e8f5e9')
    
    # Arrows
    draw_arrow(ax, 2.3, 6.75, 2.5, 6.75)
    draw_arrow(ax, 4.5, 6.75, 4.7, 6.75)
    draw_arrow(ax, 6.7, 6.75, 6.9, 6.75)
    draw_arrow(ax, 8.15, 6, 8.15, 5)
    draw_arrow(ax, 4, 6, 4, 5)
    draw_arrow(ax, 7, 3.5, 7.5, 3.5)
    
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
    print("Generating IEEE Standard Figures (v2 - Fixed Layout)")
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