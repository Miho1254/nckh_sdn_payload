#!/usr/bin/env python3
"""Generate benchmark workflow diagram - v3 - clean layout."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

DPI = 300
FONT_NAME = 'DejaVu Sans'

def main():
    output_dir = 'docs/figures'
    
    # Figure size: 7 inches wide, 4 inches tall
    fig, ax = plt.subplots(figsize=(7, 4), dpi=DPI)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Colors
    C_BG = '#f8f8f8'
    C_STEP = '#e8e8e8'
    C_WRR = '#fff8e1'
    C_PPO = '#e8f5e9'
    C_ARROW = '#333333'
    
    # Title
    ax.text(7, 7.5, 'BENCHMARK WORKFLOW', ha='center', va='center',
            fontsize=12, fontweight='bold', fontfamily=FONT_NAME)
    
    # Main container
    container = mpatches.FancyBboxPatch((0.5, 0.5), 13, 6.5,
        boxstyle="round,pad=0.1", facecolor='none', edgecolor='black', linewidth=2)
    ax.add_patch(container)
    
    # Step 1 - Clean up (top left)
    box1 = mpatches.FancyBboxPatch((1, 5.5), 5, 1.2,
        boxstyle="round,pad=0.05", facecolor=C_STEP, edgecolor='black', linewidth=1)
    ax.add_patch(box1)
    ax.text(3.5, 6.1, '1. Clean up', ha='center', va='center', fontsize=9, fontweight='bold', fontfamily=FONT_NAME)
    ax.text(3.5, 5.7, 'pkill ryu, mn -c', ha='center', va='center', fontsize=8, fontfamily=FONT_NAME)
    
    # Arrow 1->2
    ax.annotate('', xy=(3.5, 5.5), xytext=(3.5, 5.0),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
    
    # Step 2 - Verify (below step 1)
    box2 = mpatches.FancyBboxPatch((1, 3.7), 5, 1.2,
        boxstyle="round,pad=0.05", facecolor=C_STEP, edgecolor='black', linewidth=1)
    ax.add_patch(box2)
    ax.text(3.5, 4.3, '2. Verify Model', ha='center', va='center', fontsize=9, fontweight='bold', fontfamily=FONT_NAME)
    ax.text(3.5, 3.9, 'PPO + Controller', ha='center', va='center', fontsize=8, fontfamily=FONT_NAME)
    
    # Arrow 2->3
    ax.annotate('', xy=(3.5, 3.7), xytext=(3.5, 3.2),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
    
    # Step 3 - For each scenario (spans full width)
    box3 = mpatches.FancyBboxPatch((1, 2.0), 12, 1.0,
        boxstyle="round,pad=0.05", facecolor='#e3f2fd', edgecolor='black', linewidth=1)
    ax.add_patch(box3)
    ax.text(7, 2.5, '3. For each scenario (5 paired runs for WRR and PPO)', 
            ha='center', va='center', fontsize=9, fontweight='bold', fontfamily=FONT_NAME)
    
    # WRR Box (left)
    box_wrr = mpatches.FancyBboxPatch((1, 0.7), 5.5, 1.1,
        boxstyle="round,pad=0.05", facecolor=C_WRR, edgecolor='black', linewidth=1)
    ax.add_patch(box_wrr)
    ax.text(3.75, 1.5, 'WRR Baseline', ha='center', va='center', fontsize=9, fontweight='bold', fontfamily=FONT_NAME)
    ax.text(1.3, 1.1, '- LB_ALGO="RR"', ha='left', va='center', fontsize=7, fontfamily=FONT_NAME)
    ax.text(1.3, 0.85, '- Start Ryu controller', ha='left', va='center', fontsize=7, fontfamily=FONT_NAME)
    ax.text(1.3, 0.6, '- Collect flow_stats.csv', ha='left', va='center', fontsize=7, fontfamily=FONT_NAME)
    
    # PPO Box (right)
    box_ppo = mpatches.FancyBboxPatch((7.5, 0.7), 5.5, 1.1,
        boxstyle="round,pad=0.05", facecolor=C_PPO, edgecolor='black', linewidth=1)
    ax.add_patch(box_ppo)
    ax.text(10.25, 1.5, 'PPO (AI)', ha='center', va='center', fontsize=9, fontweight='bold', fontfamily=FONT_NAME)
    ax.text(7.8, 1.1, '- LB_ALGO="AI"', ha='left', va='center', fontsize=7, fontfamily=FONT_NAME)
    ax.text(7.8, 0.85, '- Ryu + PPO inference', ha='left', va='center', fontsize=7, fontfamily=FONT_NAME)
    ax.text(7.8, 0.6, '- Collect inference_log.csv', ha='left', va='center', fontsize=7, fontfamily=FONT_NAME)
    
    # Arrows from Step 3 to WRR and PPO
    ax.annotate('', xy=(3.75, 1.8), xytext=(3.5, 2.0),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
    ax.annotate('', xy=(10.25, 1.8), xytext=(10.5, 2.0),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f'{output_dir}/fig11_benchmark_workflow.png',
                dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved: {output_dir}/fig11_benchmark_workflow.png")

if __name__ == '__main__':
    main()
