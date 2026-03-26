#!/usr/bin/env python3
"""Generate benchmark workflow diagram in IEEE standard format - FIXED VERSION."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

DPI = 300
FONT = FontProperties(family='Times New Roman', size=7)

def setup_figure(width, height):
    fig = plt.figure(figsize=(width, height), dpi=DPI)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    return fig, ax

def draw_box(ax, x, y, w, h, lines, bg_color='#f5f5f5', fontsize=7):
    """Draw a box with lines of text - vertically distributed."""
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor=bg_color, edgecolor='black', linewidth=1.0
    )
    ax.add_patch(box)
    
    line_height = h / (len(lines) + 1)
    for i, line in enumerate(lines):
        text_y = y + h - (i + 0.7) * line_height
        ax.text(x + w/2, text_y, line, ha='center', va='center',
                fontproperties=FONT, fontsize=fontsize, color='black')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))

def main():
    output_dir = 'docs/figures'
    
    # Double column width for better fit
    fig, ax = setup_figure(7.0, 5.0)
    
    # Title
    ax.text(6, 9.5, "BENCHMARK WORKFLOW", ha='center', va='center',
            fontproperties=FONT, fontsize=10, fontweight='bold', color='black')
    
    # Step 1: Clean up
    draw_box(ax, 0.3, 8.0, 3.5, 1.0, ["1. Clean up", "pkill ryu, mn -c"])
    draw_arrow(ax, 2.05, 8.0, 2.05, 7.3)
    
    # Step 2: Verify
    draw_box(ax, 0.3, 6.3, 3.5, 1.0, ["2. Verify Model", "PPO + Controller"])
    draw_arrow(ax, 2.05, 6.3, 2.05, 5.6)
    
    # Step 3: For each scenario
    draw_box(ax, 0.3, 4.6, 11.4, 1.0, ["3. For each scenario (5 paired runs)"], bg_color='#e3f2fd')
    draw_arrow(ax, 2.05, 4.6, 2.05, 3.9)
    
    # WRR and PPO boxes side by side
    # WRR box
    wrr_items = [
        "WRR Baseline",
        "- LB_ALGO=RR",
        "- Ryu controller",
        "- Mininet + Artillery",
        "- Collect flow_stats.csv"
    ]
    draw_box(ax, 0.3, 1.5, 5.5, 2.4, wrr_items, bg_color='#fff3e0')
    
    # PPO box
    ppo_items = [
        "PPO (AI)",
        "- LB_ALGO=AI",
        "- Ryu + PPO inference",
        "- Mininet + Artillery",
        "- Collect inference_log.csv"
    ]
    draw_box(ax, 6.2, 1.5, 5.5, 2.4, ppo_items, bg_color='#e8f5e9')
    
    # Arrows from Step 3 to WRR/PPO
    draw_arrow(ax, 2.0, 4.6, 2.0, 3.9)
    draw_arrow(ax, 6.5, 4.6, 6.5, 3.9)
    
    # Step 4: Summary
    draw_box(ax, 0.3, 0.3, 5.5, 1.0, ["4. Tổng hợp", "mean +/- std, 95% CI"])
    draw_arrow(ax, 8.1, 3.9, 8.1, 1.3)
    
    # Step 5: Compare
    draw_box(ax, 6.2, 0.3, 5.5, 1.0, ["5. So sánh", "paired setting, winner"])
    
    # Connecting arrows
    draw_arrow(ax, 3.05, 1.5, 3.05, 1.3)
    draw_arrow(ax, 8.95, 1.5, 8.95, 1.3)
    
    # Main container
    container = mpatches.FancyBboxPatch(
        (0.1, 0.1), 11.8, 9.5,
        boxstyle="round,pad=0.2,rounding_size=0.3",
        facecolor='none', edgecolor='black', linewidth=2.0
    )
    ax.add_patch(container)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig11_benchmark_workflow.png',
                bbox_inches='tight', pad_inches=0.1, dpi=DPI)
    plt.close()
    print(f"Saved: {output_dir}/fig11_benchmark_workflow.png")

if __name__ == '__main__':
    main()
