#!/usr/bin/env python3
"""Generate benchmark workflow diagram in IEEE standard format."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

# IEEE standard settings
DPI = 300
FONT = FontProperties(family='Times New Roman', size=8)

def setup_figure(width, height):
    """Setup figure with IEEE standard settings."""
    fig = plt.figure(figsize=(width, height), dpi=DPI)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    return fig, ax

def draw_box(ax, x, y, w, h, text, bg_color='white', border_color='black', fontsize=8):
    """Draw a box with text."""
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.0
    )
    ax.add_patch(box)
    
    # Split text and distribute vertically
    lines = text.split('\n')
    line_height = h / (len(lines) + 1)
    for i, line in enumerate(lines):
        text_y = y + h - (i + 1) * line_height + line_height * 0.3
        ax.text(x + w/2, text_y, line, ha='center', va='center', 
                fontproperties=FONT, fontsize=fontsize, color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

def draw_arrow(ax, x1, y1, x2, y2, color='black', linewidth=0.8):
    """Draw an arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth))

def draw_subprocess_box(ax, x, y, w, h, title, items, bg_color='white'):
    """Draw a subprocess box with title and items."""
    # Main box
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=bg_color, edgecolor='black', linewidth=1.0
    )
    ax.add_patch(box)
    
    # Title
    title_y = y + h - 0.3
    ax.text(x + w/2, title_y, title, ha='center', va='center',
            fontproperties=FONT, fontsize=9, fontweight='bold', color='black')
    
    # Items
    item_height = (h - 0.6) / len(items)
    for i, item in enumerate(items):
        item_y = y + h - 0.5 - (i + 0.5) * item_height
        ax.text(x + 0.3, item_y, item, ha='left', va='center',
                fontproperties=FONT, fontsize=7, color='black')

def main():
    output_dir = 'docs/figures'
    
    # Create workflow diagram (single column width: 3.5 inches)
    fig, ax = setup_figure(3.5, 8)
    
    # Step 1: Clean up
    draw_box(ax, 1, 10.5, 8, 0.8, "1. Clean up\n(pkill ryu, mn -c)")
    
    # Step 2: Verify
    draw_box(ax, 1, 9.2, 8, 0.8, "2. Verify PPO model\nvà controller")
    
    # Step 3: For each scenario (5 paired runs)
    draw_box(ax, 1, 7.5, 8, 0.8, "3. For each scenario\n(5 paired runs cho WRR và PPO)")
    
    # WRR Baseline subprocess
    draw_subprocess_box(ax, 1.2, 6.2, 3.4, 2.0, "WRR Baseline", [
        "- Set LB_ALGO=\"RR\"",
        "- Start Ryu controller",
        "- Run Mininet + Artillery",
        "- Duration: ~5 minutes",
        "- Collect flow_stats.csv, port_stats.csv"
    ])
    
    # PPO subprocess
    draw_subprocess_box(ax, 5.4, 6.2, 3.4, 2.0, "PPO (AI)", [
        "- Set LB_ALGO=\"AI\"",
        "- Start Ryu with PPO inference",
        "- Run Mininet + Artillery",
        "- Collect flow_stats.csv, inference_log.csv"
    ])
    
    # Arrows from Step 3 to subprocesses
    draw_arrow(ax, 5, 7.1, 3.4, 6.2)
    draw_arrow(ax, 5, 7.1, 7.6, 6.2)
    
    # Step 4: Summary
    draw_box(ax, 1, 4.8, 8, 0.8, "4. Tổng hợp theo trung bình,\nđộ lệch chuẩn, 95% CI")
    
    # Step 5: Compare
    draw_box(ax, 1, 3.5, 8, 0.8, "5. So sánh theo paired setting\nvà xác định người thắng")
    
    # Main container box
    container = mpatches.FancyBboxPatch(
        (0.5, 0.5), 9, 11,
        boxstyle="round,pad=0.3,rounding_size=0.2",
        facecolor='none', edgecolor='black', linewidth=1.5
    )
    ax.add_patch(container)
    
    # Title
    ax.text(5, 11.8, "BENCHMARK WORKFLOW", ha='center', va='center',
            fontproperties=FONT, fontsize=10, fontweight='bold', color='black')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig11_benchmark_workflow.png', 
                bbox_inches='tight', pad_inches=0.1, dpi=DPI)
    plt.close()
    print(f"Saved: {output_dir}/fig11_benchmark_workflow.png")

if __name__ == '__main__':
    main()
