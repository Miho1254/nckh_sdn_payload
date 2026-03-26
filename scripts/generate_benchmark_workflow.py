#!/usr/bin/env python3
"""
Generate benchmark workflow diagram - v5
Using simple rectangles with text annotations - no overlapping
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure
fig, ax = plt.subplots(figsize=(3.3, 5.5), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')
ax.set_aspect('auto')

# Colors
COLOR_STEP = '#f5f5f5'
COLOR_SCENARIO = '#e3f2fd'
COLOR_WRR = '#fff8e1'
COLOR_PPO = '#e8f5e9'
COLOR_ARROW = '#333333'

def add_box(ax, x, y, w, h, text_lines, bg_color, fontsize=8):
    """Add a box with centered text lines"""
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02", facecolor=bg_color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    line_h = h / (len(text_lines) + 1)
    for i, line in enumerate(text_lines):
        ty = y + h - (i + 0.8) * line_h
        ax.text(x + w/2, ty, line, ha='center', va='center',
                fontsize=fontsize, fontweight='bold' if i == 0 else 'normal',
                wrap=True)

def add_arrow(ax, x1, y1, x2, y2):
    """Add arrow"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=1.5))

# Title
ax.text(5, 15.5, 'BENCHMARK WORKFLOW', ha='center', va='center',
        fontsize=11, fontweight='bold')

# Step 1: Clean up
add_box(ax, 1, 13.5, 8, 1.2, ['1. Clean up', 'pkill ryu, mn -c'], COLOR_STEP)
add_arrow(ax, 5, 13.5, 5, 12.8)

# Step 2: Verify
add_box(ax, 1, 11.6, 8, 1.2, ['2. Verify Model', 'PPO + Controller'], COLOR_STEP)
add_arrow(ax, 5, 11.6, 5, 10.9)

# Step 3: Scenario (wider box)
add_box(ax, 1, 9.8, 8, 1.1, ['3. For each scenario', '(5 paired runs)'], COLOR_SCENARIO, fontsize=8)
add_arrow(ax, 5, 9.8, 5, 9.0)

# WRR box (left side)
add_box(ax, 0.5, 7.2, 4.3, 1.6, ['WRR Baseline', '- LB_ALGO="RR"', '- Ryu controller', '- flow_stats.csv'], COLOR_WRR, fontsize=7)

# PPO box (right side)
add_box(ax, 5.2, 7.2, 4.3, 1.6, ['PPO (AI)', '- LB_ALGO="AI"', '- Ryu + PPO', '- inference_log.csv'], COLOR_PPO, fontsize=7)

# Arrows from Step 3 to WRR/PPO
add_arrow(ax, 2.6, 9.0, 2.6, 8.8)
add_arrow(ax, 7.4, 9.0, 7.4, 8.8)

# Arrow down from WRR/PPO to Step 4
ax.plot([5, 5], [7.2, 6.6], color=COLOR_ARROW, linewidth=1.5)
ax.annotate('', xy=(5, 6.6), xytext=(5, 7.2),
            arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=1.5))

# Step 4: Summarize
add_box(ax, 1, 5.4, 8, 1.1, ['4. Summarize', 'mean +/- std, 95% CI'], COLOR_STEP)
add_arrow(ax, 5, 5.4, 5, 4.7)

# Step 5: Compare
add_box(ax, 1, 3.6, 8, 1.1, ['5. Compare', 'Paired setting, winner'], COLOR_STEP)

# Main border
border = mpatches.FancyBboxPatch((0.3, 3.4), 9.4, 12.4,
    boxstyle="round,pad=0.1", facecolor='none', edgecolor='black', linewidth=2)
ax.add_patch(border)

plt.tight_layout(pad=0.3)
plt.savefig('docs/figures/fig11_benchmark_workflow.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: docs/figures/fig11_benchmark_workflow.png")
