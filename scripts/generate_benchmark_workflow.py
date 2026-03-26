#!/usr/bin/env python3
"""Generate benchmark workflow diagram - v4 - ultra simple."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    output_dir = 'docs/figures'
    
    fig, ax = plt.subplots(figsize=(3.5, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Title
    ax.text(5, 17.5, 'BENCHMARK WORKFLOW', ha='center', va='center',
            fontsize=11, fontweight='bold', fontfamily='sans-serif')
    
    y = 16
    step_h = 1.5
    step_gap = 0.5
    
    # Step 1
    rect = patches.Rectangle((1, y), 8, step_h, linewidth=1, edgecolor='black', facecolor='#eeeeee')
    ax.add_patch(rect)
    ax.text(5, y + 1.1, 'Step 1: Clean up', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, y + 0.4, 'pkill ryu, mn -c', ha='center', va='center', fontsize=7)
    y -= step_h + step_gap
    
    # Arrow
    ax.annotate('', xy=(5, y + step_h), xytext=(5, y + step_gap),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Step 2
    rect = patches.Rectangle((1, y), 8, step_h, linewidth=1, edgecolor='black', facecolor='#eeeeee')
    ax.add_patch(rect)
    ax.text(5, y + 1.1, 'Step 2: Verify', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, y + 0.4, 'PPO model + Controller', ha='center', va='center', fontsize=7)
    y -= step_h + step_gap
    
    # Arrow
    ax.annotate('', xy=(5, y + step_h), xytext=(5, y + step_gap),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Step 3
    rect = patches.Rectangle((1, y), 8, step_h, linewidth=1, edgecolor='black', facecolor='#bbdefb')
    ax.add_patch(rect)
    ax.text(5, y + 1.1, 'Step 3: For each scenario', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, y + 0.4, '5 paired runs (WRR + PPO)', ha='center', va='center', fontsize=7)
    y -= step_h + step_gap
    
    # WRR and PPO boxes
    # WRR
    rect_wrr = patches.Rectangle((0.5, y - 2), 4, 2, linewidth=1, edgecolor='black', facecolor='#fff3cd')
    ax.add_patch(rect_wrr)
    ax.text(2.5, y - 0.5, 'WRR Baseline', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(2.5, y - 1.0, 'LB_ALGO="RR"', ha='center', va='center', fontsize=6)
    ax.text(2.5, y - 1.4, 'Collect flow_stats', ha='center', va='center', fontsize=6)
    
    # PPO
    rect_ppo = patches.Rectangle((5.5, y - 2), 4, 2, linewidth=1, edgecolor='black', facecolor='#c8e6c9')
    ax.add_patch(rect_ppo)
    ax.text(7.5, y - 0.5, 'PPO (AI)', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(7.5, y - 1.0, 'LB_ALGO="AI"', ha='center', va='center', fontsize=6)
    ax.text(7.5, y - 1.4, 'Collect inference_log', ha='center', va='center', fontsize=6)
    
    # Arrows from Step 3 to WRR and PPO
    ax.annotate('', xy=(2.5, y), xytext=(5, y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.annotate('', xy=(7.5, y), xytext=(5, y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    y -= 2 + step_gap
    
    # Arrow down
    ax.annotate('', xy=(5, y + step_h), xytext=(5, y + step_gap),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Step 4
    rect = patches.Rectangle((1, y), 8, step_h, linewidth=1, edgecolor='black', facecolor='#eeeeee')
    ax.add_patch(rect)
    ax.text(5, y + 1.1, 'Step 4: Summarize', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, y + 0.4, 'mean +/- std, 95% CI', ha='center', va='center', fontsize=7)
    y -= step_h + step_gap
    
    # Arrow
    ax.annotate('', xy=(5, y + step_h), xytext=(5, y + step_gap),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Step 5
    rect = patches.Rectangle((1, y), 8, step_h, linewidth=1, edgecolor='black', facecolor='#eeeeee')
    ax.add_patch(rect)
    ax.text(5, y + 1.1, 'Step 5: Compare', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, y + 0.4, 'Paired setting, winner', ha='center', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig11_benchmark_workflow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/fig11_benchmark_workflow.png")

if __name__ == '__main__':
    main()
