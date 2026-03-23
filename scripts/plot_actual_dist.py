import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Cấu hình
RESULTS_DIR = "stats/benchmark_final"
MODEL = "CQL_BEST_SAMPLED"
SCENARIOS = ["golden_hour", "video_conference", "hardware_degradation", "low_rate_dos"]
BACKENDS = ["h5 (10M)", "h7 (50M)", "h8 (100M)"]
OUT_FILE = "stats/results/charts_presentation/05_actual_action_distribution.png"

def plot_actual_dist():
    data = {}
    for s in SCENARIOS:
        fpath = os.path.join(RESULTS_DIR, f"{MODEL}_{s}_metrics.json")
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                content = json.load(f)
                dist = content.get("backend_dist", {})
                # Normalize byte counts to percentages
                total = sum(dist.values()) + 1e-9
                data[s] = [dist.get("h5",0)/total*100, dist.get("h7",0)/total*100, dist.get("h8",0)/total*100]
        else:
            data[s] = [0, 0, 0]

    # Plotting
    scen_labels = ["Golden Hour", "Video Conference", "Hardware Degrad.", "Low-Rate DoS"]
    x = np.arange(len(scen_labels))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    h5_vals = [data[s][0] for s in SCENARIOS]
    h7_vals = [data[s][1] for s in SCENARIOS]
    h8_vals = [data[s][2] for s in SCENARIOS]
    
    ax.bar(x, h5_vals, width, label='h5 (10Mbps)', color='#e74c3c', alpha=0.8)
    ax.bar(x, h7_vals, width, bottom=h5_vals, label='h7 (50Mbps)', color='#f1c40f', alpha=0.8)
    ax.bar(x, h8_vals, width, bottom=np.array(h5_vals)+np.array(h7_vals), label='h8 (100Mbps)', color='#2ecc71', alpha=0.8)
    
    ax.set_ylabel('Traffic Distribution (%)')
    ax.set_title(f'Actual Load Distribution: {MODEL} (Live Benchmark)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scen_labels)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', ncol=3)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels
    for i in range(len(scen_labels)):
        # Just show the numbers for clarity
        ax.text(i, h5_vals[i]/2, f"{h5_vals[i]:.1f}%", ha='center', color='white', fontweight='bold')
        ax.text(i, h5_vals[i] + h7_vals[i]/2, f"{h7_vals[i]:.1f}%", ha='center', color='black', fontweight='bold')
        ax.text(i, h5_vals[i] + h7_vals[i] + h8_vals[i]/2, f"{h8_vals[i]:.1f}%", ha='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300)
    print(f"Realistic action distribution saved to {OUT_FILE}")

if __name__ == "__main__":
    plot_actual_dist()
