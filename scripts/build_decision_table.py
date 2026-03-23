import os
import json
import glob

res_dir = "stats/benchmark_final"
json_files = glob.glob(os.path.join(res_dir, "*.json"))

data = {}
scenarios = ["golden_hour", "video_conference", "hardware_degradation", "low_rate_dos"]

models = [
    "RR", "WRR",
    "CQL_BEST_SAMPLED", "CQL_BEST_ARGMAX",
    "CQL_EPOCH40_SAMPLED", "CQL_EPOCH40_ARGMAX",
    "CQL_EPOCH60_SAMPLED", "CQL_EPOCH60_ARGMAX"
]

for f in json_files:
    basename = os.path.basename(f)
    name_scene = basename.replace("_metrics.json", "")
    for s in scenarios:
        if name_scene.endswith(s):
            m_name = name_scene[:-(len(s)+1)]
            if m_name not in data:
                data[m_name] = {}
            with open(f, 'r') as fp:
                try:
                    data[m_name][s] = json.load(fp)
                except Exception as e:
                    print(f"Lỗi đọc file {f}: {e}")

print("### Bảng Xếp Hạng Tổng Hợp (Trung bình 4 Kịch bản)")
print("| Mô hình | Thông lượng (MBps) | Tỷ lệ Nghẽn (%) | Mất gói (%) | Độ lệch tải CV (%) |")
print("|---|---|---|---|---|")

for m in models:
    if m not in data or len(data[m]) == 0:
        continue
    
    tp_sum, hp_sum, pl_sum, cv_sum, valid_count = 0, 0, 0, 0, 0
    
    for s in scenarios:
        if s in data[m]:
            m_data = data[m][s]
            tp_sum += m_data.get('throughput_MBps', 0)
            hp_sum += m_data.get('high_pct', 0)
            pl_sum += m_data.get('packet_loss_pct', 0)
            cv_sum += m_data.get('balance_cv', 0)
            valid_count += 1
            
    if valid_count > 0:
        tp_avg = tp_sum / valid_count
        hp_avg = hp_sum / valid_count
        pl_avg = pl_sum / valid_count
        cv_avg = cv_sum / valid_count
        # Highlight AI models
        name_str = f"**{m}**" if "CQL" in m else m
        print(f"| {name_str} | {tp_avg:.2f} | {hp_avg:.2f} | {pl_avg:.2f} | {cv_avg:.2f} |")

print("\n### Chi tiết từng Kịch bản")
for s in scenarios:
    print(f"\n#### Kịch bản: {s}")
    print("| Mô hình | Thông lượng | Nghẽn (%) | Mất gói (%) | CV (%) |")
    print("|---|---|---|---|---|")
    for m in models:
        if m in data and s in data[m]:
            metrics = data[m][s]
            name_str = f"**{m}**" if "CQL" in m else m
            print(f"| {name_str} | {metrics.get('throughput_MBps', 0):.2f} | {metrics.get('high_pct', 0):.2f} | {metrics.get('packet_loss_pct', 0):.2f} | {metrics.get('balance_cv', 0):.2f} |")
