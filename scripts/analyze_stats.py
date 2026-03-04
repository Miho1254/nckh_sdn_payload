"""Phan tich thong ke AI vs RR vs WRR tu data thuc."""
import csv, os, sys, argparse, re
from collections import Counter, defaultdict

REPORT_FILE = 'stats/results/final_report.txt'

# Hàm print_log in ra màn hình và xuất file txt sạch (không có mã màu)
def print_log(*args, **kwargs):
    text = " ".join(map(str, args))
    print(text, **kwargs)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_text = ansi_escape.sub('', text)
    try:
        with open(REPORT_FILE, 'a', encoding='utf-8') as f:
            f.write(clean_text + '\n')
    except:
        pass

RESULTS_DIR = 'stats/results'
ALGOS = ['RR', 'WRR', 'AI', 'COLLECT']
SCENES = ['flash_crowd', 'predictable_ramping', 'targeted_congestion', 'gradual_shift']

# ANSI Colors
C_BLUE = '\033[1;34m'
C_GREEN = '\033[1;32m'
C_YELLOW = '\033[1;33m'
C_RED = '\033[1;31m'
C_CYAN = '\033[1;36m'
C_BOLD = '\033[1m'
C_NC = '\033[0m'

def analyze_flow_stats(csv_path):
    if not os.path.exists(csv_path):
        return None
    
    # 1. Thu thập flows để tính delta (Tránh cộng dồn lặp)
    # Key: (datapath_id, priority, match_fields...)
    flows_data = {} # (dpid, priority, in_port, eth_src, eth_dst, ipv4_src, ipv4_dst) -> [bytes]
    
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 12: continue
            
            dpid = row[1]
            priority = row[3]
            in_port = row[4]
            eth_src = row[5]
            eth_dst = row[6]
            
            # Tự động xác định index byte_count dựa trên số cột
            if len(row) >= 14:
                # Schema 14 cột có IP
                ipv4_src = row[7]
                ipv4_dst = row[8]
                byte_idx = 10
                label_idx = 13
            else:
                # Schema 12/13 cột thiếu IP
                ipv4_src = ''
                ipv4_dst = ''
                byte_idx = 8
                label_idx = 11
            
            try:
                bytes_val = int(row[byte_idx])
                label = row[label_idx].strip()
            except:
                continue
                
            key = (dpid, priority, in_port, eth_src, eth_dst, ipv4_src, ipv4_dst)
            if key not in flows_data:
                flows_data[key] = {'bytes': [], 'labels': []}
            flows_data[key]['bytes'].append(bytes_val)
            flows_data[key]['labels'].append(label)

    if not flows_data:
        return None

    # Tính delta thực tế của từng flow
    total_bytes = 0
    total_flows = len(flows_data)
    high_count = 0
    
    for key, data in flows_data.items():
        if data['bytes']:
            # Delta = Max - Min
            delta = max(data['bytes']) - min(data['bytes'])
            total_bytes += delta
        # Nếu flow từng bị gán nhãn HIGH, ta coi nó là kẹt (hoặc lấy nhãn cuối)
        if 'HIGH' in data['labels']:
            high_count += 1

    # 2. Thu thập dữ liệu từ port_stats.csv (Nguồn tin cậy nhất cho traffic thực tế)
    port_csv_path = csv_path.replace('flow_stats.csv', 'port_stats.csv')
    backend_dist = Counter({'h5': 0, 'h7': 0, 'h8': 0})
    actual_total_bytes = 0
    throughput_MBps = 0
    packet_loss_pct = 0
    
    if os.path.exists(port_csv_path):
        # Port metrics map: (dpid, port_no) -> [(rx_p, rx_b, rx_e, rx_d, tx_p, tx_b, tx_e, tx_d, dur)]
        sw_port_data = defaultdict(lambda: defaultdict(list))
        
        with open(port_csv_path) as f:
            p_reader = csv.reader(f)
            next(p_reader, None) # Header
            for p_row in p_reader:
                if len(p_row) < 13: continue
                dpid_p = p_row[1]
                p_no = p_row[2]
                rx_p = int(p_row[3])
                rx_b = int(p_row[4])
                rx_e = int(p_row[5])
                rx_d = int(p_row[6])
                tx_p = int(p_row[7])
                tx_b = int(p_row[8])
                tx_e = int(p_row[9])
                tx_d = int(p_row[10])
                dur  = float(p_row[12])
                sw_port_data[dpid_p][p_no].append((rx_p, rx_b, rx_e, rx_d, tx_p, tx_b, tx_e, tx_d, dur))
        
        def get_delta(d, p, idx):
            vals = [v[idx] for v in sw_port_data.get(d, {}).get(p, [])]
            return max(vals) - min(vals) if vals else 0
            
        def get_max(idx):
            vals = [v[idx] for d in sw_port_data.values() for p in d.values() for v in p]
            return max(vals) if vals else 0

        # Backend Traffic (What reached the targets)
        backend_dist['h5'] = get_delta('8', '2', 5) # tx_bytes
        backend_dist['h7'] = get_delta('8', '4', 5)
        backend_dist['h8'] = get_delta('8', '5', 5)
        
        # Total Real Traffic (What was sent by clients)
        for dpid_e in ['9', '10']:
            for p_e in ['2', '3', '4', '5']:
                actual_total_bytes += get_delta(dpid_e, p_e, 1) # rx_bytes

        # Packet Loss
        system_lost_packets = 0
        system_total_packets = 0
        for d in sw_port_data:
            for p in sw_port_data[d]:
                system_lost_packets += get_delta(d, p, 2) + get_delta(d, p, 3) + get_delta(d, p, 6) + get_delta(d, p, 7)
                system_total_packets += get_delta(d, p, 0) + get_delta(d, p, 4)
                
        if (system_total_packets + system_lost_packets) > 0:
            packet_loss_pct = (system_lost_packets / (system_total_packets + system_lost_packets)) * 100
            
        # Throughput
        max_duration = get_max(8)
        if max_duration > 0:
            throughput_MBps = (actual_total_bytes / 1e6) / max_duration

    if actual_total_bytes == 0:
        actual_total_bytes = total_bytes / 4 # Factor of path length

    # Tính CV theo Utilization (Fairness in Effort) thay vì Bytes (Absolute Load)
    # BW Capacities in Mbps (Tương đương 10, 50, 100) -> Chuyển đổi linh hoạt
    BW_CAPACITIES = {'h5': 10, 'h7': 50, 'h8': 100}
    utilization_dist = {}
    
    if max_duration > 0:
        for srv, bw in BW_CAPACITIES.items():
            # Tính throughput của server đó (Mbps)
            srv_mbps = (backend_dist[srv] * 8 / 1e6) / max_duration
            utilization_dist[srv] = (srv_mbps / bw) * 100 # % sử dụng đường truyền
    else:
        utilization_dist = {'h5': 0, 'h7': 0, 'h8': 0}

    # Tính CV dựa trên mức độ Utilization
    if sum(utilization_dist.values()) > 0:
        values = list(utilization_dist.values())
        mean_v = sum(values) / 3
        std_dev = (sum((v - mean_v)**2 for v in values) / 3)**0.5
        cv = (std_dev / mean_v * 100) if mean_v > 0 else 0
    else:
        cv = 0

    # AI Inference Time
    inference_csv_path = csv_path.replace('flow_stats.csv', 'inference_log.csv')
    avg_inference_ms = 0.0
    if os.path.exists(inference_csv_path):
        with open(inference_csv_path) as f:
            reader = csv.reader(f)
            next(reader, None)
            inf_times = []
            for r in reader:
                try:
                    inf_times.append(float(r[1]))
                except: pass
            if inf_times:
                avg_inference_ms = sum(inf_times) / len(inf_times)

    return {
        'total_flows': total_flows,
        'high_pct': high_count / total_flows * 100 if total_flows else 0,
        'total_bytes_MB': actual_total_bytes / 1e6,
        'throughput_MBps': throughput_MBps,
        'packet_loss_pct': packet_loss_pct,
        'inference_ms': avg_inference_ms,
        'backend_dist': dict(backend_dist),
        'utilization_dist': dict(utilization_dist),
        'balance_cv': cv,
    }

def print_grand_summary(all_results):
    """In bảng tổng hợp rút gọn cho báo cáo khoa học."""
    print_log(f"\n{C_BOLD}{C_BLUE}================================================================={C_NC}")
    print_log(f"{C_BOLD}{C_BLUE}  🏆 BẢNG TỔNG HỢP SO SÁNH CUỐI CÙNG (GRAND SUMMARY){C_NC}")
    print_log(f"{C_BOLD}{C_BLUE}================================================================={C_NC}")
    
    # Header: Scenarios | AI | RR | WRR
    algos = [a for a in ALGOS if any(a in r for r in all_results.values())]
    header = f"{C_BOLD}{'Scenario':<20}{C_NC}"
    for a in algos:
        header += f" | {C_BOLD}{a:>10}{C_NC}"
    print_log(header)
    print_log("-" * (20 + 13 * len(algos)))

    # Phần 1: Độ lệch CV% theo Công sức (Càng thấp càng tốt)
    print_log(f"{C_CYAN}[1] Độ lệch CV% (Tính theo Hiệu suất - Càng thấp = Càng Công Bằng){C_NC}")
    for scene in SCENES:
        row_res = all_results.get(scene, {})
        if not row_res: continue
        
        row = f"{scene:<20}"
        for a in algos:
            if a in row_res:
                cv = row_res[a]['balance_cv']
                is_best = cv == min(r['balance_cv'] for r in row_res.values())
                color = C_GREEN if is_best else C_NC
                indicator = "★" if is_best else " "
                row += f" | {color}{cv:>9.1f}%{indicator}{C_NC}"
            else:
                row += f" | {'-':>10}"
        print_log(row)
    
    print_log("-" * (20 + 13 * len(algos)))
    
    # Phần 2: % Nghẽn HIGH (Càng thấp càng tốt)
    print_log(f"{C_CYAN}[2] Tỷ lệ nghẽn (%) (Càng thấp = Càng thông suốt){C_NC}")
    for scene in SCENES:
        row_res = all_results.get(scene, {})
        if not row_res: continue
        
        row = f"{scene:<20}"
        for a in algos:
            if a in row_res:
                pct = row_res[a]['high_pct']
                is_best = pct == min(r['high_pct'] for r in row_res.values())
                color = C_GREEN if is_best else (C_RED if pct > 20 else C_NC)
                row += f" | {color}{pct:>10.1f}%{C_NC}"
            else:
                row += f" | {'-':>10}"
        print_log(row)

    print_log(f"{C_BOLD}{C_BLUE}================================================================={C_NC}")
    print_log(f"{C_BOLD}📍 ĐÁNH GIÁ CHUNG:{C_NC}")
    print_log(f"  - Thuật toán {C_GREEN}AI (TFT-DQN){C_NC} được kỳ vọng sẽ chiếm ưu thế (có nhiều ★ nhất)")
    print_log(f"    trong các kịch bản có tính biến động cao như {C_CYAN}flash_crowd{C_NC} và {C_CYAN}gradual_shift{C_NC}.")
    print_log(f"{C_BOLD}{C_BLUE}================================================================={C_NC}\n")

def print_report(scenes_to_run):
    all_scene_results = {}
    
    print_log(f"\n{C_BOLD}{C_BLUE}================================================================={C_NC}")
    print_log(f"{C_BOLD}{C_BLUE}  📊 BÁO CÁO PHÂN TÍCH CHI TIẾT THEO KỊCH BẢN{C_NC}")
    print_log(f"{C_BOLD}{C_BLUE}================================================================={C_NC}")

    rendered_any = False

    for scene in scenes_to_run:
        results = {}
        for algo in ALGOS:
            path = os.path.join(RESULTS_DIR, f'{algo}_{scene}', 'flow_stats.csv')
            r = analyze_flow_stats(path)
            if r:
                results[algo] = r
        
        if not results:
            continue
            
        rendered_any = True
        all_scene_results[scene] = results
        
        print_log(f"\n{C_CYAN}  Kịch bản (Scenario): {C_BOLD}{scene}{C_NC}")
        print_log(f"{C_CYAN}─────────────────────────────────────────────────────────────────{C_NC}")
        
        algos_found = list(results.keys())
        header = f"{'Metric':<25}"
        for a in algos_found:
            header += f" | {C_BOLD}{a:>10}{C_NC}"
        print_log(header)
        print_log("-" * 65)
        
        # Traffic
        row = f"{'Tổng Traffic (MB)':<25}"
        for a in algos_found:
            row += f" | {C_YELLOW}{results[a]['total_bytes_MB']:>9.1f}M{C_NC}"
        print_log(row)

        # Throughput
        row = f"{'Thông lượng (MB/s)':<25}"
        for a in algos_found:
            row += f" | {C_YELLOW}{results[a]['throughput_MBps']:>9.2f}{C_NC}"
        print_log(row)
        
        # High Flow %
        row = f"{'% Nghẽn (HIGH)':<25}"
        for a in algos_found:
            pct = results[a]['high_pct']
            color = C_RED if pct > 20 else (C_YELLOW if pct > 5 else C_GREEN)
            row += f" | {color}{pct:>9.1f}%{C_NC}"
        print_log(row)

        # Packet Loss
        row = f"{'Tỷ lệ mất gói (%)':<25}"
        for a in algos_found:
            pls = results[a]['packet_loss_pct']
            color = C_RED if pls > 5 else (C_YELLOW if pls > 1 else C_GREEN)
            row += f" | {color}{pls:>9.2f}%{C_NC}"
        print_log(row)
        
        # Backend Traffic Breakdown + Utilization
        print_log(f"{C_BOLD}Phân bổ tải tới Backend (Thực tế vs Công suất):{C_NC}")
        for server in ['h5', 'h7', 'h8']:
            row_traf = f"  -> {server} (MB){'':<13}"
            row_util = f"     - Hiệu suất (Util) {'':<4}"
            for a in algos_found:
                val = results[a]['backend_dist'].get(server, 0) / 1e6
                util = results[a]['utilization_dist'].get(server, 0)
                row_traf += f" | {C_CYAN}{val:>9.1f}M{C_NC}"
                row_util += f" | {C_YELLOW}{util:>9.1f}%{C_NC}"
            print_log(row_traf)
            print_log(row_util)
            
        # Balance score
        row = f"{'Độ lệch CV% (Càng thấp)':<25}"
        for a in algos_found:
            cv = results[a]['balance_cv']
            is_best = cv == min(r['balance_cv'] for r in results.values())
            indicator = " ★" if is_best else "  "
            color = C_GREEN if is_best else C_NC
            row += f" | {color}{cv:>8.1f}%{indicator}{C_NC}"
        print_log(row)
        
        # AI Inference Time
        if 'AI' in algos_found:
            row = f"{'AI Inference Time (ms)':<25}"
            for a in algos_found:
                if a == 'AI':
                    inf = results[a]['inference_ms']
                    row += f" | {C_YELLOW}{inf:>9.1f}{C_NC}"
                else:
                    row += f" | {'-':>10}"
            print_log(row)

    if not rendered_any:
        print_log(f"  {C_YELLOW}(Chưa có dữ liệu stats/results/ cho kịch bản này){C_NC}")
    else:
        # Sau khi in chi tiết, in bảng tổng hợp nếu chạy nhiều kịch bản
        if len(scenes_to_run) > 1:
            print_grand_summary(all_scene_results)
        else:
            print_log(f"\n{C_BOLD}{C_BLUE}================================================================={C_NC}")
            print_log(f"{C_BOLD}📍 KẾT LUẬN & ĐÁNH GIÁ:{C_NC}")
            print_log(f"  - {C_BOLD}CV%{C_NC} (Hệ số phân tán Fair-Effort): Đánh giá độ lệch về % mức độ sử")
            print_log(f"    dụng băng thông. {C_GREEN}★{C_NC} Càng thấp = Hệ thống khai thác Máy tính tương xứng")
            print_log(f"    với Năng lực của nó (Ví dụ: Tất cả cùng chạy 70% công suất).")
            print_log(f"{C_BOLD}{C_BLUE}================================================================={C_NC}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze flow stats for Load Balancing.')
    parser.add_argument('--scenario', type=str, help='Specific scenario to report on (e.g. flash_crowd)', default=None)
    parser.add_argument('--all', action='store_true', help='Report all scenarios')
    args = parser.parse_args()

    # Mở file report ở chế độ overwrite vào mỗi lần bắt đầu xử lý
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    try:
        if os.path.exists(REPORT_FILE):
            os.remove(REPORT_FILE)
    except: pass

    if args.scenario:
        print_report([args.scenario])
    else:
        print_report(SCENES)
