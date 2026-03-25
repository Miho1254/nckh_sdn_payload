#!/usr/bin/env python3
import os
import sys
import re
import csv
import glob
import numpy as np

def parse_port_stats(csv_file):
    print(f"Parsing {csv_file}...")
    max_tx = {(8, 2): 0, (8, 4): 0, (8, 5): 0}
    min_tx = {(8, 2): -1, (8, 4): -1, (8, 5): -1}
    
    if not os.path.exists(csv_file):
        print(f"[!] Không tìm thấy {csv_file}")
        return None

    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 9:
                    continue
                if row[1] == 'datapath_id' or not row[1].isdigit():
                    continue
                    
                dpid = int(row[1])
                port = int(row[2])
                tx_bytes = int(row[8])
                
                key = (dpid, port)
                if key in max_tx:
                    if min_tx[key] == -1:
                        min_tx[key] = tx_bytes
                    if tx_bytes > max_tx[key]:
                        max_tx[key] = tx_bytes
    except Exception as e:
        print(f"Error parsing {csv_file}: {e}")
        return None

    delta_tx = {}
    for k in max_tx:
        delta = max_tx[k] - (min_tx[k] if min_tx[k] != -1 else 0)
        delta_tx[k] = delta
    print(f"Done parsing {csv_file}")
    return delta_tx

def parse_artillery_logs(log_files):
    latencies = []
    total_reqs = 0
    total_errors = 0
    total_200 = 0
    print(f"Parsing {len(log_files)} log files...")
    
    for lf in log_files:
        print(f" Reading {lf}...")
        try:
            with open(lf, 'r') as f:
                lines = f.readlines()
                
            in_summary = False
            for i, line in enumerate(lines):
                if "Summary report" in line:
                    in_summary = True
                    
                if in_summary:
                    if "http.requests:" in line:
                        m = re.search(r'([\d\.]+)$', line.strip())
                        if m: total_reqs += int(float(m.group(1)))
                        
                    if "http.codes.200:" in line:
                        m = re.search(r'([\d\.]+)$', line.strip())
                        if m: total_200 += int(float(m.group(1)))
                        
                    if "errors.ETIMEDOUT:" in line or "errors.ECONNRESET:" in line:
                        m = re.search(r'([\d\.]+)$', line.strip())
                        if m: total_errors += int(float(m.group(1)))
                        
                    if "http.response_time:" in line:
                        for j in range(i+1, min(i+15, len(lines))):
                            if "p99:" in lines[j]:
                                m = re.search(r'([\d\.]+)$', lines[j].strip())
                                if m:
                                    latencies.append(float(m.group(1)))
                                    break
                            if "http" in lines[j] and not lines[j].startswith(" "):
                                break
        except Exception as e:
            print("Error", e)
            
    avg_p99 = np.mean(latencies) if latencies else 0.0
    print(f"Done parsing logs. Reqs: {total_reqs}, p99: {avg_p99}")
    return {
        'p99_latency_ms': avg_p99,
        'successes': total_200,
        'errors': total_errors,
        'total_requests': total_reqs
    }

def print_report(policy, port_data, art_data):
    print(f"\n{'='*50}")
    print(f" KẾT QUẢ: {policy.upper()}")
    print(f"{'='*50}")
    
    if port_data:
        total_bytes = sum(port_data.values())
        if total_bytes > 0:
            h5_pct = port_data[(8, 2)] / total_bytes * 100
            h7_pct = port_data[(8, 4)] / total_bytes * 100
            h8_pct = port_data[(8, 5)] / total_bytes * 100
            
            print(f"[+] Phân phối tải (Dựa trên số Byte truyền qua Switch):")
            print(f"    - h5 (10 Mbps) : {port_data[(8, 2)] / 1024 / 1024:.2f} MB ({h5_pct:.1f}%)")
            print(f"    - h7 (50 Mbps) : {port_data[(8, 4)] / 1024 / 1024:.2f} MB ({h7_pct:.1f}%)")
            print(f"    - h8 (100 Mbps): {port_data[(8, 5)] / 1024 / 1024:.2f} MB ({h8_pct:.1f}%)")
            print(f"    - Tổng Server TX: {total_bytes / 1024 / 1024:.2f} MB")
            
            target_h5, target_h7, target_h8 = 6.25, 31.25, 62.5
            mae = (abs(h5_pct - target_h5) + abs(h7_pct - target_h7) + abs(h8_pct - target_h8)) / 3
            print(f"    - Mean Absolute Error (so với capacity lý tưởng): {mae:.2f}%")
        else:
            print("[!] Không có dữ liệu truyền tải Switch.")
    else:
        print("[!] Không có file port_stats.csv")
        
    if art_data and art_data['total_requests'] > 0:
        print(f"\n[+] Chất lượng Dịch vụ (Mức Ứng dụng - Artillery):")
        print(f"    - Tổng Requests    : {art_data['total_requests']}")
        print(f"    - Thành công (200) : {art_data['successes']}")
        print(f"    - Lỗi (Timeouts/RST): {art_data['errors']}")
        print(f"    - Độ trễ P99       : {art_data['p99_latency_ms']:.2f} ms")
        
        error_rate = art_data['errors'] / art_data['total_requests'] * 100
        print(f"    - Tỉ lệ lỗi (Loss) : {error_rate:.2f}%")
    else:
        print("\n[!] Không có dữ liệu Artillery logs.")


def main():
    out_dir = sys.argv[1]
    
    print("Starting WRR evaluation")
    wrr_dir = os.path.join(out_dir, 'wrr')
    wrr_port = parse_port_stats(os.path.join(wrr_dir, 'port_stats.csv'))
    wrr_art = parse_artillery_logs(glob.glob(os.path.join(wrr_dir, 'h*_stress.log')))
    print_report("WRR (Weighted Round Robin)", wrr_port, wrr_art)
    
    print("Starting PPO evaluation")
    ppo_dir = os.path.join(out_dir, 'ppo')
    ppo_port = parse_port_stats(os.path.join(ppo_dir, 'port_stats.csv'))
    ppo_art = parse_artillery_logs(glob.glob(os.path.join(ppo_dir, 'h*_stress.log')))
    print_report("PPO (AI Policy)", ppo_port, ppo_art)
    
    print("\n" + "="*50)
    print(" BẢNG SO SÁNH CHUNG (PPO vs WRR TRÊN MININET)")
    print("="*50)
    
    if wrr_art['p99_latency_ms'] > 0 and ppo_art['p99_latency_ms'] > 0:
        lat_diff = (ppo_art['p99_latency_ms'] - wrr_art['p99_latency_ms']) / wrr_art['p99_latency_ms'] * 100
        print(f"[>] Độ trễ P99   : PPO {'giảm' if lat_diff < 0 else 'tăng'} {abs(lat_diff):.1f}% so với WRR")
        
    if wrr_art['total_requests'] > 0 and ppo_art['total_requests'] > 0:
        wrr_err = wrr_art['errors'] / wrr_art['total_requests'] * 100
        ppo_err = ppo_art['errors'] / ppo_art['total_requests'] * 100
        err_diff = ppo_err - wrr_err
        print(f"[>] Tỉ lệ Lỗi    : WRR = {wrr_err:.2f}%, PPO = {ppo_err:.2f}% (Chênh lệch: {err_diff:+.2f}%)")
        
        wrr_tpt = wrr_port[(8, 2)] + wrr_port[(8, 4)] + wrr_port[(8, 5)]
        ppo_tpt = ppo_port[(8, 2)] + ppo_port[(8, 4)] + ppo_port[(8, 5)]
        if wrr_tpt > 0:
            tpt_diff = (ppo_tpt - wrr_tpt) / wrr_tpt * 100
            print(f"[>] Throughput   : PPO {'tăng' if tpt_diff > 0 else 'giảm'} {abs(tpt_diff):.1f}% so với WRR")

if __name__ == '__main__':
    main()
