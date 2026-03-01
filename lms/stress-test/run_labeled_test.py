import time
import subprocess
import os
import sys

# Lấy cấu hình kịch bản từ môi trường
SCENARIO = os.environ.get('SCENARIO', 'artillery.yml')
SCENARIO_PATH = os.path.join(os.path.dirname(__file__), '../evaluation', SCENARIO) if SCENARIO != 'artillery.yml' else SCENARIO

LABEL_FILE = "/work/stats/current_label.txt"

def update_label(label):
    print(f"[{time.strftime('%H:%M:%S')}] Traffic Label: {label}")
    try:
        with open(LABEL_FILE, "w") as f:
            f.write(label)
    except Exception as e:
        pass

def main():
    print(f"🚀 Labeled Stress Test: {SCENARIO}")
    update_label("NORMAL")
    
    try:
        # Chạy Artillery con
        process = subprocess.Popen(
            ["artillery", "run", SCENARIO_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(f"Artillery started (PID: {process.pid})")
        
        # Đọc log realtime của Artillery để dán nhãn thông minh
        for line in process.stdout:
            print(line, end='') # In log ra terminal Mininet
            
            # Nếu thấy Artillery báo phase Burst (Mật độ 100+ req/s)
            lower_line = line.lower()
            if "burst" in lower_line or "peak" in lower_line or "rain" in lower_line or "congestion" in lower_line:
                update_label("HIGH")
                
            # Trở lại Normal nếu thấy Recovery hoặc Ramp Down
            if "recovery" in lower_line or "drizzle" in lower_line or "ramp down" in lower_line:
                update_label("NORMAL")
                
        exit_code = process.wait()
        update_label("NORMAL") # Đảm bảo reset khi xong
        print(f"Test finished with exit code {exit_code}")
        
    except FileNotFoundError:
        print("Error: 'artillery' command not found. Run 'npm install -g artillery' first.")

if __name__ == "__main__":
    main()
