#!/bin/bash
# ================================================================
# NCKH SDN - REAL-WORLD BENCHMARK (Mininet + Ryu)
# So sánh RR vs WRR vs AI (TFT-AC)
# ================================================================

BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
NC='\033[0m'

RESULTS_DIR="/work/stats/benchmark_real_world"
mkdir -p $RESULTS_DIR

# Scenarios to test
SCENARIOS=("golden_hour.yml" "video_conference.yml" "normal.yml")
ALGOS=("RR" "WRR" "AI")

clear
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}   REAL-WORLD BENCHMARK: RR vs WRR vs AI (TFT-AC)            ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${CYAN}Scenarios: ${SCENARIOS[@]}${NC}"
echo -e "${CYAN}Algorithms: ${ALGOS[@]}${NC}"
echo ""

# Function to run benchmark for one algorithm
run_benchmark() {
    local algo=$1
    local scenario=$2
    local output_file="$RESULTS_DIR/${algo}_${scenario%.yml}.json"
    
    echo -e "${YELLOW}>>> Testing $algo with $scenario <<<${NC}"
    
    # Set environment
    export LB_ALGO=$algo
    export SCENARIO=$scenario
    
    # Kill any existing Ryu processes
    pkill -f ryu-manager 2>/dev/null || true
    sleep 2
    
    # Start Ryu controller in background
    ryu-manager /work/controller_stats.py --verbose > $RESULTS_DIR/ryu_${algo}.log 2>&1 &
    RYU_PID=$!
    sleep 3
    
    # Check if Ryu started
    if ! ps -p $RYU_PID > /dev/null; then
        echo -e "${RED}Failed to start Ryu controller${NC}"
        return 1
    fi
    
    # Run Mininet experiment
    python3 /work/run_lms_mininet.py 2>&1 | tee $RESULTS_DIR/mininet_${algo}.log
    
    # Collect results
    if [ -f "/work/stats/flow_stats.csv" ]; then
        python3 << EOF
import json
import csv
import os
from datetime import datetime

# Read flow stats
flow_stats = []
try:
    with open('/work/stats/flow_stats.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            flow_stats.append(row)
except:
    pass

# Read port stats
port_stats = []
try:
    with open('/work/stats/port_stats.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            port_stats.append(row)
except:
    pass

# Read inference log (for AI)
inference_log = []
try:
    with open('/work/stats/inference_log.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inference_log.append(row)
except:
    pass

# Calculate metrics
total_bytes = sum(int(s.get('byte_count', 0)) for s in flow_stats)
total_packets = sum(int(s.get('packet_count', 0)) for s in flow_stats)

# Per-server distribution (from port stats)
server_distribution = {}
for backend in [{'name': 'h5', 'dpid': 8, 'port': 2}, 
                 {'name': 'h7', 'dpid': 8, 'port': 4}, 
                 {'name': 'h8', 'dpid': 8, 'port': 5}]:
    key = (backend['dpid'], backend['port'])
    server_tx = sum(int(s.get('tx_bytes', 0)) for s in port_stats 
                    if int(s.get('datapath_id', 0)) == backend['dpid'] 
                    and int(s.get('port_no', 0)) == backend['port'])
    server_distribution[backend['name']] = server_tx

total_server_tx = sum(server_distribution.values())
distribution_pct = {k: v/total_server_tx if total_server_tx > 0 else 0 
                    for k, v in server_distribution.items()}

# AI inference stats
avg_inference_ms = 0
if inference_log:
    inference_times = [float(row.get('inference_ms', 0)) for row in inference_log]
    avg_inference_ms = sum(inference_times) / len(inference_times)

results = {
    'algorithm': '$algo',
    'scenario': '$scenario',
    'total_bytes': total_bytes,
    'total_packets': total_packets,
    'server_distribution': server_distribution,
    'distribution_pct': distribution_pct,
    'avg_inference_ms': avg_inference_ms,
    'num_flows': len(flow_stats),
    'num_inferences': len(inference_log),
    'timestamp': datetime.now().isoformat()
}

with open('$output_file', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to $output_file")
print(f"Total bytes: {total_bytes:,}")
print(f"Total packets: {total_packets:,}")
print(f"Distribution: h5={distribution_pct.get('h5', 0):.1%}, h7={distribution_pct.get('h7', 0):.1%}, h8={distribution_pct.get('h8', 0):.1%}")
if '$algo' == 'AI':
    print(f"Avg inference time: {avg_inference_ms:.2f}ms")
EOF
    fi
    
    # Cleanup
    pkill -f ryu-manager 2>/dev/null || true
    sleep 2
    
    echo -e "${GREEN}Completed $algo with $scenario${NC}"
    echo ""
}

# Main benchmark loop
for scenario in "${SCENARIES[@]}"; do
    for algo in "${ALGOS[@]}"; do
        run_benchmark $algo $scenario
    done
done

# Generate comparison report
echo -e "${CYAN}>>> Generating Comparison Report <<<${NC}"
python3 << 'EOF'
import json
import os
import glob

results_dir = "$RESULTS_DIR"
results_files = glob.glob(f"{results_dir}/*.json")

if not results_files:
    print("No results found!")
    exit(1)

# Load all results
all_results = {}
for f in results_files:
    with open(f, 'r') as file:
        data = json.load(file)
        key = f"{data['algorithm']}_{data['scenario']}"
        all_results[key] = data

# Print comparison table
print("\n" + "="*80)
print("BENCHMARK RESULTS COMPARISON")
print("="*80)
print(f"{'Algorithm':<10} {'Scenario':<20} {'Bytes':>15} {'Packets':>12} {'h5%':>8} {'h7%':>8} {'h8%':>8}")
print("-"*80)

for key in sorted(all_results.keys()):
    r = all_results[key]
    dist = r.get('distribution_pct', {})
    print(f"{r['algorithm']:<10} {r['scenario']:<20} {r['total_bytes']:>15,} {r['total_packets']:>12,} "
          f"{dist.get('h5', 0)*100:>7.1f}% {dist.get('h7', 0)*100:>7.1f}% {dist.get('h8', 0)*100:>7.1f}%")

print("="*80)

# Calculate Jain's Fairness Index for each algorithm
print("\nJain's Fairness Index (capacity-weighted):")
for algo in ["RR", "WRR", "AI"]:
    for scenario in ["golden_hour.yml", "video_conference.yml", "normal.yml"]:
        key = f"{algo}_{scenario[:-4]}" if scenario.endswith('.yml') else f"{algo}_{scenario}"
        if key in all_results:
            r = all_results[key]
            dist = r.get('distribution_pct', {})
            # Capacity ratios: h5=1, h7=5, h8=10 -> normalized: 0.0625, 0.3125, 0.625
            capacities = [0.0625, 0.3125, 0.625]
            actual = [dist.get('h5', 0), dist.get('h7', 0), dist.get('h8', 0)]
            # Fairness: how close actual distribution is to capacity ratio
            if sum(actual) > 0:
                fairness = sum(min(a/c, c/a) for a, c in zip(actual, capacities) if a > 0 and c > 0) / 3
                print(f"  {algo} {scenario}: {fairness:.3f}")

print("\n" + "="*80)
EOF

echo -e "\n${GREEN}=================================================================${NC}"
echo -e "${GREEN} BENCHMARK COMPLETED!                                          ${NC}"
echo -e "${GREEN}=================================================================${NC}"
echo -e "${CYAN}Results: $RESULTS_DIR${NC}"