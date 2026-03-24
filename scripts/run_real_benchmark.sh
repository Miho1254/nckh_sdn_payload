#!/bin/bash
# Run Real Mininet/Ryu Benchmark
# This script starts Ryu controller and runs Mininet benchmark

set -e

echo "============================================================"
echo "REAL MININET/RYU BENCHMARK"
echo "============================================================"

# Kill any existing processes
echo "Cleaning up existing processes..."
pkill -9 ryu-manager 2>/dev/null || true
pkill -9 mn 2>/dev/null || true
pkill -9 iperf 2>/dev/null || true
mn -c 2>/dev/null || true

sleep 2

# Start Ryu controller in background
echo "Starting Ryu controller..."
ryu-manager --observe-links controller_stats.py &
RYU_PID=$!
echo "Ryu PID: $RYU_PID"

# Wait for Ryu to start
sleep 5

# Check if Ryu is running
if ! kill -0 $RYU_PID 2>/dev/null; then
    echo "ERROR: Ryu controller failed to start"
    exit 1
fi

echo "Ryu controller started successfully"

# Run Mininet benchmark
echo ""
echo "============================================================"
echo "RUNNING MININET BENCHMARK"
echo "============================================================"

# Create Python script for Mininet
cat > /tmp/mininet_bench.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import os
import sys
import time
import json
import re
from datetime import datetime
from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel, info
import numpy as np

try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except:
    PPO_AVAILABLE = False

BACKENDS = [
    {'name': 'h5', 'capacity_mbps': 10},
    {'name': 'h7', 'capacity_mbps': 50},
    {'name': 'h8', 'capacity_mbps': 100},
]
CAPACITIES = np.array([10.0, 50.0, 100.0])

class FatTreeTopo(Topo):
    def build(self):
        # Core switches
        c1 = self.addSwitch('c1', dpid='0000000000000001')
        c2 = self.addSwitch('c2', dpid='0000000000000002')
        
        # Aggregation switches
        a1 = self.addSwitch('a1', dpid='0000000000000011')
        a2 = self.addSwitch('a2', dpid='0000000000000012')
        a3 = self.addSwitch('a3', dpid='0000000000000013')
        a4 = self.addSwitch('a4', dpid='0000000000000014')
        
        # Edge switches
        e1 = self.addSwitch('e1', dpid='0000000000000021')
        e2 = self.addSwitch('e2', dpid='0000000000000022')
        e3 = self.addSwitch('e3', dpid='0000000000000023')
        e4 = self.addSwitch('e4', dpid='0000000000000024')
        
        # Hosts
        h1 = self.addHost('h1', ip='10.0.0.1')
        h2 = self.addHost('h2', ip='10.0.0.2')
        h3 = self.addHost('h3', ip='10.0.0.3')
        h4 = self.addHost('h4', ip='10.0.0.4')
        h5 = self.addHost('h5', ip='10.0.0.5')
        h7 = self.addHost('h7', ip='10.0.0.7')
        h8 = self.addHost('h8', ip='10.0.0.8')
        
        # Links
        self.addLink(c1, a1, bw=100)
        self.addLink(c1, a2, bw=100)
        self.addLink(c1, a3, bw=100)
        self.addLink(c1, a4, bw=100)
        self.addLink(c2, a1, bw=100)
        self.addLink(c2, a2, bw=100)
        self.addLink(c2, a3, bw=100)
        self.addLink(c2, a4, bw=100)
        
        self.addLink(a1, e1, bw=100)
        self.addLink(a1, e2, bw=100)
        self.addLink(a2, e1, bw=100)
        self.addLink(a2, e2, bw=100)
        self.addLink(a3, e3, bw=100)
        self.addLink(a3, e4, bw=100)
        self.addLink(a4, e3, bw=100)
        self.addLink(a4, e4, bw=100)
        
        self.addLink(e1, h1, bw=100)
        self.addLink(e1, h2, bw=100)
        self.addLink(e2, h3, bw=100)
        self.addLink(e2, h4, bw=100)
        self.addLink(e3, h5, bw=10)
        self.addLink(e3, h7, bw=50)
        self.addLink(e4, h8, bw=100)

def run_benchmark(policy='ppo', duration=15):
    print(f"\n{'=' * 60}")
    print(f"RUNNING {policy.upper()} BENCHMARK")
    print(f"{'=' * 60}")
    
    # Load PPO model
    ppo_model = None
    if policy == 'ppo' and PPO_AVAILABLE:
        model_path = 'ai_model/checkpoints/ppo_realistic_final.zip'
        if os.path.exists(model_path):
            ppo_model = PPO.load(model_path)
            print(f"Loaded PPO model from {model_path}")
    
    # Create network
    topo = FatTreeTopo()
    net = Mininet(topo=topo, controller=Controller, switch=OVSSwitch, link=TCLink)
    net.start()
    
    print("\nNetwork started")
    time.sleep(5)
    
    # Get hosts
    clients = [net.get('h1'), net.get('h2'), net.get('h3'), net.get('h4')]
    servers = [net.get('h5'), net.get('h7'), net.get('h8')]
    
    results = {
        'latencies': [],
        'throughputs': [],
        'action_distribution': [0, 0, 0]
    }
    
    # WRR weights
    weights = np.array([1.0, 5.0, 10.0])
    weights = weights / weights.sum()
    
    for i in range(duration):
        # Select server
        if policy == 'ppo' and ppo_model:
            obs = np.zeros(14, dtype=np.float32)
            obs[:3] = np.random.rand(3) * 0.3
            obs[6] = 0.3
            obs[8:11] = CAPACITIES / 150.0
            action, _ = ppo_model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action[0]) if action.ndim > 0 else int(action)
        else:
            r = np.random.random()
            if r < weights[0]:
                action = 0
            elif r < weights[0] + weights[1]:
                action = 1
            else:
                action = 2
        
        results['action_distribution'][action] += 1
        
        server = servers[action]
        client = clients[i % len(clients)]
        server_ip = server.IP()
        
        print(f"\n[{i+1}/{duration}] {policy.upper()} -> {server.name} (action={action})")
        
        # Ping test
        ping_result = client.cmd(f'ping -c 3 -i 0.1 {server_ip}')
        match = re.search(r'rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)', ping_result)
        if match:
            latency = float(match.group(2))
            results['latencies'].append(latency)
            print(f"  Latency: {latency:.2f} ms")
        
        # iperf test
        server.cmd('pkill iperf')
        server.cmd('iperf -s -p 5001 &')
        time.sleep(1)
        
        iperf_result = client.cmd(f'iperf -c {server_ip} -t 2 -b 5M -p 5001')
        match = re.search(r'(\d+\.\d+)\s*Mbits/sec', iperf_result)
        if match:
            throughput = float(match.group(1))
            results['throughputs'].append(throughput)
            print(f"  Throughput: {throughput:.2f} Mbps")
        
        time.sleep(0.5)
    
    # Print results
    print(f"\n{'=' * 60}")
    print(f"RESULTS - {policy.upper()}")
    print(f"{'=' * 60}")
    
    if results['latencies']:
        print(f"\nLatency:")
        print(f"  Mean: {np.mean(results['latencies']):.2f} ms")
        print(f"  Std: {np.std(results['latencies']):.2f} ms")
    
    if results['throughputs']:
        print(f"\nThroughput:")
        print(f"  Mean: {np.mean(results['throughputs']):.2f} Mbps")
        print(f"  Std: {np.std(results['throughputs']):.2f} Mbps")
    
    total = sum(results['action_distribution'])
    print(f"\nAction Distribution:")
    for i, count in enumerate(results['action_distribution']):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {BACKENDS[i]['name']} ({BACKENDS[i]['capacity_mbps']} Mbps): {pct:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'ai_model/real_mininet_{policy}_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'policy': policy,
            'results': results,
            'summary': {
                'mean_latency': float(np.mean(results['latencies'])) if results['latencies'] else 0,
                'mean_throughput': float(np.mean(results['throughputs'])) if results['throughputs'] else 0,
            }
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    net.stop()
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default='ppo', choices=['ppo', 'wrr'])
    parser.add_argument('--duration', type=int, default=15)
    args = parser.parse_args()
    
    setLogLevel('info')
    run_benchmark(args.policy, args.duration)
PYTHON_SCRIPT

# Run the benchmark
python3 /tmp/mininet_bench.py --policy ppo --duration 15

# Cleanup
echo ""
echo "============================================================"
echo "BENCHMARK COMPLETE"
echo "============================================================"
kill $RYU_PID 2>/dev/null || true
pkill -9 ryu-manager 2>/dev/null || true
mn -c 2>/dev/null || true

echo "Done!"