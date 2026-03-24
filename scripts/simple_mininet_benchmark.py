#!/usr/bin/env python3
"""
Simple Mininet Benchmark - No Controller Required
===================================================

This script runs a simple Mininet benchmark without requiring
an external SDN controller. It uses basic Linux networking.

Usage:
    python3 simple_mininet_benchmark.py --policy ppo
    python3 simple_mininet_benchmark.py --policy wrr
"""

import os
import sys
import time
import json
import argparse
import subprocess
import re
from datetime import datetime
from pathlib import Path

# Mininet imports
try:
    from mininet.net import Mininet
    from mininet.node import OVSSwitch, Host
    from mininet.link import TCLink
    from mininet.topo import Topo
    from mininet.log import setLogLevel, info
    MININET_AVAILABLE = True
except ImportError:
    MININET_AVAILABLE = False
    print("ERROR: Mininet not available")
    sys.exit(1)

# PPO imports
try:
    from stable_baselines3 import PPO
    import numpy as np
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False

# Configuration
BACKENDS = [
    {'name': 'h5', 'capacity_mbps': 10},
    {'name': 'h7', 'capacity_mbps': 50},
    {'name': 'h8', 'capacity_mbps': 100},
]

CAPACITIES = np.array([10.0, 50.0, 100.0])


class SimpleTopo(Topo):
    """Simple topology with 1 client and 3 servers."""
    
    def build(self):
        # Single switch
        s1 = self.addSwitch('s1')
        
        # Client
        client = self.addHost('h1', ip='10.0.0.1')
        
        # Servers with different bandwidth limits
        h5 = self.addHost('h5', ip='10.0.0.5')
        h7 = self.addHost('h7', ip='10.0.0.7')
        h8 = self.addHost('h8', ip='10.0.0.8')
        
        # Links with bandwidth limits
        self.addLink(client, s1, bw=100)  # Client link
        
        # Server links with capacity limits
        self.addLink(s1, h5, bw=10)   # 10 Mbps server
        self.addLink(s1, h7, bw=50)   # 50 Mbps server
        self.addLink(s1, h8, bw=100)  # 100 Mbps server


class SimpleMininetBenchmark:
    """Simple Mininet benchmark."""
    
    def __init__(self, policy='ppo', duration=30, traffic_rate='5M'):
        self.policy = policy
        self.duration = duration
        self.traffic_rate = traffic_rate
        self.net = None
        self.results = {
            'latencies': [],
            'throughputs': [],
            'packet_losses': [],
            'action_distribution': [0, 0, 0]
        }
        
        # Load PPO model if needed
        self.ppo_model = None
        if policy == 'ppo' and PPO_AVAILABLE:
            model_path = 'ai_model/checkpoints/ppo_realistic_final.zip'
            if os.path.exists(model_path):
                self.ppo_model = PPO.load(model_path)
                print(f"Loaded PPO model from {model_path}")
    
    def start_network(self):
        """Start Mininet network."""
        print("\n" + "=" * 60)
        print("STARTING MININET NETWORK")
        print("=" * 60)
        
        topo = SimpleTopo()
        self.net = Mininet(
            topo=topo,
            switch=OVSSwitch,
            link=TCLink,
            autoSetMacs=True,
            autoStaticArp=True
        )
        
        self.net.start()
        
        print("\nNetwork started:")
        print(f"  Hosts: {[h.name for h in self.net.hosts]}")
        print(f"  Switches: {[s.name for s in self.net.switches]}")
        
        # Wait for network to stabilize
        time.sleep(3)
        
        return self.net
    
    def stop_network(self):
        """Stop Mininet network."""
        if self.net:
            self.net.stop()
            print("\nNetwork stopped.")
    
    def measure_latency(self, client, server_ip, count=5):
        """Measure latency using ping."""
        cmd = f"ping -c {count} -i 0.1 {server_ip}"
        result = client.cmd(cmd)
        
        # Parse latency
        match = re.search(r'rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)', result)
        if match:
            return {
                'min': float(match.group(1)),
                'avg': float(match.group(2)),
                'max': float(match.group(3))
            }
        return None
    
    def run_iperf(self, client, server, duration=5, rate='5M'):
        """Run iperf test."""
        # Start iperf server
        server.cmd('pkill iperf')
        server.cmd(f'iperf -s -p 5001 &')
        time.sleep(1)
        
        # Run iperf client
        server_ip = server.IP()
        cmd = f"iperf -c {server_ip} -t {duration} -b {rate} -p 5001"
        result = client.cmd(cmd)
        
        # Parse throughput
        match = re.search(r'(\d+\.\d+)\s*Mbits/sec', result)
        if match:
            return float(match.group(1))
        return 0
    
    def run_benchmark(self):
        """Run the benchmark."""
        try:
            self.start_network()
            
            # Get hosts
            client = self.net.get('h1')
            servers = [
                self.net.get('h5'),
                self.net.get('h7'),
                self.net.get('h8')
            ]
            
            # WRR weights
            weights = np.array([1.0, 5.0, 10.0])
            weights = weights / weights.sum()
            
            print(f"\n{'=' * 60}")
            print(f"RUNNING {self.policy.upper()} BENCHMARK")
            print(f"{'=' * 60}")
            print(f"Duration: {self.duration} iterations")
            print(f"Traffic rate: {self.traffic_rate}")
            
            for i in range(self.duration):
                # Select server based on policy
                if self.policy == 'ppo':
                    # PPO decision
                    obs = np.zeros(14, dtype=np.float32)
                    obs[:3] = np.random.rand(3) * 0.3
                    obs[6] = 0.3
                    obs[8:11] = CAPACITIES / 150.0
                    
                    action, _ = self.ppo_model.predict(obs, deterministic=True)
                    if isinstance(action, np.ndarray):
                        action = int(action[0]) if action.ndim > 0 else int(action)
                else:
                    # WRR decision
                    r = np.random.random()
                    if r < weights[0]:
                        action = 0
                    elif r < weights[0] + weights[1]:
                        action = 1
                    else:
                        action = 2
                
                self.results['action_distribution'][action] += 1
                
                # Select server
                server = servers[action]
                server_name = server.name
                server_ip = server.IP()
                
                print(f"\n[{i+1}/{self.duration}] {self.policy.upper()} -> {server_name} (action={action})")
                
                # Measure latency
                latency = self.measure_latency(client, server_ip, count=3)
                if latency:
                    self.results['latencies'].append(latency['avg'])
                    print(f"  Latency: {latency['avg']:.2f} ms (min={latency['min']:.2f}, max={latency['max']:.2f})")
                
                # Run iperf
                throughput = self.run_iperf(client, server, duration=2, rate=self.traffic_rate)
                if throughput > 0:
                    self.results['throughputs'].append(throughput)
                    print(f"  Throughput: {throughput:.2f} Mbps")
                
                time.sleep(0.5)
            
            # Print results
            self.print_results()
            
        finally:
            self.stop_network()
    
    def print_results(self):
        """Print benchmark results."""
        print("\n" + "=" * 60)
        print(f"RESULTS - {self.policy.upper()}")
        print("=" * 60)
        
        if self.results['latencies']:
            print(f"\nLatency:")
            print(f"  Mean: {np.mean(self.results['latencies']):.2f} ms")
            print(f"  Std: {np.std(self.results['latencies']):.2f} ms")
            print(f"  Min: {np.min(self.results['latencies']):.2f} ms")
            print(f"  Max: {np.max(self.results['latencies']):.2f} ms")
        
        if self.results['throughputs']:
            print(f"\nThroughput:")
            print(f"  Mean: {np.mean(self.results['throughputs']):.2f} Mbps")
            print(f"  Std: {np.std(self.results['throughputs']):.2f} Mbps")
            print(f"  Min: {np.min(self.results['throughputs']):.2f} Mbps")
            print(f"  Max: {np.max(self.results['throughputs']):.2f} Mbps")
        
        total_actions = sum(self.results['action_distribution'])
        if total_actions > 0:
            print(f"\nAction Distribution:")
            for i, count in enumerate(self.results['action_distribution']):
                server_name = BACKENDS[i]['name']
                capacity = BACKENDS[i]['capacity_mbps']
                pct = count / total_actions * 100
                print(f"  {server_name} ({capacity} Mbps): {pct:.1f}%")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'ai_model/real_mininet_{self.policy}_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump({
                'policy': self.policy,
                'duration': self.duration,
                'traffic_rate': self.traffic_rate,
                'results': {
                    'latencies': self.results['latencies'],
                    'throughputs': self.results['throughputs'],
                    'action_distribution': self.results['action_distribution']
                },
                'summary': {
                    'mean_latency': float(np.mean(self.results['latencies'])) if self.results['latencies'] else 0,
                    'mean_throughput': float(np.mean(self.results['throughputs'])) if self.results['throughputs'] else 0,
                    'action_dist_pct': [c / total_actions * 100 for c in self.results['action_distribution']] if total_actions > 0 else [0, 0, 0]
                }
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Simple Mininet Benchmark')
    parser.add_argument('--policy', type=str, default='ppo', choices=['ppo', 'wrr'])
    parser.add_argument('--duration', type=int, default=20)
    parser.add_argument('--rate', type=str, default='5M')
    
    args = parser.parse_args()
    
    # Check root
    if os.getuid() != 0:
        print("ERROR: Run with sudo")
        print("  sudo python3 simple_mininet_benchmark.py --policy ppo")
        sys.exit(1)
    
    benchmark = SimpleMininetBenchmark(
        policy=args.policy,
        duration=args.duration,
        traffic_rate=args.rate
    )
    
    benchmark.run_benchmark()


if __name__ == '__main__':
    setLogLevel('info')
    main()