#!/usr/bin/env python3
"""
Generate Synthetic Data với các tình huống:
1. Server Overload - một hoặc nhiều server có util > 0.95
2. Traffic Burst - byte_rate cao đột ngột
3. Hardware Degradation - server có capacity giảm
4. Low-rate DoS - traffic bất thường từ một nguồn

Mục tiêu: Tạo data đa dạng để đánh giá AI vs WRR chính xác hơn
"""

import numpy as np
import os
import sys
import json

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BACKENDS, CAPACITIES, CAPACITY_RATIOS

# Feature schema V3 (44 features)
# Group A: 7 features (global)
# Group B: 3 * num_servers features (per-server load)
# Group C: 7 * num_servers features (per-server risk)

NUM_SERVERS = 3
NUM_FEATURES = 44
SEQUENCE_LENGTH = 5


def generate_base_features():
    """Generate base features for one timestep."""
    features = np.zeros(NUM_FEATURES)
    
    # Group A: Global features (7)
    # 0: byte_rate - normalized 0-1
    features[0] = np.random.uniform(0.01, 0.5)
    # 1: packet_rate - normalized 0-1
    features[1] = np.random.uniform(0.01, 0.3)
    # 2: flow_count
    features[2] = np.random.randint(10, 100)
    # 3: avg_packet_size
    features[3] = np.random.uniform(64, 1500)
    # 4: protocol_tcp_ratio
    features[4] = np.random.uniform(0.6, 0.9)
    # 5: protocol_udp_ratio
    features[5] = np.random.uniform(0.05, 0.3)
    # 6: protocol_other_ratio
    features[6] = 1.0 - features[4] - features[5]
    
    # Group B: Per-server load features (3 * 3 = 9)
    group_b_start = 7
    for i in range(NUM_SERVERS):
        # Load ratio based on capacity
        base_load = CAPACITY_RATIOS[i] * np.random.uniform(0.5, 1.5)
        features[group_b_start + i * 3] = min(1.0, base_load)  # load_ratio
        features[group_b_start + i * 3 + 1] = np.random.uniform(0.01, 0.1)  # packet_rate
        features[group_b_start + i * 3 + 2] = np.random.uniform(100, 1000)  # connection_count
    
    # Group C: Per-server risk features (7 * 3 = 21)
    group_c_start = 7 + 3 * NUM_SERVERS  # 16
    for i in range(NUM_SERVERS):
        # Utilization based on capacity and load
        util = min(1.0, CAPACITY_RATIOS[i] * np.random.uniform(0.3, 0.8))
        features[group_c_start + i * 7] = util  # utilization
        features[group_c_start + i * 7 + 1] = np.random.uniform(0.1, 0.5)  # error_rate
        features[group_c_start + i * 7 + 2] = 1.0 - util  # headroom
        features[group_c_start + i * 7 + 3] = np.random.uniform(0.01, 0.1)  # packet_loss_rate
        features[group_c_start + i * 7 + 4] = np.random.uniform(1, 10)  # avg_latency_ms
        features[group_c_start + i * 7 + 5] = np.random.uniform(0, 100)  # queue_length
        features[group_c_start + i * 7 + 6] = np.random.uniform(0.01, 0.1)  # drop_rate
    
    return features


def generate_overload_scenario(server_idx=0, severity=0.95):
    """Generate features for server overload scenario."""
    features = generate_base_features()
    
    # Set high utilization for specified server
    group_c_start = 7 + 3 * NUM_SERVERS  # 16
    features[group_c_start + server_idx * 7] = severity  # utilization
    features[group_c_start + server_idx * 7 + 2] = 1.0 - severity  # headroom
    features[group_c_start + server_idx * 7 + 3] = np.random.uniform(0.1, 0.3)  # packet_loss_rate
    features[group_c_start + server_idx * 7 + 4] = np.random.uniform(50, 200)  # avg_latency_ms
    features[group_c_start + server_idx * 7 + 5] = np.random.uniform(500, 1000)  # queue_length
    features[group_c_start + server_idx * 7 + 6] = np.random.uniform(0.1, 0.3)  # drop_rate
    
    # High traffic
    features[0] = np.random.uniform(0.5, 1.0)  # byte_rate
    features[1] = np.random.uniform(0.3, 0.8)  # packet_rate
    
    return features


def generate_burst_scenario(intensity=0.8):
    """Generate features for traffic burst scenario."""
    features = generate_base_features()
    
    # High traffic burst
    features[0] = intensity  # byte_rate
    features[1] = intensity * 0.6  # packet_rate
    features[2] = np.random.randint(200, 500)  # flow_count
    features[3] = np.random.uniform(500, 1500)  # avg_packet_size
    
    # All servers have moderate load
    group_c_start = 7 + 3 * NUM_SERVERS
    for i in range(NUM_SERVERS):
        util = min(0.9, CAPACITY_RATIOS[i] * intensity * 1.5)
        features[group_c_start + i * 7] = util
        features[group_c_start + i * 7 + 2] = 1.0 - util
    
    return features


def generate_degradation_scenario(server_idx=1, degradation_factor=0.5):
    """Generate features for hardware degradation scenario."""
    features = generate_base_features()
    
    # Degraded server has high utilization even with low load
    group_c_start = 7 + 3 * NUM_SERVERS
    degraded_util = min(1.0, CAPACITY_RATIOS[server_idx] * 2.0 * degradation_factor)
    features[group_c_start + server_idx * 7] = degraded_util
    features[group_c_start + server_idx * 7 + 1] = np.random.uniform(0.1, 0.3)  # error_rate
    features[group_c_start + server_idx * 7 + 2] = max(0.0, 1.0 - degraded_util)
    features[group_c_start + server_idx * 7 + 4] = np.random.uniform(100, 500)  # avg_latency_ms
    
    # Other servers have normal load
    for i in range(NUM_SERVERS):
        if i != server_idx:
            util = CAPACITY_RATIOS[i] * np.random.uniform(0.3, 0.6)
            features[group_c_start + i * 7] = util
            features[group_c_start + i * 7 + 2] = 1.0 - util
    
    return features


def generate_dos_scenario(target_server=0, intensity=0.7):
    """Generate features for low-rate DoS attack scenario."""
    features = generate_base_features()
    
    # Targeted server shows signs of stress
    group_c_start = 7 + 3 * NUM_SERVERS
    features[group_c_start + target_server * 7] = intensity  # utilization
    features[group_c_start + target_server * 7 + 1] = np.random.uniform(0.2, 0.5)  # error_rate
    features[group_c_start + target_server * 7 + 3] = np.random.uniform(0.2, 0.5)  # packet_loss_rate
    features[group_c_start + target_server * 7 + 5] = np.random.uniform(800, 1500)  # queue_length
    
    # Abnormal traffic pattern
    features[0] = np.random.uniform(0.3, 0.6)  # byte_rate (not too high)
    features[1] = np.random.uniform(0.4, 0.8)  # packet_rate (high packet count)
    features[2] = np.random.randint(50, 200)  # flow_count
    features[6] = np.random.uniform(0.3, 0.6)  # protocol_other_ratio (suspicious)
    
    return features


def generate_sequence(scenario='normal', **kwargs):
    """Generate a sequence of 5 timesteps."""
    sequence = []
    
    for t in range(SEQUENCE_LENGTH):
        if scenario == 'normal':
            features = generate_base_features()
        elif scenario == 'overload':
            # Gradually increasing overload
            severity = 0.8 + t * 0.05
            features = generate_overload_scenario(
                server_idx=kwargs.get('server_idx', 0),
                severity=min(1.0, severity)
            )
        elif scenario == 'burst':
            # Burst intensity varies
            intensity = kwargs.get('intensity', 0.8) * (0.8 + t * 0.05)
            features = generate_burst_scenario(intensity=min(1.0, intensity))
        elif scenario == 'degradation':
            features = generate_degradation_scenario(
                server_idx=kwargs.get('server_idx', 1),
                degradation_factor=kwargs.get('degradation_factor', 0.5)
            )
        elif scenario == 'dos':
            features = generate_dos_scenario(
                target_server=kwargs.get('target_server', 0),
                intensity=kwargs.get('intensity', 0.7)
            )
        else:
            features = generate_base_features()
        
        sequence.append(features)
    
    return np.array(sequence)


def generate_dataset(
    num_normal=200,
    num_overload=50,
    num_burst=50,
    num_degradation=50,
    num_dos=50,
    output_dir='ai_model/processed_data'
):
    """Generate complete dataset with various scenarios."""
    
    X_list = []
    y_list = []
    
    print(f"Generating dataset:")
    print(f"  - Normal: {num_normal}")
    print(f"  - Overload: {num_overload}")
    print(f"  - Burst: {num_burst}")
    print(f"  - Degradation: {num_degradation}")
    print(f"  - DoS: {num_dos}")
    
    # Normal traffic
    for _ in range(num_normal):
        seq = generate_sequence('normal')
        X_list.append(seq)
        y_list.append(0)  # 0 = NORMAL
    
    # Overload scenarios
    for i in range(num_overload):
        server_idx = i % 3  # Rotate through servers
        severity = np.random.uniform(0.85, 1.0)
        seq = generate_sequence('overload', server_idx=server_idx, severity=severity)
        X_list.append(seq)
        y_list.append(1)  # 1 = HIGH
    
    # Burst scenarios
    for _ in range(num_burst):
        intensity = np.random.uniform(0.6, 1.0)
        seq = generate_sequence('burst', intensity=intensity)
        X_list.append(seq)
        y_list.append(1)  # 1 = HIGH
    
    # Degradation scenarios
    for i in range(num_degradation):
        server_idx = i % 3
        degradation_factor = np.random.uniform(0.3, 0.7)
        seq = generate_sequence('degradation', server_idx=server_idx, degradation_factor=degradation_factor)
        X_list.append(seq)
        y_list.append(1)  # 1 = HIGH
    
    # DoS scenarios
    for i in range(num_dos):
        target_server = i % 3
        intensity = np.random.uniform(0.5, 0.9)
        seq = generate_sequence('dos', target_server=target_server, intensity=intensity)
        X_list.append(seq)
        y_list.append(1)  # 1 = HIGH
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Create scenarios array before shuffling
    scenarios = []
    for _ in range(num_normal):
        scenarios.append('NORMAL')
    for _ in range(num_overload):
        scenarios.append('OVERLOAD')
    for _ in range(num_burst):
        scenarios.append('BURST')
    for _ in range(num_degradation):
        scenarios.append('DEGRADATION')
    for _ in range(num_dos):
        scenarios.append('DOS')
    scenarios = np.array(scenarios)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    scenarios = scenarios[indices]
    
    print(f"\nGenerated dataset: X={X.shape}, y={y.shape}")
    print(f"Labels: {np.bincount(y)}")
    print(f"Scenarios: {np.unique(scenarios, return_counts=True)}")
    
    # Analyze utilization distribution
    group_c_start = 7 + 3 * NUM_SERVERS
    for i in range(NUM_SERVERS):
        utils = X[:, -1, group_c_start + i * 7]
        print(f"Server {i} (h{5+i*2}): Util > 0.95: {(utils > 0.95).sum()} samples")
        print(f"  Util distribution: min={utils.min():.4f}, max={utils.max():.4f}, mean={utils.mean():.4f}")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    X_path = os.path.join(output_dir, 'X_v3_synthetic.npy')
    y_path = os.path.join(output_dir, 'y_v3_synthetic.npy')
    scen_path = os.path.join(output_dir, 'scenarios_v3.npy')
    meta_path = os.path.join(output_dir, 'feature_metadata.json')
    
    np.save(X_path, X)
    np.save(y_path, y)
    np.save(scen_path, scenarios)
    
    # Create metadata
    metadata = {
        'num_features': NUM_FEATURES,
        'sequence_length': SEQUENCE_LENGTH,
        'num_servers': NUM_SERVERS,
        'version': '3.0_synthetic',
        'scenarios': {
            'NORMAL': num_normal,
            'OVERLOAD': num_overload,
            'BURST': num_burst,
            'DEGRADATION': num_degradation,
            'DOS': num_dos
        },
        'feature_groups': {
            'global': {'start': 0, 'count': 7, 'description': 'Global traffic features'},
            'server_load': {'start': 7, 'count': 9, 'description': 'Per-server load features (3 servers x 3 features)'},
            'server_risk': {'start': 16, 'count': 21, 'description': 'Per-server risk features (3 servers x 7 features)'},
            'server_util': {'start': 16, 'count': 3, 'description': 'Per-server utilization (indices 0, 7, 14 in server_risk)'}
        },
        'servers': {
            'h5': {'capacity_ratio': 0.0625, 'capacity_mbps': 100},
            'h7': {'capacity_ratio': 0.3125, 'capacity_mbps': 500},
            'h8': {'capacity_ratio': 0.625, 'capacity_mbps': 1000}
        }
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved to:")
    print(f"  - {X_path}")
    print(f"  - {y_path}")
    print(f"  - {scen_path}")
    print(f"  - {meta_path}")
    
    return X, y


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Synthetic Data for SDN Load Balancing')
    parser.add_argument('--normal', type=int, default=200, help='Number of normal samples')
    parser.add_argument('--overload', type=int, default=50, help='Number of overload samples')
    parser.add_argument('--burst', type=int, default=50, help='Number of burst samples')
    parser.add_argument('--degradation', type=int, default=50, help='Number of degradation samples')
    parser.add_argument('--dos', type=int, default=50, help='Number of DoS samples')
    parser.add_argument('--output', type=str, default='ai_model/processed_data', help='Output directory')
    
    args = parser.parse_args()
    
    np.random.seed(42)  # For reproducibility
    
    X, y = generate_dataset(
        num_normal=args.normal,
        num_overload=args.overload,
        num_burst=args.burst,
        num_degradation=args.degradation,
        num_dos=args.dos,
        output_dir=args.output
    )