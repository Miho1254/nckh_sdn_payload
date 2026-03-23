#!/usr/bin/env python3
"""
Generate Synthetic Data V2 - Với đúng action labels (0, 1, 2)
Action = 0: chọn h5 (server yếu)
Action = 1: chọn h7 (server trung bình)
Action = 2: chọn h8 (server mạnh)

Mỗi scenario sẽ có action phù hợp:
- Normal: chọn theo capacity ratio (6%, 31%, 63%)
- Overload: chọn server có headroom cao nhất
- Burst: chọn server mạnh (h8)
- Degradation: tránh server bị degradation
- DoS: chọn server yếu để phân tán traffic
"""

import numpy as np
import os
import json

# Constants
SEQUENCE_LENGTH = 5
NUM_FEATURES = 44
NUM_ACTIONS = 3

# Server capacities
CAPACITIES = np.array([1, 5, 10])  # h5, h7, h8
CAPACITY_RATIOS = CAPACITIES / np.sum(CAPACITIES)


def generate_base_features():
    """Generate base feature vector (44 features)."""
    features = np.zeros(NUM_FEATURES)
    
    # Group A: Global features (7)
    features[0] = np.random.uniform(0.1, 0.4)  # byte_rate
    features[1] = np.random.uniform(0.1, 0.3)  # packet_rate
    features[2] = np.random.randint(10, 50)   # flow_count
    features[3] = np.random.uniform(0.4, 0.7)  # tcp_ratio
    features[4] = np.random.uniform(0.2, 0.4)  # udp_ratio
    features[5] = np.random.uniform(0.05, 0.2) # icmp_ratio
    features[6] = np.random.uniform(0.0, 0.1)  # protocol_other_ratio
    
    # Group B: Per-server features (3 servers x 3 features = 9)
    for i in range(3):
        base_idx = 7 + i * 3
        features[base_idx] = np.random.uniform(0.1, 0.5)    # server_load
        features[base_idx + 1] = np.random.uniform(0.05, 0.2)  # server_packet_rate
        features[base_idx + 2] = np.random.randint(5, 20)    # server_flow_count
    
    # Group C: Per-server detailed features (3 servers x 7 features = 21)
    for i in range(3):
        base_idx = 16 + i * 7
        features[base_idx] = np.random.uniform(0.1, 0.4)     # utilization
        features[base_idx + 1] = np.random.uniform(0.6, 0.9) # headroom
        features[base_idx + 2] = np.random.uniform(0.0, 0.1) # packet_loss
        features[base_idx + 3] = np.random.uniform(10, 50)  # latency_ms
        features[base_idx + 4] = np.random.uniform(5, 30)    # queue_length
        features[base_idx + 5] = np.random.uniform(0.0, 0.05) # error_rate
        features[base_idx + 6] = np.random.uniform(0.8, 1.0) # health_score
    
    # Group D: Risk features (7)
    features[37] = np.random.uniform(0.0, 0.3)  # congestion_risk
    features[38] = np.random.uniform(0.0, 0.1)  # overload_risk
    features[39] = np.random.randint(0, 3)     # regime (0=normal, 1=high, 2=burst)
    features[40] = np.random.uniform(0.0, 0.1)  # anomaly_score
    features[41] = np.random.uniform(0.0, 0.05) # ddos_risk
    features[42] = np.random.uniform(0.0, 0.1)  # degradation_risk
    features[43] = np.random.uniform(0.9, 1.0)  # network_health
    
    return features


def generate_overload_scenario(server_idx=0, severity=0.95):
    """Generate overload scenario for specific server."""
    features = generate_base_features()
    
    # High traffic
    features[0] = np.random.uniform(0.6, 0.9)  # byte_rate
    features[1] = np.random.uniform(0.5, 0.8)   # packet_rate
    
    # Target server overloaded
    base_idx = 16 + server_idx * 7
    features[base_idx] = severity  # utilization
    features[base_idx + 1] = 1.0 - severity  # headroom
    features[base_idx + 2] = np.random.uniform(0.1, 0.3)  # packet_loss
    features[base_idx + 4] = np.random.uniform(50, 100)  # queue_length
    
    # Risk features
    features[37] = 0.8  # congestion_risk
    features[38] = 0.6  # overload_risk
    features[39] = 1    # regime = HIGH
    
    return features


def generate_burst_scenario(intensity=0.8):
    """Generate burst traffic scenario."""
    features = generate_base_features()
    
    # Very high traffic
    features[0] = np.random.uniform(0.7, 1.0) * intensity  # byte_rate
    features[1] = np.random.uniform(0.6, 0.9) * intensity   # packet_rate
    features[2] = np.random.randint(100, 300)  # flow_count
    
    # All servers have moderate load
    for i in range(3):
        base_idx = 16 + i * 7
        features[base_idx] = np.random.uniform(0.4, 0.7)  # utilization
        features[base_idx + 1] = np.random.uniform(0.3, 0.6)  # headroom
    
    features[37] = 0.6  # congestion_risk
    features[39] = 2    # regime = BURST
    
    return features


def generate_degradation_scenario(server_idx=1, degradation_factor=0.5):
    """Generate server degradation scenario."""
    features = generate_base_features()
    
    # Target server degraded
    base_idx = 16 + server_idx * 7
    features[base_idx] = np.random.uniform(0.6, 0.9)  # high utilization
    features[base_idx + 1] = np.random.uniform(0.1, 0.4)  # low headroom
    features[base_idx + 5] = np.random.uniform(0.1, 0.3)  # high error_rate
    features[base_idx + 6] = degradation_factor  # low health_score
    
    features[42] = 0.7  # degradation_risk
    features[39] = 1    # regime = HIGH
    
    return features


def generate_dos_scenario(target_server=0, intensity=0.7):
    """Generate DoS attack scenario."""
    features = generate_base_features()
    
    # High packet rate, moderate byte rate (small packets)
    features[0] = np.random.uniform(0.3, 0.5)  # byte_rate (moderate)
    features[1] = np.random.uniform(0.7, 1.0)  # packet_rate (HIGH)
    features[2] = np.random.randint(200, 500)  # flow_count (very high)
    
    # Target server under attack
    base_idx = 16 + target_server * 7
    features[base_idx] = np.random.uniform(0.8, 1.0)  # utilization
    features[base_idx + 1] = np.random.uniform(0.0, 0.2)  # headroom
    features[base_idx + 2] = np.random.uniform(0.2, 0.5)  # packet_loss
    features[base_idx + 4] = np.random.uniform(100, 200)  # queue_length
    
    features[37] = 0.9  # congestion_risk
    features[40] = 0.8  # anomaly_score
    features[41] = 0.9  # ddos_risk
    features[39] = 1    # regime = HIGH
    
    return features


def get_optimal_action(features, scenario, server_idx=None):
    """
    Determine optimal action based on scenario and server state.
    
    Returns:
        action: 0 (h5), 1 (h7), or 2 (h8)
    """
    # Extract server utilizations
    server_utils = []
    server_headrooms = []
    for i in range(3):
        base_idx = 16 + i * 7
        server_utils.append(features[base_idx])
        server_headrooms.append(features[base_idx + 1])
    
    if scenario == 'normal':
        # Normal: probabilistic selection based on capacity ratio
        # But prefer server with best headroom
        scores = []
        for i in range(3):
            score = CAPACITY_RATIOS[i] * server_headrooms[i]
            scores.append(score)
        return np.argmax(scores)
    
    elif scenario == 'overload':
        # Overload: choose server with lowest utilization (most headroom)
        return np.argmax(server_headrooms)
    
    elif scenario == 'burst':
        # Burst: choose strongest server (h8) if headroom available
        if server_headrooms[2] > 0.3:
            return 2  # h8
        elif server_headrooms[1] > 0.3:
            return 1  # h7
        else:
            return 0  # h5
    
    elif scenario == 'degradation':
        # Degradation: avoid degraded server
        if server_idx is not None:
            # Choose server with best health
            health_scores = []
            for i in range(3):
                base_idx = 16 + i * 7
                health_scores.append(features[base_idx + 6])
            return np.argmax(health_scores)
        return 2  # default to h8
    
    elif scenario == 'dos':
        # DoS: distribute to weaker servers to avoid overwhelming strong one
        # Choose server with lowest utilization
        return np.argmin(server_utils)
    
    return 2  # default to h8


def generate_sequence(scenario='normal', **kwargs):
    """Generate a sequence of 5 timesteps with optimal action."""
    sequence = []
    
    for t in range(SEQUENCE_LENGTH):
        if scenario == 'normal':
            features = generate_base_features()
        elif scenario == 'overload':
            severity = 0.8 + t * 0.05
            features = generate_overload_scenario(
                server_idx=kwargs.get('server_idx', 0),
                severity=min(1.0, severity)
            )
        elif scenario == 'burst':
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
    
    # Get optimal action for this sequence
    # Use the last timestep to determine action
    last_features = sequence[-1]
    action = get_optimal_action(
        last_features, 
        scenario,
        server_idx=kwargs.get('server_idx', None)
    )
    
    return np.array(sequence), action


def generate_dataset(
    num_normal=200,
    num_overload=100,
    num_burst=100,
    num_degradation=100,
    num_dos=100,
    output_dir='ai_model/processed_data'
):
    """Generate complete dataset with various scenarios and ACTIONS."""
    
    X_list = []
    y_list = []
    scenarios_list = []
    
    print(f"Generating dataset with ACTION labels:")
    print(f"  - Normal: {num_normal}")
    print(f"  - Overload: {num_overload}")
    print(f"  - Burst: {num_burst}")
    print(f"  - Degradation: {num_degradation}")
    print(f"  - DoS: {num_dos}")
    
    # Normal traffic - diverse actions
    for _ in range(num_normal):
        seq, action = generate_sequence('normal')
        X_list.append(seq)
        y_list.append(action)
        scenarios_list.append('normal')
    
    # Overload scenarios - choose server with headroom
    for i in range(num_overload):
        server_idx = i % 3
        severity = np.random.uniform(0.85, 1.0)
        seq, action = generate_sequence('overload', server_idx=server_idx, severity=severity)
        X_list.append(seq)
        y_list.append(action)
        scenarios_list.append('overload')
    
    # Burst scenarios - choose strong server
    for _ in range(num_burst):
        intensity = np.random.uniform(0.6, 1.0)
        seq, action = generate_sequence('burst', intensity=intensity)
        X_list.append(seq)
        y_list.append(action)
        scenarios_list.append('burst')
    
    # Degradation scenarios - avoid degraded server
    for i in range(num_degradation):
        server_idx = i % 3
        degradation_factor = np.random.uniform(0.3, 0.7)
        seq, action = generate_sequence('degradation', server_idx=server_idx, degradation_factor=degradation_factor)
        X_list.append(seq)
        y_list.append(action)
        scenarios_list.append('degradation')
    
    # DoS scenarios - distribute to weak servers
    for i in range(num_dos):
        target_server = i % 3
        intensity = np.random.uniform(0.5, 0.9)
        seq, action = generate_sequence('dos', target_server=target_server, intensity=intensity)
        X_list.append(seq)
        y_list.append(action)
        scenarios_list.append('dos')
    
    X = np.array(X_list)
    y = np.array(y_list)
    scenarios = np.array(scenarios_list)
    
    # Print action distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nAction distribution:")
    for u, c in zip(unique, counts):
        print(f"  Action {int(u)}: {c} ({c/len(y)*100:.1f}%)")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_v3.npy'), X)
    np.save(os.path.join(output_dir, 'y_v3.npy'), y)
    np.save(os.path.join(output_dir, 'scenarios_v3.npy'), scenarios)
    
    # Metadata
    metadata = {
        'num_samples': len(X),
        'sequence_length': SEQUENCE_LENGTH,
        'num_features': NUM_FEATURES,
        'num_actions': NUM_ACTIONS,
        'scenarios': {
            'normal': num_normal,
            'overload': num_overload,
            'burst': num_burst,
            'degradation': num_degradation,
            'dos': num_dos
        },
        'action_distribution': {str(int(u)): int(c) for u, c in zip(unique, counts)},
        'version': '3.0_synthetic_v2'
    }
    
    with open(os.path.join(output_dir, 'feature_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved to {output_dir}:")
    print(f"  X_v3.npy: {X.shape}")
    print(f"  y_v3.npy: {y.shape}")
    print(f"  scenarios_v3.npy: {scenarios.shape}")
    
    return X, y, scenarios


if __name__ == '__main__':
    generate_dataset(
        num_normal=200,
        num_overload=100,
        num_burst=100,
        num_degradation=100,
        num_dos=100,
        output_dir='ai_model/processed_data'
    )
