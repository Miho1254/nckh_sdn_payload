#!/usr/bin/env python3
"""
Generate Synthetic Data V3 - BALANCED ACTION DISTRIBUTION
============================================================

Vấn đề V2: Action distribution bị lệch theo CAPACITY_RATIOS [0.0625, 0.3125, 0.625]
Giải pháp V3: Ép action distribution cân bằng [0.33, 0.33, 0.33]

Cách làm:
- Normal scenario: Random chọn action với xác suất đều
- Overload scenario: Chọn server có headroom cao nhất (nhưng rotate qua các server)
- Burst scenario: Ưu tiên h8 nhưng vẫn có h5, h7
- Degradation: Tránh server bị degradation
- DoS: Ưu tiên server yếu nhưng vẫn có h7, h8

Action labels:
- Action 0: chọn h5 (server yếu)
- Action 1: chọn h7 (server trung bình)
- Action 2: chọn h8 (server mạnh)
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

# BALANCED ACTION DISTRIBUTION - Key fix for CQL OOD penalty
BALANCED_ACTIONS = [0.33, 0.33, 0.34]  # Approximately equal


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


def generate_normal_with_action(action):
    """Generate normal scenario features with specific action being optimal."""
    features = generate_base_features()
    
    # Make the chosen action look good
    # Set headroom high for chosen server
    chosen_idx = action
    base_idx = 16 + chosen_idx * 7
    features[base_idx] = np.random.uniform(0.1, 0.3)  # low utilization
    features[base_idx + 1] = np.random.uniform(0.7, 0.95)  # high headroom
    features[base_idx + 6] = np.random.uniform(0.9, 1.0)  # high health
    
    # Other servers have moderate load
    for i in range(3):
        if i != chosen_idx:
            base_idx = 16 + i * 7
            features[base_idx] = np.random.uniform(0.3, 0.6)  # moderate utilization
            features[base_idx + 1] = np.random.uniform(0.4, 0.7)  # moderate headroom
    
    return features


def generate_overload_scenario(server_idx=0, severity=0.95, action=None):
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
    
    # If action specified, make that server look good
    if action is not None:
        good_base_idx = 16 + action * 7
        features[good_base_idx] = np.random.uniform(0.2, 0.4)  # moderate utilization
        features[good_base_idx + 1] = np.random.uniform(0.6, 0.8)  # good headroom
    
    # Risk features
    features[37] = 0.8  # congestion_risk
    features[38] = 0.6  # overload_risk
    features[39] = 1    # regime = HIGH
    
    return features


def generate_burst_scenario(intensity=0.8, action=None):
    """Generate burst traffic scenario."""
    features = generate_base_features()
    
    # Very high traffic
    features[0] = np.random.uniform(0.7, 1.0) * intensity  # byte_rate
    features[1] = np.random.uniform(0.6, 0.9) * intensity   # packet_rate
    features[2] = np.random.randint(100, 300)  # flow_count
    
    # If action specified, make that server look best
    if action is not None:
        for i in range(3):
            base_idx = 16 + i * 7
            if i == action:
                features[base_idx] = np.random.uniform(0.3, 0.5)  # lower utilization
                features[base_idx + 1] = np.random.uniform(0.5, 0.7)  # better headroom
            else:
                features[base_idx] = np.random.uniform(0.5, 0.7)  # higher utilization
                features[base_idx + 1] = np.random.uniform(0.3, 0.5)  # less headroom
    else:
        # All servers have moderate load
        for i in range(3):
            base_idx = 16 + i * 7
            features[base_idx] = np.random.uniform(0.4, 0.7)  # utilization
            features[base_idx + 1] = np.random.uniform(0.3, 0.6)  # headroom
    
    features[37] = 0.6  # congestion_risk
    features[39] = 2    # regime = BURST
    
    return features


def generate_degradation_scenario(server_idx=1, degradation_factor=0.5, action=None):
    """Generate server degradation scenario."""
    features = generate_base_features()
    
    # Target server degraded
    base_idx = 16 + server_idx * 7
    features[base_idx] = np.random.uniform(0.6, 0.9)  # high utilization
    features[base_idx + 1] = np.random.uniform(0.1, 0.4)  # low headroom
    features[base_idx + 5] = np.random.uniform(0.1, 0.3)  # high error_rate
    features[base_idx + 6] = degradation_factor  # low health_score
    
    # If action specified, make that server look healthy
    if action is not None and action != server_idx:
        good_base_idx = 16 + action * 7
        features[good_base_idx] = np.random.uniform(0.2, 0.4)  # good utilization
        features[good_base_idx + 1] = np.random.uniform(0.6, 0.8)  # good headroom
        features[good_base_idx + 6] = np.random.uniform(0.9, 1.0)  # good health
    
    features[42] = 0.7  # degradation_risk
    features[39] = 1    # regime = HIGH
    
    return features


def generate_dos_scenario(target_server=0, intensity=0.7, action=None):
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
    
    # If action specified, make that server look good
    if action is not None and action != target_server:
        good_base_idx = 16 + action * 7
        features[good_base_idx] = np.random.uniform(0.2, 0.4)  # lower utilization
        features[good_base_idx + 1] = np.random.uniform(0.6, 0.8)  # better headroom
    
    features[37] = 0.9  # congestion_risk
    features[40] = 0.8  # anomaly_score
    features[41] = 0.9  # ddos_risk
    features[39] = 1    # regime = HIGH
    
    return features


def generate_sequence(scenario='normal', action=None, **kwargs):
    """Generate a sequence of 5 timesteps with specific action label."""
    sequence = []
    
    for t in range(SEQUENCE_LENGTH):
        if scenario == 'normal':
            features = generate_normal_with_action(action)
        elif scenario == 'overload':
            severity = 0.8 + t * 0.05
            features = generate_overload_scenario(
                server_idx=kwargs.get('server_idx', 0),
                severity=min(1.0, severity),
                action=action
            )
        elif scenario == 'burst':
            intensity = kwargs.get('intensity', 0.8) * (0.8 + t * 0.05)
            features = generate_burst_scenario(intensity=min(1.0, intensity), action=action)
        elif scenario == 'degradation':
            features = generate_degradation_scenario(
                server_idx=kwargs.get('server_idx', 1),
                degradation_factor=kwargs.get('degradation_factor', 0.5),
                action=action
            )
        elif scenario == 'dos':
            features = generate_dos_scenario(
                target_server=kwargs.get('target_server', 0),
                intensity=kwargs.get('intensity', 0.7),
                action=action
            )
        else:
            features = generate_normal_with_action(action)
        
        sequence.append(features)
    
    return np.array(sequence), action


def generate_balanced_dataset(
    num_per_action=200,  # Số samples cho MỖI action
    output_dir='ai_model/processed_data'
):
    """Generate dataset with BALANCED action distribution.
    
    Mỗi action (h5, h7, h8) có số lượng samples bằng nhau.
    """
    
    X_list = []
    y_list = []
    scenarios_list = []
    
    total_per_action = num_per_action
    per_scenario = total_per_action // 5  # 5 scenarios
    
    print(f"Generating BALANCED dataset:")
    print(f"  Target: {num_per_action} samples per action")
    print(f"  Scenarios: {per_scenario} each")
    
    # Scenario distribution for each action
    scenarios = ['normal', 'overload', 'burst', 'degradation', 'dos']
    
    for action in range(3):  # 0=h5, 1=h7, 2=h8
        action_samples = 0
        
        # Normal scenario
        for _ in range(per_scenario):
            seq, act = generate_sequence('normal', action=action)
            X_list.append(seq)
            y_list.append(act)
            scenarios_list.append(f'normal_a{action}')
            action_samples += 1
        
        # Overload scenario
        for i in range(per_scenario):
            server_idx = i % 3
            seq, act = generate_sequence('overload', action=action, server_idx=server_idx)
            X_list.append(seq)
            y_list.append(act)
            scenarios_list.append(f'overload_a{action}')
            action_samples += 1
        
        # Burst scenario
        for _ in range(per_scenario):
            intensity = np.random.uniform(0.6, 1.0)
            seq, act = generate_sequence('burst', action=action, intensity=intensity)
            X_list.append(seq)
            y_list.append(act)
            scenarios_list.append(f'burst_a{action}')
            action_samples += 1
        
        # Degradation scenario
        for i in range(per_scenario):
            server_idx = (action + 1) % 3  # Avoid degraded server
            degradation_factor = np.random.uniform(0.3, 0.7)
            seq, act = generate_sequence('degradation', action=action, server_idx=server_idx, degradation_factor=degradation_factor)
            X_list.append(seq)
            y_list.append(act)
            scenarios_list.append(f'degradation_a{action}')
            action_samples += 1
        
        # DoS scenario
        for i in range(per_scenario):
            target_server = (action + 1) % 3  # Different target
            intensity = np.random.uniform(0.5, 0.9)
            seq, act = generate_sequence('dos', action=action, target_server=target_server, intensity=intensity)
            X_list.append(seq)
            y_list.append(act)
            scenarios_list.append(f'dos_a{action}')
            action_samples += 1
        
        print(f"  Action {action}: {action_samples} samples")
    
    X = np.array(X_list)
    y = np.array(y_list)
    scenarios = np.array(scenarios_list)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    scenarios = scenarios[indices]
    
    # Print action distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n✅ BALANCED Action distribution:")
    for u, c in zip(unique, counts):
        print(f"  Action {int(u)}: {c} ({c/len(y)*100:.1f}%)")
    
    # Verify balance
    ratios = counts / len(y)
    print(f"\n  Distribution: [{ratios[0]:.2f}, {ratios[1]:.2f}, {ratios[2]:.2f}]")
    print(f"  Target: [0.33, 0.33, 0.34]")
    
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
            'normal': per_scenario * 3,
            'overload': per_scenario * 3,
            'burst': per_scenario * 3,
            'degradation': per_scenario * 3,
            'dos': per_scenario * 3
        },
        'action_distribution': {str(int(u)): int(c) for u, c in zip(unique, counts)},
        'version': '3.0_balanced',
        'note': 'BALANCED action distribution for CQL training - fixes OOD penalty issue'
    }
    
    with open(os.path.join(output_dir, 'feature_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Saved to {output_dir}:")
    print(f"  X_v3.npy: {X.shape}")
    print(f"  y_v3.npy: {y.shape}")
    print(f"  scenarios_v3.npy: {scenarios.shape}")
    
    return X, y, scenarios


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate BALANCED Synthetic Data for SDN Load Balancing')
    parser.add_argument('--samples', type=int, default=200, help='Samples per action (default: 200)')
    parser.add_argument('--output', type=str, default='ai_model/processed_data', help='Output directory')
    
    args = parser.parse_args()
    
    np.random.seed(42)  # For reproducibility
    
    X, y, scenarios = generate_balanced_dataset(
        num_per_action=args.samples,
        output_dir=args.output
    )
    
    print(f"\n🎉 Dataset generation complete!")
    print(f"   Total samples: {len(X)}")
    print(f"   Action distribution: BALANCED [~33%, ~33%, ~34%]")