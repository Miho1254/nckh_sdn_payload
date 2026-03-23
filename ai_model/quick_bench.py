#!/usr/bin/env python3
"""Quick benchmark script - Đánh giá nhanh TFT-CQL vs WRR.
Chỉ chạy trên 1 scenario, không cần train.
"""
import os
import sys
import numpy as np

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))

from cql_agent import CQLAgent
from evaluator import evaluate_policy, WeightedRoundRobinBaseline
from sdn_env_v2 import SDN_Offline_Env_V2
from config import NUM_ACTIONS, CAPACITIES

def main():
    print("="*60)
    print("  QUICK BENCHMARK - TFT-CQL vs WRR")
    print("="*60)
    
    # Load data
    data_dir = os.path.join(BASE_DIR, 'processed_data')
    x_path = os.path.join(data_dir, 'X_v3.npy')
    y_path = os.path.join(data_dir, 'y_v3.npy')
    
    if not os.path.exists(x_path):
        print(f"[!] Data not found: {x_path}")
        print("    Run data_processor.py first!")
        return
    
    X = np.load(x_path)
    y = np.load(y_path)
    print(f"[*] Loaded data: X={X.shape}, y={y.shape}")
    
    # Create env
    env = SDN_Offline_Env_V2(x_path, y_path, mode='eval')
    
    # Check for checkpoint
    checkpoint_path = os.path.join(BASE_DIR, 'checkpoints', 'tft_ac_best.pth')
    if not os.path.exists(checkpoint_path):
        # Try alternate path
        checkpoint_path = os.path.join(os.path.dirname(BASE_DIR), 'stats', 'tft_ac_best.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"[!] No checkpoint found!")
        print("    Training required before benchmark.")
        return
    
    print(f"[*] Loading checkpoint: {checkpoint_path}")
    num_features = X.shape[2]
    agent = CQLAgent(input_size=num_features, seq_len=5, hidden_size=64, num_actions=NUM_ACTIONS)
    agent.load_checkpoint(checkpoint_path)
    agent.model.eval()
    
    # Evaluate TFT-CQL
    print("\n[*] Evaluating TFT-CQL (sampled inference)...")
    tft_metrics = evaluate_policy(agent, X, y, env)
    
    # Evaluate WRR
    print("[*] Evaluating WRR...")
    env.reset()
    wrr = WeightedRoundRobinBaseline()
    wrr_metrics = evaluate_policy(wrr, X, y, env)
    
    # Print results
    print("\n" + "="*70)
    print("  BENCHMARK RESULTS")
    print("="*70)
    print(f"{'Metric':<25} {'TFT-CQL':>15} {'WRR':>15} {'Winner':>10}")
    print("-"*70)
    
    metrics = [
        ('Throughput', 'served_throughput', 'higher'),
        ('Overload Rate', 'overload_rate', 'lower'),
        ('QoS Efficiency', 'qos_efficiency', 'higher'),
        ('Stability Score', 'stability_score', 'higher'),
    ]
    
    for name, key, direction in metrics:
        tft_val = tft_metrics[key]
        wrr_val = wrr_metrics[key]
        if direction == 'higher':
            winner = 'TFT-CQL' if tft_val > wrr_val else 'WRR'
        else:
            winner = 'TFT-CQL' if tft_val < wrr_val else 'WRR'
        print(f"{name:<25} {tft_val:>15.4f} {wrr_val:>15.4f} {winner:>10}")
    
    # Action distribution
    tft_dist = tft_metrics['action_distribution']
    wrr_dist = wrr_metrics['action_distribution']
    print("-"*70)
    print(f"{'Action Dist (h5,h7,h8)':<25} {str([round(x,2) for x in tft_dist]):>15} {str([round(x,2) for x in wrr_dist]):>15}")
    
    # Capacity-weighted throughput
    tft_capacity = np.dot(tft_dist, CAPACITIES)
    wrr_capacity = np.dot(wrr_dist, CAPACITIES)
    print(f"{'Capacity-Weighted':<25} {tft_capacity:>15.2f} {wrr_capacity:>15.2f}")
    
    print("="*70)
    
    # Overall score
    tft_score = tft_metrics['qos_efficiency'] + tft_metrics['stability_score'] - tft_metrics['overload_rate']
    wrr_score = wrr_metrics['qos_efficiency'] + wrr_metrics['stability_score'] - wrr_metrics['overload_rate']
    
    print(f"\nTFT-CQL Score: {tft_score:.4f}")
    print(f"WRR Score:     {wrr_score:.4f}")
    
    if tft_score > wrr_score:
        print("\n✓ TFT-CQL WINS!")
    else:
        print("\n✗ WRR still wins - need more training")
    
    # Analysis
    print("\n" + "="*70)
    print("  ANALYSIS")
    print("="*70)
    
    # Check if TFT-CQL is selecting weak server too much
    if tft_dist[0] > 0.2:  # h5 > 20%
        print(f"[!] TFT-CQL selecting h5 (weak server) too much: {tft_dist[0]*100:.1f}%")
        print("    Recommendation: Increase CAPACITY_BONUS_WEIGHT or decrease diversity_bonus")
    
    if tft_dist[2] < 0.4:  # h8 < 40%
        print(f"[!] TFT-CQL not selecting h8 (strong server) enough: {tft_dist[2]*100:.1f}%")
        print("    Recommendation: Increase capacity_bonus or add weak_server_penalty")
    
    if tft_metrics['overload_rate'] > 0.01:
        print(f"[!] TFT-CQL overload rate too high: {tft_metrics['overload_rate']*100:.2f}%")
        print("    Recommendation: Increase overload_penalty in reward function")
    
    # Ideal distribution
    print(f"\nIdeal distribution (capacity ratio): [6.25%, 31.25%, 62.5%]")
    print(f"TFT-CQL distribution: [{tft_dist[0]*100:.1f}%, {tft_dist[1]*100:.1f}%, {tft_dist[2]*100:.1f}%]")
    print(f"WRR distribution: [{wrr_dist[0]*100:.1f}%, {wrr_dist[1]*100:.1f}%, {wrr_dist[2]*100:.1f}%]")

if __name__ == '__main__':
    main()