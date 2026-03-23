#!/usr/bin/env python3
"""Quick evaluation script for TFT-CQL vs WRR comparison."""
import sys
sys.path.insert(0, '/work')

import numpy as np
from ai_model.cql_agent import CQLAgent
from ai_model.evaluator import evaluate_policy, WeightedRoundRobinBaseline
from ai_model.sdn_env_v2 import SDN_Offline_Env_V2
from config import NUM_ACTIONS

def main():
    # Load data paths
    data_x_path = '/work/stats/X_v3.npy'
    data_y_path = '/work/stats/y_v3.npy'
    
    # Load model
    agent = CQLAgent(input_size=44, seq_len=5, hidden_size=64, num_actions=NUM_ACTIONS)
    agent.load_checkpoint('/work/stats/tft_ac_best.pth')
    agent.model.eval()
    
    # Create env with file paths
    env = SDN_Offline_Env_V2(data_x_path, data_y_path)
    
    # Load data for evaluation
    X = np.load(data_x_path)
    y = np.load(data_y_path)
    
    # Evaluate TFT-CQL with sampled inference
    print('Evaluating TFT-CQL with sampled inference...')
    tft_metrics = evaluate_policy(agent, X, y, env)
    
    # Evaluate WRR
    print('Evaluating WRR...')
    wrr = WeightedRoundRobinBaseline()
    env.reset()
    wrr_metrics = evaluate_policy(wrr, X, y, env)
    
    # Print comparison
    print('\n' + '='*60)
    print('RESULTS COMPARISON')
    print('='*60)
    print(f'{"Metric":<25} {"TFT-CQL":>15} {"WRR":>15}')
    print('-'*60)
    print(f'{"Throughput":<25} {tft_metrics["served_throughput"]:>15.4f} {wrr_metrics["served_throughput"]:>15.4f}')
    print(f'{"Overload Rate":<25} {tft_metrics["overload_rate"]:>15.4f} {wrr_metrics["overload_rate"]:>15.4f}')
    print(f'{"QoS Efficiency":<25} {tft_metrics["qos_efficiency"]:>15.4f} {wrr_metrics["qos_efficiency"]:>15.4f}')
    print(f'{"Stability Score":<25} {tft_metrics["stability_score"]:>15.4f} {wrr_metrics["stability_score"]:>15.4f}')
    
    tft_dist = [round(x, 2) for x in tft_metrics["action_distribution"]]
    wrr_dist = [round(x, 2) for x in wrr_metrics["action_distribution"]]
    print(f'{"Action Dist (h5,h7,h8)":<25} {str(tft_dist):>15} {str(wrr_dist):>15}')
    print('='*60)
    
    # Determine winner
    tft_score = tft_metrics["qos_efficiency"] + tft_metrics["stability_score"]
    wrr_score = wrr_metrics["qos_efficiency"] + wrr_metrics["stability_score"]
    
    print(f'\nTFT-CQL Score: {tft_score:.4f}')
    print(f'WRR Score: {wrr_score:.4f}')
    
    if tft_score > wrr_score:
        print('WINNER: TFT-CQL beats WRR!')
    else:
        print('WINNER: WRR still wins')

if __name__ == '__main__':
    main()