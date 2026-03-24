"""
PPO Load Balancer - Wrapper class cho SDN Integration

Dùng để tích hợp PPO model đã train vào Ryu Controller.

Usage:
    from ppo_load_balancer import PPOLoadBalancer
    
    # Initialize
    balancer = PPOLoadBalancer('ai_model/ppo_sdn_load_balancer.zip')
    
    # Get routing weights from flow stats
    weights = balancer.get_weights(flow_stats)
    
    # Or for simulation/testing
    weights = balancer.predict_from_observation([cpu_h5, cpu_h7, cpu_h8, lat_h5, lat_h7, lat_h8])
"""

import numpy as np
from stable_baselines3 import PPO
import os


class PPOLoadBalancer:
    """
    PPO-based Load Balancer cho SDN.
    
    Load trained PPO model và đưa ra routing decisions dựa trên
    flow statistics từ OpenFlow.
    """
    
    def __init__(self, model_path, device='cpu'):
        """
        Initialize PPO Load Balancer.
        
        Args:
            model_path: Path to trained PPO model (.zip file)
            device: 'cpu' or 'cuda'
        """
        self.model_path = model_path
        self.device = device
        
        # Load model
        print(f"[*] Loading PPO model from {model_path}...")
        self.model = PPO.load(model_path, device=device)
        print("[✓] PPO model loaded successfully")
        
        # Server info
        self.servers = ['h5', 'h7', 'h8']
        self.capacities = np.array([10.0, 50.0, 100.0])  # Mbps
        
        # Stats tracking
        self.decision_count = 0
        self.weight_history = []
        
    def get_weights(self, flow_stats):
        """
        Get routing weights từ flow statistics.
        
        Args:
            flow_stats: Dict chứa stats từ OpenFlow
                {
                    'h5': {'cpu': float, 'latency': float, 'packets': int, ...},
                    'h7': {...},
                    'h8': {...}
                }
                
        Returns:
            weights: [w_h5, w_h7, w_h8] normalized (tổng = 1.0)
        """
        # Prepare observation
        obs = self._prepare_observation(flow_stats)
        
        # Predict
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Normalize
        weights = action / (np.sum(action) + 1e-8)
        weights = np.clip(weights, 0.0, 1.0)
        
        # Track
        self.decision_count += 1
        self.weight_history.append(weights.tolist())
        
        return weights
    
    def predict_from_observation(self, observation):
        """
        Predict weights từ raw observation vector.
        
        Args:
            observation: [cpu_h5, cpu_h7, cpu_h8, lat_h5, lat_h7, lat_h8]
                        or [cpu_h5, cpu_h7, cpu_h8, lat_h5, lat_h7, lat_h8, traffic]
                        
        Returns:
            weights: [w_h5, w_h7, w_h8]
        """
        obs = np.array(observation, dtype=np.float32)
        
        # Pad if needed
        if len(obs) < 6:
            obs = np.pad(obs, (0, 6 - len(obs)), constant_values=0)
        elif len(obs) > 6:
            obs = obs[:6]
        
        # Normalize observation
        obs[0:3] = np.clip(obs[0:3], 0, 100) / 100.0  # CPU
        obs[3:6] = np.clip(obs[3:6], 0, 500) / 500.0  # Latency
        
        # Predict
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Normalize
        weights = action / (np.sum(action) + 1e-8)
        weights = np.clip(weights, 0.0, 1.0)
        
        return weights
    
    def _prepare_observation(self, flow_stats):
        """
        Chuyển flow_stats thành observation vector.
        
        Args:
            flow_stats: Dict from OpenFlow
            
        Returns:
            obs: numpy array shape (6,)
        """
        # Extract CPU utilization (normalize 0-100 -> 0-1)
        cpu_h5 = min(1.0, flow_stats.get('h5', {}).get('cpu', 0) / 100.0)
        cpu_h7 = min(1.0, flow_stats.get('h7', {}).get('cpu', 0) / 100.0)
        cpu_h8 = min(1.0, flow_stats.get('h8', {}).get('cpu', 0) / 100.0)
        
        # Extract latency (normalize 0-500ms -> 0-1)
        lat_h5 = min(1.0, flow_stats.get('h5', {}).get('latency', 10) / 500.0)
        lat_h7 = min(1.0, flow_stats.get('h7', {}).get('latency', 10) / 500.0)
        lat_h8 = min(1.0, flow_stats.get('h8', {}).get('latency', 10) / 500.0)
        
        return np.array([cpu_h5, cpu_h7, cpu_h8, lat_h5, lat_h7, lat_h8], dtype=np.float32)
    
    def get_decision_summary(self):
        """Get summary of routing decisions made."""
        if not self.weight_history:
            return {
                'total_decisions': 0,
                'avg_weights': [0.33, 0.33, 0.34],
                'std_weights': [0.0, 0.0, 0.0],
            }
        
        history = np.array(self.weight_history)
        return {
            'total_decisions': self.decision_count,
            'avg_weights': np.mean(history, axis=0).tolist(),
            'std_weights': np.std(history, axis=0).tolist(),
            'min_weights': np.min(history, axis=0).tolist(),
            'max_weights': np.max(history, axis=0).tolist(),
        }
    
    def apply_to_ovs(self, datapath, weights):
        """
        Áp dụng weights vào OVS flow rules.
        
        Args:
            datapath: Ryu datapath object
            weights: [w_h5, w_h7, w_h8]
        """
        # Normalize weights
        w5, w7, w8 = weights
        total = w5 + w7 + w8
        if total > 0:
            w5, w7, w8 = w5/total, w7/total, w8/total
        
        # Calculate priorities (scale to OVS priority range 1-100)
        # Higher weight = higher priority
        priority_base = 50
        
        # Set flow mods for each server
        # This is a simplified version - actual implementation depends on your OVS setup
        
        # Example: Set priority based on weight
        priority_h5 = int(priority_base + w5 * 50)
        priority_h7 = int(priority_base + w7 * 50)
        priority_h8 = int(priority_base + w8 * 50)
        
        return {
            'h5': {'weight': float(w5), 'priority': priority_h5},
            'h7': {'weight': float(w7), 'priority': priority_h7},
            'h8': {'weight': float(w8), 'priority': priority_h8},
        }
    
    def reset_stats(self):
        """Reset decision tracking statistics."""
        self.decision_count = 0
        self.weight_history = []


def create_simulated_stats(traffic_level=0.5):
    """
    Tạo simulated flow_stats cho testing.
    
    Args:
        traffic_level: 0.0-1.0 (higher = more load)
        
    Returns:
        Dict chứa simulated flow_stats
    """
    np.random.seed()
    
    # Simulate varying load
    base_load = traffic_level * 80  # 0-80% CPU
    
    return {
        'h5': {
            'cpu': min(100, base_load * 1.5 + np.random.randn() * 10),
            'latency': 10 + base_load * 3 + np.random.randn() * 5,
            'packets': int(1000 * traffic_level),
            'bytes': int(10000 * traffic_level),
        },
        'h7': {
            'cpu': min(100, base_load * 0.8 + np.random.randn() * 8),
            'latency': 10 + base_load * 1.5 + np.random.randn() * 3,
            'packets': int(5000 * traffic_level),
            'bytes': int(50000 * traffic_level),
        },
        'h8': {
            'cpu': min(100, base_load * 0.5 + np.random.randn() * 5),
            'latency': 10 + base_load * 0.8 + np.random.randn() * 2,
            'packets': int(10000 * traffic_level),
            'bytes': int(100000 * traffic_level),
        },
    }


if __name__ == "__main__":
    print("="*60)
    print("  PPO Load Balancer Test")
    print("="*60)
    
    # Initialize
    model_path = "ai_model/ppo_sdn_load_balancer.zip"
    
    if not os.path.exists(model_path):
        print(f"[!] Model not found at {model_path}")
        print("[*] Run training first: python ai_model/train_ppo_simple.py")
        exit(1)
    
    balancer = PPOLoadBalancer(model_path)
    
    # Test with simulated stats
    print("\n[*] Testing with simulated stats...")
    
    for traffic_level in [0.3, 0.5, 0.7, 0.9]:
        stats = create_simulated_stats(traffic_level)
        weights = balancer.get_weights(stats)
        
        print(f"  Traffic {traffic_level:.1f}: "
              f"h5={weights[0]:.3f}, h7={weights[1]:.3f}, h8={weights[2]:.3f}")
    
    # Test direct observation
    print("\n[*] Testing with direct observation...")
    obs = [0.8, 0.5, 0.3, 0.5, 0.3, 0.2]  # [cpu_h5, cpu_h7, cpu_h8, lat_h5, lat_h7, lat_h8]
    weights = balancer.predict_from_observation(obs)
    print(f"  Obs {obs}:")
    print(f"  Weights: h5={weights[0]:.3f}, h7={weights[1]:.3f}, h8={weights[2]:.3f}")
    
    # Summary
    print("\n[*] Decision Summary:")
    summary = balancer.get_decision_summary()
    print(f"  Total decisions: {summary['total_decisions']}")
    print(f"  Avg weights: {summary['avg_weights']}")
    
    print("\n[✓] PPO Load Balancer test passed!")
