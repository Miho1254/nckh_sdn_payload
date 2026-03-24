"""
SDN Load Balancer Environment - FIXED VERSION
=============================================

Các fix quan trọng:
1. Reward scaled to [-10, 10] range
2. Soft penalties instead of death penalties
3. Clipped latency (max 1000ms)
4. Curriculum learning support
5. Normalized observations
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SDNRLAdvantageEnvFixed(gym.Env):
    """
    Fixed environment - reward scaled properly so PPO can learn.
    
    Key fixes from toxic version:
    1. Reward magnitude ~ [-10, 10]
    2. Soft overload penalty (not death penalty)
    3. Latency clipped to 1000ms
    4. Curriculum learning support
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config=None):
        super().__init__()
        
        self.config = config or {}
        
        # Server capacities (Mbps)
        self.base_capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.capacity_scale = self.config.get('capacity_scale', 1.0)
        self.capacities = self.base_capacities * self.capacity_scale
        
        # Reward weights - SCALED PROPERLY
        self.W1 = self.config.get('W1_throughput', 2.0)
        self.W2 = self.config.get('W2_latency', 1.0)     # Reduced from 3.0
        self.W3 = self.config.get('W3_fairness', 0.5)
        self.W4 = self.config.get('W4_overload', 2.0)    # Reduced from 10.0
        
        # Non-linear latency (but less extreme)
        self.latency_exponent = self.config.get('latency_exponent', 2.5)  # Reduced from 4.0
        
        # Observation delay
        self.obs_delay = self.config.get('obs_delay', 2)
        
        # Burst traffic (reduced)
        self.burst_probability = self.config.get('burst_prob', 0.08)  # Reduced from 0.15
        self.burst_multiplier = self.config.get('burst_mult', 2.0)   # Reduced from 3.0
        
        # Non-stationary capacity (reduced drift)
        self.capacity_drift = self.config.get('cap_drift', 0.001)  # Reduced from 0.002
        
        # Curriculum learning
        self.curriculum_level = self.config.get('curriculum_level', 1.0)  # 1.0 = full difficulty
        
        # State history
        self.obs_history = []
        self.action_history = []
        
        # Time parameters
        self.max_steps = self.config.get('max_steps', 2000)
        
        # Action space: [-1, 1] (better for PPO)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation space: normalized
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(12,), dtype=np.float32
        )
        
        # State variables
        self.state = np.zeros(12, dtype=np.float32)
        self.traffic_intensity = 0.3
        self.traffic_trend = 0.0
        self.capacity_multiplier = np.ones(3)
        self.current_step = 0
        
        # Episode stats
        self.episode_stats = {
            'total_throughput': 0.0,
            'max_latency': 0.0,
            'overload_count': 0,
            'burst_count': 0,
        }
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Randomize capacities
        self.cap_scales = np.random.uniform(0.8, 1.2, size=3)
        self.capacities = self.base_capacities * self.cap_scales
        self.capacity_multiplier = np.ones(3)
        
        # Traffic pattern
        self.traffic_intensity = np.random.uniform(0.2, 0.5)
        self.traffic_trend = 0.0
        
        # Initialize history
        init_obs = self._get_obs()
        self.obs_history = [init_obs.copy()] * (self.obs_delay + 1)
        self.action_history = [np.zeros(3)] * self.obs_delay
        
        self.current_step = 0
        self.episode_stats = {
            'total_throughput': 0.0,
            'max_latency': 0.0,
            'overload_count': 0,
            'burst_count': 0,
        }
        
        return self.obs_history[0], {}
    
    def step(self, action):
        # Map action from [-1, 1] to [0, 1] for weights
        action = np.clip(action, -1.0, 1.0)
        action = (action + 1.0) / 2.0  # Map to [0, 1]
        action_sum = np.sum(action)
        if action_sum < 1e-8:
            action = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        else:
            action = action / action_sum
        
        w5, w7, w8 = action
        
        # Apply capacity drift
        drift = np.random.randn(3) * self.capacity_drift * self.curriculum_level
        self.capacity_multiplier = np.clip(
            self.capacity_multiplier + drift, 0.5, 1.5
        )
        current_caps = self.capacities * self.capacity_multiplier
        
        # BURST TRAFFIC (with curriculum)
        burst_now = np.random.rand() < self.burst_probability * self.curriculum_level
        if burst_now:
            self.traffic_intensity = min(1.0, self.traffic_intensity * self.burst_multiplier)
            self.episode_stats['burst_count'] += 1
        
        # Update traffic
        self._update_traffic()
        
        # Calculate load
        load_h5 = self.traffic_intensity * w5 / current_caps[0] * np.max(current_caps)
        load_h7 = self.traffic_intensity * w7 / current_caps[1] * np.max(current_caps)
        load_h8 = self.traffic_intensity * w8 / current_caps[2] * np.max(current_caps)
        
        load_h5 = float(np.clip(load_h5, 0.0, 1.0))
        load_h7 = float(np.clip(load_h7, 0.0, 1.0))
        load_h8 = float(np.clip(load_h8, 0.0, 1.0))
        
        # NON-LINEAR LATENCY (but clipped)
        base_lat = 10.0
        
        def calc_latency(load):
            if load > 0.95:
                return 1000.0  # CLIPPED - was 5000
            util = min(load, 0.99)
            return base_lat / ((1.0 - util) ** self.latency_exponent)
        
        lat_h5 = calc_latency(load_h5)
        lat_h7 = calc_latency(load_h7)
        lat_h8 = calc_latency(load_h8)
        
        # CLIP LATENCY - important for learning
        max_latency = min(max(lat_h5, lat_h7, lat_h8), 1000.0)
        lat_h5 = min(lat_h5, 1000.0)
        lat_h7 = min(lat_h7, 1000.0)
        lat_h8 = min(lat_h8, 1000.0)
        
        # Add noise (reduced)
        noise = self.config.get('noise', 0.02) * self.curriculum_level
        lat_h5 *= (1.0 + np.random.normal(0, noise))
        lat_h7 *= (1.0 + np.random.normal(0, noise))
        lat_h8 *= (1.0 + np.random.normal(0, noise))
        
        # Normalize latencies for reward
        norm_lat_h5 = lat_h5 / 1000.0
        norm_lat_h7 = lat_h7 / 1000.0
        norm_lat_h8 = lat_h8 / 1000.0
        norm_max_lat = max(norm_lat_h5, norm_lat_h7, norm_lat_h8)
        
        # REWARD - PROPERLY SCALED
        # 1. Throughput (positive)
        throughput = min(1.0, self.traffic_intensity * 2)
        throughput_reward = self.W1 * throughput
        
        # 2. Latency penalty (SCALED -10 to 0)
        # lat = 0ms → 0 penalty
        # lat = 1000ms → -10 penalty
        latency_penalty = self.W2 * norm_max_lat * 10.0  # Range: [0, -10]
        
        # 3. Overload penalty (SOFT - not death)
        # Soft penalty based on how close to overload
        overload_h5 = max(0, load_h5 - 0.85) / 0.15  # 0 to 1 when load > 0.85
        overload_h7 = max(0, load_h7 - 0.85) / 0.15
        overload_h8 = max(0, load_h8 - 0.85) / 0.15
        overload_penalty = self.W4 * max(overload_h5, overload_h7, overload_h8)
        
        # 4. Fairness (Jain's index - positive reward)
        loads = np.array([load_h5, load_h7, load_h8])
        if np.sum(loads) > 0:
            fairness = np.sum(loads) ** 2 / (3 * np.sum(loads ** 2) + 1e-8)
        else:
            fairness = 1.0
        fairness_reward = self.W3 * fairness
        
        # Total reward - should be in range [-10, 10] approximately
        reward = throughput_reward - latency_penalty - overload_penalty + fairness_reward
        
        # Update state
        new_obs = self._get_obs()
        self.obs_history.append(new_obs.copy())
        self.action_history.append(action.copy())
        
        # Return DELAYED observation
        delayed_obs = self.obs_history[0]
        self.obs_history.pop(0)
        self.action_history.pop(0)
        
        # Track stats
        self.current_step += 1
        self.episode_stats['total_throughput'] += throughput
        self.episode_stats['max_latency'] = max(self.episode_stats['max_latency'], max_latency)
        
        overload_detected = load_h5 > 0.9 or load_h7 > 0.9 or load_h8 > 0.9
        if overload_detected:
            self.episode_stats['overload_count'] += 1
        
        done = self.current_step >= self.max_steps
        
        info = {
            'throughput': float(throughput),
            'latency': float(max_latency),
            'load_h5': float(load_h5),
            'load_h7': float(load_h7),
            'load_h8': float(load_h8),
            'lat_h5': float(lat_h5),
            'lat_h7': float(lat_h7),
            'lat_h8': float(lat_h8),
            'overload': overload_detected,
            'weights': action.tolist(),
            'traffic_intensity': float(self.traffic_intensity),
            'traffic_trend': float(self.traffic_trend),
            'burst': burst_now,
            'cap_multiplier': self.capacity_multiplier.copy(),
            'reward': float(reward),
        }
        
        return delayed_obs, reward, done, False, info
    
    def _get_obs(self):
        """Get normalized observation"""
        # Current loads (delayed)
        load_h5 = self.obs_history[-1][0] if len(self.obs_history) > 0 else 0.3
        load_h7 = self.obs_history[-1][1] if len(self.obs_history) > 0 else 0.3
        load_h8 = self.obs_history[-1][2] if len(self.obs_history) > 0 else 0.3
        
        # Normalize to [-1, 1] range for better PPO performance
        # load: 0 → -1, 0.5 → 0, 1.0 → 1
        norm_load_h5 = load_h5 * 2.0 - 1.0
        norm_load_h7 = load_h7 * 2.0 - 1.0
        norm_load_h8 = load_h8 * 2.0 - 1.0
        
        return np.array([
            norm_load_h5, norm_load_h7, norm_load_h8,  # Normalized loads
            0.0, 0.0, 0.0,  # Latencies (will be computed in step)
            self.traffic_intensity * 2.0 - 1.0,  # Normalized traffic
            self.capacities[0] / 100.0 - 1.0,   # Normalized capacity
            self.capacities[1] / 100.0 - 1.0,
            self.capacities[2] / 100.0 - 1.0,
            self.traffic_trend * 2.0,  # Normalized trend
            self.burst_probability * self.curriculum_level * 2.0 - 1.0,  # Normalized burst
        ], dtype=np.float32)
    
    def _update_traffic(self):
        """Update traffic with trend"""
        old_traffic = self.traffic_intensity
        
        target = np.random.rand() * 0.6 + 0.2
        self.traffic_intensity += (target - self.traffic_intensity) * 0.15
        
        traffic_delta = self.traffic_intensity - old_traffic
        self.traffic_trend = 0.7 * self.traffic_trend + 0.3 * traffic_delta
        
        self.traffic_intensity = np.clip(self.traffic_intensity, 0.1, 1.0)
        self.traffic_trend = np.clip(self.traffic_trend, -0.5, 0.5)


def make_env(env_id='SDN-RL-Fixed'):
    """Factory function"""
    return SDNRLAdvantageEnvFixed()


if __name__ == "__main__":
    print("="*60)
    print("  Testing FIXED RL-Advantage Environment")
    print("="*60)
    
    env = SDNRLAdvantageEnvFixed()
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"Action space: {env.action_space}")
    
    print("\n[*] Running 20 steps with random actions...")
    rewards = []
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        rewards.append(reward)
        print(f"  Step {i+1}: reward={reward:7.2f}, lat={info['latency']:6.1f}ms, "
              f"overload={info['overload']}")
    
    print(f"\n[*] Reward stats: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}")
    print(f"[*] Reward range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    
    # Check if rewards are in reasonable range
    if np.max(rewards) > 20 or np.min(rewards) < -50:
        print("\n[!] WARNING: Rewards out of expected range!")
    else:
        print("\n[✓] Rewards are in reasonable range")
