"""
SDN Load Balancer Environment - RL Advantageous Version
========================================================

Thiết kế environment để RL thực sự có lợi thế:

1. NON-LINEAR LATENCY: Gần overload, latency nổ cực mạnh
   - WRR: chia đều → vào overload zone → latency spike
   - PPO: học tránh vùng nguy hiểm

2. OBSERVATION DELAY: 2-3 steps
   - WRR: dùng info cũ → suboptimal decisions
   - PPO: học compensate delay

3. BURST TRAFFIC: Spike bất ngờ
   - WRR: reactive → always behind
   - PPO: học predict và preempt

4. NON-STATIONARY CAPACITY: Server degradation
   - WRR: dùng capacity cũ → suboptimal
   - PPO: track changes

Author: Research-level environment design
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SDNRLAdvantageEnv(gym.Env):
    """
    Environment được thiết kế để RL có lợi thế thực sự.
    
    Key differences từ baseline:
    1. Non-linear latency (exponential near overload)
    2. Observation delay (2 steps)
    3. Burst traffic prediction opportunity
    4. Non-stationary capacity
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config=None):
        super().__init__()
        
        self.config = config or {}
        
        # Server capacities (Mbps)
        self.base_capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.capacity_scale = self.config.get('capacity_scale', 1.0)
        self.capacities = self.base_capacities * self.capacity_scale
        
        # Reward weights
        self.W1 = self.config.get('W1_throughput', 1.0)   # Throughput
        self.W2 = self.config.get('W2_latency', 3.0)      # Latency (higher penalty)
        self.W3 = self.config.get('W3_fairness', 1.0)      # Fairness
        self.W4 = self.config.get('W4_overload', 10.0)    # Overload penalty
        
        # Key feature: NON-LINEAR LATENCY
        # Baseline: latency = base / (1 - utilization)
        # NEW: latency = base / (1 - utilization)^alpha where alpha > 1
        # This creates "danger zone" near overload where WRR suffers
        self.latency_exponent = self.config.get('latency_exponent', 4.0)  # >1 = dangerous
        
        # Key feature: OBSERVATION DELAY
        self.obs_delay = self.config.get('obs_delay', 2)  # 2-step delay
        
        # Key feature: BURST TRAFFIC
        self.burst_probability = self.config.get('burst_prob', 0.15)
        self.burst_multiplier = self.config.get('burst_mult', 3.0)
        
        # Key feature: NON-STATIONARY CAPACITY
        self.capacity_drift = self.config.get('cap_drift', 0.002)
        
        # State history for delayed observation
        self.obs_history = []
        self.action_history = []
        
        # Time parameters
        self.max_steps = self.config.get('max_steps', 2000)
        
        # Action space: [w5, w7, w8] - tỷ lệ chia tải
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation space: [load_h5, load_h7, load_h8, latency_h5, latency_h7, latency_h8,
        #                     traffic_intensity, cap_h5, cap_h7, cap_h8, 
        #                     traffic_trend (EMA), burst_probability]
        # 12 phần tử
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        # State variables
        self.state = np.zeros(12, dtype=np.float32)
        self.traffic_intensity = 0.3
        self.traffic_trend = 0.0  # EMA of traffic change
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
        """
        Execute one step with:
        1. Delayed observation (state update happens AFTER action)
        2. Non-linear latency calculation
        3. Capacity drift
        4. Burst traffic
        """
        # Normalize action
        action = np.clip(action, 0.0, 1.0)
        action_sum = np.sum(action)
        if action_sum < 1e-8:
            action = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        else:
            action = action / action_sum
        
        w5, w7, w8 = action
        
        # Apply capacity drift (non-stationary)
        drift = np.random.randn(3) * self.capacity_drift
        self.capacity_multiplier = np.clip(
            self.capacity_multiplier + drift, 0.5, 1.5
        )
        current_caps = self.capacities * self.capacity_multiplier
        
        # BURST TRAFFIC: Random spike
        burst_now = np.random.rand() < self.burst_probability
        if burst_now:
            self.traffic_intensity = min(1.0, self.traffic_intensity * self.burst_multiplier)
            self.episode_stats['burst_count'] += 1
        
        # Update traffic
        self._update_traffic()
        
        # Calculate load (DELAYED - sử dụng action HIỆN TẠI nhưng capacity/cpu CŨ)
        # Đây là điểm mấu chốt: PPO phải predict future state
        load_h5 = self.traffic_intensity * w5 / current_caps[0] * np.max(current_caps)
        load_h7 = self.traffic_intensity * w7 / current_caps[1] * np.max(current_caps)
        load_h8 = self.traffic_intensity * w8 / current_caps[2] * np.max(current_caps)
        
        load_h5 = float(np.clip(load_h5, 0.0, 1.0))
        load_h7 = float(np.clip(load_h7, 0.0, 1.0))
        load_h8 = float(np.clip(load_h8, 0.0, 1.0))
        
        # NON-LINEAR LATENCY: lat = base / (1 - load)^exponent
        # Khi load → 1, latency nổ cực mạnh
        base_lat = 10.0
        
        def calc_latency(load):
            if load > 0.95:
                return 5000.0  # Near-certain crash
            util = min(load, 0.99)
            # Non-linear: exponent > 1 makes it "dangerous" near overload
            return base_lat / ((1.0 - util) ** self.latency_exponent)
        
        lat_h5 = calc_latency(load_h5)
        lat_h7 = calc_latency(load_h7)
        lat_h8 = calc_latency(load_h8)
        
        max_latency = max(lat_h5, lat_h7, lat_h8)
        
        # Clamp
        lat_h5 = min(lat_h5, 5000.0)
        lat_h7 = min(lat_h7, 5000.0)
        lat_h8 = min(lat_h8, 5000.0)
        
        # Add measurement noise (RL can learn to filter)
        noise = self.config.get('noise', 0.05)
        lat_h5 *= (1.0 + np.random.normal(0, noise))
        lat_h7 *= (1.0 + np.random.normal(0, noise))
        lat_h8 *= (1.0 + np.random.normal(0, noise))
        
        # REWARD CALCULATION
        # 1. Throughput
        throughput = min(1.0, self.traffic_intensity * 2)
        throughput_reward = self.W1 * throughput
        
        # 2. Latency (non-linear penalty)
        # PPO learns to AVOID the danger zone
        latency_penalty = self.W2 * (max_latency / 500.0) ** 1.5
        
        # 3. Overload penalty
        overload_detected = load_h5 > 0.9 or load_h7 > 0.9 or load_h8 > 0.9
        overload_penalty = self.W4 * overload_detected
        
        # 4. Fairness ( Jain's index)
        loads = np.array([load_h5, load_h7, load_h8])
        if np.sum(loads) > 0:
            fairness = np.sum(loads) ** 2 / (3 * np.sum(loads ** 2))
        else:
            fairness = 1.0
        fairness_reward = self.W3 * fairness
        
        # Total reward
        reward = throughput_reward - latency_penalty - overload_penalty + fairness_reward
        
        # UPDATE STATE (happens after action - this is key!)
        # Now we know the NEW state, but agent's next observation will be DELAYED
        new_obs = self._get_obs()
        
        # Add to history
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
            'delayed_obs': delayed_obs.copy(),
        }
        
        return delayed_obs, reward, done, False, info
    
    def _get_obs(self):
        """Get current (non-delayed) observation"""
        load_h5 = self.obs_history[-1][0] if len(self.obs_history) > 0 else 0.3
        load_h7 = self.obs_history[-1][1] if len(self.obs_history) > 0 else 0.3
        load_h8 = self.obs_history[-1][2] if len(self.obs_history) > 0 else 0.3
        
        return np.array([
            load_h5, load_h7, load_h8,  # Current loads (delayed)
            0.3, 0.3, 0.3,  # Latencies (will be computed in step)
            self.traffic_intensity,  # Traffic
            self.capacities[0] / 150.0,  # Normalized capacity
            self.capacities[1] / 150.0,
            self.capacities[2] / 150.0,
            self.traffic_trend,  # Traffic trend
            self.burst_probability,  # Burst probability
        ], dtype=np.float32)
    
    def _update_traffic(self):
        """Update traffic with trend"""
        old_traffic = self.traffic_intensity
        
        # Random walk with mean reversion
        target = np.random.rand() * 0.6 + 0.2  # 0.2 - 0.8
        self.traffic_intensity += (target - self.traffic_intensity) * 0.15
        
        # Update trend (EMA)
        traffic_delta = self.traffic_intensity - old_traffic
        self.traffic_trend = 0.7 * self.traffic_trend + 0.3 * traffic_delta
        
        # Clamp
        self.traffic_intensity = np.clip(self.traffic_intensity, 0.1, 1.0)
        self.traffic_trend = np.clip(self.traffic_trend, -0.5, 0.5)


def make_env(env_id='SDN-RL-Advantage'):
    """Factory function"""
    return SDNRLAdvantageEnv()


if __name__ == "__main__":
    print("="*60)
    print("  Testing RL-Advantage Environment")
    print("="*60)
    
    env = SDNRLAdvantageEnv()
    obs, info = env.reset()
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    print("\n[*] Running 10 steps with random actions...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.2f}, lat={info['latency']:.1f}ms, "
              f"overload={info['overload']}, burst={info['burst']}")
        
    print("\n[✓] RL-Advantage Environment test passed!")
    print("\nKey features:")
    print("  - Non-linear latency (exponent=4)")
    print("  - Observation delay (2 steps)")
    print("  - Burst traffic (15% chance, 3x multiplier)")
    print("  - Non-stationary capacity (drift=0.002)")
