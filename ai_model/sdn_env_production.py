#!/usr/bin/env python3
"""
SDN Load Balancer Environment - Production Grade
================================================

Thiết kế theo hướng production-grade:
- SLA-based reward (không có fairness penalty, ideal_ratio)
- Burst traffic model (non-stationary, unpredictable)
- Delayed observation (partial observability)
- Realistic state (queue length, RTT, packet loss history)

Mục tiêu: Tạo environment mà RL có thể thắng WRR trong điều kiện thực tế.

Author: Research Team
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque
import random


class SDNProductionEnv(gym.Env):
    """
    Production-grade SDN Load Balancer Environment.
    
    Key differences from simple simulation:
    1. SLA-based reward (no fairness penalty, no ideal_ratio)
    2. Burst traffic with correlation
    3. Delayed observation (2-5 steps)
    4. Noisy measurements
    5. Non-stationary conditions
    
    Observation (22 dims):
    - Queue length per server (3)
    - Latency per server (3)
    - Packet loss rate per server (3)
    - RTT estimate per server (3)
    - Traffic trend (up/down/stable) (3)
    - Recent throughput (1)
    - Server capacity (3)
    - Global load (1)
    - Time since last burst (1)
    - Suspicious ratio (1)
    
    Action (3 dims):
    - Weight distribution [w5, w7, w8] (softmax normalized)
    
    Reward:
    - SLA compliance bonus
    - Latency penalty (p99)
    - Packet loss penalty
    - Overload penalty
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config=None):
        super().__init__()
        
        self.config = config or {}
        
        # ═══════════════════════════════════════════════════════════════
        # SERVER CONFIGURATION
        # ═══════════════════════════════════════════════════════════════
        
        # Capacities (Mbps) - can be randomized per episode
        self.base_capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.capacities = self.base_capacities.copy()
        self.max_capacity = float(np.max(self.capacities))
        
        # Buffer sizes (requests) - smaller = more realistic
        self.buffer_sizes = np.array([50.0, 250.0, 500.0], dtype=np.float32)
        
        # ═══════════════════════════════════════════════════════════════
        # SLA PARAMETERS (Production-grade)
        # ═══════════════════════════════════════════════════════════════
        
        self.sla_latency_threshold = self.config.get('sla_latency_ms', 100.0)  # ms
        self.sla_packet_loss_threshold = self.config.get('sla_packet_loss', 0.01)  # 1%
        self.sla_throughput_min = self.config.get('sla_throughput_min', 0.5)  # 50% of capacity
        
        # ═══════════════════════════════════════════════════════════════
        # TRAFFIC MODEL (Non-stationary + Burst)
        # ═══════════════════════════════════════════════════════════════
        
        self.traffic_intensity = 0.3
        self.traffic_trend = 0.0  # -1 (down), 0 (stable), +1 (up)
        self.burst_probability = self.config.get('burst_prob', 0.15)
        self.burst_intensity = 0.0
        self.steps_since_burst = 0
        
        # Correlation for burst traffic
        self.burst_correlation = 0.7  # Burst tends to continue
        
        # ═══════════════════════════════════════════════════════════════
        # DELAYED OBSERVATION (Partial Observability)
        # ═══════════════════════════════════════════════════════════════
        
        self.observation_delay = self.config.get('obs_delay', 2)  # steps
        self.state_history = deque(maxlen=self.observation_delay + 1)
        
        # ═══════════════════════════════════════════════════════════════
        # NOISE MODEL (Realistic Measurements)
        # ═══════════════════════════════════════════════════════════════
        
        self.measurement_noise = self.config.get('measurement_noise', 0.1)
        self.latency_noise = self.config.get('latency_noise', 0.15)
        
        # ═══════════════════════════════════════════════════════════════
        # STATE TRACKING
        # ═══════════════════════════════════════════════════════════════
        
        self.max_steps = self.config.get('max_steps', 1000)
        self.current_step = 0
        
        # Queue state per server
        self.queue_lengths = np.zeros(3, dtype=np.float32)
        self.packet_loss_history = deque(maxlen=10)
        self.rtt_estimates = np.ones(3, dtype=np.float32) * 20.0  # ms
        self.throughput_history = deque(maxlen=20)
        
        # Load history for trend detection
        self.load_history = deque(maxlen=10)
        self.latency_history = deque(maxlen=10)
        
        # Suspicious traffic (DoS detection)
        self.suspicious_ratio = 0.0
        
        # Episode stats
        self.episode_stats = {
            'total_throughput': 0.0,
            'total_packet_loss': 0.0,
            'sla_violations': 0,
            'overload_events': 0,
            'burst_events': 0,
            'p99_latency': [],
            'p50_latency': [],
        }
        
        # ═══════════════════════════════════════════════════════════════
        # ACTION & OBSERVATION SPACE
        # ═══════════════════════════════════════════════════════════════
        
        # Action: weight distribution [w5, w7, w8]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Observation: 22 dimensions
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(22,),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = np.zeros(22, dtype=np.float32)
        self._reset_state_history()
    
    def _reset_state_history(self):
        """Reset state history for delayed observation."""
        initial_state = np.zeros(22, dtype=np.float32)
        for _ in range(self.observation_delay + 1):
            self.state_history.append(initial_state.copy())
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        # Randomize capacities slightly (±10%)
        if self.config.get('randomize_capacity', True):
            capacity_noise = np.random.uniform(0.9, 1.1, size=3)
            self.capacities = self.base_capacities * capacity_noise
            self.max_capacity = float(np.max(self.capacities))
        
        # Reset traffic
        self.traffic_intensity = np.random.uniform(0.2, 0.4)
        self.traffic_trend = 0.0
        self.burst_intensity = 0.0
        self.steps_since_burst = 100
        
        # Reset state
        self.queue_lengths = np.zeros(3, dtype=np.float32)
        self.packet_loss_history = deque(maxlen=10)
        self.rtt_estimates = np.ones(3, dtype=np.float32) * 20.0
        self.throughput_history = deque(maxlen=20)
        self.load_history = deque(maxlen=10)
        self.latency_history = deque(maxlen=10)
        self.suspicious_ratio = 0.0
        self.current_step = 0
        
        # Reset episode stats
        self.episode_stats = {
            'total_throughput': 0.0,
            'total_packet_loss': 0.0,
            'sla_violations': 0,
            'overload_events': 0,
            'burst_events': 0,
            'p99_latency': [],
            'p50_latency': [],
        }
        
        self._reset_state_history()
        self._update_state()
        
        return self._get_delayed_observation(), {}
    
    def step(self, action):
        """Execute one timestep with production-grade dynamics."""
        
        # ═══════════════════════════════════════════════════════════════
        # ACTION PROCESSING
        # ═══════════════════════════════════════════════════════════════
        
        # Normalize action
        action = np.clip(action, 0.0, 1.0)
        action_sum = np.sum(action)
        if action_sum < 1e-8:
            action = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        else:
            action = action / action_sum
        
        w5, w7, w8 = action
        
        # ═══════════════════════════════════════════════════════════════
        # TRAFFIC MODEL (Non-stationary + Burst)
        # ═══════════════════════════════════════════════════════════════
        
        self._update_traffic_model()
        
        # Calculate load per server
        # Load = traffic * weight / capacity (normalized)
        load_h5 = self.traffic_intensity * w5 / self.capacities[0] * self.max_capacity
        load_h7 = self.traffic_intensity * w7 / self.capacities[1] * self.max_capacity
        load_h8 = self.traffic_intensity * w8 / self.capacities[2] * self.max_capacity
        
        loads = np.array([load_h5, load_h7, load_h8], dtype=np.float32)
        loads = np.clip(loads, 0.0, 1.0)
        
        # ═══════════════════════════════════════════════════════════════
        # QUEUE DYNAMICS (M/M/1 with finite buffer)
        # ═══════════════════════════════════════════════════════════════
        
        # Incoming requests (based on traffic + burst)
        incoming_rate = self.traffic_intensity * 1000 * (1 + self.burst_intensity)
        incoming_per_server = incoming_rate * np.array([w5, w7, w8])
        
        # Queue update (simplified)
        # queue_new = queue_old + incoming - processed
        processing_rate = self.capacities * 10  # Simplified: capacity proportional to processing
        self.queue_lengths = np.clip(
            self.queue_lengths + incoming_per_server - processing_rate,
            0, self.buffer_sizes
        )
        
        # ═══════════════════════════════════════════════════════════════
        # LATENCY MODEL (M/M/1 + Queueing delay)
        # ═══════════════════════════════════════════════════════════════
        
        base_lat = 10.0  # ms
        
        # M/M/1 latency: L = base / (1 - utilization)
        latencies = []
        for i, (load, queue) in enumerate(zip(loads, self.queue_lengths)):
            util = min(load, 0.99)
            queue_delay = queue / self.buffer_sizes[i] * 50  # Queue adds delay
            mm1_lat = base_lat / (1.0 - util) if util < 0.99 else 1000.0
            total_lat = mm1_lat + queue_delay
            
            # Add measurement noise
            total_lat *= (1.0 + np.random.normal(0, self.latency_noise))
            total_lat = min(total_lat, 1000.0)
            latencies.append(total_lat)
        
        lat_h5, lat_h7, lat_h8 = latencies
        
        # ═══════════════════════════════════════════════════════════════
        # PACKET LOSS (Buffer overflow)
        # ═══��═══════════════════════════════════════════════════════════
        
        # Packet loss = overflow / incoming
        packet_loss = np.zeros(3, dtype=np.float32)
        for i in range(3):
            if incoming_per_server[i] > self.buffer_sizes[i]:
                overflow = incoming_per_server[i] - self.buffer_sizes[i]
                packet_loss[i] = min(1.0, overflow / incoming_per_server[i])
        
        total_packet_loss = np.mean(packet_loss)
        self.packet_loss_history.append(total_packet_loss)
        
        # ═══════════════════════════════════════════════════════════════
        # RTT ESTIMATE (Exponential moving average)
        # ═══════════════════════════════════════════════════════════════
        
        for i, lat in enumerate([lat_h5, lat_h7, lat_h8]):
            self.rtt_estimates[i] = 0.7 * self.rtt_estimates[i] + 0.3 * lat
        
        # ═══════════════════════════════════════════════════════════════
        # THROUGHPUT CALCULATION
        # ═══════════════════════════════════════════════════════════════
        
        # Throughput = traffic * (1 - packet_loss)
        throughput = self.traffic_intensity * (1 - total_packet_loss)
        self.throughput_history.append(throughput)
        
        # ═══════════════════════════════════════════════════════════════
        # SLA COMPLIANCE CHECK
        # ═══════════════════════════════════════════════════════════════
        
        p99_latency = max(lat_h5, lat_h7, lat_h8)
        p50_latency = np.median([lat_h5, lat_h7, lat_h8])
        
        sla_violation = 0
        if p99_latency > self.sla_latency_threshold:
            sla_violation += 1
        if total_packet_loss > self.sla_packet_loss_threshold:
            sla_violation += 1
        if throughput < self.sla_throughput_min:
            sla_violation += 1
        
        # ══════════════════════���════════════════════════════════════════
        # OVERLOAD DETECTION
        # ═══════════════════════════════════════════════════════════════
        
        overload_threshold = 0.95
        overload_events = np.sum(loads > overload_threshold)
        
        # ═══════════════════════════════════════════════════════════════
        # REWARD CALCULATION (SLA-based, NO fairness penalty)
        # SCALED: Reward in range [-10, 10] for stable training
        # ═══════════════════════════════════════════════════════════════
        
        # Core KPIs (Production-grade)
        # 1. Latency penalty (p99) - SCALED
        latency_penalty = (p99_latency / 500.0) ** 2  # Quadratic, scaled
        
        # 2. Packet loss penalty - SCALED
        packet_loss_penalty = 5.0 * total_packet_loss
        
        # 3. SLA violation penalty - SCALED
        sla_penalty = 2.0 * sla_violation
        
        # 4. Overload penalty - SCALED
        overload_penalty = 5.0 * overload_events
        
        # 5. Throughput reward - SCALED
        throughput_reward = throughput * 2.0 if sla_violation == 0 else throughput * 0.5
        
        # Total reward (scaled to [-10, 10] range)
        reward = throughput_reward - latency_penalty - packet_loss_penalty - sla_penalty - overload_penalty
        
        # Clip reward for stability
        reward = np.clip(reward, -20.0, 10.0)
        
        # ═══════════════════════════════════════════════════════════════
        # STATE UPDATE
        # ═══════════════════════════════════════════════════════════════
        
        self._update_state()
        
        # Update history
        self.load_history.append(loads.copy())
        self.latency_history.append([lat_h5, lat_h7, lat_h8])
        
        # Episode stats
        self.episode_stats['total_throughput'] += throughput
        self.episode_stats['total_packet_loss'] += total_packet_loss
        self.episode_stats['sla_violations'] += sla_violation
        self.episode_stats['overload_events'] += overload_events
        if self.burst_intensity > 0.3:
            self.episode_stats['burst_events'] += 1
        self.episode_stats['p99_latency'].append(p99_latency)
        self.episode_stats['p50_latency'].append(p50_latency)
        
        # Next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Info
        info = {
            'throughput': float(throughput),
            'p99_latency': float(p99_latency),
            'p50_latency': float(p50_latency),
            'packet_loss': float(total_packet_loss),
            'sla_violation': int(sla_violation),
            'overload_events': int(overload_events),
            'burst_intensity': float(self.burst_intensity),
            'traffic_intensity': float(self.traffic_intensity),
            'loads': loads.tolist(),
            'latencies': [lat_h5, lat_h7, lat_h8],
            'weights': action.tolist(),
        }
        
        return self._get_delayed_observation(), reward, done, False, info
    
    def _update_traffic_model(self):
        """
        Update traffic with non-stationary + burst model.
        
        Key features:
        1. Random walk with mean reversion
        2. Burst events (correlated)
        3. Trend persistence
        """
        
        # ═══════════════════════════════════════════════════════════════
        # BASE TRAFFIC (Random walk with mean reversion)
        # ═══════════════════════════════════════════════════════════════
        
        target = np.random.uniform(0.2, 0.6)
        self.traffic_intensity += (target - self.traffic_intensity) * 0.05
        
        # ═══════════════════════════════════════════════════════════════
        # BURST MODEL (Correlated spikes)
        # ═══════════════════════════════════════════════════════════════
        
        self.steps_since_burst += 1
        
        # Burst probability increases with time since last burst
        burst_prob = self.burst_probability * (1 + self.steps_since_burst / 100.0)
        
        if np.random.rand() < burst_prob:
            # New burst event
            self.burst_intensity = np.random.uniform(0.3, 0.8)
            self.steps_since_burst = 0
            self.traffic_intensity = min(1.0, self.traffic_intensity + self.burst_intensity)
        elif self.burst_intensity > 0:
            # Burst continuation (correlated)
            if np.random.rand() < self.burst_correlation:
                # Burst continues
                self.burst_intensity *= 0.9
                self.traffic_intensity = min(1.0, self.traffic_intensity + self.burst_intensity * 0.5)
            else:
                # Burst ends
                self.burst_intensity *= 0.5
        
        # ═══════════════════════════════════════════════════════════════
        # TREND DETECTION
        # ═══════════════════════════════════════════════════════════════
        
        if len(self.load_history) >= 3:
            recent_loads = np.mean(list(self.load_history)[-3:], axis=0)
            older_loads = np.mean(list(self.load_history)[-6:-3], axis=0) if len(self.load_history) >= 6 else recent_loads
            trend = np.mean(recent_loads - older_loads)
            self.traffic_trend = np.clip(trend * 10, -1, 1)
        
        # Clamp traffic
        self.traffic_intensity = np.clip(self.traffic_intensity, 0.1, 1.0)
        
        # ═══════════════════════════════════════════════════════════════
        # SUSPICIOUS TRAFFIC (DoS detection)
        # ═══════════════════════════════════════════════════════════════
        
        if self.burst_intensity > 0.5:
            # High burst = potential DoS
            self.suspicious_ratio = min(1.0, self.suspicious_ratio + 0.1)
        else:
            self.suspicious_ratio = max(0.0, self.suspicious_ratio - 0.05)
    
    def _update_state(self):
        """Update state vector (22 dimensions)."""
        
        # Queue length (normalized)
        queue_norm = self.queue_lengths / self.buffer_sizes
        
        # Latency (normalized)
        if len(self.latency_history) > 0:
            recent_lat = np.array(list(self.latency_history)[-1], dtype=np.float32)
            lat_norm = recent_lat / 200.0
        else:
            lat_norm = np.zeros(3, dtype=np.float32)
        
        # Packet loss (normalized)
        if len(self.packet_loss_history) > 0:
            loss_norm = np.mean(list(self.packet_loss_history)[-5:])
        else:
            loss_norm = 0.0
        loss_per_server = np.ones(3) * loss_norm
        
        # RTT (normalized)
        rtt_norm = self.rtt_estimates / 200.0
        
        # Traffic trend
        trend_onehot = np.zeros(3)
        if self.traffic_trend > 0.3:
            trend_onehot[2] = 1  # Up
        elif self.traffic_trend < -0.3:
            trend_onehot[0] = 1  # Down
        else:
            trend_onehot[1] = 1  # Stable
        
        # Recent throughput
        if len(self.throughput_history) > 0:
            recent_throughput = np.mean(list(self.throughput_history)[-5:])
        else:
            recent_throughput = 0.0
        
        # Capacity (normalized)
        cap_norm = self.capacities / 150.0
        
        # Global load
        global_load = self.traffic_intensity
        
        # Time since last burst (normalized)
        burst_time = min(1.0, self.steps_since_burst / 50.0)
        
        # Suspicious ratio
        suspicious = self.suspicious_ratio
        
        # Combine into state (22 dims)
        self.state = np.array([
            queue_norm[0], queue_norm[1], queue_norm[2],  # 3
            lat_norm[0], lat_norm[1], lat_norm[2],  # 3
            loss_per_server[0], loss_per_server[1], loss_per_server[2],  # 3
            rtt_norm[0], rtt_norm[1], rtt_norm[2],  # 3
            trend_onehot[0], trend_onehot[1], trend_onehot[2],  # 3
            float(recent_throughput),  # 1
            cap_norm[0], cap_norm[1], cap_norm[2],  # 3
            float(global_load),  # 1
            float(burst_time),  # 1
            float(suspicious),  # 1
        ], dtype=np.float32)
        
        # Add to history for delayed observation
        self.state_history.append(self.state.copy())
    
    def _get_delayed_observation(self):
        """Get observation with delay (partial observability)."""
        if len(self.state_history) > self.observation_delay:
            return self.state_history[-self.observation_delay - 1]
        return self.state_history[0]
    
    def get_stats(self):
        """Get episode statistics."""
        stats = self.episode_stats.copy()
        if len(stats['p99_latency']) > 0:
            stats['avg_p99_latency'] = np.mean(stats['p99_latency'])
            stats['avg_p50_latency'] = np.mean(stats['p50_latency'])
        else:
            stats['avg_p99_latency'] = 0.0
            stats['avg_p50_latency'] = 0.0
        return stats


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO VARIANTS
# ═══════════════════════════════════════════════════════════════════════

class BurstTrafficEnv(SDNProductionEnv):
    """High burst probability environment."""
    
    def __init__(self, config=None):
        config = config or {}
        config['burst_prob'] = config.get('burst_prob', 0.25)  # Higher burst
        config['obs_delay'] = config.get('obs_delay', 3)  # More delay
        super().__init__(config)


class HighNoiseEnv(SDNProductionEnv):
    """High measurement noise environment."""
    
    def __init__(self, config=None):
        config = config or {}
        config['measurement_noise'] = config.get('measurement_noise', 0.2)
        config['latency_noise'] = config.get('latency_noise', 0.3)
        super().__init__(config)


class LowSLAEnv(SDNProductionEnv):
    """Strict SLA requirements environment."""
    
    def __init__(self, config=None):
        config = config or {}
        config['sla_latency_ms'] = config.get('sla_latency_ms', 50.0)  # Stricter
        config['sla_packet_loss'] = config.get('sla_packet_loss', 0.005)  # 0.5%
        super().__init__(config)


class DynamicCapacityEnv(SDNProductionEnv):
    """Capacity fluctuates over time (hardware degradation)."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.degradation_rate = 0.001
    
    def step(self, action):
        # Gradual capacity degradation
        self.capacities *= (1 - self.degradation_rate)
        self.capacities = np.clip(self.capacities, 5.0, None)  # Min 5 Mbps
        return super().step(action)


# ═══════════════════════════════════════════════════════════════════════
# ENVIRONMENT REGISTRATION
# ═══════════════════════════════════════════════════════════════════════

def make_production_env(env_id='SDNProduction-v0', seed=None, **kwargs):
    """Create production environment."""
    
    env_map = {
        'SDNProduction-v0': SDNProductionEnv,
        'SDNBurst-v0': BurstTrafficEnv,
        'SDNHighNoise-v0': HighNoiseEnv,
        'SDNLowSLA-v0': LowSLAEnv,
        'SDNDynamicCapacity-v0': DynamicCapacityEnv,
    }
    
    env_class = env_map.get(env_id, SDNProductionEnv)
    env = env_class(config=kwargs)
    
    if seed is not None:
        env.reset(seed=seed)
    
    return env


if __name__ == '__main__':
    # Test environment
    print("Testing SDN Production Environment...")
    
    env = make_production_env('SDNProduction-v0', seed=42)
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Run random policy
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 20 == 0:
            print(f"Step {i}: reward={reward:.2f}, latency={info['p99_latency']:.1f}ms, "
                  f"loss={info['packet_loss']:.3f}, burst={info['burst_intensity']:.2f}")
        
        if done:
            break
    
    stats = env.get_stats()
    print(f"\nEpisode stats:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Avg p99 latency: {stats['avg_p99_latency']:.1f}ms")
    print(f"  Total packet loss: {stats['total_packet_loss']:.3f}")
    print(f"  SLA violations: {stats['sla_violations']}")
    print(f"  Burst events: {stats['burst_events']}")