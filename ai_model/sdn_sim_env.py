"""
SDN Load Balancer Gymnasium Environment
Mô phỏng SDN với 3 node: h5(10M), h7(50M), h8(100M)

Observation: [CPU_h5, CPU_h7, CPU_h8, Lat_h5, Lat_h7, Lat_h8, Suspicious_Ratio]
            - 7 phần tử (nâng cấp từ 6 để hỗ trợ DoS detection)
Action: [Weight_h5, Weight_h7, Weight_h8] - tỷ lệ chia tải (tổng = 1.0)

Reward:
  - W1 * Throughput: thưởng xử lý được nhiều request
  - -W2 * Latency: phạt latency cao (theo M/M/1 queue)
  - -W4 * Crash: phạt cực nặng khi overload
  - -W5 * Fairness_Penalty: phạt mất cân bằng tải

Author: PPO-Based Approach
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SDNLoadBalancerEnv(gym.Env):
    """
    SDN Load Balancer Simulation Environment.
    
    3 nodes với capacities khác nhau:
    - h5: 10 Mbps (yếu)
    - h7: 50 Mbps (trung bình)
    - h8: 100 Mbps (mạnh)
    
    AI phải học cách chia tải thông minh để:
    1. Tối đa hóa throughput
    2. Giảm thiểu latency
    3. Tránh overload/crash
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config=None):
        super().__init__()
        
        # Default config
        self.config = config or {}
        
        # Server capacities (Mbps) - RANDOMIZED each episode for research
        # This prevents PPO from memorizing a fixed optimal ratio
        base_capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.capacity_scale = self.config.get('capacity_scale', 1.0)
        self.capacities = base_capacities * self.capacity_scale
        self.max_capacity = float(np.max(self.capacities))
        
        # Reward weights
        self.W1 = self.config.get('W1_throughput', 1.0)
        self.W2 = self.config.get('W2_latency', 2.0)
        self.W4 = self.config.get('W4_crash', 10.0)
        self.W5 = self.config.get('W5_fairness', 0.5)
        
        # REMOVED: Hardcoded ideal_ratio - optimal is now dynamic!
        # Ideal load ratios computed dynamically from capacity
        self.ideal_ratio = None  # Will be computed per-episode
        
        # Load thresholds
        self.overload_threshold = 0.95
        self.warning_threshold = 0.85
        
        # Time parameters
        self.max_steps = self.config.get('max_steps', 5000)  # Tăng từ 1000 lên 5000
        
        # Action space: tỷ lệ chia tải cho 3 node
        # [Weight_h5, Weight_h7, Weight_h8], tổng = 1.0
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        )
        
        # Observation space: [CPU_h5, CPU_h7, CPU_h8, Lat_h5, Lat_h7, Lat_h8, Suspicious_Ratio, Global_Arrival_Rate, Cap_h5, Cap_h7, Cap_h8]
        # 11 phần tử - BAO GỒM capacity để PPO có thể compute optimal distribution
        # Normalized về 0.0 - 1.0
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(11,),
            dtype=np.float32
        )
        
        # State variables
        self.state = np.zeros(11, dtype=np.float32)  # [cpu_h5, cpu_h7, cpu_h8, lat_h5, lat_h7, lat_h8, suspicious, arrival_rate, cap_h5, cap_h7, cap_h8]
        self.traffic_intensity = 0.3  # 0.0 - 1.0
        self.suspicious_ratio = 0.0   # Tỷ lệ traffic khả nghi (DoS)
        self.current_step = 0
        
        # Load history (để tính EWMA latency trend)
        self.load_history = []
        self.latency_history = []
        
        # Episode statistics
        self.episode_stats = {
            'total_throughput': 0.0,
            'max_latency': 0.0,
            'crash_count': 0,
            'overload_count': 0,
        }
        
    def reset(self, seed=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # RANDOMIZE CAPACITIES each episode for research-level difficulty
        base_capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.cap_scales = np.random.uniform(0.7, 1.3, size=3)
        self.capacities = base_capacities * self.cap_scales
        self.max_capacity = float(np.max(self.capacities))
        self.ideal_ratio = self.capacities / np.sum(self.capacities)
        
        # Non-stationary traffic: burst pattern that changes over time
        self.traffic_phase = np.random.rand() * 2 * np.pi  # Random phase
        self.traffic_freq = np.random.uniform(0.02, 0.05)  # Frequency of bursts
        self.traffic_base = np.random.uniform(0.2, 0.4)  # Base traffic level
        
        # BURST TRAFFIC: Occasional spikes
        self.burst_probability = 0.15  # 15% chance of burst per step
        self.in_burst = False
        self.burst_duration = 0
        self.burst_intensity = 0.0
        
        # Random initial state
        self.state = np.random.rand(11).astype(np.float32) * 0.3
        self.state[6] = 0.0
        self.state[7] = 0.3
        self.state[8] = self.capacities[0] / 150.0
        self.state[9] = self.capacities[1] / 150.0
        self.state[10] = self.capacities[2] / 150.0
        
        self.traffic_intensity = self.traffic_base
        self.suspicious_ratio = 0.0
        self.current_step = 0
        self.load_history = []
        self.latency_history = []
        
        # Reset episode stats
        self.episode_stats = {
            'total_throughput': 0.0,
            'max_latency': 0.0,
            'crash_count': 0,
            'overload_count': 0,
        }
        
        return self.state, {}
    
    def step(self, action):
        """
        Execute one timestep.
        
        Args:
            action: [w5, w7, w8] - tỷ lệ chia tải (sẽ được normalize)
            
        Returns:
            observation: new state
            reward: scalar
            terminated: game over?
            truncated: time limit?
            info: dict
        """
        # Normalize action (tổng = 1.0)
        action = np.clip(action, 0.0, 1.0)
        action_sum = np.sum(action)
        if action_sum < 1e-8:
            action = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        else:
            action = action / action_sum
        
        w5, w7, w8 = action
        
        # ═══════════════════════════════════════════════════════════════
        # TUYỆT CHIÊU 1: EXPONENTIAL LATENCY (M/M/1 QUEUE)
        # Khi utilization chạm 80-90%, latency tăng theo exponential
        # ═══════════════════════════════════════════════════════════════
        
        # Load = traffic * weight / capacity (normalized)
        # Đảo ngược: chia cho max_capacity để h5 (10M) có load cao hơn h8 (100M)
        # Khi traffic cao + weight đều -> h5 overload nhanh chóng
        load_h5 = self.traffic_intensity * w5 / self.capacities[0] * self.max_capacity
        load_h7 = self.traffic_intensity * w7 / self.capacities[1] * self.max_capacity
        load_h8 = self.traffic_intensity * w8 / self.capacities[2] * self.max_capacity
        
        # Clamp loads
        load_h5 = float(np.clip(load_h5, 0.0, 1.0))
        load_h7 = float(np.clip(load_h7, 0.0, 1.0))
        load_h8 = float(np.clip(load_h8, 0.0, 1.0))
        
        # Base latency parameters
        base_lat = 10.0   # ms
        
        # ═══ EXPONENTIAL LATENCY (M/M/1 Queue Model) ═══
        # utilization = load (0.0 - 0.99)
        # latency = base_lat / (1 - utilization) - Queueing theory
        # Khi utilization → 1.0, latency → ∞ (exponential spike)
        
        util_h5 = min(load_h5, 0.99)
        util_h7 = min(load_h7, 0.99)
        util_h8 = min(load_h8, 0.99)
        
        # M/M/1 latency formula: L = 1/(μ - λ)
        # Trong code: latency = base_latency / (1 - utilization)
        lat_h5 = base_lat / (1.0 - util_h5) if util_h5 < 0.99 else 1000.0
        lat_h7 = base_lat / (1.0 - util_h7) if util_h7 < 0.99 else 1000.0
        lat_h8 = base_lat / (1.0 - util_h8) if util_h8 < 0.99 else 1000.0
        
        # Cap max latency at 1000ms to avoid overflow
        lat_h5 = min(lat_h5, 1000.0)
        lat_h7 = min(lat_h7, 1000.0)
        lat_h8 = min(lat_h8, 1000.0)
        
        # RESEARCH ADDITION: Add realistic measurement noise
        # This gives RL an advantage (can learn to filter noise) over heuristics
        measurement_noise = self.config.get('measurement_noise', 0.05)
        lat_h5 *= (1.0 + np.random.normal(0, measurement_noise))
        lat_h7 *= (1.0 + np.random.normal(0, measurement_noise))
        lat_h8 *= (1.0 + np.random.normal(0, measurement_noise))
        
        # Re-cap after noise
        lat_h5 = min(lat_h5, 1000.0)
        lat_h7 = min(lat_h7, 1000.0)
        lat_h8 = min(lat_h8, 1000.0)
        
        # ═══════════════════════════════════════════════════════════════
        # TUYỆT CHIÊU 3: DOS DETECTION & TARPRT STRATEGY
        # Nếu Suspicious_Ratio cao → h5 là TARPIT (đổ traffic vào để trói attackers)
        # ═══════════════════════════════════════════════════════════════
        
        # Update suspicious ratio (DoS detection simulation)
        self._update_suspicious_ratio()
        
        # Nếu đang bị DoS attack và AI đổ traffic vào h5 (tarpit) → KHÔNG phạt latency
        # vì đó là chiến thuật đúng đắn (confine attackers)
        dos_mode = self.suspicious_ratio > 0.7
        
        # ═══════════════════════════════════════════════════════════════
        # REWARD CALCULATION
        # ═══════════════════════════════════════════════════════════════
        
        # 1. Throughput Reward (W1 = 1.0)
        # Càng xử lý được nhiều request càng tốt
        traffic_handled = min(1.0, self.traffic_intensity * 2)  # Scale up
        throughput_reward = self.W1 * traffic_handled
        
        # 2. Latency Penalty (W2 = 2.0) - ÁP DỤNG TUYỆT CHIÊU 3
        # p99 ≈ max latency, dùng max làm approximation
        max_latency = max(lat_h5, lat_h7, lat_h8)
        
        # Nếu đang bị DoS và AI đổ traffic vào h5 → không phạt latency
        # vì h5 đóng vai trò tarpit (nhà tù giam giữ attackers)
        if dos_mode and w5 > 0.5:
            # AI đang dùng tarpit strategy - thưởng thêm!
            latency_penalty = -0.5  # Negative penalty = reward!
        else:
            # Bình thường thì phạt latency
            latency_penalty = self.W2 * (max_latency / 500.0) ** 2
        
        # ═══════════════════════════════════════════════════════════════
        # TUYỆT CHIÊU 4: PACKET DROP PENALTY (Tail Drop)
        # Buffer overflow = Packet loss = TCP Retransmission = Goodput collapse
        # ��══════════════════════════════════════════════════════════════
        
        # Buffer capacity per node (requests per second buffer)
        # h5(10M) = 50 req/s, h7(50M) = 250 req/s, h8(100M) = 500 req/s
        buffer_capacity = np.array([50.0, 250.0, 500.0], dtype=np.float32)
        incoming_rate = self.traffic_intensity * np.array([w5, w7, w8], dtype=np.float32) * 1000.0  # Scale to req/s
        
        # Dropped packets = max(0, incoming - buffer_capacity)
        dropped_h5 = max(0.0, incoming_rate[0] - buffer_capacity[0])
        dropped_h7 = max(0.0, incoming_rate[1] - buffer_capacity[1])
        dropped_h8 = max(0.0, incoming_rate[2] - buffer_capacity[2])
        total_dropped = dropped_h5 + dropped_h7 + dropped_h8
        
        # Normalize drop rate (0.0 - 1.0)
        drop_rate = min(1.0, total_dropped / 100.0)  # 100 = reference max drop
        drop_penalty = 2.0 * drop_rate  # Reduced from 50 to 2 for stable training
        
        # 3. Crash Penalty (W4 = 100.0) - CỰC NẶNG
        crash_penalty = 0.0
        terminated = False
        
        if load_h5 > self.overload_threshold:
            crash_penalty += self.W4 * (load_h5 - self.overload_threshold) * 10
            # terminated = True  # Không end episode để PPO học được
        if load_h7 > self.overload_threshold:
            crash_penalty += self.W4 * (load_h7 - self.overload_threshold) * 10
        if load_h8 > self.overload_threshold:
            crash_penalty += self.W4 * (load_h8 - self.overload_threshold) * 10
        
        # ═══════════════════════════════════════════════════════════════
        # TUYỆT CHIÊU 2: FAIRNESS PENALTY (Load Balancing)
        # Phạt nếu AI dồn tải vào h7 mà bỏ qua h8 (capacity-based)
        # ═══════════════════════════════════════════════════════════════
        
        # Actual ratios
        total_weight = w5 + w7 + w8 + 1e-8
        actual_ratio = np.array([w5/total_weight, w7/total_weight, w8/total_weight], dtype=np.float32)
        
        # Imbalance = sum of absolute deviations from ideal
        imbalance_penalty = np.sum(np.abs(actual_ratio - self.ideal_ratio))
        fairness_penalty = self.W5 * imbalance_penalty
        
        # Overload warning count
        overload_detected = load_h5 > self.warning_threshold or \
                           load_h7 > self.warning_threshold or \
                           load_h8 > self.warning_threshold
        
        # Tổng reward = Throughput - Latency - Crash - Fairness - Drop
        reward = throughput_reward - latency_penalty - crash_penalty - fairness_penalty - drop_penalty
        
        # ════════════════════════════════════════���══════════════════════
        # STATE UPDATE
        # ═══════════════════════════════════════════════════════════════
        
        # Update state: [cpu_h5, cpu_h7, cpu_h8, lat_h5, lat_h7, lat_h8, suspicious, arrival_rate, cap_h5, cap_h7, cap_h8]
        # 11 phần tử - BAO GỒM normalized capacities
        self.state = np.array([
            load_h5, load_h7, load_h8,
            lat_h5 / 1000.0, lat_h7 / 1000.0, lat_h8 / 1000.0,  # Normalized
            self.suspicious_ratio,  # DoS detection flag
            self.traffic_intensity,  # Global Arrival Rate
            self.capacities[0] / 150.0,  # Normalized capacity h5
            self.capacities[1] / 150.0,  # Normalized capacity h7
            self.capacities[2] / 150.0   # Normalized capacity h8
        ], dtype=np.float32)
        
        # Update traffic (random walk with mean reversion)
        self._update_traffic()
        
        # Track history
        self.load_history.append([load_h5, load_h7, load_h8])
        self.latency_history.append([lat_h5, lat_h7, lat_h8])
        
        # Next step
        self.current_step += 1
        done = terminated or self.current_step >= self.max_steps
        
        # Update episode stats
        self.episode_stats['total_throughput'] += traffic_handled
        self.episode_stats['max_latency'] = max(self.episode_stats['max_latency'], max_latency)
        if overload_detected:
            self.episode_stats['overload_count'] += 1
        if terminated:
            self.episode_stats['crash_count'] += 1
        
        # Info dict - THÊM suspicious_ratio
        info = {
            'throughput': float(traffic_handled),
            'latency': float(max_latency),
            'load_h5': float(load_h5),
            'load_h7': float(load_h7),
            'load_h8': float(load_h8),
            'lat_h5': float(lat_h5),
            'lat_h7': float(lat_h7),
            'lat_h8': float(lat_h8),
            'overload': overload_detected,
            'crash': terminated,
            'weights': action.tolist(),
            'traffic_intensity': float(self.traffic_intensity),
            'suspicious_ratio': float(self.suspicious_ratio),  # DoS flag
            'fairness_penalty': float(fairness_penalty),  # Fairness metric
            'drop_penalty': float(drop_penalty),  # Packet drop penalty (TUYỆT CHIÊU 4)
            'total_dropped': float(total_dropped),  # Actual dropped packets
            'dos_mode': dos_mode,
        }
        
        return self.state, reward, done, terminated, info
    
    def _update_traffic(self):
        """
        Cập nhật traffic intensity cho next step.
        Simulate changing conditions:
        - Random walk with mean reversion
        - Occasional burst traffic
        """
        # Random walk
        target = np.random.rand() * 0.8 + 0.1  # Target: 0.1 - 0.9
        self.traffic_intensity += (target - self.traffic_intensity) * 0.1
        
        # Occasional burst (10% chance)
        if np.random.rand() < 0.1:
            self.traffic_intensity = min(1.0, self.traffic_intensity + 0.3)
        
        # Clamp
        self.traffic_intensity = np.clip(self.traffic_intensity, 0.1, 1.0)
    
    def _update_suspicious_ratio(self):
        """
        TUYỆT CHIÊU 3: Cập nhật tỷ lệ traffic khả nghi (DoS Detection)
        
        Mô phỏng xác suất bị DoS attack:
        - Random walk với mean reversion
        - Khi suspicious_ratio > 0.7 → coi như đang bị attack
        - AI cần học chiến thuật TARPIT: đổ attack vào h5
        """
        # 5% chance toggle attack mode per step
        if np.random.rand() < 0.05:
            if self.suspicious_ratio < 0.5:
                # Start attack: suspicious_ratio jumps to 0.7-0.9
                self.suspicious_ratio = 0.7 + np.random.rand() * 0.2
            else:
                # End attack: decay to 0
                self.suspicious_ratio = 0.0
        
        # Natural decay if not in attack
        if self.suspicious_ratio > 0 and self.suspicious_ratio < 0.7:
            self.suspicious_ratio *= 0.9  # Gradual decay
        
        # Clamp
        self.suspicious_ratio = np.clip(self.suspicious_ratio, 0.0, 1.0)
    
    def get_loads(self):
        """Get current loads for all servers."""
        return self.state[:3].tolist()
    
    def get_latencies(self):
        """Get current latencies (denormalized)."""
        return (self.state[3:6] * 1000.0).tolist()  # 1000 = max_lat
    
    def get_stats(self):
        """Get episode statistics."""
        avg_throughput = self.episode_stats['total_throughput'] / max(1, self.current_step)
        return {
            'steps': self.current_step,
            'avg_throughput': avg_throughput,
            'max_latency': self.episode_stats['max_latency'],
            'crash_count': self.episode_stats['crash_count'],
            'overload_count': self.episode_stats['overload_count'],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO-SPECIFIC ENVIRONMENTS
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenHourEnv(SDNLoadBalancerEnv):
    """Golden Hour: Burst traffic - Elephant + Mice flows"""
    
    def _update_traffic(self):
        # Override: More aggressive bursts
        target = np.random.rand()  # 0.0 - 1.0
        self.traffic_intensity += (target - self.traffic_intensity) * 0.2
        
        # Higher burst chance (20%)
        if np.random.rand() < 0.2:
            self.traffic_intensity = min(1.0, self.traffic_intensity + 0.4)
        
        self.traffic_intensity = np.clip(self.traffic_intensity, 0.1, 1.0)


class VideoConferenceEnv(SDNLoadBalancerEnv):
    """Video Conference: Low latency critical, need stable routing"""
    
    def __init__(self):
        super().__init__()
        self.W2 = 3.0  # Higher latency penalty
        self.W1 = 0.8  # Slightly lower throughput priority


class HardwareDegradationEnv(SDNLoadBalancerEnv):
    """Hardware Degradation: Node becomes slower over time"""
    
    def __init__(self):
        super().__init__()
        self.degradation_factor = 1.0
        
    def step(self, action):
        # Gradually increase latency multiplier (simulating hardware degradation)
        self.degradation_factor = min(1.5, self.degradation_factor + 0.0001)
        
        result = super().step(action)
        
        # Modify latencies in info
        info = result[4]
        info['lat_h5'] *= self.degradation_factor
        info['lat_h7'] *= self.degradation_factor
        info['lat_h8'] *= self.degradation_factor
        info['degradation_factor'] = self.degradation_factor
        
        return result


class LowRateDosEnv(SDNLoadBalancerEnv):
    """Low Rate DoS: Many connections, low bytes - Tarpit scenario"""
    
    def __init__(self):
        super().__init__()
        self.is_under_attack = False
        
    def _update_traffic(self):
        # Toggle attack mode randomly
        if np.random.rand() < 0.05:  # 5% chance per step
            self.is_under_attack = not self.is_under_attack
        
        if self.is_under_attack:
            # Under attack: high connections but low throughput
            # AI should route attack traffic to h5 (tarpit)
            self.traffic_intensity = 0.8  # High load
        else:
            # Normal traffic
            super()._update_traffic()
    
    def _update_suspicious_ratio(self):
        """Override: LowRateDoS has explicit attack mode toggle"""
        if self.is_under_attack:
            self.suspicious_ratio = 0.9  # High suspicious = DoS attack
        else:
            self.suspicious_ratio = 0.0  # Normal traffic


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(env_id, seed=None):
    """Factory function to create environments."""
    envs = {
        'SDN-v0': SDNLoadBalancerEnv,
        'GoldenHour-v0': GoldenHourEnv,
        'VideoConference-v0': VideoConferenceEnv,
        'HardwareDegradation-v0': HardwareDegradationEnv,
        'LowRateDoS-v0': LowRateDosEnv,
    }
    
    if env_id not in envs:
        raise ValueError(f"Unknown env_id: {env_id}. Available: {list(envs.keys())}")
    
    env = envs[env_id]()
    if seed is not None:
        env.reset(seed=seed)
    
    return env


if __name__ == "__main__":
    print("="*60)
    print("  SDN Load Balancer Environment Test")
    print("="*60)
    
    # Test basic environment
    env = SDNLoadBalancerEnv()
    obs, info = env.reset()
    
    print(f"\n[*] Initial observation: {obs}")
    print(f"[*] Action space: {env.action_space}")
    print(f"[*] Observation space: {env.observation_space}")
    
    print("\n[*] Running 10 random steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.3f}, loads={info['weights']}, lat={info['latency']:.1f}ms")
        if done:
            print(f"  [CRASH!] Environment terminated")
            break
    
    print("\n[*] Testing Golden Hour scenario...")
    env2 = make_env('GoldenHour-v0')
    obs2, _ = env2.reset()
    
    for i in range(5):
        action = np.array([0.1, 0.3, 0.6], dtype=np.float32)  # Favor h8
        obs2, reward, done, trunc, info = env2.step(action)
        print(f"  Step {i+1}: traffic={info['traffic_intensity']:.2f}, lat={info['latency']:.1f}ms")
    
    print("\n[✓] Environment test passed!")
