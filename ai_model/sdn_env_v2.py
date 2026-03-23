"""
SDN Offline Environment V2 cho TFT-CQL Actor-Critic.

Khác biệt so với V1 (sdn_env.py):
  - step() trả (next_state, reward_main, done, info)
  - reward_main = throughput utility đơn giản (không gánh constraints)
  - info chứa tất cả constraint signals + evaluation metrics
  - Hỗ trợ mode='train' và mode='eval'

V6 - ROBIN HOOD REWARD:
  - Bỏ capacity_bonus (bias toward h8)
  - Bỏ burst_bonus (bias toward h8)
  - Bỏ weak_server_penalty (thay bằng wastage_penalty)
  - Thêm efficiency_bonus (thưởng hiệu quả)
  - Thêm wastage_penalty (phạt lãng phí)
  - Thêm saving_bonus (thưởng tiết kiệm)
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (CAPACITIES, NUM_ACTIONS, BACKENDS, CAPACITY_RATIOS,
                    UTIL_THRESHOLD, SAFETY_THRESHOLD, FAIRNESS_TOLERANCE)


class SDN_Offline_Env_V2:
    """
    Offline replay environment cho TFT-CQL.

    State: [42 features] x 5 timesteps (V3 feature schema)
    Action: 0=h5, 1=h7, 2=h8 (hoặc theo config)
    Reward: served throughput utility (tách khỏi constraints)

    step() → (next_state, reward_main, done, info)
    """
    def __init__(self, data_x_path, data_y_path, mode='train', metadata=None):
        self.X_data = np.load(data_x_path)
        self.y_labels = np.load(data_y_path)
        self.mode = mode
        self.metadata = metadata

        self.num_samples = len(self.X_data)
        self.current_step = 0
        self.num_actions = NUM_ACTIONS
        self.num_features = self.X_data.shape[2] if self.X_data.ndim == 3 else self.X_data.shape[1]
        self.prev_action = None

        # Evaluation metrics accumulator
        self.eval_metrics = {
            'total_reward': 0.0,
            'overload_count': 0,
            'utilizations': [],
            'fairness_devs': [],
            'action_switches': 0,
            'action_counts': np.zeros(self.num_actions),
            # New metrics for fair evaluation
            'throughputs': [],
            'burst_count': 0,
            'burst_handled': 0,
            'stability_bonus_total': 0.0,
            'diversity_bonus_total': 0.0,
            'burst_bonus_total': 0.0,
            # V6 metrics
            'efficiency_bonus_total': 0.0,
            'wastage_penalty_total': 0.0,
            'saving_bonus_total': 0.0,
        }

        num_features_str = f"{self.num_features} features"
        print(f"[*] SDN Env V2 ({mode}): {self.num_samples} states | "
              f"{num_features_str} | {self.num_actions} actions")

    def reset(self):
        self.current_step = 0
        self.prev_action = None
        self.eval_metrics = {
            'total_reward': 0.0,
            'overload_count': 0,
            'utilizations': [],
            'fairness_devs': [],
            'action_switches': 0,
            'action_counts': np.zeros(self.num_actions),
            # New metrics for fair evaluation
            'throughputs': [],
            'burst_count': 0,
            'burst_handled': 0,
            'stability_bonus_total': 0.0,
            'diversity_bonus_total': 0.0,
            'burst_bonus_total': 0.0,
            # V6 metrics
            'efficiency_bonus_total': 0.0,
            'wastage_penalty_total': 0.0,
            'saving_bonus_total': 0.0,
        }
        return self.X_data[self.current_step]

    def step(self, action):
        state = self.X_data[self.current_step]
        label = self.y_labels[self.current_step]
        last_step = state[-1]

        # ── EXTRACT FEATURES from last timestep ──
        # Feature indices depend on schema; use safe extraction
        byte_rate = last_step[0] if self.num_features > 0 else 0.0
        packet_rate = last_step[1] if self.num_features > 1 else 0.0

        # Per-server utilization (from Group C features if available)
        server_utils = self._extract_utilizations(last_step)
        chosen_util = server_utils[action] if action < len(server_utils) else 0.0

        # Per-server headroom
        server_headrooms = self._extract_headrooms(last_step)

        # ═══════════════════════════════════════════════════════════
        # REWARD FUNCTION V6 - "ROBIN HOOD" 
        # Phiên bản công bằng: Thưởng hiệu quả, phạt lãng phí
        # Nguyên tắc: "Lấy từ người giàu (h8), chia cho người nghèo (h5, h7)"
        # ═══════════════════════════════════════════════════════════
        
        # Capacity weights: h5=0.1, h7=0.5, h8=1.0
        capacity_weights = np.array(CAPACITIES) / np.max(CAPACITIES)
        
        # Throughput thực tế: càng chọn server mạnh + headroom cao = throughput cao
        headroom = max(0.1, 1.0 - chosen_util)
        effective_capacity = CAPACITIES[action] * headroom
        real_throughput = byte_rate * effective_capacity / np.max(CAPACITIES)
        
        # Packet loss rate từ features (Group C)
        group_c_start = 7 + 3 * self.num_actions
        packet_loss_idx = group_c_start + action * 7 + 3
        packet_loss_rate = float(last_step[packet_loss_idx]) if packet_loss_idx < len(last_step) else 0.0
        
        # Queue length từ features (Group C)
        queue_length_idx = group_c_start + action * 7 + 5
        queue_length = float(last_step[queue_length_idx]) if queue_length_idx < len(last_step) else 0.0
        
        # Latency từ features (Group C)
        latency_idx = group_c_start + action * 7 + 4
        latency_ms = float(last_step[latency_idx]) if latency_idx < len(last_step) else 1.0
        
        # Response time = latency + queue_delay
        queue_delay = queue_length / max(1, CAPACITIES[action]) * 10  # ms
        response_time = latency_ms + queue_delay
        
        # Throughput base (không phụ thuộc action)
        throughput = byte_rate + packet_rate
        
        # ═══════════════════════════════════════════════════════════
        # ROBIN HOOD REWARD COMPONENTS
        # ═══════════════════════════════════════════════════════════
        
        # 1. ĐỊNH NGHĨA CHI PHÍ TÀI NGUYÊN (RESOURCE TAX)
        # h5 (10MB) = 10.0, h7 (50MB) = 50.0, h8 (100MB) = 100.0
        capacity_costs = [10.0, 50.0, 100.0]
        action_cost = capacity_costs[action]
        
        # 2. THROUGHPUT REWARD - V10: SCALED cho normalized data
        # V10: 10.0 -> 0.01 (giảm 1000 lần để phù hợp với normalized input)
        throughput_reward = real_throughput * 0.01
        
        # 3. EFFICIENCY BONUS - V10: SCALED
        # efficiency = throughput / cost (càng tiết kiệm càng tốt)
        if action_cost > 0:
            efficiency_bonus = (real_throughput / action_cost) * 0.02  # V10: 20.0 -> 0.02
        else:
            efficiency_bonus = 0.0
        
        # 4. OVERLOAD PENALTY - V10: SCALED CHO NORMALIZED DATA
        # Vấn đề V9: overload_penalty=20000 + StandardScaler = Gradient Explosion!
        # Giải pháp V10: Scale down để phù hợp với normalized input (mean=0, std=1)
        # Giữ nguyên CÁN CÂN: rớt mạng vẫn đáng sợ hơn lãng phí (nhưng số nhỏ hơn)
        overload_penalty = 0.0
        if chosen_util > UTIL_THRESHOLD:
            # V10: 5000.0 -> 5.0 (chia 1000 để tránh gradient explosion)
            base_penalty = 5.0 * (chosen_util - UTIL_THRESHOLD) ** 2
            # Server yếu (h5) bị phạt gấp 3 lần khi overload
            if action == 0:  # h5
                base_penalty *= 3.0
            elif action == 1:  # h7
                base_penalty *= 1.5
            overload_penalty = base_penalty
        if chosen_util > SAFETY_THRESHOLD:
            # V10: 20000.0 -> 20.0 (chia 1000)
            base_penalty = 20.0 * (chosen_util - SAFETY_THRESHOLD) ** 2
            # Server yếu (h5) bị phạt gấp 5 lần khi critically overload
            if action == 0:  # h5
                base_penalty *= 5.0
            elif action == 1:  # h7
                base_penalty *= 2.0
            overload_penalty = base_penalty
        
        # 5. WASTAGE PENALTY - V14: CÂN BẰNG TUYỆT ĐỐI
        # V13: h5 bị bỏ rơi vì thưởng quá thấp
        # V14: Điều chỉnh để h5/h7/h8 đều có cơ hội
        wastage = max(0.0, action_cost - real_throughput * 100.0)
        wastage_penalty = wastage * 0.015  # V14: 0.01 -> 0.015
        
        # 6. SAVING BONUS - V14: ĐỦ LỚN ĐỂ Dám ĐỐI MẶT RỦI RO
        # V13: saving_bonus=0.5 quá nhỏ, AI chọn h7 thay vì h5
        # V14: Tăng để cân bằng với wastage penalty
        saving_bonus = 0.0
        if packet_loss_rate < 0.01:  # Không có packet loss
            if action == 0:  # h5 - server yếu nhất
                saving_bonus = 1.0  # V14: 0.5 -> 1.0
            elif action == 1:  # h7 - server trung bình
                saving_bonus = 0.3  # V14: 0.2 -> 0.3
        
        # 7. STABILITY BONUS - Thưởng khi giữ server ổn định
        stability_bonus = 0.0
        if self.prev_action is not None and action == self.prev_action:
            stability_bonus = 0.0005 * throughput  # V10: 0.5 -> 0.0005
        
        # 8. DIVERSITY BONUS - Giữ thấp để không conflict với CQL
        diversity_bonus = 0.0
        if self.mode == 'train':
            total_actions = max(1, sum(self.eval_metrics['action_counts']))
            action_dist = self.eval_metrics['action_counts'] / total_actions
            eps = 1e-8
            entropy = -np.sum(action_dist * np.log(action_dist + eps))
            max_entropy = np.log(self.num_actions)
            entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0
            diversity_bonus = 0.5 * entropy_ratio  # GIỮ THẤP để CQL hoạt động tốt
        
        # 9. DOS HANDLING - V10: SCALED cho normalized data
        dos_penalty = 0.0
        if packet_rate > 0 and byte_rate > 0:
            packet_byte_ratio = packet_rate / (byte_rate + 0.001)
            if packet_byte_ratio > 100 and byte_rate > 0.3:  # DOS detected
                if action == 2:  # Chọn h8 trong DOS -> PHẠT
                    dos_penalty = 0.03 * (1 + throughput)  # V10: 30.0 -> 0.03
                elif action == 0:  # Chọn h5 trong DOS -> THƯỞNG
                    dos_penalty = -0.01  # V10: -10.0 -> -0.01
        
        # ═══════════════════════════════════════════════════════════
        # REWARD TỔNG - ROBIN HOOD VERSION
        # Bỏ capacity_bonus (bias toward h8)
        # Bỏ burst_bonus (bias toward h8)
        # Bỏ weak_server_penalty (đã có wastage_penalty)
        # ═══════════════════════════════════════════════════════════
        reward_main = (
            throughput_reward           # Thưởng throughput
            + efficiency_bonus           # Thưởng hiệu quả (NEW!)
            - overload_penalty           # Phạt overload
            - wastage_penalty            # Phạt lãng phí (NEW!)
            + saving_bonus               # Thưởng tiết kiệm (NEW!)
            + stability_bonus            # Thưởng ổn định
            + diversity_bonus            # Khuyến khích đa dạng (nhỏ)
            - dos_penalty                # Xử lý DOS
        )
        
        overload_flag = 1 if chosen_util > SAFETY_THRESHOLD else 0
        util_breach = max(0.0, chosen_util - UTIL_THRESHOLD)

        # Fairness deviation: so sánh action distribution vs capacity ratio
        action_ratio = self._get_action_ratio(action)
        fairness_dev = abs(action_ratio - CAPACITY_RATIOS[action])
        
        # Throughput improvement: so sánh với WRR baseline
        wrr_distribution = np.array([0.0625, 0.3125, 0.625])
        ai_distribution = (self.eval_metrics['action_counts'] + (self.eval_metrics['action_counts'][action] + 1) / max(1, sum(self.eval_metrics['action_counts']) + 2)) / max(1, sum(self.eval_metrics['action_counts']) + 1)
        throughput_improvement = float(np.dot(ai_distribution, CAPACITIES) - np.dot(wrr_distribution, CAPACITIES))

        # Congestion penalty components
        congestion_risk = chosen_util ** 2

        # Action churn
        action_switched = 0
        if self.prev_action is not None and action != self.prev_action:
            action_switched = 1

        # Regime
        regime_label = "HIGH" if label == 1 else "NORMAL"

        # Per-server stats snapshot
        per_server = {}
        for i, b in enumerate(BACKENDS):
            per_server[b['name']] = {
                'utilization': float(server_utils[i]) if i < len(server_utils) else 0.0,
                'headroom': float(server_headrooms[i]) if i < len(server_headrooms) else 1.0,
                'capacity': float(CAPACITIES[i]),
            }

        info = {
            'chosen_server': action,
            'chosen_server_name': BACKENDS[action]['name'] if action < len(BACKENDS) else 'unknown',
            'chosen_util': float(chosen_util),
            'overload_flag': overload_flag,
            'util_breach': float(util_breach),
            'headroom': float(server_headrooms[action]) if action < len(server_headrooms) else 1.0,
            'fairness_dev': float(fairness_dev),
            'congestion_risk': float(congestion_risk),
            'action_switched': action_switched,
            'regime_label': regime_label,
            'per_server_stats': per_server,
            'throughput': float(throughput),
            'throughput_improvement': throughput_improvement,
            # New metrics for fair evaluation
            'throughput_reward': float(throughput_reward),
            'overload_penalty': float(overload_penalty),
            'stability_bonus': float(stability_bonus),
            'diversity_bonus': float(diversity_bonus),
            'burst_bonus': 0.0,  # Removed in V6
            'is_burst': byte_rate > 0.5,  # High traffic flag
            # Action-dependent metrics
            'real_throughput': float(real_throughput),
            'effective_capacity': float(effective_capacity),
            'packet_loss_rate': float(packet_loss_rate),
            'queue_length': float(queue_length),
            'response_time': float(response_time),
            'latency_ms': float(latency_ms),
            # V6 metrics
            'efficiency_bonus': float(efficiency_bonus),
            'wastage_penalty': float(wastage_penalty),
            'saving_bonus': float(saving_bonus),
        }

        # ── Update eval metrics ──
        self.eval_metrics['total_reward'] += reward_main
        self.eval_metrics['overload_count'] += overload_flag
        self.eval_metrics['utilizations'].append(chosen_util)
        self.eval_metrics['fairness_devs'].append(fairness_dev)
        self.eval_metrics['action_switches'] += action_switched
        self.eval_metrics['action_counts'][action] += 1
        
        # New metrics tracking
        self.eval_metrics['throughputs'].append(throughput)
        if byte_rate > 0.5:  # High traffic (burst)
            self.eval_metrics['burst_count'] += 1
            if chosen_util < 0.8:  # Handled well
                self.eval_metrics['burst_handled'] += 1
        self.eval_metrics['stability_bonus_total'] += stability_bonus
        self.eval_metrics['diversity_bonus_total'] += diversity_bonus
        self.eval_metrics['burst_bonus_total'] += 0.0  # Removed in V6
        # V6 metrics
        self.eval_metrics['efficiency_bonus_total'] += efficiency_bonus
        self.eval_metrics['wastage_penalty_total'] += wastage_penalty
        self.eval_metrics['saving_bonus_total'] += saving_bonus

        self.prev_action = action

        # ── NEXT STATE ──
        self.current_step += 1
        done = self.current_step >= self.num_samples - 1
        next_state = self.X_data[min(self.current_step, self.num_samples - 1)]

        return next_state, reward_main, done, info

    def get_state_shape(self):
        return self.X_data[0].shape

    def get_eval_summary(self):
        """Tổng hợp evaluation metrics sau khi chạy hết episode."""
        utils = np.array(self.eval_metrics['utilizations'])
        fairness = np.array(self.eval_metrics['fairness_devs'])
        throughputs = np.array(self.eval_metrics['throughputs'])
        total_steps = max(1, sum(self.eval_metrics['action_counts']))
        
        # Calculate new metrics
        # QoS Efficiency = Throughput / (1 + Overload_Rate)
        overload_rate = self.eval_metrics['overload_count'] / max(1, len(utils))
        avg_throughput = float(np.mean(throughputs)) if len(throughputs) > 0 else 0.0
        qos_efficiency = avg_throughput / (1 + overload_rate)
        
        # Burst Handling Ratio = burst_handled / burst_count
        burst_handling_ratio = (
            self.eval_metrics['burst_handled'] / max(1, self.eval_metrics['burst_count'])
            if self.eval_metrics['burst_count'] > 0 else 1.0
        )
        
        # Stability Score = 1 / (1 + std_throughput)
        throughput_std = float(np.std(throughputs)) if len(throughputs) > 1 else 0.0
        stability_score = 1.0 / (1 + throughput_std)

        return {
            'total_reward': self.eval_metrics['total_reward'],
            'avg_reward': self.eval_metrics['total_reward'] / max(1, len(utils)),
            'overload_count': self.eval_metrics['overload_count'],
            'overload_rate': overload_rate,
            'p95_utilization': float(np.percentile(utils, 95)) if len(utils) > 0 else 0.0,
            'mean_utilization': float(np.mean(utils)) if len(utils) > 0 else 0.0,
            'mean_fairness_dev': float(np.mean(fairness)) if len(fairness) > 0 else 0.0,
            'policy_churn_rate': self.eval_metrics['action_switches'] / max(1, len(utils)),
            'action_distribution': (self.eval_metrics['action_counts'] / total_steps).tolist(),
            'throughput_improvement': float(np.dot(self.eval_metrics['action_counts'] / max(1, total_steps), CAPACITIES) - np.dot(np.array([0.0625, 0.3125, 0.625]), CAPACITIES)),
            # New metrics for fair evaluation
            'qos_efficiency': qos_efficiency,
            'burst_handling_ratio': burst_handling_ratio,
            'stability_score': stability_score,
            'avg_throughput': avg_throughput,
            'throughput_std': throughput_std,
            'burst_count': self.eval_metrics['burst_count'],
            'burst_handled': self.eval_metrics['burst_handled'],
            # V6 metrics
            'efficiency_bonus_total': self.eval_metrics['efficiency_bonus_total'],
            'wastage_penalty_total': self.eval_metrics['wastage_penalty_total'],
            'saving_bonus_total': self.eval_metrics['saving_bonus_total'],
        }

    # ── Private helpers ──

    def _extract_utilizations(self, features):
        """Extract per-server utilization from V3 feature vector."""
        # Group C starts after Group A (7) + Group B (3 * num_servers)
        group_c_start = 7 + 3 * self.num_actions
        utils = []
        for i in range(self.num_actions):
            idx = group_c_start + i * 7  # util is first in each server's 7 risk features
            if idx < len(features):
                utils.append(float(features[idx]))
            else:
                utils.append(0.0)
        return utils

    def _extract_headrooms(self, features):
        """Extract per-server headroom from V3 feature vector."""
        group_c_start = 7 + 3 * self.num_actions
        headrooms = []
        for i in range(self.num_actions):
            idx = group_c_start + i * 7 + 2  # headroom is 3rd in risk features
            if idx < len(features):
                headrooms.append(float(features[idx]))
            else:
                headrooms.append(1.0)
        return headrooms

    def _get_action_ratio(self, action):
        """Compute running ratio of how often this action was chosen."""
        total = max(1, sum(self.eval_metrics['action_counts']) + 1)
        return (self.eval_metrics['action_counts'][action] + 1) / total
