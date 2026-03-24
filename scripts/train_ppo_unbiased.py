#!/usr/bin/env python3
"""
Train PPO with Stable Baselines3 - UNBIASED VERSION
====================================================

This script trains a PPO model WITHOUT reward bias.
- No capacity_bonus
- Pure throughput - latency penalty - overload penalty
"""

import os
import sys
import numpy as np
import torch
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from scipy import stats

sys.path.insert(0, '/work')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class SDNEnvUnbiased(gym.Env):
    """UNBIASED SDN environment - No reward shaping bias."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, scenario='normal'):
        super().__init__()
        
        self.scenario = scenario
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.float32
        )
        
        # Server capacities
        self.capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        self.max_capacity = float(np.max(self.capacities))
        
        # State
        self.state = None
        self.current_step = 0
        self.max_steps = 200
        
        # Traffic
        self.traffic_intensity = 0.3
        self.burst_probability = 0.15
        self.in_burst = False
        self.burst_duration = 0
        self.burst_intensity = 0.0
        
        # Scenario-specific params
        if scenario == 'burst':
            self.burst_probability = 0.3
        elif scenario == 'high_traffic':
            self.traffic_intensity = 0.6
        elif scenario == 'dynamic':
            self.capacity_drift = 0.001
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random initial state
        self.state = np.random.rand(11).astype(np.float32) * 0.3
        self.state[6] = 0.0
        self.state[7] = 0.3
        self.state[8] = self.capacities[0] / 150.0
        self.state[9] = self.capacities[1] / 150.0
        self.state[10] = self.capacities[2] / 150.0
        
        self.current_step = 0
        
        # Scenario-specific reset
        if self.scenario == 'normal':
            self.traffic_intensity = np.random.uniform(0.2, 0.4)
        elif self.scenario == 'burst':
            self.traffic_intensity = np.random.uniform(0.2, 0.4)
            self.burst_probability = 0.3
        elif self.scenario == 'high_traffic':
            self.traffic_intensity = np.random.uniform(0.5, 0.8)
        elif self.scenario == 'dynamic':
            self.traffic_intensity = np.random.uniform(0.2, 0.4)
            self.capacities = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        
        self.in_burst = False
        self.burst_duration = 0
        
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1
        
        # Convert discrete action to weight vector
        if isinstance(action, np.ndarray):
            action = int(action[0]) if action.ndim > 0 else int(action)
        
        weights = np.zeros(3, dtype=np.float32)
        weights[action] = 1.0
        
        # Update traffic based on scenario
        if self.scenario == 'burst':
            if np.random.random() < self.burst_probability and not self.in_burst:
                self.in_burst = True
                self.burst_duration = np.random.randint(5, 15)
                self.burst_intensity = np.random.uniform(0.5, 0.9)
        elif self.scenario == 'high_traffic':
            self.traffic_intensity = np.random.uniform(0.5, 0.8)
        elif self.scenario == 'dynamic':
            # Capacity drift
            self.capacities *= (1 - np.random.uniform(-0.01, 0.01))
            self.capacities = np.clip(self.capacities, 5.0, 100.0)
        
        if self.in_burst:
            self.traffic_intensity = min(0.95, self.traffic_intensity + self.burst_intensity)
            self.burst_duration -= 1
            if self.burst_duration <= 0:
                self.in_burst = False
        else:
            if self.scenario != 'high_traffic':
                self.traffic_intensity = max(0.1, self.traffic_intensity + np.random.uniform(-0.05, 0.05))
        
        # Calculate load
        load_h5 = self.traffic_intensity * weights[0] / self.capacities[0] * self.max_capacity
        load_h7 = self.traffic_intensity * weights[1] / self.capacities[1] * self.max_capacity
        load_h8 = self.traffic_intensity * weights[2] / self.capacities[2] * self.max_capacity
        
        load_h5 = np.clip(load_h5, 0, 1)
        load_h7 = np.clip(load_h7, 0, 1)
        load_h8 = np.clip(load_h8, 0, 1)
        
        # Latency (M/M/1 queue)
        base_lat = 10.0
        lat_h5 = base_lat / (1 - load_h5 + 1e-8) if load_h5 < 0.99 else 1000.0
        lat_h7 = base_lat / (1 - load_h7 + 1e-8) if load_h7 < 0.99 else 1000.0
        lat_h8 = base_lat / (1 - load_h8 + 1e-8) if load_h8 < 0.99 else 1000.0
        
        avg_latency = (lat_h5 + lat_h7 + lat_h8) / 3
        
        # Throughput
        throughput = self.traffic_intensity * (1 - 0.5 * (load_h5 + load_h7 + load_h8) / 3)
        
        # UNBIASED REWARD - No capacity bonus
        latency_penalty = -0.1 * avg_latency
        overload_penalty = -10.0 * (max(0, load_h5 - 0.9) + max(0, load_h7 - 0.9) + max(0, load_h8 - 0.9))
        
        # Pure reward: throughput - latency - overload
        reward = throughput + latency_penalty + overload_penalty
        
        # Update state
        self.state = np.array([
            load_h5, load_h7, load_h8,
            lat_h5 / 100.0, lat_h7 / 100.0, lat_h8 / 100.0,
            self.traffic_intensity,
            avg_latency / 100.0,
            self.capacities[0] / 150.0,
            self.capacities[1] / 150.0,
            self.capacities[2] / 150.0
        ], dtype=np.float32)
        
        done = self.current_step >= self.max_steps
        
        info = {
            'throughput': throughput,
            'latency': avg_latency,
            'load_h5': load_h5,
            'load_h7': load_h7,
            'load_h8': load_h8,
            'overload': load_h5 > 0.9 or load_h7 > 0.9 or load_h8 > 0.9
        }
        
        return self.state, float(reward), done, False, info


class MultiScenarioCallback(BaseCallback):
    """Callback to switch between scenarios during training."""
    
    def __init__(self, scenarios, switch_freq=10000, verbose=0):
        super().__init__(verbose)
        self.scenarios = scenarios
        self.switch_freq = switch_freq
        self.current_scenario_idx = 0
        self.step_count = 0
    
    def _on_step(self):
        self.step_count += 1
        if self.step_count % self.switch_freq == 0:
            self.current_scenario_idx = (self.current_scenario_idx + 1) % len(self.scenarios)
            if self.verbose > 0:
                print(f"Switching to scenario: {self.scenarios[self.current_scenario_idx]}")
        return True


def train_ppo_unbiased(
    total_timesteps=500_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    save_dir='ai_model/checkpoints'
):
    """Train PPO with unbiased reward."""
    
    print("=" * 60)
    print("TRAINING PPO - UNBIASED VERSION")
    print("=" * 60)
    print("NO capacity bonus - Pure throughput/latency/overload reward")
    print("=" * 60)
    
    # Create environment
    env = SDNEnvUnbiased(scenario='normal')
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        device='cpu'
    )
    
    print(f"\nModel architecture:")
    print(model.policy)
    
    # Action distribution callback
    class ActionDistCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.action_counts = [0, 0, 0]
            self.episode_rewards = []
            self.current_episode_reward = 0
            
        def _on_step(self):
            actions = self.locals.get('actions', None)
            rewards = self.locals.get('rewards', None)
            
            if actions is not None:
                for action in actions.flatten():
                    if action < 3:
                        self.action_counts[int(action)] += 1
            
            if rewards is not None:
                for reward in rewards.flatten():
                    self.current_episode_reward += reward
            
            # Check for episode end
            dones = self.locals.get('dones', None)
            if dones is not None:
                for done in dones.flatten():
                    if done:
                        self.episode_rewards.append(self.current_episode_reward)
                        self.current_episode_reward = 0
            
            return True
        
        def get_distribution(self):
            total = sum(self.action_counts)
            if total > 0:
                return [c / total for c in self.action_counts]
            return [0.33, 0.33, 0.34]
        
        def get_mean_reward(self):
            if len(self.episode_rewards) > 0:
                return np.mean(self.episode_rewards), np.std(self.episode_rewards)
            return 0, 0
    
    action_callback = ActionDistCallback()
    
    # Train
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[action_callback],
        progress_bar=False
    )
    
    elapsed = datetime.now() - start_time
    print(f"\nTraining completed in {elapsed}")
    
    # Get action distribution
    action_dist = action_callback.get_distribution()
    mean_reward, std_reward = action_callback.get_mean_reward()
    
    print(f"\nAction distribution during training:")
    print(f"  Action 0 (h5 - weakest):  {action_dist[0]:.1%}")
    print(f"  Action 1 (h7 - medium):   {action_dist[1]:.1%}")
    print(f"  Action 2 (h8 - strongest): {action_dist[2]:.1%}")
    print(f"\nMean episode reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Save final model
    final_path = f"{save_dir}/ppo_unbiased_final.zip"
    model.save(final_path)
    print(f"\nSaved final model to: {final_path}")
    
    return model, action_dist, mean_reward, std_reward


def benchmark_unbiased(model_path, n_episodes=100, scenarios=['normal', 'burst', 'high_traffic', 'dynamic']):
    """Benchmark unbiased model with statistical tests."""
    
    print("\n" + "=" * 60)
    print("BENCHMARK - UNBIASED MODEL")
    print("=" * 60)
    
    model = PPO.load(model_path)
    
    all_results = {}
    
    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario.upper()} ---")
        
        env = SDNEnvUnbiased(scenario=scenario)
        
        # PPO results
        ppo_rewards = []
        ppo_actions = [0, 0, 0]
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):
                    action = int(action[0]) if action.ndim > 0 else int(action)
                
                ppo_actions[action] += 1
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
            
            ppo_rewards.append(total_reward)
        
        # WRR results
        wrr_rewards = []
        wrr_actions = [0, 0, 0]
        weights = np.array([1.0, 5.0, 10.0])
        weights = weights / weights.sum()
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                r = np.random.random()
                if r < weights[0]:
                    action = 0
                elif r < weights[0] + weights[1]:
                    action = 1
                else:
                    action = 2
                
                wrr_actions[action] += 1
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
            
            wrr_rewards.append(total_reward)
        
        # Statistical tests
        ppo_mean = np.mean(ppo_rewards)
        ppo_std = np.std(ppo_rewards)
        wrr_mean = np.mean(wrr_rewards)
        wrr_std = np.std(wrr_rewards)
        
        # t-test
        t_stat, p_value = stats.ttest_ind(ppo_rewards, wrr_rewards)
        
        # Cohen's d
        pooled_std = np.sqrt((ppo_std**2 + wrr_std**2) / 2)
        cohens_d = (ppo_mean - wrr_mean) / pooled_std if pooled_std > 0 else 0
        
        # 95% confidence interval
        ppo_ci = stats.t.interval(0.95, len(ppo_rewards)-1, loc=ppo_mean, scale=stats.sem(ppo_rewards))
        wrr_ci = stats.t.interval(0.95, len(wrr_rewards)-1, loc=wrr_mean, scale=stats.sem(wrr_rewards))
        
        all_results[scenario] = {
            'PPO': {
                'mean': ppo_mean,
                'std': ppo_std,
                'ci_95': ppo_ci,
                'action_dist': [c / sum(ppo_actions) for c in ppo_actions]
            },
            'WRR': {
                'mean': wrr_mean,
                'std': wrr_std,
                'ci_95': wrr_ci,
                'action_dist': [c / sum(wrr_actions) for c in wrr_actions]
            },
            'statistical': {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            }
        }
        
        print(f"\nPPO: {ppo_mean:.2f} ± {ppo_std:.2f} (95% CI: [{ppo_ci[0]:.2f}, {ppo_ci[1]:.2f}])")
        print(f"WRR: {wrr_mean:.2f} ± {wrr_std:.2f} (95% CI: [{wrr_ci[0]:.2f}, {wrr_ci[1]:.2f}])")
        print(f"t-test: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"Cohen's d: {cohens_d:.3f}")
        print(f"Significant: {'YES' if p_value < 0.05 else 'NO'}")
        
        winner = "PPO" if ppo_mean > wrr_mean else "WRR"
        improvement = ((ppo_mean - wrr_mean) / abs(wrr_mean) * 100) if wrr_mean != 0 else 0
        print(f"Winner: {winner} ({improvement:+.1f}%)")
    
    return all_results


if __name__ == '__main__':
    # Train
    model, action_dist, mean_reward, std_reward = train_ppo_unbiased(
        total_timesteps=500_000,
        learning_rate=3e-4,
        ent_coef=0.01
    )
    
    # Benchmark
    results = benchmark_unbiased(
        'ai_model/checkpoints/ppo_unbiased_final.zip',
        n_episodes=100,
        scenarios=['normal', 'burst', 'high_traffic', 'dynamic']
    )
    
    # Save results
    import json
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'ai_model/benchmark_unbiased_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: ai_model/benchmark_unbiased_{timestamp}.json")