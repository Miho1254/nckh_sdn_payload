#!/usr/bin/env python3
"""
PPO Training Script - 2M Samples Multi-Scenario Training

Train PPO với 2M samples trên nhiều scenarios để học chính sách robust.
Mục tiêu: AI phải học được capacity-weighted distribution và beat Random baseline.

Chạy: python ai_model/train_ppo_2m.py
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdn_sim_env import SDNLoadBalancerEnv, GoldenHourEnv, VideoConferenceEnv, HardwareDegradationEnv, LowRateDosEnv, make_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList


class MultiScenarioCallback(BaseCallback):
    """Callback để switch giữa các scenarios trong training."""
    
    def __init__(self, envs, switch_freq=10000, verbose=1):
        super().__init__(verbose)
        self.envs = envs  # List of env names
        self.switch_freq = switch_freq
        self.current_env_idx = 0
        self.step_count = 0
        
    def _on_step(self):
        self.step_count += 1
        if self.step_count % self.switch_freq == 0:
            self.current_env_idx = (self.current_env_idx + 1) % len(self.envs)
            if self.verbose > 0:
                print(f"\n[SWITCH] Switching to scenario: {self.envs[self.current_env_idx]}")
        return True


class DetailedRewardLogger(BaseCallback):
    """Callback để log chi tiết reward và action distribution."""
    
    def __init__(self, verbose=1, log_freq=1000):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = np.zeros(3)  # 3 actions: h5, h7, h8
        self.total_steps = 0
        self.log_freq = log_freq
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self):
        self.total_steps += 1
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Track action distribution (continuous actions - probabilities)
        # For continuous action space, actions are probabilities for each server
        # PPO outputs raw actions that need to be normalized
        actions = self.locals['actions'][0]  # Shape: (3,) for 3 servers
        
        # Ensure actions are positive and sum to 1 (probability distribution)
        if isinstance(actions, np.ndarray):
            # Clip negative values and normalizeize
            actions_clipped = np.clip(actions, 0, None)
            if actions_clipped.sum() > 0:
                actions_normalized = actions_clipped / actions_clipped.sum()
            else:
                actions_normalized = np.array([0.33, 0.33, 0.34])
            self.action_counts += actions_normalized
        else:
            # Discrete action
            self.action_counts[int(actions)] += 1
        
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Log progress
        if self.total_steps % self.log_freq == 0 and self.verbose > 0:
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                # Calculate action distribution from accumulated probabilities
                total = self.action_counts.sum()
                if total > 0:
                    action_dist = self.action_counts / total
                else:
                    action_dist = np.array([0.33, 0.33, 0.34])
                print(f"  Step {self.total_steps}: avg_reward={avg_reward:.2f} | actions: h5={action_dist[0]*100:.1f}%, h7={action_dist[1]*100:.1f}%, h8={action_dist[2]*100:.1f}%")
        
        return True


class CapacityWeightedRewardWrapper:
    """Wrapper để thêm capacity-weighted reward shaping."""
    
    def __init__(self, env):
        self.env = env
        # Capacity ratios: h5=6.25%, h7=31.25%, h8=62.5%
        self.target_dist = np.array([0.0625, 0.3125, 0.625])
        
    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        
        # Add capacity-weighted bonus
        # Reward for choosing high-capacity servers
        capacity_bonus = 0.0
        if action == 2:  # h8 (highest capacity)
            capacity_bonus = 0.1
        elif action == 1:  # h7 (medium capacity)
            capacity_bonus = 0.05
        else:  # h5 (lowest capacity)
            capacity_bonus = -0.05  # Small penalty
        
        # Add to reward
        shaped_reward = reward + capacity_bonus
        
        return obs, shaped_reward, done, trunc, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def train_ppo_2m_multi_scenario():
    """Train PPO với 2M samples trên nhiều scenarios."""
    
    print("="*70)
    print("  PPO TRAINING - 2M SAMPLES MULTI-SCENARIO")
    print("="*70)
    print("\n[*] Mục tiêu: Học capacity-weighted distribution")
    print("    Target: h5=6.25%, h7=31.25%, h8=62.5%")
    print("    Baseline: Random (33.3% each)")
    print()
    
    # Create directories
    os.makedirs("ai_model/models", exist_ok=True)
    os.makedirs("ai_model/logs", exist_ok=True)
    
    # Training configuration
    total_timesteps = 2_000_000  # 2M samples
    eval_freq = 50_000
    n_eval_episodes = 10
    
    # Scenarios to train on (weighted by importance)
    scenarios = [
        ('SDN-v0', 0.3),           # 30% - Normal traffic
        ('GoldenHour-v0', 0.25),   # 25% - Burst traffic
        ('VideoConference-v0', 0.2), # 20% - Low latency
        ('HardwareDegradation-v0', 0.15), # 15% - Degradation
        ('LowRateDoS-v0', 0.1),    # 10% - DoS attack
    ]
    
    print("[*] Training scenarios:")
    for scenario, weight in scenarios:
        print(f"    {scenario}: {weight*100:.0f}%")
    print()
    
    # Create environment
    print("[*] Creating environment...")
    env = make_env('SDN-v0')
    
    # Check environment
    print("[*] Checking environment...")
    obs, _ = env.reset()
    print(f"    Observation shape: {obs.shape}")
    print(f"    Action space: {env.action_space}")
    print(f"    Observation sample: {obs[:5]}...")
    print()
    
    # Create evaluation environments
    eval_envs = {scenario: make_env(scenario) for scenario, _ in scenarios}
    
    # PPO Hyperparameters - Tuned for capacity-weighted learning
    print("[*] Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,        # Lower LR for stable learning
        n_steps=4096,              # Larger batch for better gradient
        batch_size=128,            # Larger batch size
        n_epochs=20,               # More epochs per update
        gamma=0.99,                # Discount factor
        gae_lambda=0.95,           # GAE lambda
        clip_range=0.2,            # PPO clip range
        ent_coef=0.005,            # Lower entropy for more deterministic policy
        vf_coef=0.5,               # Value function coefficient
        max_grad_norm=0.5,         # Gradient clipping
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],  # Larger network
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        tensorboard_log=None  # Disable tensorboard
    )
    
    print(f"    Policy network: [256, 256, 128]")
    print(f"    Learning rate: {model.learning_rate}")
    print(f"    N steps: {model.n_steps}")
    print(f"    Batch size: {model.batch_size}")
    print(f"    N epochs: {model.n_epochs}")
    print(f"    Entropy coef: {model.ent_coef}")
    print()
    
    # Callbacks
    reward_logger = DetailedRewardLogger(verbose=1, log_freq=5000)
    
    # Eval callback for main scenario
    eval_callback = EvalCallback(
        eval_envs['SDN-v0'],
        best_model_save_path="./ai_model/models/best_2m/",
        log_path="./ai_model/logs/eval_2m/",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=1
    )
    
    # Training
    print("[*] Starting training...")
    print(f"[*] Total timesteps: {total_timesteps:,}")
    print(f"[*] Estimated time: {total_timesteps/10000:.0f}-{total_timesteps/5000:.0f} minutes")
    print()
    
    start_time = time.time()
    
    # Multi-scenario training
    timesteps_per_scenario = [int(total_timesteps * weight) for _, weight in scenarios]
    
    for i, (scenario, weight) in enumerate(scenarios):
        scenario_timesteps = timesteps_per_scenario[i]
        print(f"\n{'='*60}")
        print(f"  TRAINING ON: {scenario} ({scenario_timesteps:,} steps)")
        print(f"{'='*60}")
        
        # Create scenario-specific env
        scenario_env = make_env(scenario)
        
        # Set new environment
        model.set_env(scenario_env)
        
        # Train on this scenario
        model.learn(
            total_timesteps=scenario_timesteps,
            callback=[reward_logger],
            reset_num_timesteps=False,
            progress_bar=False  # Disable progress bar
        )
        
        print(f"\n[*] Completed {scenario}")
        
        # Quick evaluation
        print(f"[*] Quick eval on {scenario}:")
        eval_rewards = []
        for _ in range(5):
            obs, _ = scenario_env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = scenario_env.step(action)
                total_reward += reward
            eval_rewards.append(total_reward)
        print(f"    Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"[*] Total training time: {training_time/60:.1f} minutes")
    print(f"[*] Total timesteps: {reward_logger.total_steps:,}")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"ai_model/models/ppo_2m_multi_scenario_{timestamp}.zip"
    model.save(save_path)
    print(f"[✓] Model saved: {save_path}")
    
    # Final evaluation on all scenarios
    print("\n[*] Final evaluation on all scenarios:")
    print("-" * 60)
    
    for scenario, _ in scenarios:
        eval_env = make_env(scenario)
        eval_rewards = []
        action_counts = np.zeros(3)
        
        for _ in range(20):
            obs, _ = eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_counts[action] += 1
                obs, reward, done, trunc, info = eval_env.step(action)
                total_reward += reward
            eval_rewards.append(total_reward)
        
        action_dist = action_counts / (20 * 1000)  # Assuming ~1000 steps per episode
        print(f"\n{scenario}:")
        print(f"  Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"  Action dist: h5={action_dist[0]*100:.1f}%, h7={action_dist[1]*100:.1f}%, h8={action_dist[2]*100:.1f}%")
        print(f"  Target dist: h5=6.25%, h7=31.25%, h8=62.5%")
        
        # Check if close to target
        target_dist = np.array([0.0625, 0.3125, 0.625])
        deviation = np.sum(np.abs(action_dist - target_dist)) / 2
        print(f"  Deviation from target: {deviation:.3f}")
    
    # Compare with Random baseline
    print("\n" + "="*60)
    print("  COMPARISON WITH RANDOM BASELINE")
    print("="*60)
    
    random_rewards = []
    ppo_rewards = []
    
    env = make_env('SDN-v0')
    
    # Random baseline
    for _ in range(20):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
        random_rewards.append(total_reward)
    
    # PPO
    for _ in range(20):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
        ppo_rewards.append(total_reward)
    
    print(f"\nRandom baseline: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print(f"PPO agent:      {np.mean(ppo_rewards):.2f} ± {np.std(ppo_rewards):.2f}")
    print(f"Improvement:     {(np.mean(ppo_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100:.1f}%")
    
    return model


def train_ppo_curriculum():
    """Train PPO với curriculum learning - từ dễ đến khó."""
    
    print("="*70)
    print("  PPO CURRICULUM LEARNING - 2M SAMPLES")
    print("="*70)
    print("\n[*] Curriculum: Easy -> Medium -> Hard")
    print()
    
    # Create directories
    os.makedirs("ai_model/models", exist_ok=True)
    os.makedirs("ai_model/logs", exist_ok=True)
    
    # Curriculum stages
    stages = [
        # Stage 1: Learn capacity-weighted distribution (easy)
        {
            'name': 'capacity_learning',
            'env': 'SDN-v0',
            'timesteps': 500_000,
            'description': 'Learn basic capacity-weighted routing'
        },
        # Stage 2: Handle burst traffic (medium)
        {
            'name': 'burst_handling',
            'env': 'GoldenHour-v0',
            'timesteps': 500_000,
            'description': 'Learn to handle burst traffic'
        },
        # Stage 3: Low latency optimization (medium)
        {
            'name': 'latency_optimization',
            'env': 'VideoConference-v0',
            'timesteps': 400_000,
            'description': 'Optimize for low latency'
        },
        # Stage 4: Hardware degradation (hard)
        {
            'name': 'degradation_handling',
            'env': 'HardwareDegradation-v0',
            'timesteps': 300_000,
            'description': 'Handle hardware degradation'
        },
        # Stage 5: DoS attack (hard)
        {
            'name': 'dos_defense',
            'env': 'LowRateDoS-v0',
            'timesteps': 300_000,
            'description': 'Defend against DoS attacks'
        },
    ]
    
    # Create base environment
    env = make_env('SDN-v0')
    
    # PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        tensorboard_log=None  # Disable tensorboard
    )
    
    print("[*] Starting curriculum training...")
    print()
    
    start_time = time.time()
    
    for i, stage in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"  STAGE {i+1}/{len(stages)}: {stage['name'].upper()}")
        print(f"{'='*60}")
        print(f"  Environment: {stage['env']}")
        print(f"  Timesteps: {stage['timesteps']:,}")
        print(f"  Description: {stage['description']}")
        print()
        
        # Create stage environment
        stage_env = make_env(stage['env'])
        model.set_env(stage_env)
        
        # Train
        model.learn(
            total_timesteps=stage['timesteps'],
            reset_num_timesteps=False,
            progress_bar=False  # Disable progress bar
        )
        
        print(f"\n[✓] Stage {i+1} completed")
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"  CURRICULUM TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"[*] Total training time: {training_time/60:.1f} minutes")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"ai_model/models/ppo_curriculum_2m_{timestamp}.zip"
    model.save(save_path)
    print(f"[✓] Model saved: {save_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description='PPO Training - 2M Samples')
    parser.add_argument('--mode', type=str, default='multi',
                       choices=['multi', 'curriculum'],
                       help='Training mode: multi-scenario or curriculum')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test only')
    
    args = parser.parse_args()
    
    if args.test:
        print("\n[*] Quick test mode...")
        env = make_env('SDN-v0')
        obs, _ = env.reset()
        print(f"    Observation shape: {obs.shape}")
        print(f"    Action space: {env.action_space}")
        print("[✓] Test passed!")
    elif args.mode == 'multi':
        train_ppo_2m_multi_scenario()
    elif args.mode == 'curriculum':
        train_ppo_curriculum()
    
    print("\n[*] Done!")