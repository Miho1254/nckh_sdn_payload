#!/usr/bin/env python3
"""
Curriculum Learning for PPO on Production SDN Environment
==========================================================

Strategy: Train PPO on scenarios where WRR is weak:
1. SDNHighNoise-v0: WRR loses 6% to AdaptiveWRR
2. SDNLowSLA-v0: WRR loses 4% to AdaptiveWRR

Key insight: We don't need PPO to win everywhere.
We only need PPO to win where WRR is weak.

Author: Research Team
"""

import os
import sys
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Import production environment
from sdn_env_production import (
    make_production_env,
    SDNProductionEnv,
    BurstTrafficEnv,
    HighNoiseEnv,
    LowSLAEnv,
    DynamicCapacityEnv,
)


class CurriculumCallback(BaseCallback):
    """Callback để track training progress với curriculum learning."""
    
    def __init__(self, verbose=1, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Track specific metrics
        self.p99_latencies = []
        self.packet_losses = []
        self.sla_violations = []
        
        # Action distribution
        self.action_counts = np.zeros(3)
        self.action_samples = 0
    
    def _on_step(self):
        # Track episode
        self.current_episode_reward += self.locals['rewards'][0]
        
        # Track action distribution
        actions = self.locals['actions'][0]
        if isinstance(actions, np.ndarray):
            actions_clipped = np.clip(actions, 0, None)
            if actions_clipped.sum() > 0:
                actions_normalized = actions_clipped / actions_clipped.sum()
                self.action_counts += actions_normalized
                self.action_samples += 1
        
        # Check episode end
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        
        # Log metrics
        if self.n_calls % self.log_freq == 0:
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            
            # Action distribution
            if self.action_samples > 0:
                action_dist = self.action_counts / self.action_samples
                h5_pct = action_dist[0] * 100
                h7_pct = action_dist[1] * 100
                h8_pct = action_dist[2] * 100
            else:
                h5_pct = h7_pct = h8_pct = 33.33
            
            print(f"Step {self.n_calls:,} | "
                  f"Reward: {avg_reward:.1f} | "
                  f"Action: h5={h5_pct:.1f}% h7={h7_pct:.1f}% h8={h8_pct:.1f}%")
        
        return True


def make_curriculum_envs():
    """Create curriculum environments."""
    
    def make_env_fn(env_class, config=None):
        def _init():
            return env_class(config=config)
        return _init
    
    # Curriculum stages:
    # Stage 1: Easy (Production) - learn basic load balancing
    # Stage 2: Medium (HighNoise) - learn to handle noise
    # Stage 3: Hard (LowSLA) - learn to optimize under strict constraints
    
    env_classes = [
        (SDNProductionEnv, {'max_steps': 500}),  # Stage 1
        (HighNoiseEnv, {'max_steps': 500}),      # Stage 2
        (LowSLAEnv, {'max_steps': 500}),          # Stage 3
    ]
    
    return env_classes


def train_curriculum(
    total_timesteps=1_000_000,
    stage_steps=None,
    num_envs=4,
    seed=42,
    save_path=None
):
    """
    Train PPO with curriculum learning.
    
    Curriculum:
    1. SDNProduction-v0 (easy): Learn basic load balancing
    2. SDNHighNoise-v0 (medium): Learn to handle measurement noise
    3. SDNLowSLA-v0 (hard): Learn to optimize under strict SLA
    
    Args:
        total_timesteps: Total training steps
        stage_steps: Steps per stage (if None, split equally)
        num_envs: Number of parallel environments
        seed: Random seed
        save_path: Path to save model
    """
    
    print("=" * 70)
    print("CURRICULUM LEARNING FOR PPO")
    print("=" * 70)
    print("Target: Scenarios where WRR is weak")
    print("  - SDNHighNoise-v0: WRR loses 6% to AdaptiveWRR")
    print("  - SDNLowSLA-v0: WRR loses 4% to AdaptiveWRR")
    print("=" * 70)
    
    if stage_steps is None:
        stage_steps = total_timesteps // 3
    
    np.random.seed(seed)
    
    # Curriculum stages
    stages = [
        {
            'name': 'SDNProduction-v0 (Easy)',
            'env_class': SDNProductionEnv,
            'config': {'max_steps': 500},
            'timesteps': stage_steps,
        },
        {
            'name': 'SDNHighNoise-v0 (Medium)',
            'env_class': HighNoiseEnv,
            'config': {'max_steps': 500, 'measurement_noise': 0.2, 'latency_noise': 0.3},
            'timesteps': stage_steps,
        },
        {
            'name': 'SDNLowSLA-v0 (Hard)',
            'env_class': LowSLAEnv,
            'config': {'max_steps': 500, 'sla_latency_ms': 50.0, 'sla_packet_loss': 0.005},
            'timesteps': stage_steps,
        },
    ]
    
    # Create initial environment
    print(f"\nStage 1: {stages[0]['name']}")
    print("-" * 40)
    
    def make_env_fn():
        return stages[0]['env_class'](config=stages[0]['config'])
    
    if num_envs > 1:
        env = SubprocVecEnv([make_env_fn for _ in range(num_envs)])
    else:
        env = DummyVecEnv([make_env_fn])
    
    # Create PPO model
    print("Creating PPO model...")
    import torch
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device='cpu',
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128, 64],
                vf=[128, 128, 64]
            ),
            activation_fn=torch.nn.ReLU,
        ),
        verbose=0,
        seed=seed,
    )
    
    # Train through curriculum
    all_metrics = {}
    total_steps_trained = 0
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n{'='*70}")
        print(f"STAGE {stage_idx + 1}: {stage['name']}")
        print(f"Steps: {stage['timesteps']:,}")
        print(f"{'='*70}")
        
        # Create environment for this stage
        def make_stage_env_fn():
            return stage['env_class'](config=stage['config'])
        
        if num_envs > 1:
            stage_env = SubprocVecEnv([make_stage_env_fn for _ in range(num_envs)])
        else:
            stage_env = DummyVecEnv([make_stage_env_fn])
        
        # Set environment
        model.set_env(stage_env)
        
        # Create callback
        callback = CurriculumCallback(verbose=1, log_freq=1000)
        
        # Train
        print(f"Training on {stage['name']}...")
        start_time = datetime.now()
        
        try:
            model.learn(
                total_timesteps=stage['timesteps'],
                callback=callback,
                progress_bar=False,
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            break
        except Exception as e:
            print(f"\nTraining error: {e}")
            import traceback
            traceback.print_exc()
            break
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Track metrics
        all_metrics[stage['name']] = {
            'duration': duration,
            'episode_rewards': callback.episode_rewards.copy() if callback.episode_rewards else [],
            'action_counts': callback.action_counts.copy(),
            'action_samples': callback.action_samples,
        }
        
        total_steps_trained += stage['timesteps']
        
        # Print stage summary
        if callback.episode_rewards:
            avg_reward = np.mean(callback.episode_rewards[-100:])
            print(f"\nStage {stage_idx + 1} Summary:")
            print(f"  Average reward: {avg_reward:.1f}")
            print(f"  Duration: {duration:.1f}s")
            
            if callback.action_samples > 0:
                action_dist = callback.action_counts / callback.action_samples
                print(f"  Action distribution: h5={action_dist[0]*100:.1f}% h7={action_dist[1]*100:.1f}% h8={action_dist[2]*100:.1f}%")
    
    # Save model
    if save_path is None:
        save_path = f"models/ppo_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("CURRICULUM TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total steps: {total_steps_trained:,}")
    
    for stage_name, metrics in all_metrics.items():
        if metrics['episode_rewards']:
            avg_reward = np.mean(metrics['episode_rewards'][-100:])
            print(f"\n{stage_name}:")
            print(f"  Final avg reward: {avg_reward:.1f}")
            if metrics['action_samples'] > 0:
                action_dist = metrics['action_counts'] / metrics['action_samples']
                print(f"  Action dist: h5={action_dist[0]*100:.1f}% h7={action_dist[1]*100:.1f}% h8={action_dist[2]*100:.1f}%")
    
    return model, all_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO with Curriculum Learning')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Total training timesteps')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    model, metrics = train_curriculum(
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        seed=args.seed,
    )