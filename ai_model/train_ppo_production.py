#!/usr/bin/env python3
"""
Train PPO on Production-Grade SDN Environment
==============================================

Key differences from simple simulation:
1. SLA-based reward (no fairness penalty, no ideal_ratio)
2. Burst traffic with correlation
3. Delayed observation (partial observability)
4. Realistic state (22 dims)

Goal: Demonstrate that PPO can beat WRR in realistic conditions.

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


class ProductionTrainingCallback(BaseCallback):
    """Callback để track training progress trên production environment."""
    
    def __init__(self, verbose=1, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Track specific metrics
        self.p99_latencies = []
        self.packet_losses = []
        self.sla_violations = []
        self.burst_events = []
        
        # Action distribution
        self.action_counts = np.zeros(3)
        self.action_samples = 0
    
    def _on_step(self):
        # Track episode
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
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
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Log metrics
        if self.n_calls % self.log_freq == 0:
            # Get info from environment
            if hasattr(self.training_env, 'get_attr'):
                try:
                    stats = self.training_env.get_attr('episode_stats')
                    if stats and len(stats) > 0:
                        latest_stats = stats[-1]
                        if 'avg_p99_latency' in latest_stats:
                            self.p99_latencies.append(latest_stats['avg_p99_latency'])
                        if 'total_packet_loss' in latest_stats:
                            self.packet_losses.append(latest_stats['total_packet_loss'])
                        if 'sla_violations' in latest_stats:
                            self.sla_violations.append(latest_stats['sla_violations'])
                        if 'burst_events' in latest_stats:
                            self.burst_events.append(latest_stats['burst_events'])
                except:
                    pass
            
            # Calculate averages
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
            
            # Action distribution
            if self.action_samples > 0:
                action_dist = self.action_counts / self.action_samples
                h5_pct = action_dist[0] * 100
                h7_pct = action_dist[1] * 100
                h8_pct = action_dist[2] * 100
            else:
                h5_pct = h7_pct = h8_pct = 33.33
            
            # Latency stats
            avg_p99 = np.mean(self.p99_latencies[-100:]) if self.p99_latencies else 0
            avg_loss = np.mean(self.packet_losses[-100:]) if self.packet_losses else 0
            total_sla = sum(self.sla_violations[-100:]) if self.sla_violations else 0
            total_burst = sum(self.burst_events[-100:]) if self.burst_events else 0
            
            print(f"Step {self.n_calls:,} | "
                  f"Reward: {avg_reward:.1f} | "
                  f"Len: {avg_length:.0f} | "
                  f"P99: {avg_p99:.1f}ms | "
                  f"Loss: {avg_loss:.4f} | "
                  f"SLA: {total_sla} | "
                  f"Burst: {total_burst} | "
                  f"Action: h5={h5_pct:.1f}% h7={h7_pct:.1f}% h8={h8_pct:.1f}%")
        
        return True


def make_parallel_envs(env_id, num_envs=4):
    """Create multiple parallel environments."""
    
    def make_env_fn():
        return make_production_env(env_id)
    
    if num_envs > 1:
        return SubprocVecEnv([make_env_fn for _ in range(num_envs)])
    else:
        return DummyVecEnv([make_env_fn])


def train_ppo_production(
    env_id='SDNProduction-v0',
    total_timesteps=2_000_000,
    num_envs=4,
    save_path=None,
    seed=42
):
    """
    Train PPO on production environment.
    
    Args:
        env_id: Environment ID
        total_timesteps: Total training steps
        num_envs: Number of parallel environments
        save_path: Path to save model
        seed: Random seed
    """
    
    print("=" * 70)
    print("TRAINING PPO ON PRODUCTION SDN ENVIRONMENT")
    print("=" * 70)
    print(f"Environment: {env_id}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {num_envs}")
    print(f"Seed: {seed}")
    print("=" * 70)
    
    # Set random seed
    np.random.seed(seed)
    
    # Create environments
    print("\nCreating parallel environments...")
    env = make_parallel_envs(env_id, num_envs)
    
    # Create PPO model
    print("Creating PPO model...")
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
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        device='cpu',  # PPO is faster on CPU for simple environments
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128, 64],  # Larger network for complex environment
                vf=[128, 128, 64]
            ),
            activation_fn=torch.nn.ReLU,
        ),
        verbose=0,
        seed=seed,
    )
    
    # Create callback
    callback = ProductionTrainingCallback(verbose=1, log_freq=1000)
    
    # Train
    print("\nStarting training...")
    print("-" * 70)
    
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("-" * 70)
    print(f"Training completed in {duration:.1f} seconds")
    
    # Save model
    if save_path is None:
        save_path = f"models/ppo_production_{env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to: {save_path}")
    
    # Final stats
    print("\n" + "=" * 70)
    print("FINAL TRAINING STATS")
    print("=" * 70)
    
    if callback.episode_rewards:
        print(f"Average reward (last 100 episodes): {np.mean(callback.episode_rewards[-100:]):.2f}")
        print(f"Max reward: {max(callback.episode_rewards):.2f}")
        print(f"Min reward: {min(callback.episode_rewards):.2f}")
    
    if callback.action_samples > 0:
        action_dist = callback.action_counts / callback.action_samples
        print(f"\nFinal action distribution:")
        print(f"  h5 (10M): {action_dist[0]*100:.1f}%")
        print(f"  h7 (50M): {action_dist[1]*100:.1f}%")
        print(f"  h8 (100M): {action_dist[2]*100:.1f}%")
    
    if callback.p99_latencies:
        print(f"\nLatency stats:")
        print(f"  Average p99: {np.mean(callback.p99_latencies):.1f}ms")
        print(f"  Min p99: {min(callback.p99_latencies):.1f}ms")
    
    if callback.packet_losses:
        print(f"\nPacket loss stats:")
        print(f"  Average loss: {np.mean(callback.packet_losses):.4f}")
    
    print("=" * 70)
    
    return model, callback


def train_multi_scenario_production(total_timesteps=500_000, seed=42):
    """
    Train PPO on multiple production scenarios.
    
    Scenarios:
    1. SDNProduction-v0: Base production environment
    2. SDNBurst-v0: High burst traffic
    3. SDNHighNoise-v0: High measurement noise
    4. SDNLowSLA-v0: Strict SLA requirements
    """
    
    print("=" * 70)
    print("MULTI-SCENARIO PRODUCTION TRAINING")
    print("=" * 70)
    
    scenarios = [
        ('SDNProduction-v0', 'Base production'),
        ('SDNBurst-v0', 'High burst traffic'),
        ('SDNHighNoise-v0', 'High measurement noise'),
        ('SDNLowSLA-v0', 'Strict SLA'),
    ]
    
    all_models = {}
    
    for env_id, desc in scenarios:
        print(f"\n{'='*70}")
        print(f"Training on: {desc} ({env_id})")
        print(f"{'='*70}")
        
        save_path = f"models/ppo_{env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        model, callback = train_ppo_production(
            env_id=env_id,
            total_timesteps=total_timesteps,
            num_envs=4,
            save_path=save_path,
            seed=seed,
        )
        
        all_models[env_id] = {
            'model': model,
            'callback': callback,
            'save_path': save_path,
        }
    
    return all_models


# Import torch for policy kwargs
import torch


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO on Production SDN Environment')
    parser.add_argument('--env', type=str, default='SDNProduction-v0',
                        choices=['SDNProduction-v0', 'SDNBurst-v0', 'SDNHighNoise-v0', 
                                 'SDNLowSLA-v0', 'SDNDynamicCapacity-v0'],
                        help='Environment to train on')
    parser.add_argument('--timesteps', type=int, default=2_000_000,
                        help='Total training timesteps')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--multi-scenario', action='store_true',
                        help='Train on all scenarios')
    
    args = parser.parse_args()
    
    if args.multi_scenario:
        train_multi_scenario_production(
            total_timesteps=args.timesteps // 4,  # Split timesteps across scenarios
            seed=args.seed,
        )
    else:
        train_ppo_production(
            env_id=args.env,
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            seed=args.seed,
        )