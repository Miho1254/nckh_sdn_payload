#!/usr/bin/env python3
"""
PPO Training Script cho SDN Load Balancer

Chạy: python ai_model/train_ppo_simple.py

Dependencies:
    pip install gymnasium stable-baselines3 numpy matplotlib
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdn_sim_env import SDNLoadBalancerEnv, GoldenHourEnv, make_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env


class RewardLoggerCallback(BaseCallback):
    """Callback để log reward trong training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self):
        # Track rewards
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.episode_count += 1
            
            # Print progress every 10 episodes
            if self.episode_count % 10 == 0 and self.verbose > 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"  Episode {self.episode_count}: avg reward (last 10) = {avg_reward:.2f}")
        
        return True


def train_ppo(env_id='SDN-v0', total_timesteps=50_000, save_path=None):
    """
    Train PPO agent cho SDN Load Balancer.
    
    Args:
        env_id: Environment ID
        total_timesteps: Số steps training
        save_path: Path để save model
    """
    print("="*60)
    print(f"  PPO TRAINING - {env_id}")
    print("="*60)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create environment
    print("\n[*] Creating environment...")
    env = make_env(env_id)
    
    # Check environment (for debugging)
    print("[*] Checking environment compatibility...")
    try:
        check_env(env, warn=True)
        print("[✓] Environment check passed!")
    except Exception as e:
        print(f"[!] Environment warning: {e}")
    
    # Create evaluation environment
    eval_env = make_env(env_id)
    
    # PPO Hyperparameters
    # MLP Policy với 64x64 hidden layers
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,      # Standard PPO lr
        n_steps=2048,            # Steps per update
        batch_size=64,           # Batch size cho update
        n_epochs=10,              # PPO epochs per update
        gamma=0.99,              # Discount factor
        gae_lambda=0.95,         # GAE lambda
        clip_range=0.2,          # PPO clip range
        ent_coef=0.01,           # Entropy coefficient (exploration)
        normalize_advantage=True,
        max_grad_norm=0.5,       # Gradient clipping
        verbose=1,
        tensorboard_log=f"./logs/tensorboard_{env_id}/"
    )
    
    print(f"\n[*] Model architecture:")
    print(f"    Policy: MLP (64x64)")
    print(f"    Learning rate: {model.learning_rate}")
    print(f"    Gamma: {model.gamma}")
    print(f"    N steps: {model.n_steps}")
    print(f"    Batch size: {model.batch_size}")
    
    # Callbacks
    reward_logger = RewardLoggerCallback(verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/best_{env_id}/",
        log_path=f"./logs/eval_{env_id}/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Training
    print(f"\n[*] Starting training...")
    print(f"[*] Total timesteps: {total_timesteps}")
    print(f"[*] Estimated time: {total_timesteps/1000:.0f}-{total_timesteps/500:.0f} seconds")
    print()
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, eval_callback],
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    print(f"\n[*] Training completed in {training_time:.1f} seconds")
    
    # Save model
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"ai_model/ppo_sdn_{env_id}_{timestamp}.zip"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"[✓] Model saved: {save_path}")
    
    # Final evaluation
    print("\n[*] Final evaluation (10 episodes)...")
    eval_rewards = []
    eval_lengths = []
    
    for i in range(10):
        obs, _ = eval_env.reset()
        total_reward = 0
        done = False
        length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = eval_env.step(action)
            total_reward += reward
            length += 1
        
        eval_rewards.append(total_reward)
        eval_lengths.append(length)
    
    print(f"\n[*] Evaluation Results:")
    print(f"    Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"    Mean episode length: {np.mean(eval_lengths):.0f} ± {np.std(eval_lengths):.0f}")
    
    # Training curve
    if len(reward_logger.episode_rewards) > 0:
        print(f"\n[*] Training curve (last 20 episodes):")
        last_20 = reward_logger.episode_rewards[-20:]
        for i, r in enumerate(last_20):
            bar = "█" * int(max(1, r / 5))
            print(f"    Episode {reward_logger.episode_count - 20 + i + 1}: {r:7.2f} |{bar}")
    
    return model, reward_logger


def train_multi_scenario():
    """Train PPO trên nhiều scenarios để có robust model."""
    
    scenarios = ['SDN-v0', 'GoldenHour-v0']
    
    print("\n" + "="*60)
    print("  MULTI-SCENARIO TRAINING")
    print("="*60)
    
    all_models = {}
    
    for scenario in scenarios:
        model, logger = train_ppo(
            env_id=scenario,
            total_timesteps=30_000,
            save_path=f"ai_model/ppo_{scenario.lower().replace('-', '_')}.zip"
        )
        all_models[scenario] = model
        
        print(f"\n[*] Trained on {scenario}")
    
    print("\n" + "="*60)
    print("  ALL SCENARIOS TRAINED!")
    print("="*60)
    
    return all_models


def quick_test():
    """Quick test để verify environment hoạt động."""
    
    print("\n[*] Quick environment test...")
    env = SDNLoadBalancerEnv()
    obs, _ = env.reset()
    
    print(f"    Observation shape: {obs.shape}")
    print(f"    Action space: {env.action_space}")
    
    # Test 10 random steps
    print("\n[*] Running 10 random steps:")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        print(f"    Step {i+1}: r={reward:7.3f} | weights=[{info['weights'][0]:.2f}, {info['weights'][1]:.2f}, {info['weights'][2]:.2f}] | lat={info['latency']:6.1f}ms")
        
        if done:
            print(f"    [CRASH!] Episode ended early")
            break
    
    print("\n[✓] Quick test passed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO Training for SDN Load Balancer')
    parser.add_argument('--scenario', type=str, default='SDN-v0',
                       choices=['SDN-v0', 'GoldenHour-v0', 'VideoConference-v0', 
                               'HardwareDegradation-v0', 'LowRateDoS-v0', 'all'],
                       help='Scenario to train on')
    parser.add_argument('--timesteps', type=int, default=50_000,
                       help='Total training timesteps')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test only')
    
    args = parser.parse_args()
    
    if args.test:
        quick_test()
    elif args.scenario == 'all':
        train_multi_scenario()
    else:
        train_ppo(
            env_id=args.scenario,
            total_timesteps=args.timesteps,
            save_path=f"ai_model/ppo_sdn_load_balancer.zip"
        )
    
    print("\n[*] Done!")
