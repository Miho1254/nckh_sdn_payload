#!/usr/bin/env python3
"""
ROBUST PPO Training Script cho SDN Load Balancer
- Sử dụng model.learn() với callbacks thay vì manual loop
- Checkpointing để tránh mất progress khi crash
- Logging metrics để theo dõi training progress

Chạy: python ai_model/train_robust_500k.py --steps 1000000
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from collections import deque

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdn_sim_env import SDNLoadBalancerEnv, make_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure


class MetricsCallback(BaseCallback):
    """Callback để track training metrics."""
    
    def __init__(self, eval_freq=1000, save_freq=50000, save_path="models/ppo_checkpoint", verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_path = save_path
        self.checkpoint_count = 0
        self.total_steps = 0
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.reward_window = deque(maxlen=100)
        self.length_window = deque(maxlen=100)
        
        # For tensorboard logging
        self._last_log_time = time.time()
        
    def _on_step(self):
        # Track episode rewards from infos
        if self.locals.get("infos") is not None:
            for info in self.locals["infos"]:
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.reward_window.append(ep_reward)
                    self.length_window.append(ep_length)
        
        self.total_steps = self.num_timesteps
        
        # Print progress
        if self.verbose > 0 and self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            if len(self.reward_window) > 0:
                mean_reward = np.mean(self.reward_window)
                std_reward = np.std(self.reward_window)
                mean_length = np.mean(self.length_window)
                
                elapsed = time.time() - self._last_log_time
                steps_per_sec = self.eval_freq / elapsed if elapsed > 0 else 0
                
                print(f"  Steps: {self.num_timesteps:,} | "
                      f"Mean Reward (last 100): {mean_reward:8.2f} ± {std_reward:7.2f} | "
                      f"Ep Length: {mean_length:6.0f} | "
                      f"{steps_per_sec:6.0f} steps/s")
                
                self._last_log_time = time.time()
        
        # Save checkpoint
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            checkpoint_path = f"{self.save_path}_step{self.num_timesteps}.zip"
            self.model.save(checkpoint_path)
            self.checkpoint_count += 1
            
            # Save metrics
            metrics_path = f"{self.save_path}_metrics{self.num_timesteps}.json"
            metrics = {
                'step': int(self.num_timesteps),
                'mean_reward': float(np.mean(self.reward_window)) if self.reward_window else 0,
                'std_reward': float(np.std(self.reward_window)) if self.reward_window else 0,
                'mean_length': float(np.mean(self.length_window)) if self.length_window else 0,
                'total_episodes': len(self.episode_rewards)
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            
            if self.verbose > 0:
                print(f"\n[CHECKPOINT #{self.checkpoint_count}] Saved at step {self.num_timesteps:,}")
                print(f"  -> {checkpoint_path}")
            
            # Clean old checkpoints (keep last 3)
            try:
                checkpoints = sorted([f for f in os.listdir(os.path.dirname(self.save_path)) 
                                      if f.startswith(os.path.basename(self.save_path)) and f.endswith('.zip')])
                while len(checkpoints) > 3:
                    old = checkpoints.pop(0)
                    try:
                        os.remove(os.path.join(os.path.dirname(self.save_path), old))
                        if self.verbose > 0:
                            print(f"  [CLEANUP] Removed old checkpoint: {old}")
                    except:
                        pass
            except:
                pass
        
        return True
    
    def _on_training_end(self):
        if self.verbose > 0:
            print(f"\n[*] Training ended at step {self.num_timesteps:,}")
            if len(self.episode_rewards) > 0:
                print(f"[*] Total episodes: {len(self.episode_rewards)}")
                print(f"[*] Final mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")


class EvalCallback(BaseCallback):
    """Callback để đánh giá model định kỳ."""
    
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            eval_rewards = []
            eval_lengths = []
            
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                total_reward = 0
                done = False
                length = 0
                
                while not done and length < 5000:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, trunc, info = self.eval_env.step(action)
                    total_reward += reward
                    length += 1
                
                eval_rewards.append(total_reward)
                eval_lengths.append(length)
            
            mean_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)
            
            if self.verbose > 0:
                print(f"\n[EVAL @ {self.num_timesteps:,}] "
                      f"Reward: {mean_reward:8.2f} ± {std_reward:7.2f} | "
                      f"Length: {np.mean(eval_lengths):.0f}")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = os.path.join(os.path.dirname(self.model.save_path or "models/ppo_best"), "ppo_best")
                self.model.save(best_path)
                if self.verbose > 0:
                    print(f"  [NEW BEST] Saved best model with reward {mean_reward:.2f}")
        
        return True


def train_robust_ppo(env_id, total_timesteps, checkpoint_freq=50000):
    """
    Training PPO với checkpointing và metrics tracking.
    """
    print("=" * 70)
    print(f"  ROBUST PPO TRAINING - {env_id}")
    print(f"  Total steps: {total_timesteps:,}")
    print(f"  Checkpoint every: {checkpoint_freq:,} steps")
    print("=" * 70)
    
    # Create directories
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    print("\n[*] Creating training environment...")
    env = make_env(env_id)
    
    # Create evaluation environment (separate instance)
    print("[*] Creating evaluation environment...")
    eval_env = make_env(env_id)
    
    # Model save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(model_dir, f"ppo_{env_id}_{timestamp}")
    
    # Create PPO model with good defaults for SDN
    print("[*] Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,          # Collect 2048 steps before update
        batch_size=64,         # Mini-batch size
        n_epochs=10,           # 10 passes over the data per update
        gamma=0.99,            # Discount factor
        gae_lambda=0.95,       # GAE lambda
        clip_range=0.2,        # PPO clipping
        ent_coef=0.01,         # Entropy coefficient for exploration
        normalize_advantage=True,
        max_grad_norm=0.5,     # Gradient clipping
        verbose=0,
        device='cpu',          # Force CPU (MlpPolicy)
        tensorboard_log=os.path.join(log_dir, f"ppo_{env_id}_{timestamp}"),
    )
    model.save_path = save_path  # Store for callbacks
    
    # Check for existing checkpoint to resume
    base_checkpoint = os.path.join(model_dir, f"ppo_{env_id}_checkpoint")
    start_step = 0
    
    # Setup logger (skip tensorboard if not installed)
    try:
        new_logger = configure(os.path.join(log_dir, f"ppo_{env_id}_{timestamp}"), ["stdout", "csv"])
        model.set_logger(new_logger)
    except:
        print("[*] Warning: Could not setup CSV logger, continuing without file logging")
    
    # Setup callbacks
    metrics_callback = MetricsCallback(
        eval_freq=5000,
        save_freq=checkpoint_freq,
        save_path=os.path.join(model_dir, f"ppo_{env_id}_checkpoint"),
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=max(10000, checkpoint_freq // 5),
        n_eval_episodes=3,
        verbose=1
    )
    
    print(f"\n[*] Starting training from step {start_step:,}...")
    print(f"[*] Target: {total_timesteps:,} steps")
    print(f"[*] TensorBoard log: {os.path.join(log_dir, f'ppo_{env_id}_{timestamp}')}")
    print()
    print("  " + "-" * 66)
    print(f"  {'Steps':<12} | {'Mean Reward':<18} | {'Ep Length':<10} | {'Speed':<8}")
    print("  " + "-" * 66)
    
    start_time = time.time()
    
    # TRAIN using model.learn() - this is the proper way!
    model.learn(
        total_timesteps=total_timesteps,
        callback=[metrics_callback, eval_callback],
        progress_bar=False,  # Disable for compatibility
        reset_num_timesteps=(start_step == 0)
    )
    
    elapsed = time.time() - start_time
    
    # Final save
    final_path = f"{save_path}_final_{total_timesteps}.zip"
    model.save(final_path)
    
    print("\n" + "=" * 70)
    print(f"[*] Training completed!")
    print(f"[*] Total steps: {total_timesteps:,}")
    print(f"[*] Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[*] Steps per second: {total_timesteps/elapsed:.1f}")
    print(f"[✓] Final model saved: {final_path}")
    
    # Final evaluation
    print("\n[*] Final evaluation (10 episodes)...")
    final_eval_rewards = []
    final_eval_lengths = []
    
    for i in range(10):
        obs, _ = eval_env.reset()
        total_reward = 0
        done = False
        length = 0
        
        while not done and length < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = eval_env.step(action)
            total_reward += reward
            length += 1
        
        final_eval_rewards.append(total_reward)
        final_eval_lengths.append(length)
    
    print(f"\n[*] Evaluation Results:")
    print(f"    Mean reward: {np.mean(final_eval_rewards):.2f} ± {np.std(final_eval_rewards):.2f}")
    print(f"    Mean episode length: {np.mean(final_eval_lengths):.0f}")
    
    # Training curve summary
    if len(metrics_callback.episode_rewards) > 0:
        print(f"\n[*] Training summary:")
        print(f"    Total episodes: {len(metrics_callback.episode_rewards)}")
        
        # Show rewards in phases
        n = len(metrics_callback.episode_rewards)
        if n >= 50:
            first_50 = np.mean(metrics_callback.episode_rewards[:50])
            last_50 = np.mean(metrics_callback.episode_rewards[-50:])
            print(f"    First 50 eps mean: {first_50:.2f}")
            print(f"    Last 50 eps mean:  {last_50:.2f}")
            print(f"    Improvement: {last_50 - first_50:+.2f}")
    
    return model, metrics_callback


def benchmark_vs_wrr(model_path, env_id='SDN-v0'):
    """So sánh PPO model với WRR baseline."""
    print("\n" + "=" * 70)
    print("  BENCHMARK: PPO vs WRR")
    print("=" * 70)
    
    # Load PPO model
    model = PPO.load(model_path)
    
    # Test on multiple scenarios
    scenarios = ['SDN-v0', 'GoldenHour-v0', 'LowRateDoS-v0']
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n[*] Scenario: {scenario}")
        
        # PPO evaluation
        env = make_env(scenario)
        ppo_latencies = []
        ppo_rewards = []
        
        for ep in range(5):
            obs, _ = env.reset()
            latencies = []
            rewards = []
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = env.step(action)
                latencies.append(info['latency'])
                rewards.append(reward)
            
            if latencies:
                ppo_latencies.append(np.mean(latencies))
                ppo_rewards.append(np.sum(rewards))
        
        # WRR evaluation (round-robin)
        env = make_env(scenario)
        wrr_latencies = []
        wrr_rewards = []
        
        for ep in range(5):
            obs, _ = env.reset()
            latencies = []
            rewards = []
            done = False
            step = 0
            
            while not done:
                # WRR: cycle through servers - convert to [w5, w7, w8] array
                server_idx = step % 3
                if server_idx == 0:
                    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # h5 only
                elif server_idx == 1:
                    action = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # h7 only
                else:
                    action = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # h8 only
                obs, reward, done, trunc, info = env.step(action)
                latencies.append(info['latency'])
                rewards.append(reward)
                step += 1
            
            if latencies:
                wrr_latencies.append(np.mean(latencies))
                wrr_rewards.append(np.sum(rewards))
        
        ppo_lat_mean = np.mean(ppo_latencies)
        wrr_lat_mean = np.mean(wrr_latencies)
        ppo_rew_mean = np.mean(ppo_rewards)
        wrr_rew_mean = np.mean(wrr_rewards)
        
        lat_improvement = ((wrr_lat_mean - ppo_lat_mean) / wrr_lat_mean * 100) if wrr_lat_mean > 0 else 0
        rew_improvement = ((ppo_rew_mean - wrr_rew_mean) / abs(wrr_rew_mean) * 100) if wrr_rew_mean != 0 else 0
        
        print(f"    PPO: latency={ppo_lat_mean:.1f}ms, reward={ppo_rew_mean:.1f}")
        print(f"    WRR: latency={wrr_lat_mean:.1f}ms, reward={wrr_rew_mean:.1f}")
        print(f"    Latency improvement: {lat_improvement:+.1f}%")
        print(f"    Reward improvement: {rew_improvement:+.1f}%")
        
        all_results.append({
            'scenario': scenario,
            'ppo_latency': ppo_lat_mean,
            'wrr_latency': wrr_lat_mean,
            'ppo_reward': ppo_rew_mean,
            'wrr_reward': wrr_rew_mean,
            'lat_improvement': lat_improvement,
            'rew_improvement': rew_improvement
        })
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Robust PPO Training for SDN Load Balancer')
    parser.add_argument('--steps', type=int, default=1000000,
                       help='Total training timesteps (default: 1000000)')
    parser.add_argument('--scenario', type=str, default='SDN-v0',
                       choices=['SDN-v0', 'GoldenHour-v0', 'VideoConference-v0',
                               'HardwareDegradation-v0', 'LowRateDoS-v0'],
                       help='Scenario to train on')
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                       help='Checkpoint frequency in steps (default: 50000)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark after training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    
    args = parser.parse_args()
    
    print(f"[*] Target: {args.steps:,} steps")
    print(f"[*] Scenario: {args.scenario}")
    print(f"[*] Checkpoint every: {args.checkpoint_freq:,} steps")
    print()
    
    # Train
    model, metrics = train_robust_ppo(
        env_id=args.scenario,
        total_timesteps=args.steps,
        checkpoint_freq=args.checkpoint_freq
    )
    
    # Benchmark
    if args.benchmark:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/ppo_{args.scenario}_{timestamp}_final_{args.steps}.zip"
        results = benchmark_vs_wrr(save_path, args.scenario)
        
        print("\n" + "=" * 70)
        print("  BENCHMARK SUMMARY")
        print("=" * 70)
        for r in results:
            print(f"  {r['scenario']}:")
            print(f"    PPO latency: {r['ppo_latency']:.1f}ms vs WRR: {r['wrr_latency']:.1f}ms ({r['lat_improvement']:+.1f}%)")
            print(f"    PPO reward:  {r['ppo_reward']:.1f} vs WRR: {r['wrr_reward']:.1f} ({r['rew_improvement']:+.1f}%)")
    
    print("\n[*] Done!")
