#!/bin/bash
# ============================================
# PPO SIM-TO-REAL TRAINING: 500K STEPS
# Multi-Scenario + Noise + Diversity
# ============================================
# Chạy: bash ai_model/train_sim2real_500k.sh
# Hoặc: cd ai_model && python train_sim2real_500k.py

cd "$(dirname "$0")"

echo "=============================================="
echo "  PPO SIM-TO-REAL: 500K STEPS"
echo "  Multi-Scenario Training"
echo "=============================================="

python3 << 'TRAINING_SCRIPT'
import os, time, numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sdn_sim_env import *

print('='*65)
print('  SIM-TO-REAL: 500K STEPS, MULTI-SCENARIO')
print('='*65)

def make_mixed_env():
    """Tạo random environment từ nhiều scenarios"""
    scenarios = [
        'SDN-v0',           # Normal traffic
        'GoldenHour-v0',    # Flash crowd / burst traffic
        'LowRateDoS-v0',    # Slowloris DoS attack
        'HardwareDegradation-v0',  # Hardware failure
    ]
    return make_env(np.random.choice(scenarios))

# Tạo 4 parallel environments để train nhanh hơn
env = DummyVecEnv([make_mixed_env for _ in range(4)])

# PPO hyperparameters
model = PPO(
    'MlpPolicy', 
    env,
    learning_rate=3e-4,      # Learning rate
    n_steps=4096,           # Steps per update
    batch_size=256,         # Batch size lớn
    n_epochs=15,            # More epochs
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,        # GAE lambda
    clip_range=0.2,         # PPO clip range
    ent_coef=0.02,          # Entropy coefficient (thêm noise)
    vf_coef=0.5,            # Value function coefficient
    max_grad_norm=0.5,      # Gradient clipping
    verbose=1,
    device='cpu'
)

# Training loop
TOTAL_STEPS = 500000
STEPS_PER_ITER = 4096

print(f'\n[*] Target: {TOTAL_STEPS:,} steps')
print(f'[*] Starting training...')
print('[*] (Ctrl+C để dừng và save model)')
print()

start = time.time()
last_time = start

while model.num_timesteps < TOTAL_STEPS:
    model.learn(total_timesteps=STEPS_PER_ITER, progress_bar=False)
    
    # Progress every 50k steps
    if model.num_timesteps % 50000 == 0:
        elapsed = time.time() - start
        progress = model.num_timesteps / TOTAL_STEPS * 100
        eta = elapsed / progress * (100 - progress) / 60 if progress > 0 else 0
        
        print(f'  [{model.num_timesteps:>7,}/{TOTAL_STEPS:,}] {progress:>5.1f}% | '
              f'Elapsed: {elapsed/60:.1f}min | ETA: {eta:.1f}min')
        
        last_time = time.time()

elapsed = time.time() - start
print(f'\n[*] Training completed in {elapsed/60:.1f} minutes')

# Save model
os.makedirs('checkpoints', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f'checkpoints/ppo_sim2real_500k_{timestamp}.zip'
model.save(save_path)
print(f'[+] Model saved: {save_path}')

# ============================================
# BENCHMARK: So sánh PPO vs WRR
# ============================================
print('\n' + '='*65)
print('  BENCHMARK: PPO vs WRR')
print('='*65)

scenarios = ['SDN-v0', 'GoldenHour-v0', 'LowRateDoS-v0']

for s in scenarios:
    test_env = make_env(s)
    
    # Test PPO
    ppo_latencies = []
    for episode in range(50):
        obs, _ = test_env.reset()
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, d, t, info = test_env.step(action)
            ppo_latencies.append(info['latency'])
            if d:
                break
    
    # Test WRR (Round Robin = 1/3, 1/3, 1/3)
    wrr_latencies = []
    for episode in range(50):
        obs, _ = test_env.reset()
        for step in range(100):
            action = np.array([0.33, 0.33, 0.34], dtype=np.float32)
            obs, r, d, t, info = test_env.step(action)
            wrr_latencies.append(info['latency'])
            if d:
                break
    
    ppo_mean = np.mean(ppo_latencies)
    wrr_mean = np.mean(wrr_latencies)
    margin = wrr_mean - ppo_mean
    
    status = 'PPO WIN' if margin > 0 else 'WRR WIN'
    print(f'  {s:22s} | PPO: {ppo_mean:7.1f}ms | WRR: {wrr_mean:7.1f}ms | {status} ({margin:+.1f}ms)')

print('\n' + '='*65)
print(f'[OK] DONE! Model: {save_path}')
print('='*65)
TRAINING_SCRIPT

echo ""
echo "Training hoàn tất!"
echo "Model đã lưu trong thư mục ai_model/checkpoints/"
