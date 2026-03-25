#!/bin/bash
# PPO SIM-TO-REAL 500K STEPS - FIXED
cd "$(dirname "$0")"

echo "=========================================="
echo "  PPO SIM-TO-REAL 500K - FIXED VERSION"
echo "=========================================="

python3 << 'TRAINING_SCRIPT'
import os, sys, time, numpy as np
from datetime import datetime

# Force stdout flush
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

from sdn_sim_env import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

print('Starting...', flush=True)

def make_mixed_env():
    scenarios = ['SDN-v0', 'GoldenHour-v0', 'LowRateDoS-v0']
    return make_env(np.random.choice(scenarios))

# Single env for now (avoid vec env complexity)
env = make_mixed_env()

model = PPO(
    'MlpPolicy', env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    verbose=0,  # Silent mode
    device='cpu'
)

TOTAL = 500000
print(f'Target: {TOTAL:,} steps', flush=True)

start = time.time()
last_step = 0

while model.num_timesteps < TOTAL:
    # Learn in chunks
    model.learn(total_timesteps=10000, progress_bar=False)
    
    current = model.num_timesteps
    if current - last_step >= 50000:
        elapsed = time.time() - start
        pct = current / TOTAL * 100
        eta = elapsed / pct * (100 - pct) / 60 if pct > 0 else 0
        print(f'  {current:>7,}/{TOTAL:,} ({pct:5.1f}%) | {elapsed/60:.1f}min elapsed | ETA: {eta:.1f}min', flush=True)
        last_step = current

elapsed = time.time() - start
print(f'\nTraining done in {elapsed/60:.1f} min', flush=True)

# Save
os.makedirs('checkpoints', exist_ok=True)
path = f'checkpoints/ppo_sim2real_500k_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
model.save(path)
print(f'Saved: {path}', flush=True)

# Benchmark
print('\n=== BENCHMARK ===', flush=True)
for s in ['SDN-v0', 'GoldenHour-v0', 'LowRateDoS-v0']:
    te = make_env(s)
    
    ppo_lats = []
    for _ in range(30):
        o, _ = te.reset()
        for _ in range(100):
            a, _ = model.predict(o, deterministic=True)
            o, r, d, t, i = te.step(a)
            ppo_lats.append(i['latency'])
            if d: break
    
    wrr_lats = []
    for _ in range(30):
        o, _ = te.reset()
        for _ in range(100):
            o, r, d, t, i = te.step(np.array([0.33, 0.33, 0.34]))
            wrr_lats.append(i['latency'])
            if d: break
    
    margin = np.mean(wrr_lats) - np.mean(ppo_lats)
    status = 'PPO WIN' if margin > 0 else 'WRR WIN'
    print(f'  {s:20s} | PPO: {np.mean(ppo_lats):6.1f}ms | WRR: {np.mean(wrr_lats):6.1f}ms | {status} ({margin:+.0f}ms)', flush=True)

print('\nDONE!', flush=True)
TRAINING_SCRIPT

echo ""
echo "Hoan tat! Model trong ai_model/checkpoints/"
