#!/bin/bash
cd "$(dirname "$0")"
echo "PPO 500K - Starting..."

python3 -u << 'PYEOF'
import os, sys, time, numpy as np
from datetime import datetime

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

print('Importing...', flush=True)
from sdn_sim_env import make_env
from stable_baselines3 import PPO

print('Creating env...', flush=True)
env = make_env('GoldenHour-v0')

print('Creating model...', flush=True)
model = PPO(
    'MlpPolicy', env,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.01,
    verbose=0,
    device='cpu'
)

TOTAL = 500000
print(f'Training {TOTAL:,} steps...', flush=True)
start = time.time()

while model.num_timesteps < TOTAL:
    model.learn(total_timesteps=10240, progress_bar=False)
    current = model.num_timesteps
    elapsed = time.time() - start
    pct = current / TOTAL * 100
    eta = elapsed / pct * (100 - pct) / 60 if pct > 0 else 0
    print(f'  {current:>7,}/{TOTAL:,} ({pct:5.1f}%) | {elapsed/60:.1f}min | ETA:{eta:.1f}min', flush=True)

elapsed = time.time() - start
print(f'\nDone in {elapsed/60:.1f} min!', flush=True)

path = f'checkpoints/ppo_500k_{datetime.now().strftime("%H%M%S")}.zip'
model.save(path)
print(f'Saved: {path}', flush=True)

# Benchmark
print('\nBenchmark:', flush=True)
for s in ['SDN-v0', 'GoldenHour-v0', 'LowRateDoS-v0']:
    te = make_env(s)
    ppo_l, wrr_l = [], []
    for _ in range(15):
        o, _ = te.reset()
        for _ in range(50):
            a, _ = model.predict(o, deterministic=True)
            o, r, d, t, i = te.step(a)
            ppo_l.append(i['latency'])
            if d: break
        o, _ = te.reset()
        for _ in range(50):
            o, r, d, t, i = te.step(np.array([0.33, 0.33, 0.34]))
            wrr_l.append(i['latency'])
            if d: break
    m = np.mean(wrr_l) - np.mean(ppo_l)
    print(f'  {s:20s} | PPO:{np.mean(ppo_l):6.1f} | WRR:{np.mean(wrr_l):6.1f} | {"PPO WIN" if m>0 else "WRR WIN"} ({m:+.0f}ms)', flush=True)

print('\nCOMPLETE!', flush=True)
PYEOF

echo "Done!"
