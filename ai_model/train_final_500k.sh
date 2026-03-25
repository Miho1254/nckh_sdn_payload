#!/bin/bash
# PPO 500K STEPS - FINAL VERSION
cd "$(dirname "$0")"

echo "=========================================="
echo "  PPO 500K STEPS - FINAL"
echo "=========================================="

python3 << 'PYEOF'
import os, time, numpy as np
from datetime import datetime
from sdn_sim_env import make_env
from stable_baselines3 import PPO

print('Starting PPO 500K training...')
env = make_env('GoldenHour-v0')

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
start = time.time()
last_report = 0

while model.num_timesteps < TOTAL:
    model.learn(total_timesteps=10240, progress_bar=False)
    current = model.num_timesteps
    if current - last_report >= 50000:
        elapsed = time.time() - start
        pct = current / TOTAL * 100
        eta_min = elapsed / pct * (100 - pct) / 60 if pct > 0 else 0
        print(f'  {current:>7,}/{TOTAL:,} ({pct:5.1f}%) | {elapsed/60:.1f}min | ETA: {eta_min:.1f}min')
        last_report = current

elapsed = time.time() - start
print(f'\nDone in {elapsed/60:.1f} min!')

path = f'checkpoints/ppo_500k_{datetime.now().strftime("%H%M%S")}.zip'
model.save(path)
print(f'Saved: {path}')

# Quick benchmark
print('\n=== BENCHMARK ===')
for s in ['SDN-v0', 'GoldenHour-v0', 'LowRateDoS-v0']:
    te = make_env(s)
    ppo_l = []; wrr_l = []
    for _ in range(20):
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
    margin = np.mean(wrr_l) - np.mean(ppo_l)
    print(f'  {s:20s} | PPO:{np.mean(ppo_l):6.1f}ms | WRR:{np.mean(wrr_l):6.1f}ms | {"PPO WIN" if margin>0 else "WRR WIN"} ({margin:+.0f}ms)')

print('\nCOMPLETE!')
PYEOF
