"""
TFT-CQL Training Pipeline — 3 Phase.

Phase 1: Pretrain temporal encoder (forecast task)
Phase 2: Offline actor-critic CQL training
Phase 3: Constraint tuning + model selection

Usage:
    python train_actor_critic.py --epochs 100 --phase all
    python train_actor_critic.py --phase pretrain --epochs 30
    python train_actor_critic.py --phase train --encoder_ckpt checkpoints/encoder_best.pth
"""
import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cql_agent import CQLAgent
from sdn_env_v2 import SDN_Offline_Env_V2
from config import (NUM_ACTIONS, CQL_ALPHA, ACTOR_LR, CRITIC_LR,
                    FORECAST_LOSS_WEIGHT, ENTROPY_COEFF, CONSTRAINT_WEIGHTS,
                    SEQUENCE_LENGTH, CAPACITY_RATIOS, CAPACITIES, CAPACITY_PRIOR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'training_logs')

# Ensure directories exist and are writable
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def load_v3_data():
    """Load V3 dataset + metadata."""
    x_path = os.path.join(DATA_DIR, 'X_v3.npy')
    y_path = os.path.join(DATA_DIR, 'y_v3.npy')
    s_path = os.path.join(DATA_DIR, 'scenarios_v3.npy')
    meta_path = os.path.join(DATA_DIR, 'feature_metadata.json')

    if not os.path.exists(x_path):
        print(f"[!] V3 data not found at {x_path}")
        print("    Run data_processor.py first to generate V3 features.")
        sys.exit(1)

    X = np.load(x_path)
    y = np.load(y_path)
    scen = np.load(s_path) if os.path.exists(s_path) else np.array(['UNKNOWN']*len(y))

    metadata = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    print(f"[*] Loaded V3 data: X={X.shape}, y={y.shape}")
    if metadata:
        print(f"    Features: {metadata['num_features']}, Version: {metadata['version']}")
    return X, y, scen, metadata


def phase0_capacity_ratio_pretraining(agent, X, y, scen, env, epochs=1000, batch_size=64):
    """Phase 0: Supervised pretraining để AI học balanced distribution từ dữ liệu.
    
    V7 - BALANCED: Target distribution là [0.33, 0.33, 0.34] để match với balanced dataset.
    
    Mục tiêu: Force AI learn the BALANCED action distribution
    bằng cách supervised learning trên action distribution.
    
    Cách hoạt động:
    - Với mỗi state, AI output policy distribution
    - Target distribution là BALANCED: [0.33, 0.33, 0.34]
    - Sử dụng KL divergence loss với soft targets để buộc policy khớp chính xác
    - Chỉ update actor head (không update encoder)
    - Không có entropy bonus - chỉ dùng KL divergence loss để force AI học đúng distribution
    """
    print("\n" + "="*60)
    print("  PHASE 0: BALANCED DISTRIBUTION KL DIVERGENCE PRETRAINING")
    print("="*60)
    print(f"  Target distribution: {CAPACITY_PRIOR} (BALANCED)")
    
    num_samples = len(X) - 1
    best_loss = float('inf')
    patience, patience_limit = 0, 50
    losses = []
    
    # Target distribution là BALANCED prior [0.33, 0.33, 0.34]
    target_dist = np.array(CAPACITY_PRIOR, dtype=np.float32)
    target_tensor = torch.from_numpy(target_dist).unsqueeze(0).to(agent.device)  # (1, num_actions)
    
    # KL divergence loss coefficient - TĂNG CAO HƠN ĐỂ ÉP AI HỌC CAPACITY RATIO
    # V5: 500.0 không đủ, distribution vẫn gần uniform [27%, 34%, 39%]
    # V6: Tăng lên 1000.0 và thêm cross-entropy loss để ép chặt hơn
    kl_coeff = 1000.0
    
    # TĂNG learning rate cho Phase 0 để hội tụ nhanh hơn
    # V5: 1e-4 không đủ
    # V6: Tăng lên 3e-4
    for param_group in agent.actor_optimizer.param_groups:
        param_group['lr'] = 3e-4
    
    # Đặt model vào train mode
    agent.model.train()
    
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        epoch_loss = 0.0
        num_batches = 0
        
        for start in range(0, num_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 2:
                continue
            
            states = X[batch_idx]
            states_tensor = torch.from_numpy(states).float().to(agent.device)
            
            # Forward pass: get policy logits
            policy_logits = agent.model.actor_head(agent.model.encode(states_tensor))
            policy_logits = policy_logits.clamp(-10.0, 10.0)
            
            # Compute log probabilities
            log_policy = F.log_softmax(policy_logits, dim=-1)
            
            # KL divergence loss: buộc policy phải khớp chính xác với capacity ratios
            # KL(P||Q) = sum(P * log(P/Q)) - khuyến khích policy hội tụ về target
            capacity_loss = kl_coeff * F.kl_div(
                log_policy, target_tensor.expand(len(batch_idx), -1),
                reduction='batchmean')
            
            # Total loss = KL divergence loss (không có entropy bonus)
            total_loss = capacity_loss
            
            # Backward pass - chỉ update actor head
            agent.actor_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.model.actor_head.parameters()) +
                list(agent.model.safety_head.parameters()),
                max_norm=1.0)
            agent.actor_optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(1, num_batches)
        losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            agent.save_checkpoint(
                os.path.join(CKPT_DIR, 'encoder_best.pth'),
                epoch=epoch, metrics={'capacity_kl_loss': avg_loss})
        else:
            patience += 1
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            # Evaluate current policy distribution
            agent.model.eval()
            with torch.no_grad():
                test_states = X[:10]  # Test với 10 samples
                test_states_t = torch.from_numpy(test_states).float().to(agent.device)
                test_logits = agent.model.actor_head(agent.model.encode(test_states_t))
                test_probs = F.softmax(test_logits, dim=-1)
                avg_test_probs = test_probs.mean(dim=0).cpu().numpy()
            agent.model.train()
            
            print(f"  Epoch {epoch+1:3d}/{epochs} | Capacity Loss: {avg_loss:.6f} | "
                  f"Best: {best_loss:.6f} | Patience: {patience}/{patience_limit}")
            print(f"          Current policy: h5={avg_test_probs[0]:.2%} h7={avg_test_probs[1]:.2%} h8={avg_test_probs[2]:.2%}")
        
        if patience >= patience_limit:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    agent.model.eval()
    with torch.no_grad():
        test_states = X[:100] if len(X) > 100 else X
        test_states_t = torch.from_numpy(test_states).float().to(agent.device)
        test_logits = agent.model.actor_head(agent.model.encode(test_states_t))
        test_probs = F.softmax(test_logits, dim=-1)
        avg_test_probs = test_probs.mean(dim=0).cpu().numpy()
    agent.model.train()
    
    print(f"  Final capacity loss: {best_loss:.6f}")
    print(f"  Final policy distribution: h5={avg_test_probs[0]:.2%} h7={avg_test_probs[1]:.2%} h8={avg_test_probs[2]:.2%}")
    return losses


def phase1_pretrain_encoder(agent, X, y, epochs=30, batch_size=64):
    """Phase 1: Pretrain temporal encoder via forecast task."""
    print("\n" + "="*60)
    print("  PHASE 1: ENCODER PRETRAINING (Forecast)")
    print("="*60)

    num_samples = len(X) - 1
    best_loss = float('inf')
    patience, patience_limit = 0, 10
    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 2:
                continue
            states = X[batch_idx]
            next_states = X[batch_idx + 1]

            loss = agent.pretrain_encoder(states, next_states)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            agent.save_checkpoint(
                os.path.join(CKPT_DIR, 'encoder_best.pth'),
                epoch=epoch, metrics={'forecast_loss': avg_loss})
        else:
            patience += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Forecast Loss: {avg_loss:.6f} | "
                  f"Best: {best_loss:.6f} | Patience: {patience}/{patience_limit}")

        if patience >= patience_limit:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return losses


def phase2_offline_training(agent, X, y, scen, env, epochs=100, batch_size=64):
    print("\\n" + "="*60)
    print("  PHASE 2: OFFLINE CQL ACTOR-CRITIC TRAINING")
    print("="*60)

    num_samples = len(X) - 1
    all_metrics = []
    best_composite = -float('inf')
    
    # Early stopping trackers
    patience = 0
    patience_limit = 10

    # ═══════════════════════════════════════════════════════════════
    # V15: ENTROPY WARMUP DECAY SCHEDULE
    # Ban đầu: entropy_coeff = 2.0 (ép AI khám phá h5/h7/h8)
    # Cuối:    entropy_coeff = 0.5 (cho phép policy converge)
    # Decay:   tuyến tính sau 80% epochs
    # ═══════════════════════════════════════════════════════════════
    initial_entropy_coeff = agent.entropy_coeff  # 2.0 (from config)
    final_entropy_coeff = 0.5
    warmup_ratio = 0.8  # Giữ cao trong 80% epochs đầu

    for epoch in range(epochs):
        # V15: Entropy decay schedule
        if epoch < int(epochs * warmup_ratio):
            # Phase warmup: giữ entropy_coeff cao
            agent.entropy_coeff = initial_entropy_coeff
        else:
            # Phase decay: giảm tuyến tính về final
            decay_progress = (epoch - int(epochs * warmup_ratio)) / max(1, epochs - int(epochs * warmup_ratio))
            agent.entropy_coeff = initial_entropy_coeff + (final_entropy_coeff - initial_entropy_coeff) * decay_progress

        indices = np.random.permutation(num_samples)
        epoch_losses = {}
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 2:
                continue

            states = X[batch_idx]
            next_states = X[np.minimum(batch_idx + 1, num_samples)]

            # Behavior policy actions (use sampling for offline RL dataset collection simulation)
            actions = []
            rewards = []
            dones = []
            infos = []
            for j, idx in enumerate(batch_idx):
                env.current_step = idx
                env.prev_action = None
                action = agent.select_action(states[j], deterministic=False)
                _, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                dones.append(float(done))
                infos.append(info)

            actions = np.array(actions)
            rewards = np.array(rewards)
            dones = np.array(dones)

            losses = agent.train_step(states, next_states, rewards, actions, dones, infos)
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            num_batches += 1

        avg_losses = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}
        
        # Evaluate 3 modes
        eval_res = _evaluate_checkpoint(agent, X, y, scen, env)
        eval_score = eval_res['score']
        
        avg_losses['eval_score'] = eval_score
        avg_losses.update(eval_res)
        all_metrics.append({'epoch': epoch, **avg_losses})

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Critic: {avg_losses.get('critic_loss', 0):.4f} | "
                  f"Actor: {avg_losses.get('actor_loss', 0):.4f} | "
                  f"Raw(H): {eval_res['raw_entropy']:.3f} | "
                  f"Samp(H): {eval_res['samp_entropy']:.3f} | "
                  f"Arg(H): {eval_res['arg_entropy']:.3f} | "
                  f"Eval: {eval_score:.4f}")
            
            raw_str = f"[{eval_res['raw_mean'][0]:.2f}, {eval_res['raw_mean'][1]:.2f}, {eval_res['raw_mean'][2]:.2f}]"
            smp_str = f"h5={eval_res['samp_freq'][0]:.0%} h7={eval_res['samp_freq'][1]:.0%} h8={eval_res['samp_freq'][2]:.0%}"
            arg_str = f"h5={eval_res['arg_freq'][0]:.0%} h7={eval_res['arg_freq'][1]:.0%} h8={eval_res['arg_freq'][2]:.0%}"
            print(f"          Raw:  {raw_str}")
            print(f"          Samp: {smp_str}")
            print(f"          Arg:  {arg_str}")

            for k, sc in eval_res['scen_breakdown'].items():
                sc_str = f"h5={sc['samp'][0]:.0%} h7={sc['samp'][1]:.0%} h8={sc['samp'][2]:.0%}"
                print(f"            ↳ {k}: {sc_str}")

        # Checkpoint selection: Must have healthy sampled entropy (> 0.5) to be considered 'best'
        # Early stopping logic - DISABLED V8: Trong Offline RL, Entropy thường tụt mạnh ban đầu
        # trước khi AI nhận ra sai và tăng lại. Phải ép chạy đủ 100 epochs để AI nếm hình phạt.
        # if eval_res['samp_entropy'] < 0.3:
        #     patience += 1
        #     if patience >= patience_limit:
        #         print(f"  [!] Early stopping triggered at epoch {epoch+1} due to prolonged entropy collapse (H < 0.3).")
        #         break
        # else:
        #     patience = max(0, patience - 1)
            
        if eval_score > best_composite and eval_res['samp_entropy'] > 0.5:
            best_composite = eval_score
            agent.save_checkpoint(
                os.path.join(CKPT_DIR, 'tft_ac_best.pth'),
                epoch=epoch, metrics=avg_losses)

        if (epoch + 1) % 20 == 0:
            agent.save_checkpoint(
                os.path.join(CKPT_DIR, f'tft_ac_epoch{epoch+1}.pth'),
                epoch=epoch, metrics=avg_losses)

    agent.save_checkpoint(
        os.path.join(CKPT_DIR, 'tft_ac_final.pth'),
        epoch=epochs-1, metrics=avg_losses)

    return all_metrics


def phase3_constraint_tuning(agent, X, y, scen, env, epochs=20, batch_size=64):
    """Phase 3: Fine-tune constraint weights for safe policy."""
    print("\n" + "="*60)
    print("  PHASE 3: CONSTRAINT TUNING")
    print("="*60)

    # Load best checkpoint from phase 2
    best_path = os.path.join(CKPT_DIR, 'tft_ac_best.pth')
    if os.path.exists(best_path):
        try:
            agent.load_checkpoint(best_path)
            print("  Loaded best checkpoint from Phase 2")
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
            print("  Continuing with current model state")

    # Increase constraint weights for fine-tuning
    agent.constraint_weights = {
        "util_breach": 4.0,
        "fairness_dev": 2.0,
        "action_churn": 1.0,
    }

    metrics = phase2_offline_training(agent, X, y, scen, env, epochs=epochs, batch_size=batch_size)

    # Save final model
    agent.save_checkpoint(
        os.path.join(CKPT_DIR, 'tft_ac_final.pth'),
        epoch=epochs, metrics=metrics[-1] if metrics else {})

    return metrics


def _get_action_distribution(agent, X_subset):
    import torch
    import numpy as np
    from config import NUM_ACTIONS
    raw_probs_sum = np.zeros(NUM_ACTIONS)
    sampled_counts = np.zeros(NUM_ACTIONS)
    argmax_counts  = np.zeros(NUM_ACTIONS)

    agent.model.eval()
    for i in range(len(X_subset)):
        state = X_subset[i]
        dist = agent.get_policy_distribution(state)
        raw_probs_sum += dist
        
        # Sampled action
        action_samp = agent.select_action(state, deterministic=False)
        sampled_counts[action_samp] += 1
        
        # Argmax action
        action_arg = agent.select_action(state, deterministic=True)
        argmax_counts[action_arg] += 1
        
    agent.model.train()

    n = len(X_subset)
    raw_mean = raw_probs_sum / max(1, n)
    served_freq = sampled_counts / max(1, sampled_counts.sum())
    argmax_freq = argmax_counts / max(1, argmax_counts.sum())
    return raw_mean, served_freq, argmax_freq


def _evaluate_checkpoint(agent, X, y, scen, env):
    import numpy as np
    from config import CAPACITY_RATIOS, CAPACITIES, SCALING_LOAD, NUM_ACTIONS
    total_steps = min(len(X) - 1, 500)

    action_counts_samp = np.zeros(NUM_ACTIONS)
    action_counts_arg  = np.zeros(NUM_ACTIONS)
    raw_probs_sum = np.zeros(NUM_ACTIONS)
    
    overloads_samp = 0
    overloads_arg = 0
    churn_samp = 0
    churn_arg = 0
    prev_samp = None
    prev_arg = None
    
    # Store per-scenario results
    scen_counts = {}

    agent.model.eval()
    for i in range(total_steps):
        state = X[i]
        s_val = scen[i] if (scen is not None and i < len(scen)) else 'UNKNOWN'
        if s_val not in scen_counts:
            scen_counts[s_val] = {'count': 0, 'raw': np.zeros(NUM_ACTIONS), 'samp': np.zeros(NUM_ACTIONS), 'arg': np.zeros(NUM_ACTIONS)}
            
        scen_counts[s_val]['count'] += 1
        
        dist = agent.get_policy_distribution(state)
        raw_probs_sum += dist
        scen_counts[s_val]['raw'] += dist
        
        a_samp = agent.select_action(state, deterministic=False)
        action_counts_samp[a_samp] += 1
        scen_counts[s_val]['samp'][a_samp] += 1
        
        a_arg = agent.select_action(state, deterministic=True)
        action_counts_arg[a_arg] += 1
        scen_counts[s_val]['arg'][a_arg] += 1

        # Overload check
        if X.shape[2] >= 37:
            state_last = state[-1] if state.ndim == 2 else state
            for s_idx in range(NUM_ACTIONS):
                idx = 7 + s_idx * 3
                if idx < len(state_last):
                    load_norm = state_last[idx]
                    util = (load_norm * SCALING_LOAD) / (CAPACITIES[s_idx] * 1e6) if CAPACITIES[s_idx] > 0 else 0
                    if util > 0.95:
                        if s_idx == a_samp: overloads_samp += 1
                        if s_idx == a_arg: overloads_arg += 1

        if prev_samp is not None and a_samp != prev_samp: churn_samp += 1
        if prev_arg is not None and a_arg != prev_arg: churn_arg += 1
        prev_samp = a_samp
        prev_arg = a_arg

    agent.model.train()

    # RAW metrics
    raw_mean = raw_probs_sum / max(1, total_steps)
    ideal = np.array(CAPACITY_RATIOS[:NUM_ACTIONS])
    raw_entropy = -np.sum(raw_mean * np.log(raw_mean + 1e-8))

    # SAMPLED metrics
    samp_freq = action_counts_samp / max(1, action_counts_samp.sum())
    samp_entropy = -np.sum(samp_freq * np.log(samp_freq + 1e-8))
    samp_fairness = np.abs(samp_freq - ideal).mean()
    samp_overload_rate = overloads_samp / max(1, total_steps)
    samp_churn_rate = churn_samp / max(1, total_steps)

    # ARGMAX metrics
    arg_freq = action_counts_arg / max(1, action_counts_arg.sum())
    arg_entropy = -np.sum(arg_freq * np.log(arg_freq + 1e-8))

    max_entropy = np.log(NUM_ACTIONS)

    # Hard-action eval score (using Sampling mapping, because LB uses sampling)
    # Thêm throughput component để khuyến khích AI chọn server có capacity cao hơn
    # Capacity-weighted throughput: AI nên chọn h8 (capacity=10) nhiều hơn h5 (capacity=1)
    wrr_distribution = np.array([0.0625, 0.3125, 0.625])  # h5:h7:h8
    ai_distribution = samp_freq
    wrr_capacity_weighted = np.dot(wrr_distribution, CAPACITIES)
    ai_capacity_weighted = np.dot(ai_distribution, CAPACITIES)
    throughput_improvement = (ai_capacity_weighted - wrr_capacity_weighted) / np.max(CAPACITIES)
    
    score = (
        0.3 * (raw_entropy / max_entropy)
        + 0.3 * (samp_entropy / max_entropy)
        + 0.2 * max(0, throughput_improvement)  # Throughput improvement component
        - 2.0 * samp_overload_rate
        - 1.0 * samp_fairness
        - 0.5 * samp_churn_rate
    )
    
    # Store scenario breakdowns
    scen_breakdown = {}
    for k, v in scen_counts.items():
        c = max(1, v['count'])
        scen_breakdown[k] = {
            'raw': v['raw']/c,
            'samp': v['samp']/c,
            'arg': v['arg']/c
        }

    return {
        'score': score,
        'raw_entropy': raw_entropy,
        'samp_entropy': samp_entropy,
        'arg_entropy': arg_entropy,
        'samp_freq': samp_freq,
        'raw_mean': raw_mean,
        'arg_freq': arg_freq,
        'scen_breakdown': scen_breakdown
    }


def plot_training_curves(metrics, save_path):
    """Plot training metrics dashboard."""
    if not metrics:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('TFT-CQL Training Dashboard', fontsize=14)

    keys = ['critic_loss', 'cql_penalty', 'actor_loss', 'entropy', 'forecast_loss', 'eval_composite']
    titles = ['Critic Loss', 'CQL Penalty', 'Actor Loss', 'Policy Entropy', 'Forecast Loss', 'Eval Composite']

    for ax, key, title in zip(axes.flat, keys, titles):
        values = [m.get(key, 0) for m in metrics]
        ax.plot(values, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training curves saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='TFT-CQL Training Pipeline')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['pretrain', 'train', 'tune', 'all', 'phase0'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--encoder_ckpt', type=str, default=None,
                        help='Pretrained encoder checkpoint for phase 2')
    parser.add_argument('--augment', action='store_true',
                        help='Use augmented training data')
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load data
    if args.augment:
        print("[*] Using AUGMENTED training data")
        X_path = os.path.join(DATA_DIR, 'X_v3_aug.npy')
        y_path = os.path.join(DATA_DIR, 'y_v3_aug.npy')
        s_path = os.path.join(DATA_DIR, 'scenarios_v3_aug.npy')
        X = np.load(X_path)
        y = np.load(y_path)
        scen = np.load(s_path) if os.path.exists(s_path) else None
        metadata = {'version': 'v3_aug', 'num_features': X.shape[2]}
    else:
        X, y, scen, metadata = load_v3_data()
    num_features = X.shape[2] if X.ndim == 3 else X.shape[1]

    # ═════════════════════════════════════════════════════════════════════
    # V9: STATE NORMALIZATION - "MỞ MẮT" CHO NEURAL NETWORK
    # ═════════════════════════════════════════════════════════════════════
    # Vấn đề: 44 features có scale khác nhau cực đại (VD: bytes=10M vs latency=0.05s)
    #          -> Neural network bị "mù", chỉ nhìn features lớn, bỏ qua features nhỏ
    # Giải pháp: StandardScaler ép TẤT CẢ features về mean=0, std=1
    #             -> AI thực sự phân biệt được normal vs dos scenarios
    # ═════════════════════════════════════════════════════════════════════
    # V9: STATE NORMALIZATION - "MỞ MẮT" CHO NEURAL NETWORK
    # ═════════════════════════════════════════════════════════════════════
    # Vấn đề: 44 features có scale khác nhau cực đại (VD: bytes=10M vs latency=0.05s)
    #          -> Neural network bị "mù", chỉ nhìn features lớn, bỏ qua features nhỏ
    # Giải pháp: StandardScaler ép TẤT CẢ features về mean=0, std=1
    # Quan trọng: PHẢI save normalized X vào file để env đọc đúng dữ liệu!
    print("\n[*] V9: Applying StandardScaler normalization to features...")
    num_samples, seq_len, num_feat = X.shape
    X_flat = X.reshape(-1, num_feat)  # (samples*seq_len, features)
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)  # NORMALIZE!
    X_scaled = X_flat_scaled.reshape(num_samples, seq_len, num_feat)  # Reshape back
    
    # Save normalized data to file for env to read (FIX MISMATCH!)
    if args.augment:
        norm_x_path = os.path.join(DATA_DIR, 'X_v3_aug_normalized.npy')
        np.save(norm_x_path, X_scaled)
        norm_y_path = os.path.join(DATA_DIR, 'y_v3_aug_normalized.npy')
        np.save(norm_y_path, y)
        print(f"    Saved normalized data: {norm_x_path}")
    else:
        norm_x_path = os.path.join(DATA_DIR, 'X_v3_normalized.npy')
        np.save(norm_x_path, X_scaled)
        norm_y_path = os.path.join(DATA_DIR, 'y_v3_normalized.npy')
        np.save(norm_y_path, y)
        print(f"    Saved normalized data: {norm_x_path}")
    
    # Use normalized X for training
    X = X_scaled
    print(f"    Features scaled: mean≈0, std≈1 for all {num_feat} features")
    print(f"    Original scale: min={X_flat[:, 0].min():.2f}, max={X_flat[:, 0].max():.2f}")
    print(f"    Scaled scale: min={X_flat_scaled[:, 0].min():.2f}, max={X_flat_scaled[:, 0].max():.2f}")

    # Init agent with capacity prior for KL regularization
    agent = CQLAgent(
        input_size=num_features,
        seq_len=SEQUENCE_LENGTH,
        hidden_size=args.hidden_size,
        num_actions=NUM_ACTIONS,
        actor_lr=5e-5,       # Much slower than critic (prevents encoder corruption)
        critic_lr=CRITIC_LR,
        cql_alpha=CQL_ALPHA,
        entropy_coeff=ENTROPY_COEFF,
        kl_coeff=0.05,  # V9 FIX: 0.5 -> 0.05 (khớp với config.py)
        target_entropy_ratio=0.4,
        forecast_loss_weight=FORECAST_LOSS_WEIGHT,
        constraint_weights=CONSTRAINT_WEIGHTS,
        capacity_prior=CAPACITY_PRIOR,
    )

    # Init env with NORMALIZED data (FIX MISMATCH!)
    if args.augment:
        env = SDN_Offline_Env_V2(
            os.path.join(DATA_DIR, 'X_v3_aug_normalized.npy'),
            os.path.join(DATA_DIR, 'y_v3_aug_normalized.npy'),
            mode='train', metadata=metadata)
    else:
        env = SDN_Offline_Env_V2(
            os.path.join(DATA_DIR, 'X_v3_normalized.npy'),
            os.path.join(DATA_DIR, 'y_v3_normalized.npy'),
            mode='train', metadata=metadata)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_metrics = []

    # Phase 0: Capacity ratio supervised pretraining (MỚI)
    if args.phase in ('all', 'phase0', 'pretrain', 'train'):
        print("\n" + "="*60)
        print("  EXECUTING PHASE 0: CAPACITY RATIO SUPERVISED PRETRAINING")
        print("="*60)
        cap_losses = phase0_capacity_ratio_pretraining(
            agent, X, y, scen, env, epochs=20, batch_size=args.batch_size)
        all_metrics.extend([{'epoch': -i-1, 'capacity_kl_loss': l} for i, l in enumerate(cap_losses[::-1])])

    if args.phase in ('pretrain', 'all'):
        pretrain_losses = phase1_pretrain_encoder(
            agent, X, y, epochs=min(30, args.epochs), batch_size=args.batch_size)
        all_metrics.extend([{'epoch': i, 'forecast_loss': l} for i, l in enumerate(pretrain_losses)])

    if args.encoder_ckpt:
        agent.load_checkpoint(args.encoder_ckpt)
        print(f"  Loaded encoder checkpoint: {args.encoder_ckpt}")

    if args.phase in ('train', 'all'):
        train_metrics = phase2_offline_training(
            agent, X, y, scen, env, epochs=args.epochs, batch_size=args.batch_size)
        all_metrics.extend(train_metrics)

    if args.phase in ('tune', 'all'):
        tune_metrics = phase3_constraint_tuning(
            agent, X, y, scen, env, epochs=min(20, args.epochs // 5), batch_size=args.batch_size)
        all_metrics.extend(tune_metrics)

    # Plot and save
    if all_metrics:
        plot_training_curves(all_metrics,
                             os.path.join(LOG_DIR, f'training_{timestamp}.png'))

        # Save metrics JSON
        metrics_path = os.path.join(LOG_DIR, f'metrics_{timestamp}.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"\n  Metrics saved: {metrics_path}")

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"  Checkpoints in: {CKPT_DIR}/")
    print(f"  Logs in: {LOG_DIR}/")


if __name__ == '__main__':
    main()
