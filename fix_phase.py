import sys
import re

with open('ai_model/train_actor_critic.py', 'r') as f:
    text = f.read()

# Replace phase2
phase2_impl = """def phase2_offline_training(agent, X, y, scen, env, epochs=100, batch_size=64):
    print("\\n" + "="*60)
    print("  PHASE 2: OFFLINE CQL ACTOR-CRITIC TRAINING")
    print("="*60)

    num_samples = len(X) - 1
    all_metrics = []
    best_composite = -float('inf')
    
    # Early stopping trackers
    patience = 0
    patience_limit = 10

    for epoch in range(epochs):
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
        # Early stopping logic:
        if eval_res['samp_entropy'] < 0.3:
            patience += 1
            if patience >= patience_limit:
                print(f"  [!] Early stopping triggered at epoch {epoch+1} due to prolonged entropy collapse (H < 0.3).")
                break
        else:
            patience = max(0, patience - 1)
            
        if eval_score > best_composite and eval_res['samp_entropy'] > 0.5:
            best_composite = eval_score
            agent.save_checkpoint(
                os.path.join("ai_model/checkpoints", 'tft_ac_best.pth'),
                epoch=epoch, metrics=avg_losses)

        if (epoch + 1) % 20 == 0:
            agent.save_checkpoint(
                os.path.join("ai_model/checkpoints", f'tft_ac_epoch{epoch+1}.pth'),
                epoch=epoch, metrics=avg_losses)

    agent.save_checkpoint(
        os.path.join("ai_model/checkpoints", 'tft_ac_final.pth'),
        epoch=epochs-1, metrics=avg_losses)

    return all_metrics
"""
text = re.sub(r'def phase2_offline_training.*?return all_metrics\n', phase2_impl, text, flags=re.DOTALL)


# Replace phase3
phase3_impl = """def phase3_constraint_tuning(agent, X, y, scen, env, epochs=20, batch_size=64):
    print("\\n" + "="*60)
    print("  PHASE 3: CONSTRAINT TUNING (Fine-tuning)")
    print("="*60)

    num_samples = len(X) - 1
    all_metrics = []
    
    agent.critic_optimizer.param_groups[0]['lr'] = 1e-4
    agent.actor_optimizer.param_groups[0]['lr'] = 1e-4

    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        epoch_losses = {}
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            batch_idx = indices[start:start+batch_size]
            if len(batch_idx) < 2: continue
            states = X[batch_idx]
            next_states = X[np.minimum(batch_idx + 1, num_samples)]
            actions, rewards, dones, infos = [], [], [], []
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
        eval_res = _evaluate_checkpoint(agent, X, y, scen, env)
        eval_score = eval_res['score']
        
        avg_losses['eval_score'] = eval_score
        avg_losses.update(eval_res)
        all_metrics.append({'epoch': epoch, 'phase': 3, **avg_losses})

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [Tune] Epoch {epoch+1:2d}/{epochs} | Score: {eval_score:.4f} | Samp H: {eval_res['samp_entropy']:.3f} | KL: {avg_losses.get('kl_to_prior', 0):.4f}")

    agent.save_checkpoint(
        os.path.join("ai_model/checkpoints", 'tft_ac_tuned.pth'),
        epoch=epochs-1, metrics=avg_losses)

    return all_metrics
"""
text = re.sub(r'def phase3_constraint_tuning.*?return all_metrics\n', phase3_impl, text, flags=re.DOTALL)

with open('ai_model/train_actor_critic.py', 'w') as f:
    f.write(text)
print("Applied phase 3/4 replacements")
