import sys
import re

with open('ai_model/train_actor_critic.py', 'r') as f:
    text = f.read()

# Replace _get_action_distribution
def_dist = """def _get_action_distribution(agent, X_subset):
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
"""
text = re.sub(r'def _get_action_distribution.*?return raw_mean, served_freq\n', def_dist, text, flags=re.DOTALL)

# Replace _evaluate_checkpoint
def_eval = """def _evaluate_checkpoint(agent, X, y, scen, env):
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
        s_val = scen[i] if i < len(scen) else 'UNKNOWN'
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
    score = (
        0.5 * (raw_entropy / max_entropy)
        + 0.5 * (samp_entropy / max_entropy)
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
"""
text = re.sub(r'def _evaluate_checkpoint.*?return score\n', def_eval, text, flags=re.DOTALL)

with open('ai_model/train_actor_critic.py', 'w') as f:
    f.write(text)
print("Applied phase 2 replacements")
