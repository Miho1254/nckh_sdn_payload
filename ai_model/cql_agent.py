"""
CQL (Conservative Q-Learning) Agent cho SDN Load Balancing.

Thay thế hoàn toàn DQNAgent:
  - Offline dataset sampling (không dùng epsilon-greedy exploration)
  - Twin critic update với CQL conservative penalty
  - Actor update với advantage-weighted policy gradient
  - Constraint-aware optimization
  - Auxiliary forecast loss

FIX ROUND 2:
  - Encoder tách khỏi actor optimizer (chỉ update qua critic + pretrain)
  - KL-to-capacity-prior regularization
  - Min entropy target constraint
  - Actor LR giảm xuống 5e-5
  - Logits clamp + temperature scaling
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ai_model.tft_ac_net import TFT_ActorCritic_Model


class OfflineDataset:
    """Offline transition dataset cho CQL training."""
    def __init__(self, states, labels, env_class=None):
        self.states = states                   # [N, seq_len, num_features]
        self.labels = labels                   # [N]
        self.num_samples = len(states)

    def sample(self, batch_size):
        indices = np.random.choice(self.num_samples - 1, size=batch_size, replace=False)
        states = self.states[indices]
        next_states = self.states[indices + 1]
        labels = self.labels[indices]
        return states, next_states, labels, indices

    def __len__(self):
        return self.num_samples


class CQLAgent:
    """
    Conservative Q-Learning Agent cho offline RL.

    KEY DESIGN (Round 2):
      - Encoder params chỉ trong critic_optimizer (actor không phá encoder)
      - KL regularization: kéo policy về capacity-ratio prior
      - Min entropy target: ngăn policy collapse sớm
      - Actor LR << Critic LR
    """
    def __init__(self, input_size, seq_len=5, hidden_size=64, num_actions=3,
                 actor_lr=5e-5, critic_lr=3e-4, gamma=0.99,
                 cql_alpha=1.0, entropy_coeff=0.1,
                 kl_coeff=1.0, target_entropy_ratio=0.4,
                 forecast_loss_weight=0.1,
                 constraint_weights=None,
                 capacity_prior=None,
                 tau=0.005):
        self.num_actions = num_actions
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.entropy_coeff = entropy_coeff
        self.kl_coeff = kl_coeff
        self.forecast_loss_weight = forecast_loss_weight
        self.tau = tau
        self.constraint_weights = constraint_weights or {
            "util_breach": 2.0,
            "fairness_dev": 1.0,
            "action_churn": 0.5,
        }

        # Target entropy = ratio * max_entropy (0.4 * ln(3) ≈ 0.44)
        self.target_entropy = target_entropy_ratio * np.log(num_actions)

        # Capacity prior for KL regularization
        if capacity_prior is not None:
            self.capacity_prior = torch.FloatTensor(capacity_prior)
        else:
            self.capacity_prior = torch.ones(num_actions) / num_actions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.capacity_prior = self.capacity_prior.to(self.device)
        print(f"[*] CQL Agent initialized on {self.device}")
        print(f"    Actor LR: {actor_lr}, Critic LR: {critic_lr}")
        print(f"    Entropy coeff: {entropy_coeff}, KL coeff: {kl_coeff}")
        print(f"    Target entropy: {self.target_entropy:.4f}")
        print(f"    Capacity prior: {self.capacity_prior.cpu().numpy()}")

        # Policy network
        self.model = TFT_ActorCritic_Model(
            input_size=input_size, seq_len=seq_len,
            hidden_size=hidden_size, num_actions=num_actions
        ).to(self.device)

        # Target network (for stable critic targets)
        self.target_model = TFT_ActorCritic_Model(
            input_size=input_size, seq_len=seq_len,
            hidden_size=hidden_size, num_actions=num_actions
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # ═══════════════════════════════════════════
        # CRITICAL FIX: Separate optimizers
        # Encoder updated by CRITIC only (stable representations)
        # Actor only updates its own head (prevents encoder corruption)
        # ═══════════════════════════════════════════
        encoder_params = list(self.model.vsn.parameters()) + \
                         list(self.model.lstm.parameters()) + \
                         list(self.model.temporal_attn.parameters())
        actor_head_params = list(self.model.actor_head.parameters())
        critic_params = list(self.model.critic1.parameters()) + \
                        list(self.model.critic2.parameters())
        forecast_params = list(self.model.forecast_head.parameters())
        safety_params = list(self.model.safety_head.parameters())

        # Critic optimizer includes encoder (encoder learns from TD signal)
        self.critic_optimizer = optim.Adam(
            encoder_params + critic_params + forecast_params,
            lr=critic_lr)

        # Actor optimizer: ONLY actor head + safety head (NO encoder!)
        self.actor_optimizer = optim.Adam(
            actor_head_params + safety_params,
            lr=actor_lr)

        # Separate encoder optimizer for pretrain phase
        self.pretrain_optimizer = optim.Adam(
            encoder_params + forecast_params,
            lr=critic_lr)

        self.forecast_loss_fn = nn.MSELoss()

    def select_action(self, state, deterministic=True):
        """Inference: chọn action từ policy distribution (không epsilon)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            probs = self.model.get_policy(state_t)
        if deterministic:
            return probs.argmax(dim=-1).item()
        return torch.multinomial(probs, 1).item()

    def get_policy_distribution(self, state):
        """Get full probability distribution for logging."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            probs = self.model.get_policy(state_t)
        return probs.cpu().numpy()[0]

    def train_step(self, states, next_states, rewards, actions, dones, info_batch=None):
        """
        One training step: critic update + actor update.

        Round 2 fixes:
          - Encoder ONLY updated via critic_optimizer
          - KL(policy || capacity_prior) regularization
          - Min-entropy constraint
          - Advantages normalized + clamped
          - Logits clamped to prevent softmax saturation
        """
        self.model.train()

        states_t = torch.FloatTensor(states).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        action_onehot = F.one_hot(actions_t, self.num_actions).float()

        # ═══════════════════════════════════════════
        #  CRITIC UPDATE (includes encoder)
        # ═══════════════════════════════════════════
        with torch.no_grad():
            next_probs = self.target_model.get_policy(next_states_t)
            next_q_all = self.target_model.get_q_values(next_states_t)
            next_v = (next_probs * next_q_all).sum(dim=-1)
            target_q = rewards_t + self.gamma * next_v * (1.0 - dones_t)

        _, q1, q2, forecast, _ = self.model(states_t, action_onehot)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # CQL Conservative Penalty
        all_q1_values, all_q2_values = [], []
        for a in range(self.num_actions):
            a_oh = torch.zeros_like(action_onehot)
            a_oh[:, a] = 1.0
            _, q1_a, q2_a, _, _ = self.model(states_t, a_oh)
            all_q1_values.append(q1_a)
            all_q2_values.append(q2_a)

        all_q1 = torch.stack(all_q1_values, dim=-1)
        all_q2 = torch.stack(all_q2_values, dim=-1)

        cql_penalty = (torch.logsumexp(all_q1, dim=-1).mean() - q1.mean() +
                        torch.logsumexp(all_q2, dim=-1).mean() - q2.mean())

        # Forecast auxiliary loss (encoder learns temporal patterns)
        actual_future = next_states_t[:, -1, :]
        forecast_loss = self.forecast_loss_fn(forecast, actual_future)

        total_critic_loss = (critic_loss +
                             self.cql_alpha * cql_penalty +
                             self.forecast_loss_weight * forecast_loss)

        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ═══════════════════════════════════════════
        #  ACTOR UPDATE (head only, no encoder)
        # ═══════════════════════════════════════════
        # Detach encoder output so actor gradient doesn't flow to encoder
        with torch.no_grad():
            context = self.model.encode(states_t)
        context_detached = context.detach()

        # Get actor logits from detached context
        policy_logits = self.model.actor_head(context_detached)
        policy_logits = policy_logits.clamp(-10.0, 10.0)
        policy_probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)

        # Advantage-weighted policy gradient
        with torch.no_grad():
            q_values = self.model.get_q_values(states_t)
            q_taken = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            baseline = (policy_probs.detach() * q_values).sum(dim=-1)
            advantages = q_taken - baseline
            # Normalize advantages
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - advantages.mean()) / adv_std
            advantages = advantages.clamp(-3.0, 3.0)

        log_prob_taken = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        actor_loss = -(advantages * log_prob_taken).mean()

        # ── Entropy regularization ──
        entropy = -(policy_probs * log_probs).sum(dim=-1).mean()

        # Min-entropy constraint: penalize if entropy drops below target
        # This prevents premature collapse
        entropy_deficit = torch.relu(self.target_entropy - entropy)
        entropy_penalty = self.entropy_coeff * entropy_deficit

        # Also add standard entropy bonus
        entropy_bonus = -0.01 * entropy  # small standard bonus

        # ── KL-to-capacity-prior ──
        # Prevent policy from drifting too far from hardware capacity distribution
        prior = self.capacity_prior.unsqueeze(0).expand_as(policy_probs)
        kl_to_prior = (policy_probs * (log_probs - torch.log(prior + 1e-8))).sum(dim=-1).mean()
        kl_loss = self.kl_coeff * kl_to_prior

        # Constraint penalties
        constraint_loss = torch.tensor(0.0, device=self.device)
        if info_batch is not None:
            util_breaches = torch.FloatTensor(
                [info.get('util_breach', 0.0) for info in info_batch]).to(self.device)
            fairness_devs = torch.FloatTensor(
                [info.get('fairness_dev', 0.0) for info in info_batch]).to(self.device)
            churn_flags = torch.FloatTensor(
                [info.get('action_switched', 0.0) for info in info_batch]).to(self.device)

            constraint_loss = (
                self.constraint_weights["util_breach"] * util_breaches.mean() +
                self.constraint_weights["fairness_dev"] * fairness_devs.mean() +
                self.constraint_weights["action_churn"] * churn_flags.mean()
            )

        # Total actor loss
        total_actor_loss = (actor_loss + entropy_penalty + entropy_bonus +
                            kl_loss + constraint_loss)

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.actor_head.parameters()) +
            list(self.model.safety_head.parameters()),
            max_norm=0.5)
        self.actor_optimizer.step()

        # ═══════════════════════════════════════════
        #  SOFT UPDATE TARGET
        # ═══════════════════════════════════════════
        self._soft_update_target()

        return {
            'critic_loss': critic_loss.item(),
            'cql_penalty': cql_penalty.item(),
            'actor_loss': actor_loss.item(),
            'entropy': entropy.item(),
            'entropy_deficit': entropy_deficit.item(),
            'kl_to_prior': kl_to_prior.item(),
            'constraint_loss': constraint_loss.item() if isinstance(constraint_loss, torch.Tensor) else constraint_loss,
            'forecast_loss': forecast_loss.item(),
            'total_critic_loss': total_critic_loss.item(),
            'total_actor_loss': total_actor_loss.item(),
        }

    def pretrain_encoder(self, states, next_states):
        """Phase 1: pretrain temporal encoder via forecast task."""
        self.model.train()
        states_t = torch.FloatTensor(states).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)

        context = self.model.encode(states_t)
        forecast = self.model.forecast_head(context)
        actual = next_states_t[:, -1, :]

        loss = self.forecast_loss_fn(forecast, actual)

        self.pretrain_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.pretrain_optimizer.step()

        return loss.item()

    def save_checkpoint(self, path, epoch=0, metrics=None):
        """Save model + optimizer state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'metrics': metrics or {},
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load model + optimizer state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_state_dict'])
        if 'actor_optimizer' in checkpoint:
            try:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            except Exception:
                pass
        if 'critic_optimizer' in checkpoint:
            try:
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            except Exception:
                pass
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})

    def _soft_update_target(self):
        for tp, pp in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)
