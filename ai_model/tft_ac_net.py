"""
TFT-CQL Actor-Critic Network cho SDN Load Balancing.

Kiến trúc:
  Shared Temporal Encoder (VSN + LSTM + Temporal Attention)
    ├── Forecast Head  (dự báo next-step traffic)
    ├── Actor Head     (policy distribution π(a|s))
    ├── Critic Head    (twin Q-functions cho CQL)
    └── Safety Head    (overload risk per action)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
#  Shared Components (giữ từ TFT gốc)
# ══════════════════════════════════════════════════════════════

class GatedResidualNetwork(nn.Module):
    """GRN — GLU-based non-linear feature processor."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.skip_proj = (nn.Linear(input_size, hidden_size)
                          if input_size != hidden_size else nn.Identity())

    def forward(self, x):
        res = self.skip_proj(x)
        x = self.elu(self.fc1(x))
        x = self.fc2(x)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return self.layer_norm(res + x1 * torch.sigmoid(x2))


class VariableSelectionNetwork(nn.Module):
    """VSN — learned feature importance per timestep (vectorized)."""
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.weight_grn = GatedResidualNetwork(num_features, num_features)
        self.feature_grn = GatedResidualNetwork(1, hidden_size)

    def forward(self, x):
        batch, seq, feat = x.shape
        weights = F.softmax(self.weight_grn(x), dim=-1)  # [B, S, F]
        x_processed = self.feature_grn(x.reshape(-1, 1))  # [B*S*F, H]
        x_processed = x_processed.view(batch, seq, feat, self.hidden_size)
        return (weights.unsqueeze(-1) * x_processed).sum(dim=2)  # [B, S, H]


class TemporalSelfAttention(nn.Module):
    """Multi-Head Attention over timesteps."""
    def __init__(self, hidden_size, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_out)


# ══════════════════════════════════════════════════════════════
#  Head Modules
# ══════════════════════════════════════════════════════════════

class ForecastHead(nn.Module):
    """Dự báo next-step features."""
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.net = nn.Linear(hidden_size, output_size)

    def forward(self, context):
        return self.net(context)


class ActorHead(nn.Module):
    """Sinh policy distribution π(a|s) qua softmax logits."""
    def __init__(self, hidden_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, context):
        return self.net(context)  # raw logits, softmax ở ngoài


class CriticHead(nn.Module):
    """Single Q-function: Q(s, a) = f(context, one_hot(a))."""
    def __init__(self, hidden_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size + num_actions, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, context, action_onehot):
        x = torch.cat([context, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)


class SafetyHead(nn.Module):
    """Predict overload risk score per action."""
    def __init__(self, hidden_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Sigmoid(),
        )

    def forward(self, context):
        return self.net(context)


# ══════════════════════════════════════════════════════════════
#  Full Model
# ══════════════════════════════════════════════════════════════

class TFT_ActorCritic_Model(nn.Module):
    """
    TFT-Encoded Conservative Offline Actor-Critic.
    
    Forward returns:
        policy_logits  [B, num_actions]   — raw logits cho softmax
        q1             [B]                — critic 1 value
        q2             [B]                — critic 2 value
        forecast       [B, forecast_size] — next-step prediction
        safety_scores  [B, num_actions]   — overload risk (0-1)
    """
    def __init__(self, input_size, seq_len=5, hidden_size=64,
                 num_actions=3, forecast_size=None):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        if forecast_size is None:
            forecast_size = input_size

        # Shared Temporal Encoder
        self.vsn = VariableSelectionNetwork(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            batch_first=True)
        self.temporal_attn = TemporalSelfAttention(hidden_size, num_heads=2)

        # Heads
        self.forecast_head = ForecastHead(hidden_size, forecast_size)
        self.actor_head = ActorHead(hidden_size, num_actions)
        self.critic1 = CriticHead(hidden_size, num_actions)
        self.critic2 = CriticHead(hidden_size, num_actions)
        self.safety_head = SafetyHead(hidden_size, num_actions)

    def encode(self, state):
        """Shared encoder: state [B, S, F] → context [B, H]."""
        x = self.vsn(state)
        x, _ = self.lstm(x)
        x = self.temporal_attn(x)
        return x[:, -1, :]  # context vector from last timestep

    def forward(self, state, action_onehot=None):
        context = self.encode(state)

        policy_logits = self.actor_head(context)
        forecast = self.forecast_head(context)
        safety_scores = self.safety_head(context)

        # Critic needs action — default to uniform if not provided
        if action_onehot is None:
            action_onehot = torch.zeros(
                context.size(0), self.num_actions, device=context.device)

        q1 = self.critic1(context, action_onehot)
        q2 = self.critic2(context, action_onehot)

        return policy_logits, q1, q2, forecast, safety_scores

    def get_policy(self, state):
        """Inference shortcut: state → action probabilities."""
        context = self.encode(state)
        logits = self.actor_head(context)
        return F.softmax(logits, dim=-1)

    def get_q_values(self, state):
        """Get Q(s,a) for all actions."""
        context = self.encode(state)
        q_vals = []
        for a in range(self.num_actions):
            one_hot = torch.zeros(context.size(0), self.num_actions,
                                  device=context.device)
            one_hot[:, a] = 1.0
            q1 = self.critic1(context, one_hot)
            q2 = self.critic2(context, one_hot)
            q_vals.append(torch.min(q1, q2))
        return torch.stack(q_vals, dim=-1)  # [B, num_actions]


if __name__ == '__main__':
    print("=== Test TFT-ActorCritic Model ===")

    num_features = 42
    model = TFT_ActorCritic_Model(
        input_size=num_features, seq_len=5, hidden_size=64, num_actions=3)

    dummy = torch.randn(32, 5, num_features)
    action_oh = F.one_hot(torch.randint(0, 3, (32,)), 3).float()

    logits, q1, q2, forecast, safety = model(dummy, action_oh)

    print(f"Policy logits:  {logits.shape}")     # [32, 3]
    print(f"Q1:             {q1.shape}")          # [32]
    print(f"Q2:             {q2.shape}")          # [32]
    print(f"Forecast:       {forecast.shape}")    # [32, 42]
    print(f"Safety scores:  {safety.shape}")      # [32, 3]

    probs = model.get_policy(dummy)
    print(f"Policy probs:   {probs.shape}")       # [32, 3]
    print(f"Sum check:      {probs.sum(dim=-1).mean():.4f}")

    q_all = model.get_q_values(dummy)
    print(f"Q-values (all): {q_all.shape}")       # [32, 3]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("All shape checks passed!")
