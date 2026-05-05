import math
from typing import Tuple


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for GAP_SAC networks. Install torch before training or inference."
        ) from exc
    return torch, nn, F


torch, nn, F = _require_torch()


class GatedAttentionFeatureExtractor(nn.Module):
    """Paper-derived gated attention block used before actor and critic heads."""

    def __init__(self, state_dim: int, hidden_units: int = 256, num_tokens: int = 8):
        super().__init__()
        if hidden_units % num_tokens != 0:
            raise ValueError("hidden_units must be divisible by num_tokens")
        self.num_tokens = num_tokens
        self.token_dim = hidden_units // num_tokens
        self.input_layer = nn.Linear(state_dim, hidden_units)
        self.q_layer = nn.Linear(self.token_dim, self.token_dim)
        self.k_layer = nn.Linear(self.token_dim, self.token_dim)
        self.v_layer = nn.Linear(self.token_dim, self.token_dim)
        self.output_layer = nn.Linear(self.token_dim, self.token_dim)
        self.layer_norm = nn.LayerNorm(self.token_dim)
        self.gate_layer = nn.Linear(self.token_dim, self.token_dim)
        self.flatten_layer = nn.Linear(hidden_units, hidden_units)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        tokens = x.view(x.shape[0], self.num_tokens, self.token_dim)
        q = self.q_layer(tokens)
        k = self.k_layer(tokens)
        v = self.v_layer(tokens)

        # Paper attention equation: softmax(QK^T / sqrt(d_k)) V.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.token_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = attention_weights @ v

        transformed = self.layer_norm(self.output_layer(attention_output) + attention_output)
        gate = torch.sigmoid(self.gate_layer(transformed))
        gated_output = transformed * gate
        return F.relu(self.flatten_layer(gated_output.reshape(state.shape[0], -1)))


class GaussianActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_units: int = 256,
        max_linear_velocity: float = 0.22,
        max_angular_velocity: float = 2.84,
        allow_reverse: bool = False,
    ):
        super().__init__()
        self.feature_extractor = GatedAttentionFeatureExtractor(state_dim, hidden_units)
        self.hidden = nn.Linear(hidden_units, hidden_units)
        self.mean_layer = nn.Linear(hidden_units, action_dim)
        self.log_std_layer = nn.Linear(hidden_units, action_dim)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.allow_reverse = bool(allow_reverse)

    def forward(self, state):
        features = self.feature_extractor(state)
        hidden = F.relu(self.hidden(features))
        mean = self.mean_layer(hidden)
        log_std = torch.clamp(self.log_std_layer(hidden), -20.0, 2.0)
        return mean, log_std

    def _scale_action(self, tanh_action):
        angular = tanh_action[:, 1:2] * self.max_angular_velocity
        if self.allow_reverse:
            linear = tanh_action[:, 0:1] * self.max_linear_velocity
        else:
            linear = (tanh_action[:, 0:1] + 1.0) * 0.5 * self.max_linear_velocity
        return torch.cat([linear, angular], dim=-1)

    def sample(self, state) -> Tuple[object, object, object]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        tanh_action = torch.tanh(raw_action)
        action = self._scale_action(tanh_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1.0 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic_action = self._scale_action(torch.tanh(mean))
        return action, log_prob, deterministic_action


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 2, hidden_units: int = 256):
        super().__init__()
        self.feature_extractor = GatedAttentionFeatureExtractor(state_dim, hidden_units)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_units + action_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, state, action):
        features = self.feature_extractor(state)
        return self.q_head(torch.cat([features, action], dim=-1))


class TwinQCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 2, hidden_units: int = 256):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_units)
        self.q2 = QNetwork(state_dim, action_dim, hidden_units)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)
