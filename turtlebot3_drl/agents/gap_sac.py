from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from turtlebot3_drl.agents.prioritized_replay import PrioritizedReplayBuffer
from turtlebot3_drl.agents.networks import GaussianActor, TwinQCritic, torch, F


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class GAPSACAgent:
    """Soft Actor-Critic with gated attention and Prioritized Experience Replay."""

    def __init__(self, state_dim: int, config: Dict[str, Any]):
        algorithm = config.get("algorithm", {})
        action_cfg = config.get("action", {})
        per_cfg = config.get("per", {})

        requested_device = str(algorithm.get("device", "cpu")).lower()
        if requested_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.gamma = float(algorithm.get("gamma", 0.98))
        self.tau = float(algorithm.get("tau", 0.2))
        self.batch_size = int(algorithm.get("batch_size", 100))
        self.policy_update_delay = int(algorithm.get("policy_update_delay", 2))
        self.automatic_entropy_tuning = bool(algorithm.get("automatic_entropy_tuning", True))
        self.update_step = 0

        hidden_units = int(algorithm.get("hidden_units", 256))
        learning_rate = float(algorithm.get("learning_rate", 0.0008))
        self.actor = GaussianActor(
            state_dim,
            hidden_units=hidden_units,
            max_linear_velocity=float(action_cfg.get("max_linear_velocity", 0.22)),
            max_angular_velocity=float(action_cfg.get("max_angular_velocity", 2.84)),
            allow_reverse=bool(action_cfg.get("allow_reverse", False)),
        ).to(self.device)
        self.critic = TwinQCritic(state_dim, hidden_units=hidden_units).to(self.device)
        self.target_critic = TwinQCritic(state_dim, hidden_units=hidden_units).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        if self.automatic_entropy_tuning:
            self.target_entropy = -2.0
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.log_alpha = None
            self.alpha = torch.tensor(0.2, device=self.device)

        self.replay_buffer = PrioritizedReplayBuffer(
            int(algorithm.get("replay_buffer_size", 100000)),
            alpha=float(per_cfg.get("per_alpha", 0.6)),
            beta_start=float(per_cfg.get("per_beta_start", 0.4)),
            beta_frames=int(per_cfg.get("per_beta_frames", 100000)),
            epsilon=float(per_cfg.get("per_epsilon", 1e-6)),
        )
        self.state_dim = int(state_dim)
        self.action_metadata = {
            "max_linear_velocity": float(action_cfg.get("max_linear_velocity", 0.22)),
            "max_angular_velocity": float(action_cfg.get("max_angular_velocity", 2.84)),
            "allow_reverse": bool(action_cfg.get("allow_reverse", False)),
        }

    @property
    def entropy_alpha(self):
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()
        return self.alpha

    def select_action(self, state: Sequence[float], deterministic: bool = False) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, deterministic_action = self.actor.sample(state_tensor)
        selected = deterministic_action if deterministic else action
        return selected.squeeze(0).cpu().numpy()

    def add_transition(self, transition: Transition, td_error: float | None = None) -> None:
        self.replay_buffer.add(transition, td_error)

    def update(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(np.stack([t.state for t in batch.transitions]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([t.action for t in batch.transitions]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([[t.reward] for t in batch.transitions], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([t.next_state for t in batch.transitions]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([[float(t.done)] for t in batch.transitions], dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(batch.weights, dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.entropy_alpha.detach() * next_log_probs
            target_q = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        td_errors = (target_q - torch.min(current_q1, current_q2)).detach().abs().cpu().numpy().flatten()
        critic_loss = (weights * F.mse_loss(current_q1, target_q, reduction="none")).mean()
        critic_loss += (weights * F.mse_loss(current_q2, target_q, reduction="none")).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.replay_buffer.update_priorities(batch.indices, td_errors)

        actor_loss_value = 0.0
        alpha_loss_value = 0.0
        if self.update_step % self.policy_update_delay == 0:
            new_actions, log_probs, _ = self.actor.sample(states)
            q1_pi, q2_pi = self.critic(states, new_actions)
            actor_loss = (self.entropy_alpha.detach() * log_probs - torch.min(q1_pi, q2_pi)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu())

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha_loss_value = float(alpha_loss.detach().cpu())

            self.soft_update(self.critic, self.target_critic, self.tau)

        self.update_step += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.entropy_alpha.detach().cpu()),
        }

    @staticmethod
    def soft_update(source, target, tau: float) -> None:
        # Paper soft target update: theta_target <- tau theta_source + (1 - tau) theta_target.
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save_checkpoint(self, path: str, metadata: Dict[str, Any] | None = None) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "log_alpha": None if self.log_alpha is None else self.log_alpha.detach().cpu(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": None
                if not self.automatic_entropy_tuning
                else self.alpha_optimizer.state_dict(),
                "update_step": self.update_step,
                "state_dim": self.state_dim,
                "action": self.action_metadata,
                "metadata": metadata or {},
            },
            path,
        )

    def load_checkpoint(self, path: str, map_location: str | None = None) -> None:
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
        if "target_critic" in checkpoint:
            self.target_critic.load_state_dict(checkpoint["target_critic"])
        if self.automatic_entropy_tuning and checkpoint.get("log_alpha") is not None:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if self.automatic_entropy_tuning and checkpoint.get("alpha_optimizer") is not None:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        if "update_step" in checkpoint:
            self.update_step = int(checkpoint["update_step"])
