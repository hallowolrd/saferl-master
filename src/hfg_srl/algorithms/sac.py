"""Vanilla SAC baseline (no safety modifications).

Standard Soft Actor-Critic with twin Q-networks and automatic entropy tuning.
Serves as the base algorithm for all SAC-family variants.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent, register_agent
from ..models.actor import SquashedGaussianActor
from ..models.critic import TwinQNetwork
from ..utils.config import HFGConfig

logger = logging.getLogger(__name__)


@register_agent("sac")
class SAC(BaseAgent):
    """Standard Soft Actor-Critic (SAC) baseline.

    No safety mechanisms — pure reward maximization.
    Used as a baseline to measure the cost of adding safety constraints.
    """

    def __init__(
        self,
        cfg: HFGConfig,
        obs_dim: int,
        act_dim: int,
        action_space_high: np.ndarray,
        device: str,
    ):
        super().__init__(cfg, obs_dim, act_dim, action_space_high, device)
        a = cfg.algorithm

        # Networks
        self.actor = SquashedGaussianActor(
            obs_dim, act_dim, a.hidden_dims, a.activation
        ).to(device)

        self.critic = TwinQNetwork(
            obs_dim, act_dim, a.hidden_dims, a.activation
        ).to(device)

        self.critic_target = TwinQNetwork(
            obs_dim, act_dim, a.hidden_dims, a.activation
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy temperature
        self.target_entropy = -act_dim
        self.log_alpha = torch.tensor(
            np.log(a.alpha), requires_grad=True, device=device
        )

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=a.lr_critic)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=a.lr_alpha)

        self.gamma = a.gamma
        self.tau = a.tau
        self.auto_alpha = a.auto_alpha
        self._training = True

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.actor.sample(state_t, deterministic=deterministic)
        action = action.cpu().numpy()[0]
        action = action * self.action_space_high.cpu().numpy()
        return action

    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        # Batch already on device (replay buffer stores tensors on device).
        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_observations"]
        dones = batch["dones"]

        # --- Critic Update ---
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_obs)
            next_q1, next_q2 = self.critic_target(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * next_q.squeeze(-1)

        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = (
            F.mse_loss(current_q1.squeeze(-1), target_q)
            + F.mse_loss(current_q2.squeeze(-1), target_q)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        new_actions, log_probs, _ = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha Update ---
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)

        # --- Target update ---
        if step % self.cfg.algorithm.target_update_interval == 0:
            self._soft_update_target()

        self.total_steps += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def _soft_update_target(self) -> None:
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha,
                "actor_optim": self.actor_optimizer.state_dict(),
                "critic_optim": self.critic_optimizer.state_dict(),
                "alpha_optim": self.alpha_optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            os.path.join(path, "sac.pt"),
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "sac.pt"), map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"].data
        self.actor_optimizer.load_state_dict(ckpt["actor_optim"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optim"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optim"])
        self.total_steps = ckpt["total_steps"]

    def train(self) -> None:
        self.actor.train()
        self.critic.train()
        self._training = True

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self._training = False
