"""Safe SAC baseline with standard Lagrangian constraint handling.

Standard constrained SAC using Lagrangian relaxation as a baseline
for comparison with HFG-SAC.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent, register_agent
from ..models.actor import SquashedGaussianActor
from ..models.critic import TwinQNetwork
from ..utils.config import HFGConfig

logger = logging.getLogger(__name__)


@register_agent("safe_sac")
class SafeSAC(BaseAgent):
    """Safe SAC with Lagrangian constraint relaxation.

    Standard SAC + constraint cost + Lagrange multiplier for constraint satisfaction.
    Baseline for comparison with HFG-SAC.
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

        # Entropy temperature (alpha)
        self.target_entropy = -act_dim  # heuristic from SAC paper
        self.log_alpha = torch.tensor(np.log(a.alpha), requires_grad=True, device=device)

        # Lagrange multiplier for constraint
        self.target_constraint = 0.0  # zero violation target
        self.log_lambda = torch.tensor(np.log(1.0), requires_grad=True, device=device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=a.lr_critic)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=a.lr_alpha)
        self.lambda_optimizer = optim.Adam([self.log_lambda], lr=1e-3)

        self.gamma = a.gamma
        self.tau = a.tau
        self.auto_alpha = a.auto_alpha
        self._training = True

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def lambda_val(self) -> torch.Tensor:
        return self.log_lambda.exp().clamp(min=0.0, max=100.0)

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.actor.sample(state_t, deterministic=deterministic)
        action = action.cpu().numpy()[0]
        # Scale to action space range
        action = action * self.action_space_high.cpu().numpy()
        return action

    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        # Batch already on device (replay buffer stores tensors on device).
        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_observations"]
        dones = batch["dones"]
        constraint_costs = batch.get("constraint_costs", torch.zeros_like(rewards))

        # --- Critic Update ---
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_obs)
            next_q1, next_q2 = self.critic_target(next_obs, next_actions)
            next_q = (
                torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            ).squeeze(-1)
            target_q = rewards + self.gamma * (1 - dones) * next_q
            # Subtract lambda * constraint cost
            target_q = target_q - self.lambda_val * constraint_costs

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

        # Actor loss: maximize Q - alpha * log_prob - lambda * constraint_estimate
        actor_loss = (self.alpha * log_probs - min_q_new).mean()
        # Note: constraint cost for new actions would need model; approximate via batch

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha Update ---
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)

        # --- Lambda Update (Lagrangian) ---
        avg_constraint = constraint_costs.mean()
        lambda_loss = -self.lambda_val * (self.target_constraint - avg_constraint)
        # We want to maximize lambda * (constraint - target), i.e. push constraint down
        # Gradient ascent on lambda: dL/d(lambda) = (constraint - target)
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()

        # --- Target update ---
        if step % self.cfg.algorithm.target_update_interval == 0:
            self._soft_update_target()

        self.total_steps += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "lambda": self.lambda_val.item(),
            "avg_constraint_cost": avg_constraint.item(),
        }

    def _soft_update_target(self) -> None:
        """Soft update critic target network."""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha,
            "log_lambda": self.log_lambda,
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
            "lambda_optim": self.lambda_optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, os.path.join(path, "safe_sac.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "safe_sac.pt"), map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"].data
        self.log_lambda.data = ckpt["log_lambda"].data
        self.actor_optimizer.load_state_dict(ckpt["actor_optim"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optim"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optim"])
        self.lambda_optimizer.load_state_dict(ckpt["lambda_optim"])
        self.total_steps = ckpt["total_steps"]

    def train(self) -> None:
        self.actor.train()
        self.critic.train()
        self._training = True

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self._training = False
