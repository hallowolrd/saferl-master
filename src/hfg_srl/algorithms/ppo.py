"""PPO baseline (Proximal Policy Optimization).

On-policy RL algorithm widely used as a baseline for safe RL comparisons.
Uses clipped surrogate objective and GAE for advantage estimation.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from .base_agent import BaseAgent, register_agent
from .onpolicy_utils import RolloutStorage, compute_gae, normalize_advantage
from ..utils.config import HFGConfig


class PPOPolicy(nn.Module):
    """Gaussian policy network for PPO (tanh-squashed)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.mean_net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_net(obs)
        std = self.log_std.clamp(min=-20, max=2).exp()
        return mean, std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        raw = dist.rsample()
        log_prob = dist.log_prob(raw).sum(dim=-1)
        action = torch.tanh(raw)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob, torch.tanh(mean)

    def log_prob_from_squashed(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Normal]:
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        atanh_actions = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        log_probs = dist.log_prob(atanh_actions).sum(dim=-1)
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)
        return log_probs, dist


class PPOValue(nn.Module):
    """Value function network for PPO."""

    def __init__(self, obs_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@register_agent("ppo")
class PPO(BaseAgent):
    """Proximal Policy Optimization baseline.

    On-policy algorithm with clipped surrogate and GAE.
    No safety mechanism — used as standard RL baseline.
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
        a, p = cfg.algorithm, cfg.ppo

        self.clip_eps = p.clip_eps
        self.gae_lambda = p.gae_lambda
        self.entropy_coef = p.entropy_coef
        self.value_coef = p.value_coef
        self.max_grad_norm = p.max_grad_norm
        self.n_epochs = p.n_epochs
        self.batch_size = p.minibatch_size

        self.policy = PPOPolicy(obs_dim, act_dim, a.hidden_dims).to(device)
        self.value_net = PPOValue(obs_dim, a.hidden_dims).to(device)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=a.lr_actor,
        )

        self.gamma = a.gamma
        self._training = True
        self.rollout = RolloutStorage(with_cost=False)

    @staticmethod
    def compute_gae_static(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        gamma: float,
        lam: float,
        bootstrap: float,
    ) -> np.ndarray:
        return compute_gae(rewards, values, dones, gamma, lam, bootstrap)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                _, _, mean = self.policy.sample(state_t)
                return (mean.cpu().numpy()[0] * self.action_space_high.cpu().numpy())
            else:
                action, log_prob, _ = self.policy.sample(state_t)
                val = self.value_net(state_t)
                self.rollout.obs.append(state.copy())
                self.rollout.log_probs.append(float(log_prob.item()))
                self.rollout.values.append(float(val.item()))
            out = action.cpu().numpy()[0]
            self.rollout.actions.append(out.copy())
        return out * self.action_space_high.cpu().numpy()

    def store_reward_and_done(self, reward: float, done: bool) -> None:
        """Store reward and done flag after env.step() — called by trainer."""
        self.rollout.rewards.append(reward)
        self.rollout.dones.append(float(done))

    def finish_rollout(self, last_obs: np.ndarray, truncated: bool) -> None:
        """Bootstrap V(s_last) for time-limit truncation before the update."""
        bootstrap = 0.0
        if truncated:
            state_t = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                bootstrap = float(self.value_net(state_t).item())
        self.rollout.finish(bootstrap)

    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        r = self.rollout
        n = len(r)
        obs_arr = np.asarray(r.obs, dtype=np.float32)
        actions_arr = np.asarray(r.actions, dtype=np.float32)
        rewards_arr = np.asarray(r.rewards, dtype=np.float32)
        log_probs_arr = np.asarray(r.log_probs, dtype=np.float32)
        values_arr = np.asarray(r.values, dtype=np.float32)
        dones_arr = np.asarray(r.dones, dtype=np.float32)

        advantages = compute_gae(
            rewards_arr, values_arr, dones_arr,
            self.gamma, self.gae_lambda, r.bootstrap_r,
        )
        returns = advantages + values_arr
        advantages = normalize_advantage(advantages, normalize=True)

        obs_t = torch.as_tensor(obs_arr, device=self.device)
        actions_t = torch.as_tensor(actions_arr, device=self.device)
        old_log_probs = torch.as_tensor(log_probs_arr, device=self.device)
        advantages_t = torch.as_tensor(advantages, device=self.device)
        returns_t = torch.as_tensor(returns, device=self.device)

        policy_loss_total = 0.0
        value_loss_total = 0.0
        n_batches = 0
        batch_size = min(self.batch_size, n)

        for _ in range(self.n_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = indices[start:start + batch_size]
                new_log_probs, dist = self.policy.log_prob_from_squashed(obs_t[idx], actions_t[idx])

                ratio = (new_log_probs - old_log_probs[idx]).exp()
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss -= self.entropy_coef * dist.entropy().sum(dim=-1).mean()

                value_loss = F.mse_loss(self.value_net(obs_t[idx]), returns_t[idx])
                loss = policy_loss + self.value_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                policy_loss_total += float(policy_loss.item())
                value_loss_total += float(value_loss.item())
                n_batches += 1

        self.rollout = RolloutStorage(with_cost=False)
        self.total_steps += 1
        return {
            "policy_loss": policy_loss_total / max(n_batches, 1),
            "value_loss": value_loss_total / max(n_batches, 1),
        }

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            os.path.join(path, "ppo.pt"),
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "ppo.pt"), map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.value_net.load_state_dict(ckpt["value"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]

    def train(self) -> None:
        self.policy.train()
        self.value_net.train()
        self._training = True

    def eval(self) -> None:
        self.policy.eval()
        self.value_net.eval()
        self._training = False
