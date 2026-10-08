"""PPO-Lagrangian baseline.

PPO with Lagrangian constraint relaxation — a standard safe RL baseline.
Adds a Lagrange multiplier for constraint costs to the PPO objective.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent, register_agent
from .onpolicy_utils import RolloutStorage, compute_gae, normalize_advantage
from .ppo import PPOPolicy, PPOValue
from ..utils.config import HFGConfig


@register_agent("ppo_lagrangian")
class PPOLagrangian(BaseAgent):
    """PPO with Lagrangian constraint handling.

    Standard safe RL baseline. Adds a cost value function and Lagrange
    multiplier to PPO for constraint satisfaction.
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

        # Maximum allowed per-step violation rate.
        self.cost_limit = p.cost_limit_rate
        self._normalize_cost = p.normalize_cost_advantage

        self.log_lambda = nn.Parameter(torch.tensor(0.0, device=device))
        self.lambda_lr = 1e-2

        self.policy = PPOPolicy(obs_dim, act_dim, a.hidden_dims).to(device)
        self.value_net = PPOValue(obs_dim, a.hidden_dims).to(device)
        self.cost_value_net = PPOValue(obs_dim, a.hidden_dims).to(device)

        self.optimizer = optim.Adam(
            list(self.policy.parameters())
            + list(self.value_net.parameters())
            + list(self.cost_value_net.parameters()),
            lr=a.lr_actor,
        )
        self.lambda_optimizer = optim.Adam([self.log_lambda], lr=self.lambda_lr)

        self.gamma = a.gamma
        self._training = True
        self.rollout = RolloutStorage(with_cost=True)

    @property
    def lambda_val(self) -> torch.Tensor:
        return self.log_lambda.exp().clamp(min=0.0, max=100.0)

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

    @staticmethod
    def normalize_advantage(adv: np.ndarray, normalize: bool) -> np.ndarray:
        return normalize_advantage(adv, normalize=normalize)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy.forward(state_t)
                return torch.tanh(mean).cpu().numpy()[0] * self.action_space_high.cpu().numpy()
            action, log_prob, _ = self.policy.sample(state_t)
            val = self.value_net(state_t)
            cval = self.cost_value_net(state_t)
            out = action.cpu().numpy()[0]
            self.rollout.obs.append(state.copy())
            self.rollout.actions.append(out.copy())
            self.rollout.log_probs.append(float(log_prob.item()))
            self.rollout.values.append(float(val.item()))
            self.rollout.cost_values.append(float(cval.item()))
        return out * self.action_space_high.cpu().numpy()

    def store_reward_and_done(
        self, reward: float, cost: float, done: bool
    ) -> None:
        self.rollout.rewards.append(reward)
        self.rollout.costs.append(cost)
        self.rollout.dones.append(float(done))

    def finish_rollout(self, last_obs: np.ndarray, truncated: bool) -> None:
        bootstrap_r = bootstrap_c = 0.0
        if truncated:
            state_t = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                bootstrap_r = float(self.value_net(state_t).item())
                bootstrap_c = float(self.cost_value_net(state_t).item())
        self.rollout.finish(bootstrap_r, bootstrap_c)

    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        r = self.rollout
        n = len(r)
        if n < 2:
            return {
                "policy_loss": 0.0, "value_loss": 0.0,
                "lambda": self.lambda_val.item(),
            }

        obs_arr = np.asarray(r.obs, dtype=np.float32)
        actions_arr = np.asarray(r.actions, dtype=np.float32)
        rewards_arr = np.asarray(r.rewards, dtype=np.float32)
        costs_arr = np.asarray(r.costs, dtype=np.float32)
        log_probs_arr = np.asarray(r.log_probs, dtype=np.float32)
        values_arr = np.asarray(r.values, dtype=np.float32)
        cost_values_arr = np.asarray(r.cost_values, dtype=np.float32)
        dones_arr = np.asarray(r.dones, dtype=np.float32)

        adv_r = compute_gae(rewards_arr, values_arr, dones_arr,
                            self.gamma, self.gae_lambda, r.bootstrap_r)
        returns_r = adv_r + values_arr
        adv_c = compute_gae(costs_arr, cost_values_arr, dones_arr,
                            self.gamma, self.gae_lambda, r.bootstrap_c)
        returns_c = adv_c + cost_values_arr

        adv_r = normalize_advantage(adv_r, normalize=True)
        # Cost advantages preserve scale (center only).
        adv_c = normalize_advantage(adv_c, normalize=self._normalize_cost)

        obs_t = torch.as_tensor(obs_arr, device=self.device)
        actions_t = torch.as_tensor(actions_arr, device=self.device)
        old_log_probs = torch.as_tensor(log_probs_arr, device=self.device)
        adv_r_t = torch.as_tensor(adv_r, device=self.device)
        adv_c_t = torch.as_tensor(adv_c, device=self.device)
        returns_r_t = torch.as_tensor(returns_r, device=self.device)
        returns_c_t = torch.as_tensor(returns_c, device=self.device)

        # Constraint statistic: per-step violation rate vs limit.
        cost_rate = float(costs_arr.mean())

        policy_loss_total = 0.0
        value_loss_total = 0.0
        n_batches = 0
        batch_size = min(self.batch_size, n)
        lam = self.lambda_val.detach()

        for _ in range(self.n_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = indices[start:start + batch_size]
                new_log_probs, dist = self.policy.log_prob_from_squashed(obs_t[idx], actions_t[idx])

                ratio = (new_log_probs - old_log_probs[idx]).exp()
                combined = adv_r_t[idx] - lam * adv_c_t[idx]
                surr1 = ratio * combined
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * combined
                policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss -= self.entropy_coef * dist.entropy().sum(dim=-1).mean()

                value_loss = F.mse_loss(self.value_net(obs_t[idx]), returns_r_t[idx])
                cost_value_loss = F.mse_loss(self.cost_value_net(obs_t[idx]), returns_c_t[idx])
                loss = policy_loss + self.value_coef * (value_loss + cost_value_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())
                    + list(self.value_net.parameters())
                    + list(self.cost_value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                policy_loss_total += float(policy_loss.item())
                value_loss_total += float(value_loss.item()) + float(cost_value_loss.item())
                n_batches += 1

        # Dual ascent: increase lambda when violation rate exceeds limit.
        lambda_loss = -self.lambda_val * (cost_rate - self.cost_limit)
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()

        self.rollout = RolloutStorage(with_cost=True)
        self.total_steps += 1
        return {
            "policy_loss": policy_loss_total / max(n_batches, 1),
            "value_loss": value_loss_total / max(n_batches, 1),
            "lambda": self.lambda_val.item(),
            "cost_rate": cost_rate,
        }

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value_net.state_dict(),
                "cost_value": self.cost_value_net.state_dict(),
                "log_lambda": self.log_lambda,
                "optimizer": self.optimizer.state_dict(),
                "lambda_optimizer": self.lambda_optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            os.path.join(path, "ppo_lagrangian.pt"),
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "ppo_lagrangian.pt"), map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.value_net.load_state_dict(ckpt["value"])
        self.cost_value_net.load_state_dict(ckpt["cost_value"])
        self.log_lambda.data = ckpt["log_lambda"].data
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lambda_optimizer.load_state_dict(ckpt["lambda_optimizer"])
        self.total_steps = ckpt["total_steps"]

    def train(self) -> None:
        self.policy.train()
        self.value_net.train()
        self.cost_value_net.train()
        self._training = True

    def eval(self) -> None:
        self.policy.eval()
        self.value_net.eval()
        self.cost_value_net.eval()
        self._training = False
