"""Constrained Policy Optimization (Achiam et al., 2017).

Faithful implementation:
- diagonal Gaussian policy (un-squashed; the environment clips actions),
- conjugate gradients on a Fisher-vector product (mean-KL Hessian),
- analytic solution of the trust-region + cost dual problem,
- backtracking line search that accepts a step only when the actual
  mean-KL stays inside the trust region and the surrogate improves.

Policy update: one trust-region step per rollout. Value/cost-value
networks are fitted by gradient descent over the rollout.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from .base_agent import BaseAgent, register_agent
from .onpolicy_utils import RolloutStorage, compute_gae, normalize_advantage
from .ppo import PPOValue
from ..utils.config import HFGConfig


# --------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------- #

class CPOActor(nn.Module):
    """Diagonal Gaussian policy with state-independent log std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        # Small final-layer init -> start near the origin.
        nn.init.uniform_(layers[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(layers[-1].bias)
        self.mean_net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = self.log_std.clamp(-20, 2).exp()
        return Normal(mean, std)


# --------------------------------------------------------------------- #
# Flat parameter helpers
# --------------------------------------------------------------------- #

def _flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in module.parameters()])


def _set_flat_params(module: nn.Module, flat: torch.Tensor) -> None:
    i = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view(p.shape))
        i += n


def flat_grad(loss: torch.Tensor, module: nn.Module,
              create_graph: bool = False, retain: bool = True) -> torch.Tensor:
    grads = torch.autograd.grad(loss, list(module.parameters()),
                                create_graph=create_graph, retain_graph=retain)
    return torch.cat([g.contiguous().view(-1) for g in grads])


# --------------------------------------------------------------------- #
# Conjugate gradient and dual solution
# --------------------------------------------------------------------- #

def conjugate_gradient(Hvp: Callable[[torch.Tensor], torch.Tensor],
                       b: torch.Tensor, iters: int = 10,
                       tol: float = 1e-10) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(iters):
        Ap = Hvp(p)
        alpha = rdotr / (torch.dot(p, Ap) + 1e-12)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x


def cpo_direction(g: torch.Tensor, b: torch.Tensor,
                  Hvp: Callable[[torch.Tensor], torch.Tensor],
                  cost_slack: float, target_kl: float
                  ) -> Tuple[torch.Tensor, int]:
    """Analytic dual solution of  max g·x s.t. 1/2 x'Hx<=δ, b·x<=c.

    Returns (step direction, flag) with flag 0 = optimal,
    1 = boundary with cost multiplier active, 2 = feasibility recovery.
    """
    eps = 1e-8
    x = conjugate_gradient(Hvp, g)               # H^-1 g
    y = conjugate_gradient(Hvp, b)               # H^-1 b
    q = torch.dot(g, x)
    r = torch.dot(g, y)
    s = torch.dot(b, y)

    c = cost_slack
    if c < 0 and c ** 2 > 2 * target_kl * (s + eps):
        # Constraint currently infeasible: pure cost-reduction step.
        step = -torch.sqrt(2 * target_kl / (s + eps)) * y
        return step.to(torch.float32), 2

    A = q - r ** 2 / (s + eps)
    B = 2 * target_kl - c ** 2 / (s + eps)
    A, B = max(A.item(), eps), max(B.item(), eps)
    lam = np.sqrt(A / B)
    nu = max(0.0, (r.item() - lam * c) / (s.item() + eps))
    step = (1.0 / lam) * (x - nu * y)
    return step.to(torch.float32), (1 if nu > 0 else 0)


# --------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------- #

@register_agent("cpo")
class CPO(BaseAgent):
    """Constrained Policy Optimization (Achiam et al., 2017)."""

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

        self.target_kl = p.target_kl
        self.cg_iters = p.cg_iters
        self.cg_damping = p.cg_damping
        self.backtrack_coef = p.backtrack_coef
        self.backtrack_iters = p.backtrack_iters
        self.value_coef = p.value_coef
        self.n_epochs = p.n_epochs
        self.batch_size = p.minibatch_size
        self.cost_limit_rate = p.cost_limit_rate

        self.actor = CPOActor(obs_dim, act_dim, a.hidden_dims).to(device)
        self.value_net = PPOValue(obs_dim, a.hidden_dims).to(device)
        self.cost_value_net = PPOValue(obs_dim, a.hidden_dims).to(device)

        self.value_optimizer = optim.Adam(
            list(self.value_net.parameters()) + list(self.cost_value_net.parameters()),
            lr=a.lr_actor,
        )

        self.gamma = a.gamma
        self.gae_lambda = p.gae_lambda
        self._training = True
        self.rollout = RolloutStorage(with_cost=True)

    # -- KL / FVP ------------------------------------------------------- #

    def _mean_kl(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            old_dist = self.actor.distribution(obs)
        new_dist = self.actor.distribution(obs)
        kl = torch.distributions.kl_divergence(old_dist, new_dist)
        return kl.sum(-1).mean()

    def _fisher_vector_product(self, obs: torch.Tensor) -> Callable:
        def Hvp(v: torch.Tensor) -> torch.Tensor:
            kl = self._mean_kl(obs)
            grad = flat_grad(kl, self.actor, create_graph=True)
            gv = (grad * v).sum()
            hvp = flat_grad(gv, self.actor, retain=True)
            return hvp + self.cg_damping * v
        return Hvp

    # -- Data collection ------------------------------------------------ #

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor.distribution(state_t)
            raw = dist.mean if deterministic else dist.sample()
            action = raw.cpu().numpy()[0]
            if not deterministic:
                log_prob = dist.log_prob(raw).sum(-1)
                val = self.value_net(state_t)
                cval = self.cost_value_net(state_t)
                self.rollout.obs.append(state.copy())
                self.rollout.actions.append(action.copy())
                self.rollout.log_probs.append(float(log_prob.item()))
                self.rollout.values.append(float(val.item()))
                self.rollout.cost_values.append(float(cval.item()))
        return action * self.action_space_high.cpu().numpy()

    def store_reward_and_done(self, reward: float, cost: float, done: bool) -> None:
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

    # -- Surrogates ----------------------------------------------------- #

    def _surrogates(self, obs_t, actions_t, old_log_probs, adv_r, adv_c):
        dist = self.actor.distribution(obs_t)
        log_probs = dist.log_prob(actions_t).sum(-1)
        ratio = torch.exp(log_probs - old_log_probs)
        surr_r = (ratio * adv_r).mean()
        surr_c = (ratio * adv_c).mean()
        return surr_r, surr_c

    # -- Update --------------------------------------------------------- #

    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        r = self.rollout
        n = len(r)
        if n < 2:
            return {"kl": 0.0, "accepted": 0.0}

        obs_t = torch.as_tensor(np.asarray(r.obs, dtype=np.float32), device=self.device)
        actions_t = torch.as_tensor(np.asarray(r.actions, dtype=np.float32), device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(r.log_probs, dtype=np.float32), device=self.device)
        dones_arr = np.asarray(r.dones, dtype=np.float32)
        values_arr = np.asarray(r.values, dtype=np.float32)
        cost_values_arr = np.asarray(r.cost_values, dtype=np.float32)
        rewards_arr = np.asarray(r.rewards, dtype=np.float32)
        costs_arr = np.asarray(r.costs, dtype=np.float32)

        adv_r = compute_gae(rewards_arr, values_arr, dones_arr,
                            self.gamma, self.gae_lambda, r.bootstrap_r)
        returns_r = adv_r + values_arr
        adv_c = compute_gae(costs_arr, cost_values_arr, dones_arr,
                            self.gamma, self.gae_lambda, r.bootstrap_c)
        returns_c = adv_c + cost_values_arr

        adv_r_t = torch.as_tensor(normalize_advantage(adv_r, True), device=self.device)
        adv_c_t = torch.as_tensor(normalize_advantage(adv_c, False), device=self.device)

        # Baseline surrogate values and current discounted cost return.
        surr_r0, surr_c0 = self._surrogates(
            obs_t, actions_t, old_log_probs, adv_r_t, adv_c_t)
        g = flat_grad(surr_r0, self.actor, retain=True)
        b = flat_grad(surr_c0, self.actor, retain=True)

        j_c = float(returns_c.mean())
        disc_horizon = (1 - self.gamma ** n) / (1 - self.gamma)
        cost_limit = self.cost_limit_rate * disc_horizon
        cost_slack = cost_limit - j_c

        Hvp = self._fisher_vector_product(obs_t)
        direction, flag = cpo_direction(
            g, b, Hvp, cost_slack, self.target_kl)

        # Backtracking line search on the real KL and surrogate values.
        old_params = _flat_params(self.actor)
        accepted = 0.0
        alpha = 1.0
        kl_val = 0.0
        expected_r = torch.dot(g, direction)
        for i in range(self.backtrack_iters):
            _set_flat_params(self.actor, old_params + alpha * direction)
            kl_val = float(self._mean_kl(obs_t).item())
            new_r, new_c = self._surrogates(
                obs_t, actions_t, old_log_probs, adv_r_t, adv_c_t)
            if flag == 2:
                ok = new_c < surr_c0 and kl_val <= self.target_kl
            else:
                ok = (new_r >= surr_r0 + 1e-9
                      and kl_val <= self.target_kl
                      and (flag != 1 or float(new_c) <= cost_limit))
            if ok:
                accepted = 1.0
                break
            alpha *= self.backtrack_coef
        if not accepted:
            _set_flat_params(self.actor, old_params)
            kl_val = 0.0

        # Fit value / cost-value networks.
        returns_r_t = torch.as_tensor(returns_r, device=self.device)
        returns_c_t = torch.as_tensor(returns_c, device=self.device)
        batch_size = min(self.batch_size, n)
        for _ in range(self.n_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = indices[start:start + batch_size]
                vl = F.mse_loss(self.value_net(obs_t[idx]), returns_r_t[idx])
                cl = F.mse_loss(self.cost_value_net(obs_t[idx]), returns_c_t[idx])
                loss = self.value_coef * (vl + cl)
                self.value_optimizer.zero_grad()
                loss.backward()
                self.value_optimizer.step()

        self.rollout = RolloutStorage(with_cost=True)
        self.total_steps += 1
        return {
            "kl": kl_val,
            "accepted": accepted,
            "line_search_alpha": alpha,
            "flag": float(flag),
            "j_cost": j_c,
            "cost_slack": cost_slack,
        }

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "value": self.value_net.state_dict(),
                "cost_value": self.cost_value_net.state_dict(),
                "total_steps": self.total_steps,
            },
            os.path.join(path, "cpo.pt"),
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "cpo.pt"), map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.value_net.load_state_dict(ckpt["value"])
        self.cost_value_net.load_state_dict(ckpt["cost_value"])
        self.total_steps = ckpt["total_steps"]

    def train(self) -> None:
        self.actor.train()
        self.value_net.train()
        self.cost_value_net.train()
        self._training = True

    def eval(self) -> None:
        self.actor.eval()
        self.value_net.eval()
        self.cost_value_net.eval()
        self._training = False
