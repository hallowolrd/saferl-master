"""Fuzzy SAC baseline — SAC with fuzzy reward shaping only.

SAC + fuzzy knowledge reward shaping, but without the upper fuzzy
constraint protection layer. Used as an ablation to measure the
contribution of fuzzy constraint protection.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent, register_agent
from ..fuzzy_system.reward_shaper import FuzzyRewardShaper
from ..fuzzy_system.fuzzy_rules import expert_rule_base
from ..models.actor import SquashedGaussianActor
from ..models.critic import TwinQNetwork
from ..utils.config import HFGConfig

logger = logging.getLogger(__name__)


@register_agent("fuzzy_sac")
class FuzzySAC(BaseAgent):
    """Fuzzy SAC: SAC + fuzzy reward shaping (no constraint protection).

    Ablation variant: keeps the lower fuzzy knowledge layer but
    removes the upper fuzzy constraint protection layer.
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
        e = cfg.env

        # SAC Networks
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

        # Fuzzy reward shaping (no constraint protection)
        rule_base = expert_rule_base() if cfg.fuzzy.use_expert_rules else None
        self.fuzzy_reward_shaper = FuzzyRewardShaper(
            cfg.fuzzy,
            num_inputs=obs_dim,
            rule_base=rule_base,
        ).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=a.lr_critic)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=a.lr_alpha)

        self.gamma = a.gamma
        self.tau = a.tau
        self.auto_alpha = a.auto_alpha
        self._training = True

        # --- SOC shield (shared with hfg_sac) ---
        margin = getattr(e, "soc_shield_margin", 0.02)
        self.soc_min_shield = e.soc_min + margin
        self.soc_max_shield = e.soc_max - margin
        dt_h = 24.0 / e.time_steps_per_day
        self._ess_disc_per_act = e.ess_power_kw * dt_h / max(e.ess_capacity_kwh, 1e-6)
        self.soc_obs_idx = 0
        self.ess_act_idx = 0
        self._shield_act_penalty_mu = getattr(cfg.fuzzy, "shield_act_penalty_mu", 8.0)

        # --- B1 freq/voltage hard shield (islanded only) ---
        self._is_islanded = (obs_dim == 9) and (act_dim == 4)
        self._fv_shield_enabled = (
            getattr(cfg.fuzzy, "freq_volt_shield_enabled", False)
            and self._is_islanded
        )
        self._fv_safety_ratio = getattr(cfg.fuzzy, "freq_volt_shield_safety_ratio", 0.175)
        if self._fv_shield_enabled:
            self._pv_cap = float(e.pv_capacity_kw)
            self._wt_cap = float(e.wt_capacity_kw)
            self._base_load = float(e.base_load_kw)
            self._ess_power_kw = float(e.ess_power_kw)
            self._de_rated_kw = float(e.de_rated_power_kw)
            self._interruptible_load_kw = float(e.interruptible_load_kw)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.actor.sample(state_t, deterministic=deterministic)
            action = self._apply_shield(state_t, action)
        out = (action * self.action_space_high).cpu().numpy()
        return out[0]

    def _apply_shield(
        self, state_t: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """SOC + freq/voltage shield (islanded-only). Same logic as hfg_sac."""
        soc = state_t[..., self.soc_obs_idx]
        disc = self._ess_disc_per_act
        if disc > 1e-9:
            a_charge_max = ((self.soc_max_shield - soc) / disc).clamp(-1.0, 1.0)
            a_discharge_max = ((soc - self.soc_min_shield) / disc).clamp(-1.0, 1.0)
            ess = action[..., self.ess_act_idx]
            ess = torch.minimum(ess, a_charge_max)
            ess = torch.maximum(ess, -a_discharge_max)
            action = action.clone()
            action[..., self.ess_act_idx] = ess

        if self._fv_shield_enabled:
            action = self._apply_freq_voltage_shield(state_t, action)
        return action

    def _apply_freq_voltage_shield(
        self, state_t: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Same B1 freq/voltage shield as hfg_sac (islanded only)."""
        pv_kw = state_t[..., 1] * self._pv_cap
        wt_kw = state_t[..., 2] * self._wt_cap
        load_kw = state_t[..., 3] * self._base_load

        ess_f = action[..., 0]
        de_f = action[..., 1].clamp(0.0, 1.0)
        shed_f = action[..., 2].clamp(0.0, 1.0)
        dump_f = action[..., 3].clamp(0.0, 1.0)

        ess_kw = ess_f * self._ess_power_kw
        de_kw = de_f * self._de_rated_kw
        shed_kw = shed_f * self._interruptible_load_kw
        dump_kw = dump_f * self._pv_cap * 0.5

        gen = pv_kw + wt_kw + de_kw + torch.relu(-ess_kw)
        cons = load_kw - shed_kw + dump_kw + torch.relu(ess_kw)
        imbalance = gen - cons
        imb_ratio = imbalance / max(self._base_load, 1e-6)

        over = (imb_ratio.abs() - self._fv_safety_ratio).clamp(min=0.0)
        # Removed per-step .item() GPU sync — same rationale as hfg_sac.
        adj_kw = over * self._base_load
        pos = (imbalance > 0).float()
        neg = (imbalance < 0).float()

        action = action.clone()

        # Over-generation: reduce DE, then reduce dump.
        de_reduction = torch.minimum(de_kw, adj_kw * pos)
        de_kw_new = de_kw - de_reduction
        remaining_pos = adj_kw * pos - de_reduction
        dump_reduction = torch.minimum(dump_kw, remaining_pos)
        dump_kw_new = dump_kw - dump_reduction

        # Under-generation: raise DE, then shed.
        headroom_de = self._de_rated_kw - de_kw_new
        de_increase = torch.minimum(headroom_de, adj_kw * neg)
        de_kw_new = de_kw_new + de_increase
        remaining_neg = adj_kw * neg - de_increase
        headroom_shed = self._interruptible_load_kw - shed_kw
        shed_increase = torch.minimum(headroom_shed, remaining_neg)
        shed_kw_new = shed_kw + shed_increase

        action[..., 1] = de_kw_new / max(self._de_rated_kw, 1e-6)
        action[..., 2] = shed_kw_new / max(self._interruptible_load_kw, 1e-6)
        action[..., 3] = dump_kw_new / max(self._pv_cap * 0.5, 1e-6)
        return action

    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        # Batch already on device (replay buffer stores tensors on device).
        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"].view(-1)
        next_obs = batch["next_observations"]
        dones = batch["dones"].view(-1)

        # Step fuzzy reward shaper weight decay
        if self._training:
            self.fuzzy_reward_shaper.step()

        # Fuzzy reward shaping (pass dones so terminal transitions use
        # F = -Phi(s), preserving Ng et al. policy-invariance — issue 2.5).
        with torch.no_grad():
            shaped_rewards = self.fuzzy_reward_shaper.shape_reward(
                rewards.unsqueeze(-1), obs, next_obs, self.gamma,
                dones=dones,
            ).squeeze(-1).view(-1)

        # --- Critic Update ---
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_obs)
            # Apply shield before evaluating critic target (shared with hfg_sac).
            next_actions = self._apply_shield(next_obs, next_actions)
            next_actions_scaled = next_actions * self.action_space_high
            next_q1, next_q2 = self.critic_target(next_obs, next_actions_scaled)
            next_q = torch.min(next_q1, next_q2).view(-1) - self.alpha * next_log_probs.view(-1)
            target_q = shaped_rewards + self.gamma * (1.0 - dones) * next_q

        current_q1, current_q2 = self.critic(obs, actions)
        # Huber (smooth L1) loss: constant gradient for large TD errors,
        # preventing outlier transitions from destabilizing Q networks.
        critic_loss = (
            F.smooth_l1_loss(current_q1.view(-1), target_q)
            + F.smooth_l1_loss(current_q2.view(-1), target_q)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- Actor Update ---
        new_actions, log_probs, _ = self.actor.sample(obs)
        shielded_actions = self._apply_shield(obs, new_actions)
        shielded_actions_scaled = shielded_actions * self.action_space_high
        q1_new, q2_new = self.critic(obs, shielded_actions_scaled)
        min_q_new = torch.min(q1_new, q2_new).view(-1)

        actor_sac_loss = (self.alpha * log_probs.view(-1) - min_q_new).mean()

        # Shield-activation penalty (shared with hfg_sac): pushes raw ESS
        # action inside the safe band instead of relying on the hard clamp.
        soc = obs[..., self.soc_obs_idx]
        disc = self._ess_disc_per_act
        if disc > 1e-9:
            a_charge_max = ((self.soc_max_shield - soc) / disc).clamp(-1.0, 1.0)
            a_discharge_max = ((soc - self.soc_min_shield) / disc).clamp(-1.0, 1.0)
            ess_raw = new_actions[..., self.ess_act_idx]
            over_charge = torch.relu(ess_raw - a_charge_max)
            over_discharge = torch.relu(-a_discharge_max - ess_raw)
            shield_penalty = (
                self._shield_act_penalty_mu * (over_charge + over_discharge)
            ).mean()
        else:
            shield_penalty = torch.tensor(0.0, device=self.device)

        actor_loss = actor_sac_loss + shield_penalty

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # --- Alpha Update ---
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs.view(-1) + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        # NOTE: the TSK potential is intentionally *not* fine-tuned here.
        fuzzy_loss = torch.tensor(0.0, device=self.device)

        # --- Target update ---
        if step % self.cfg.algorithm.target_update_interval == 0:
            self._soft_update_target()

        self.total_steps += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "actor_sac_loss": actor_sac_loss.item(),
            "shield_penalty": shield_penalty.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "fuzzy_loss": fuzzy_loss.item(),
            "shaping_weight": self.fuzzy_reward_shaper.current_weight,
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
                "fuzzy_reward_shaper": self.fuzzy_reward_shaper.state_dict(),
                "actor_optim": self.actor_optimizer.state_dict(),
                "critic_optim": self.critic_optimizer.state_dict(),
                "alpha_optim": self.alpha_optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            os.path.join(path, "fuzzy_sac.pt"),
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(
            os.path.join(path, "fuzzy_sac.pt"), map_location=self.device
        )
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"].data
        self.fuzzy_reward_shaper.load_state_dict(ckpt["fuzzy_reward_shaper"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optim"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optim"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optim"])
        self.total_steps = ckpt["total_steps"]

    def train(self) -> None:
        self.actor.train()
        self.critic.train()
        self.fuzzy_reward_shaper.train()
        self._training = True

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self.fuzzy_reward_shaper.eval()
        self._training = False
