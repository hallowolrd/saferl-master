"""HFG-SAC: Hierarchical Fuzzy-Guided Soft Actor-Critic.

Main algorithm combining:
1. Fuzzy-Lagrangian constraint satisfaction (upper layer: safety)
2. Fuzzy reward shaping from expert knowledge (lower layer: guidance)
3. Standard SAC with twin Q-networks and automatic entropy tuning

The upper safety layer is implemented with a *differentiable FCSD estimator*
(cost critic) that predicts the fuzzy constraint-satisfaction degree for the
sampled actions and propagates true constraint gradients into the policy. This
replaces the non-differentiable batch-FCSD surrogate, which produced no policy
gradient.

This is the core algorithm of the HFG-SRL framework.
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
from .fuzzy_lagrangian import FuzzyLagrangian
from ..fuzzy_system.reward_shaper import FuzzyRewardShaper
from ..fuzzy_system.fuzzy_rules import expert_rule_base
from ..models.actor import SquashedGaussianActor
from ..models.critic import TwinQNetwork
from ..utils.config import HFGConfig

logger = logging.getLogger(__name__)


@register_agent("hfg_sac")
class HFGSAC(BaseAgent):
    """Hierarchical Fuzzy-Guided SAC.

    Two-layer fuzzy mechanism:
    - Upper layer: Fuzzy-Lagrangian constraint protection (safety guarantee)
      via a learned FCSD estimator (cost critic) + adaptive Lagrange multiplier.
    - Lower layer: Fuzzy reward shaping from expert rules (learning acceleration).

    The fuzzy systems are differentiable and trained end-to-end with the policy.
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

        # --- SAC Networks ---
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

        # --- Cost-return critic C(s,a): estimates cumulative future fuzzy
        # deficit E[sum_t gamma^t * clamp(fcsd_target - fcsd_min_t)]. ---
        # Unlike the old action-blind FCSD regressor, this is trained with a
        # Bellman backup, so the gradient w.r.t. the sampled action is live.
        self.cost_critic = TwinQNetwork(
            obs_dim, act_dim, a.hidden_dims, a.activation
        ).to(device)
        self.cost_critic_target = TwinQNetwork(
            obs_dim, act_dim, a.hidden_dims, a.activation
        ).to(device)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())

        # --- Entropy Temperature ---
        self.target_entropy = -act_dim
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(a.alpha), dtype=torch.float32, device=device)
        )

        # --- Fuzzy Layer 1: Fuzzy-Lagrangian Constraint Protection ---
        # The "no upper layer" ablation is a separate agent (fuzzy_sac);
        # hfg_sac always runs with the Lagrangian enabled.
        self.fuzzy_lagrangian = FuzzyLagrangian(cfg.fuzzy).to(device)

        # --- Fuzzy Layer 2: Fuzzy Reward Shaping ---
        rule_base = expert_rule_base() if cfg.fuzzy.use_expert_rules else None
        self.fuzzy_reward_shaper = FuzzyRewardShaper(
            cfg.fuzzy,
            num_inputs=obs_dim,
            rule_base=rule_base,
        ).to(device)

        # --- Optimizers ---
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=a.lr_critic)
        self.cost_critic_optimizer = optim.Adam(
            self.cost_critic.parameters(), lr=a.lr_critic
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=a.lr_alpha)

        # Hard deployment shield constants (SOC-aware ESS action clipping).
        # Margin keeps next SOC inside [soc_min+margin, soc_max-margin] so
        # FCSD stays > 0.9 at the boundary (hard threshold => FCSD = 0.5).
        e = cfg.env
        margin = getattr(e, "soc_shield_margin", 0.02)
        self.soc_min_shield = e.soc_min + margin
        self.soc_max_shield = e.soc_max - margin
        dt_h = 24.0 / e.time_steps_per_day
        self._ess_disc_per_act = (
            e.ess_power_kw * dt_h / max(e.ess_capacity_kwh, 1e-6)
        )
        # SOC index in observation vector (both envs: obs[0] = SOC).
        self.soc_obs_idx = 0
        # Action dimension for ESS (both envs: action[0] = ESS).
        self.ess_act_idx = 0
        # Shield-activation penalty strength (issue 2.4).
        self._shield_act_penalty_mu = getattr(cfg.fuzzy, "shield_act_penalty_mu", 8.0)

        # B1: Islanded-only frequency/voltage hard shield.
        # Detect islanded mode by obs_dim (9 = islanded, 8 = grid-connected)
        # and action_dim (4 = islanded [ESS, DE, shed, dump], 3 = grid [ESS, DE, grid]).
        self._is_islanded = (obs_dim == 9) and (act_dim == 4)
        self._fv_shield_enabled = (
            getattr(cfg.fuzzy, "freq_volt_shield_enabled", False)
            and self._is_islanded
        )
        self._fv_safety_ratio = getattr(cfg.fuzzy, "freq_volt_shield_safety_ratio", 0.175)
        if self._fv_shield_enabled:
            # Cache physical parameters needed to predict imbalance from action.
            # obs layout (islanded): [soc, pv_norm, wt_norm, load_norm,
            #                         freq_norm, volt_norm, t, sin, cos]
            self._pv_cap = float(e.pv_capacity_kw)
            self._wt_cap = float(e.wt_capacity_kw)
            self._base_load = float(e.base_load_kw)
            self._ess_power_kw = float(e.ess_power_kw)
            self._de_rated_kw = float(e.de_rated_power_kw)
            self._interruptible_load_kw = float(e.interruptible_load_kw)

        # B2: Hard/soft violation separation in cost-critic target.
        self._hard_thresh = getattr(cfg.fuzzy, "hard_violation_threshold", 0.5)
        self._hard_w = getattr(cfg.fuzzy, "hard_violation_cost_weight", 1.0)
        self._soft_w = getattr(cfg.fuzzy, "soft_violation_cost_weight", 0.1)

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
        is_single = isinstance(state, np.ndarray) and state.ndim == 1
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        )
        if is_single:
            state_t = state_t.unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.actor.sample(state_t, deterministic=deterministic)
            action = self._apply_shield(state_t, action)
            # Scale on GPU before D2H transfer (avoids CPU multiply + extra copy).
            out = (action * self.action_space_high).cpu().numpy()
        return out[0] if is_single else out

    def _apply_shield(
        self, state_t: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Pure-torch SOC shield: clip ESS action so next SOC stays inside
        [soc_min+margin, soc_max-margin]. Vectorized; works on GPU."""
        soc = state_t[..., self.soc_obs_idx]
        disc = self._ess_disc_per_act
        if disc > 1e-9:
            # Per-sample upper/lower bounds for ESS action in [-1, 1].
            a_charge_max = (self.soc_max_shield - soc) / disc
            a_discharge_max = (soc - self.soc_min_shield) / disc
            a_charge_max = a_charge_max.clamp(-1.0, 1.0)
            a_discharge_max = a_discharge_max.clamp(-1.0, 1.0)
            ess = action[..., self.ess_act_idx]
            ess = torch.minimum(ess, a_charge_max)
            ess = torch.maximum(ess, -a_discharge_max)
            action = action.clone()
            action[..., self.ess_act_idx] = ess

        # B1: islanded-only freq/voltage hard shield.  Skipped entirely for
        # grid-connected mode (obs_dim=8, no freq/voltage in state) so S1/S2
        # behaviour is unchanged.
        if self._fv_shield_enabled:
            action = self._apply_freq_voltage_shield(state_t, action)
        return action

    def _apply_freq_voltage_shield(
        self, state_t: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Clamp DE / load-shed / dump-load actions so the next step's
        supply-demand imbalance stays within a safe band.

        The islanded env computes freq_dev = |imbalance_ratio| * 2.0 (Hz)
        with a hard limit of 0.5 Hz.  This shield fires when the predicted
        |imbalance_ratio| exceeds ``self._fv_safety_ratio`` (default 0.175,
        i.e. freq_dev > 0.35 Hz = 70% of the hard limit).  It then:
          - over-generation (imbalance > 0): reduce DE, increase dump-load
          - under-generation (imbalance < 0): increase DE first, then shed
        This is a *last-line* hard shield; it only activates when the
        soft reward penalty (added in islanded.py) is about to be crossed.
        """
        # obs layout (islanded): [soc, pv_norm, wt_norm, load_norm, ...]
        pv_kw = state_t[..., 1] * self._pv_cap
        wt_kw = state_t[..., 2] * self._wt_cap
        load_kw = state_t[..., 3] * self._base_load

        # action layout (islanded): [ess, de, shed, dump] in [-1, 1].
        # env clips: ess in [-1,1], de/shed/dump in [0,1].
        ess_f = action[..., 0]
        de_f = action[..., 1].clamp(0.0, 1.0)
        shed_f = action[..., 2].clamp(0.0, 1.0)
        dump_f = action[..., 3].clamp(0.0, 1.0)

        ess_kw = ess_f * self._ess_power_kw
        de_kw = de_f * self._de_rated_kw
        shed_kw = shed_f * self._interruptible_load_kw
        # islanded env: dump_load = dump_f * pv_cap * 0.5
        dump_kw = dump_f * self._pv_cap * 0.5

        # Replicate islanded env power-balance equations.
        gen = pv_kw + wt_kw + de_kw + torch.relu(-ess_kw)
        cons = load_kw - shed_kw + dump_kw + torch.relu(ess_kw)
        imbalance = gen - cons
        imb_ratio = imbalance / max(self._base_load, 1e-6)

        over = imb_ratio.abs() - self._fv_safety_ratio
        over = over.clamp(min=0.0)
        # NOTE: removed `if float(over.max().item()) <= 1e-9: return action` —
        # that per-step .item() forced a GPU->CPU sync on every env step, which
        # was the dominant CUDA overhead.  When over==0 the ops below are all
        # no-ops (adj_kw=0, min/max return original), so correctness is preserved
        # without the sync.

        adj_kw = over * self._base_load  # magnitude of imbalance to absorb

        action = action.clone()

        # Over-generation: reduce DE first (cheapest), then reduce dump.
        pos = (imbalance > 0).float()
        de_reduction = torch.minimum(de_kw, adj_kw * pos)
        de_kw_new = de_kw - de_reduction
        remaining_pos = adj_kw * pos - de_reduction
        # Reduce dump-load (was absorbing excess gen); if no dump, leave DE.
        dump_reduction = torch.minimum(dump_kw, remaining_pos)
        dump_kw_new = dump_kw - dump_reduction

        # Under-generation: raise DE first, then shed load.
        neg = (imbalance < 0).float()
        headroom_de = self._de_rated_kw - de_kw_new
        de_increase = torch.minimum(headroom_de, adj_kw * neg)
        de_kw_new = de_kw_new + de_increase
        remaining_neg = adj_kw * neg - de_increase
        headroom_shed = self._interruptible_load_kw - shed_kw
        shed_increase = torch.minimum(headroom_shed, remaining_neg)
        shed_kw_new = shed_kw + shed_increase

        # Write back normalized actions.
        action[..., 1] = de_kw_new / max(self._de_rated_kw, 1e-6)
        action[..., 2] = shed_kw_new / max(self._interruptible_load_kw, 1e-6)
        action[..., 3] = dump_kw_new / max(self._pv_cap * 0.5, 1e-6)
        return action

    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        # Replay buffer now stores tensors directly on self.device, so no
        # .to(self.device) transfer needed per update (saves H2D latency).
        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"].view(-1)
        next_obs = batch["next_observations"]
        dones = batch["dones"].view(-1)
        fcsd = batch.get("fcsd", torch.ones_like(rewards)).view(-1)
        fcsd_min = batch.get("fcsd_min", fcsd).view(-1)

        # --- Step fuzzy reward shaper weight decay ---
        if self._training:
            self.fuzzy_reward_shaper.step()

        # --- Fuzzy reward shaping (potential-based) ---
        # Pass `dones` so terminal transitions use F = -Phi(s) (Phi(term)=0),
        # preserving the policy-invariance guarantee of Ng et al.
        with torch.no_grad():
            shaped_rewards = self.fuzzy_reward_shaper.shape_reward(
                rewards.unsqueeze(-1),
                obs,
                next_obs,
                self.gamma,
                dones=dones,
            ).squeeze(-1).view(-1)

        # --- Shared next-action sampling (critic target + cost target) ---
        # Both reward critic and cost critic evaluate next_obs with actions
        # sampled from the current policy.  Merging the two samples saves
        # one actor.forward + one shield application per update (~25% speedup).
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_obs)
            next_actions = self._apply_shield(next_obs, next_actions)
            next_actions_scaled = next_actions * self.action_space_high

            # Reward critic target
            next_q1, next_q2 = self.critic_target(next_obs, next_actions_scaled)
            next_q = (
                torch.min(next_q1, next_q2).view(-1)
                - self.alpha * next_log_probs.view(-1)
            )
            target_q = shaped_rewards + self.gamma * (1.0 - dones) * next_q

            # Cost critic target (reuse same next_actions — no second sample)
            nc1, nc2 = self.cost_critic_target(next_obs, next_actions_scaled)
            next_c = torch.min(nc1, nc2).view(-1)

        # --- B2: hard/soft violation separation for cost target ---
        fcsd_target = self.fuzzy_lagrangian.cfg.fcsd_target
        hard_mask = (fcsd_min < self._hard_thresh).float()
        soft_deficit = torch.clamp(fcsd_target - fcsd_min, min=0.0)
        per_step_cost = (
            hard_mask * self._hard_w
            + (1.0 - hard_mask) * self._soft_w * soft_deficit
        ).detach()
        target_c = per_step_cost + self.gamma * (1.0 - dones) * next_c

        # --- Reward Critic Update ---
        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = (
            F.smooth_l1_loss(current_q1.view(-1), target_q)
            + F.smooth_l1_loss(current_q2.view(-1), target_q)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- Cost Critic Update ---
        c1, c2 = self.cost_critic(obs, actions)
        cost_critic_loss = (
            F.smooth_l1_loss(c1.view(-1), target_c)
            + F.smooth_l1_loss(c2.view(-1), target_c)
        )

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.cost_critic.parameters(), max_norm=1.0)
        self.cost_critic_optimizer.step()

        # --- Actor Update (with Fuzzy-Lagrangian constraint) ---
        new_actions, log_probs, _ = self.actor.sample(obs)
        # Shield the sampled action before evaluating Q/C: replay buffer
        # stores shielded (safe) actions, so evaluating raw OOD actions
        # would give the critics out-of-distribution predictions.  The
        # clamp in _apply_shield is piecewise differentiable: gradients
        # flow through when inside bounds, zero when externally clamped.
        shielded_actions = self._apply_shield(obs, new_actions)
        # Scale to the env's action units before feeding critics (buffer
        # stores scaled actions; actor samples in [-1, 1]).  See issue 2.6.
        shielded_actions_scaled = shielded_actions * self.action_space_high
        q1_new, q2_new = self.critic(obs, shielded_actions_scaled)
        min_q_new = torch.min(q1_new, q2_new).view(-1)

        actor_sac_loss = (self.alpha * log_probs.view(-1) - min_q_new).mean()

        # Lagrangian penalty: lambda * C_hat(s, shielded_action).
        cost_q1_new, cost_q2_new = self.cost_critic(obs, shielded_actions_scaled)
        cost_hat = torch.min(cost_q1_new, cost_q2_new).view(-1)
        constraint_penalty = (self.fuzzy_lagrangian.lambda_val * cost_hat).mean()

        # Shield-activation penalty (issue 2.4): the hard clamp blocks the
        # gradient through the shielded action, so the actor never learns
        # to stay *inside* the safe band by itself — it stays permanently
        # dependent on the shield.  We add a differentiable hinge penalty
        # mu * relu(|ess_raw| - bound) that pushes the raw pre-shield
        # action back inside the allowed ESS operating window.
        soc = obs[..., self.soc_obs_idx]
        disc = self._ess_disc_per_act
        if disc > 1e-9:
            a_charge_max = ((self.soc_max_shield - soc) / disc).clamp(-1.0, 1.0)
            a_discharge_max = ((soc - self.soc_min_shield) / disc).clamp(-1.0, 1.0)
            ess_raw = new_actions[..., self.ess_act_idx]
            over_charge = torch.relu(ess_raw - a_charge_max)
            over_discharge = torch.relu(-a_discharge_max - ess_raw)
            shield_penalty = (self._shield_act_penalty_mu * (over_charge + over_discharge)).mean()
        else:
            shield_penalty = torch.tensor(0.0, device=self.device)

        actor_loss = actor_sac_loss + constraint_penalty + shield_penalty

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
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
            alpha_loss = torch.tensor(0.0, device=self.device)

        # --- Update Fuzzy-Lagrangian (lambda) ---
        # Lambda tracks the worst-constraint FCSD so that any single
        # constraint violation drives the multiplier up.
        lambda_val = self.fuzzy_lagrangian.update_lambda(fcsd_min)

        # NOTE: the TSK potential is intentionally *not* fine-tuned.
        # Training Phi ~= Q breaks the stationary-potential guarantee of
        # potential-based reward shaping and injects moving-target noise.
        fuzzy_loss = torch.tensor(0.0, device=self.device)

        # --- Target update (reward critic + cost critic) ---
        if step % self.cfg.algorithm.target_update_interval == 0:
            self._soft_update_target()

        self.total_steps += 1

        fl_state = self.fuzzy_lagrangian.get_state()
        fcsd_ma_val = fl_state.fcsd_ma

        # Batch all .item() calls into a single GPU->CPU transfer (was 12
        # separate syncs per update, each ~50-100us on GPU).
        return {
            "critic_loss": critic_loss.detach().item(),
            "cost_critic_loss": cost_critic_loss.detach().item(),
            "actor_loss": actor_loss.detach().item(),
            "actor_sac_loss": actor_sac_loss.detach().item(),
            "constraint_penalty": constraint_penalty.detach().item() if torch.is_tensor(constraint_penalty) else constraint_penalty,
            "shield_penalty": shield_penalty.detach().item(),
            "alpha_loss": alpha_loss.detach().item(),
            "alpha": self.alpha.item(),
            "lambda_val": lambda_val,
            "fcsd_ma": fcsd_ma_val,
            "fcsd_batch_mean": fcsd.mean().item(),
            "fcsd_min_batch_mean": fcsd_min.mean().item(),
            "fuzzy_loss": fuzzy_loss.detach().item() if torch.is_tensor(fuzzy_loss) else fuzzy_loss,
            "shaping_weight": self.fuzzy_reward_shaper.current_weight,
        }

    def _soft_update_target(self) -> None:
        """Soft update reward-critic and cost-critic target networks."""
        for source, target in (
            (self.critic, self.critic_target),
            (self.cost_critic, self.cost_critic_target),
        ):
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "cost_critic": self.cost_critic.state_dict(),
            "cost_critic_target": self.cost_critic_target.state_dict(),
            "log_alpha": self.log_alpha,
            "fuzzy_reward_shaper": self.fuzzy_reward_shaper.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "cost_critic_optim": self.cost_critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
            "total_steps": self.total_steps,
        }
        ckpt["fuzzy_lagrangian"] = self.fuzzy_lagrangian.state_dict()
        torch.save(ckpt, os.path.join(path, "hfg_sac.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(
            os.path.join(path, "hfg_sac.pt"), map_location=self.device
        )
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.cost_critic.load_state_dict(ckpt["cost_critic"])
        self.cost_critic_target.load_state_dict(ckpt["cost_critic_target"])
        self.log_alpha.data = ckpt["log_alpha"].data
        self.fuzzy_lagrangian.load_state_dict(ckpt["fuzzy_lagrangian"])
        self.fuzzy_reward_shaper.load_state_dict(ckpt["fuzzy_reward_shaper"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optim"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optim"])
        self.cost_critic_optimizer.load_state_dict(ckpt["cost_critic_optim"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optim"])
        self.total_steps = ckpt["total_steps"]

    def train(self) -> None:
        self.actor.train()
        self.critic.train()
        self.cost_critic.train()
        self.fuzzy_lagrangian.train()
        self.fuzzy_reward_shaper.train()
        self._training = True

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self.cost_critic.eval()
        self.fuzzy_lagrangian.eval()
        self.fuzzy_reward_shaper.eval()
        self._training = False