"""Safety Layer baseline (Dalal et al., 2018-style action projection).

Monitors constraint values and projects the base SAC action onto the
safe set at every decision:
- SOC band: analytic per-step projection of the ESS action,
- islanded mode: frequency/voltage projection via the power-balance
  model (same physics as the HFG-SAC hard shield).

All thresholds come from the environment/fuzzy config; nothing is
hardcoded to a scenario.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch

from .base_agent import BaseAgent, register_agent
from .sac import SAC
from ..utils.config import HFGConfig


@register_agent("safety_layer")
class SafetyLayerSAC(BaseAgent):
    """SAC with a constraint-based safe-action projection layer."""

    def __init__(
        self,
        cfg: HFGConfig,
        obs_dim: int,
        act_dim: int,
        action_space_high: np.ndarray,
        device: str,
    ):
        super().__init__(cfg, obs_dim, act_dim, action_space_high, device)
        self.base_agent = SAC(cfg, obs_dim, act_dim, action_space_high, device)

        e = cfg.env
        margin = getattr(e, "soc_shield_margin", 0.02)
        self.soc_min_shield = e.soc_min + margin
        self.soc_max_shield = e.soc_max - margin

        dt_h = 24.0 / e.time_steps_per_day
        self._ess_disc_per_act = e.ess_power_kw * dt_h / max(e.ess_capacity_kwh, 1e-6)

        # Islanded detection and physical model (dims-based).
        self._is_islanded = (obs_dim == 9) and (act_dim == 4)
        self._fv_ratio = getattr(cfg.fuzzy, "freq_volt_shield_safety_ratio", 0.175)
        if self._is_islanded:
            self._pv_cap = float(e.pv_capacity_kw)
            self._wt_cap = float(e.wt_capacity_kw)
            self._base_load = float(e.base_load_kw)
            self._ess_power_kw = float(e.ess_power_kw)
            self._de_rated_kw = float(e.de_rated_power_kw)
            self._interruptible_load_kw = float(e.interruptible_load_kw)

    @property
    def actor(self):
        return self.base_agent.actor

    @property
    def critic(self):
        return self.base_agent.critic

    def _project_soc(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Project ESS action so the next SOC stays inside the band."""
        soc = float(state[0])
        disc = self._ess_disc_per_act
        if disc <= 1e-9:
            return action
        a_charge_max = min(1.0, max(-1.0, (self.soc_max_shield - soc) / disc))
        a_discharge_max = min(1.0, max(-1.0, (soc - self.soc_min_shield) / disc))
        action = action.copy()
        action[0] = min(float(action[0]), a_charge_max)
        action[0] = max(float(action[0]), -a_discharge_max)
        return action

    def _project_freq_voltage(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Project DE/shed/dump to keep supply-demand imbalance in the band."""
        pv_kw = float(state[1]) * self._pv_cap
        wt_kw = float(state[2]) * self._wt_cap
        load_kw = float(state[3]) * self._base_load

        ess_f = float(action[0])
        de_f = float(np.clip(action[1], 0.0, 1.0))
        shed_f = float(np.clip(action[2], 0.0, 1.0))
        dump_f = float(np.clip(action[3], 0.0, 1.0))

        ess_kw = ess_f * self._ess_power_kw
        de_kw = de_f * self._de_rated_kw
        shed_kw = shed_f * self._interruptible_load_kw
        dump_kw = dump_f * self._pv_cap * 0.5

        gen = pv_kw + wt_kw + de_kw + max(0.0, -ess_kw)
        cons = load_kw - shed_kw + dump_kw + max(0.0, ess_kw)
        imbalance = gen - cons
        imb_ratio = imbalance / max(self._base_load, 1e-6)

        over = max(0.0, abs(imb_ratio) - self._fv_ratio)
        adj_kw = over * self._base_load

        action = action.copy()
        if imbalance > 0:  # over-generation: reduce DE, then dump
            de_red = min(de_kw, adj_kw)
            de_kw -= de_red
            dump_red = min(dump_kw, adj_kw - de_red)
            dump_kw -= dump_red
        else:              # under-generation: raise DE, then shed
            de_inc = min(self._de_rated_kw - de_kw, adj_kw)
            de_kw += de_inc
            shed_inc = min(self._interruptible_load_kw - shed_kw, adj_kw - de_inc)
            shed_kw += shed_inc

        action[1] = de_kw / max(self._de_rated_kw, 1e-6)
        action[2] = shed_kw / max(self._interruptible_load_kw, 1e-6)
        action[3] = dump_kw / max(self._pv_cap * 0.5, 1e-6)
        return action

    def _project(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        action = self._project_soc(state, action)
        if self._is_islanded:
            action = self._project_freq_voltage(state, action)
        return action

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        scaled = self.base_agent.select_action(state, deterministic=deterministic)
        normalized = scaled / self.action_space_high.cpu().numpy()
        projected = self._project(state, normalized)
        return projected * self.action_space_high.cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        return self.base_agent.update(batch, step)

    def save(self, path: str) -> None:
        self.base_agent.save(path)

    def load(self, path: str) -> None:
        self.base_agent.load(path)

    def train(self) -> None:
        self.base_agent.train()
        self._training = True

    def eval(self) -> None:
        self.base_agent.eval()
        self._training = False
