"""Grid-connected microgrid environment.

Implements a grid-connected microgrid with PV, WT, ESS, DE, and grid exchange.
Data is sourced from a pluggable BaseDataLoader (synthetic, CSV, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base_env import MicrogridEnv, register_env
from .fuzzy_constraints import FuzzyConstraint, FuzzyConstraintSet
from ..data import BaseDataLoader
from ..utils.config import EnvConfig

logger = logging.getLogger(__name__)


@register_env("grid_connected")
class GridConnectedEnv(MicrogridEnv):
    """Grid-connected microgrid environment.

    Observation space: [SOC, PV_output, WT_output, load, time_of_day,
                        day_of_year_sin, day_of_year_cos, grid_price]
    Action space: [ESS_power, DE_power, grid_power] (all normalized)

    Fuzzy constraints:
        - SOC upper/lower (soft)
        - Grid power limit (soft)
        - Voltage deviation (soft)
    """

    def __init__(self, cfg: EnvConfig, data_loader: Optional[BaseDataLoader] = None):
        super().__init__(cfg, data_loader=data_loader)
        self._build_fuzzy_constraints()
        self._init_state()

    def _build_fuzzy_constraints(self) -> None:
        """Initialize fuzzy constraint set."""
        self.fuzzy_constraints = FuzzyConstraintSet(
            agg_method="weighted_average",
            constraints=[
                FuzzyConstraint(
                    name="soc_upper",
                    threshold=self.cfg.soc_max,
                    width=0.05,
                    direction="upper",
                    weight=1.5,
                ),
                FuzzyConstraint(
                    name="soc_lower",
                    threshold=self.cfg.soc_min,
                    width=0.05,
                    direction="lower",
                    weight=1.5,
                ),
                FuzzyConstraint(
                    name="grid_power_limit",
                    threshold=self.cfg.grid_power_limit_kw,
                    width=200.0,
                    direction="upper",
                    weight=1.0,
                ),
            ],
        )

    def _init_state(self) -> None:
        """Initialize internal state variables."""
        self.soc = 0.5  # Start at 50% SOC
        self.pv_output = 0.0
        self.wt_output = 0.0
        self.load = 0.0
        self.grid_power = 0.0
        self.ess_power = 0.0
        self.de_power = 0.0
        self.grid_price = self.cfg.grid_buy_price

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        return (8,)

    @property
    def action_space_shape(self) -> Tuple[int, ...]:
        return (3,)  # ess_power, de_power, grid_power

    @property
    def action_high(self) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)

    @property
    def action_low(self) -> np.ndarray:
        return np.array([-1.0, 0.0, -1.0], dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        self.current_day = 0
        self.soc = 0.5
        # Fix-3: pre-load whole episode into numpy arrays.
        self._preload_episode_data()
        self.pv_output, self.wt_output, self.load, self.grid_price = self._get_data_at(0)

        obs = self._get_observation()
        info = {
            "fcsd": 1.0,
            "constraint_violation": 0.0,
            "violated": False,
            "soc": self.soc,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step.

        Action is in [-1, 1]^3:
            [ESS_power, DE_power, grid_power_norm]
            - ESS: negative = discharge, positive = charge
            - DE: [0, 1] fraction of rated power
            - grid: negative = export, positive = import
        """
        # Fix-3: read current step data from pre-loaded numpy series.
        self.pv_output, self.wt_output, self.load, self.grid_price = self._get_data_at(
            self.current_step
        )

        # Vectorized denormalize: action already clipped by caller, but re-clip for safety.
        a = np.clip(action, self.action_low, self.action_high).astype(np.float64)
        self.ess_power = a[0] * self.cfg.ess_power_kw
        self.de_power = a[1] * self.cfg.de_rated_power_kw
        self.grid_power = a[2] * self.cfg.grid_power_limit_kw

        # Power balance — vectorized max(0, x) via np.maximum.
        ess_disch = np.maximum(0.0, -self.ess_power)
        ess_charge = np.maximum(0.0, self.ess_power)
        grid_import = np.maximum(0.0, self.grid_power)
        grid_export = np.maximum(0.0, -self.grid_power)

        total_gen = self.pv_output + self.wt_output + self.de_power + ess_disch + grid_import
        total_load = self.load + ess_charge + grid_export
        imbalance = total_gen - total_load
        imbalance_penalty = abs(imbalance)

        # Update SOC
        dt_hours = 24.0 / self.cfg.time_steps_per_day
        soc_change = self.ess_power * dt_hours / self.cfg.ess_capacity_kwh
        self.soc = float(np.clip(self.soc + soc_change, 0.0, 1.0))

        # Compute cost — vectorized.
        buy_price = self.grid_price
        sell_price = buy_price * 0.5
        grid_buy_cost = grid_import * buy_price * dt_hours
        grid_sell_revenue = grid_export * sell_price * dt_hours
        de_cost = self.de_power * self.cfg.de_fuel_cost * dt_hours
        ess_deg_cost = abs(self.ess_power) * self.cfg.ess_degradation_cost * dt_hours
        total_cost = grid_buy_cost - grid_sell_revenue + de_cost + ess_deg_cost + imbalance_penalty

        # Fuzzy constraints — vectorized.
        constraint_values = [
            self.soc,
            self.soc,
            abs(self.grid_power),
        ]
        fcsd, fcsd_min, violation, hard_violated, _ = self.fuzzy_constraints.compute_all(
            constraint_values
        )

        # Reward
        reward = -total_cost - violation * 100.0

        # Step forward
        self.current_step += 1
        if self.current_step % self.cfg.time_steps_per_day == 0:
            self.current_day += 1

        terminated = False
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = {
            "cost": total_cost,
            "fcsd": fcsd,
            "fcsd_min": fcsd_min,
            "constraint_violation": violation,
            "violated": hard_violated,
            "soc": self.soc,
            "pv": self.pv_output,
            "wt": self.wt_output,
            "load": self.load,
            "grid_power": self.grid_power,
            "grid_price": self.grid_price,
            "ess_power": self.ess_power,
            "de_power": self.de_power,
            "imbalance": imbalance,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        t = self._time_of_day()
        day_norm = self._day_of_year() / 365.0
        obs = np.array([
            self.soc,
            self.pv_output / max(self.cfg.pv_capacity_kw, 1e-6),
            self.wt_output / max(self.cfg.wt_capacity_kw, 1e-6),
            self.load / max(self.cfg.base_load_kw, 1e-6),
            t,
            np.sin(2 * np.pi * day_norm),
            np.cos(2 * np.pi * day_norm),
            self.grid_price / max(self.cfg.grid_buy_price, 1e-6),  # normalized price
        ], dtype=np.float32)
        return obs
