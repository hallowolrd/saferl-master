"""Islanded microgrid environment.

Islanded mode has stricter safety requirements:
- No grid support, so supply-demand balance is critical
- Frequency and voltage stability constraints
- Higher priority on ESS and DE safety

Data is sourced from a pluggable BaseDataLoader.
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


@register_env("islanded")
class IslandedEnv(MicrogridEnv):
    """Islanded microgrid environment.

    Observation space: [SOC, PV_output, WT_output, load, freq_deviation,
                        voltage_deviation, time_of_day, day_sin, day_cos]
    Action space: [ESS_power, DE_power, load_shedding, dump_load]

    Fuzzy constraints (stricter than grid-connected):
        - SOC upper/lower (harder constraint)
        - Frequency deviation
        - Voltage deviation
        - Supply-demand balance
    """

    def __init__(self, cfg: EnvConfig, data_loader: Optional[BaseDataLoader] = None):
        super().__init__(cfg, data_loader=data_loader)
        self._build_fuzzy_constraints()
        self._init_state()

    def _build_fuzzy_constraints(self) -> None:
        self.fuzzy_constraints = FuzzyConstraintSet(
            agg_method="weighted_average",
            constraints=[
                FuzzyConstraint(
                    name="soc_upper",
                    threshold=self.cfg.soc_max,
                    width=0.03,  # narrower = harder constraint
                    direction="upper",
                    weight=2.0,
                ),
                FuzzyConstraint(
                    name="soc_lower",
                    threshold=self.cfg.soc_min,
                    width=0.03,
                    direction="lower",
                    weight=2.0,
                ),
                FuzzyConstraint(
                    name="freq_deviation",
                    threshold=self.cfg.frequency_limit_hz,
                    width=0.1,
                    direction="upper",
                    weight=3.0,  # highest weight for frequency
                ),
                FuzzyConstraint(
                    name="voltage_deviation",
                    threshold=0.05,  # 5% deviation
                    width=0.01,
                    direction="upper",
                    weight=2.5,
                ),
            ],
        )

    def _init_state(self) -> None:
        self.soc = 0.6  # Start higher for islanded (more reserve)
        self.pv_output = 0.0
        self.wt_output = 0.0
        self.load = 0.0
        self.grid_price = self.cfg.grid_buy_price
        self.freq_deviation = 0.0
        self.voltage_deviation = 0.0
        self.ess_power = 0.0
        self.de_power = 0.0
        self.load_shed = 0.0
        self.dump_load = 0.0

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        return (9,)

    @property
    def action_space_shape(self) -> Tuple[int, ...]:
        return (4,)

    @property
    def action_high(self) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    @property
    def action_low(self) -> np.ndarray:
        return np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)

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
        self.soc = 0.6
        self.freq_deviation = 0.0
        self.voltage_deviation = 0.0
        # Fix-3: pre-load whole episode into numpy arrays (vectorized).
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
        # Fix-3: read current step data from pre-loaded numpy series.
        self.pv_output, self.wt_output, self.load, self.grid_price = self._get_data_at(
            self.current_step
        )

        # Vectorized denormalize.
        a = np.clip(action, self.action_low, self.action_high).astype(np.float64)
        self.ess_power = a[0] * self.cfg.ess_power_kw
        self.de_power = a[1] * self.cfg.de_rated_power_kw
        self.load_shed = a[2] * self.cfg.interruptible_load_kw
        self.dump_load = a[3] * self.cfg.pv_capacity_kw * 0.5

        # Power balance — vectorized.
        net_load = self.load - self.load_shed + self.dump_load
        ess_disch = np.maximum(0.0, -self.ess_power)
        ess_charge = np.maximum(0.0, self.ess_power)
        total_gen = self.pv_output + self.wt_output + self.de_power + ess_disch
        total_cons = net_load + ess_charge

        imbalance = total_gen - total_cons
        imbalance_ratio = imbalance / max(self.cfg.base_load_kw, 1e-6)

        # Frequency/voltage deviation — vectorized.
        abs_imb = abs(imbalance_ratio)
        self.freq_deviation = float(np.clip(abs_imb * 2.0, 0.0, 2.0))
        self.voltage_deviation = float(np.clip(abs_imb * 0.08, 0.0, 0.1))

        # Update SOC
        dt_hours = 24.0 / self.cfg.time_steps_per_day
        soc_change = self.ess_power * dt_hours / self.cfg.ess_capacity_kwh
        self.soc = float(np.clip(self.soc + soc_change, 0.0, 1.0))

        # Costs — vectorized.
        de_cost = self.de_power * self.cfg.de_fuel_cost * dt_hours
        ess_deg_cost = abs(self.ess_power) * self.cfg.ess_degradation_cost * dt_hours
        shed_unit_cost = getattr(self.cfg, "load_shed_cost", 2.0)
        load_shed_cost = self.load_shed * shed_unit_cost * dt_hours
        dump_cost = self.dump_load * 0.05 * dt_hours
        total_cost = de_cost + ess_deg_cost + load_shed_cost + dump_cost

        # Fuzzy constraints — vectorized.
        constraint_values = [
            self.soc,
            self.soc,
            self.freq_deviation,
            self.voltage_deviation,
        ]
        fcsd, fcsd_min, violation, hard_violated, _ = self.fuzzy_constraints.compute_all(
            constraint_values
        )

        # Reward — vectorized hinge penalties.
        freq_limit = self.cfg.frequency_limit_hz
        volt_limit = 0.05
        freq_excess = max(0.0, self.freq_deviation - 0.3 * freq_limit)
        volt_excess = max(0.0, self.voltage_deviation - 0.3 * volt_limit)
        freq_slope = getattr(self.cfg, "freq_penalty_slope", 1000.0)
        freq_volt_pen = freq_excess * freq_slope + volt_excess * 4000.0

        reward = -total_cost - violation * 200.0 - freq_volt_pen

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
            "freq_deviation": self.freq_deviation,
            "voltage_deviation": self.voltage_deviation,
            "load_shed": self.load_shed,
            "dump_load": self.dump_load,
            "imbalance_ratio": imbalance_ratio,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        t = self._time_of_day()
        day_norm = self._day_of_year() / 365.0
        obs = np.array([
            self.soc,
            self.pv_output / max(self.cfg.pv_capacity_kw, 1e-6),
            self.wt_output / max(self.cfg.wt_capacity_kw, 1e-6),
            self.load / max(self.cfg.base_load_kw, 1e-6),
            self.freq_deviation / max(self.cfg.frequency_limit_hz, 1e-6),
            self.voltage_deviation / 0.05,
            t,
            np.sin(2 * np.pi * day_norm),
            np.cos(2 * np.pi * day_norm),
        ], dtype=np.float32)
        return obs
