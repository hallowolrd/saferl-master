"""Scenario slicing — extract subsets of data for specific operating scenarios.

Supports common microgrid experiment scenarios:
- Seasonal: summer, winter, spring, autumn
- Mode-specific: normal, extreme (high PV, high load, low wind, etc.)
- Day-type: typical_day, cloudy_day, windy_day
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .base_loader import BaseDataLoader, DataPoint


@dataclass
class ScenarioConfig:
    """Configuration for a data scenario slice.

    Attributes:
        name: Scenario identifier (e.g., 'summer_normal').
        months: List of months (1-12) to include. None = all months.
        day_indices: Optional explicit list of day indices to use.
        extreme_multiplier: If set, scale PV/load for extreme scenarios.
            Dict with keys like {'pv': 1.3, 'load': 1.2, 'wt': 0.5}.
        start_day: Starting day index (overrides months if set).
        num_days: Number of days to include.
    """
    name: str = "full"
    months: Optional[List[int]] = None
    day_indices: Optional[List[int]] = None
    extreme_multiplier: Optional[dict] = None
    start_day: Optional[int] = None
    num_days: Optional[int] = None


class ScenarioLoader(BaseDataLoader):
    """Wraps a BaseDataLoader to expose only a scenario-specific subset.

    This allows the same raw dataset to be reused across many experiment
    scenarios (seasonal, extreme, etc.) without duplicating data.

    Args:
        base_loader: The underlying full data loader.
        scenario: Scenario configuration.
        seed: Random seed (passed through).
    """

    def __init__(
        self,
        base_loader: BaseDataLoader,
        scenario: ScenarioConfig,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.base_loader = base_loader
        self.scenario = scenario
        self._step_map: List[int] = self._build_step_map()
        self._total_steps = len(self._step_map)

    def _build_step_map(self) -> List[int]:
        """Build a mapping from scenario-step to base-loader-step indices."""
        spd = self.base_loader.steps_per_day
        total_days = self.base_loader.num_days

        # Determine which day indices to include
        if self.scenario.day_indices is not None:
            day_idxs = [d % total_days for d in self.scenario.day_indices]
        elif self.scenario.months is not None:
            day_idxs = []
            # Use calendar-accurate month mapping
            month_days = _month_day_counts(total_days >= 366)  # leap year?
            for d in range(total_days):
                d_mod = d % 365 if total_days >= 365 else d
                month = _day_of_year_to_month(d_mod, month_days)
                if month in self.scenario.months:
                    day_idxs.append(d)
        elif self.scenario.start_day is not None:
            n = self.scenario.num_days or total_days
            day_idxs = [(self.scenario.start_day + i) % total_days for i in range(n)]
        else:
            day_idxs = list(range(total_days))

        # Expand days to step indices
        step_map = []
        for d in day_idxs:
            step_map.extend(range(d * spd, (d + 1) * spd))
        return step_map

    def __len__(self) -> int:
        return self._total_steps

    @property
    def steps_per_day(self) -> int:
        return self.base_loader.steps_per_day

    def __getitem__(self, step_idx: int) -> DataPoint:
        step_idx = step_idx % self._total_steps
        base_idx = self._step_map[step_idx]
        dp = self.base_loader[base_idx]

        # Apply extreme scenario multipliers
        if self.scenario.extreme_multiplier is not None:
            pv_mult = self.scenario.extreme_multiplier.get("pv", 1.0)
            wt_mult = self.scenario.extreme_multiplier.get("wt", 1.0)
            load_mult = self.scenario.extreme_multiplier.get("load", 1.0)
            dp = DataPoint(
                pv_kw=dp.pv_kw * pv_mult,
                wt_kw=dp.wt_kw * wt_mult,
                load_kw=dp.load_kw * load_mult,
                grid_price=dp.grid_price,
                extra={**dp.extra, "scenario": self.scenario.name},
            )
        return dp


# --- Predefined scenario presets ---

PRESET_SCENARIOS = {
    "full_year": ScenarioConfig(name="full_year"),
    "summer": ScenarioConfig(
        name="summer",
        months=[6, 7, 8],  # June-August
    ),
    "winter": ScenarioConfig(
        name="winter",
        months=[12, 1, 2],  # Dec-Feb
    ),
    "summer_extreme": ScenarioConfig(
        name="summer_extreme",
        months=[7],  # Peak summer
        extreme_multiplier={"pv": 1.2, "load": 1.3, "wt": 0.6},
    ),
    "winter_extreme": ScenarioConfig(
        name="winter_extreme",
        months=[1],  # Peak winter
        extreme_multiplier={"pv": 0.5, "load": 1.25, "wt": 1.3},
    ),
    "typical_week_summer": ScenarioConfig(
        name="typical_week_summer",
        start_day=182,  # ~July 1
        num_days=7,
    ),
    "typical_week_winter": ScenarioConfig(
        name="typical_week_winter",
        start_day=1,  # ~Jan 1
        num_days=7,
    ),
}


def _month_day_counts(leap_year: bool = False) -> List[int]:
    """Return number of days in each month (index 0 = January)."""
    if leap_year:
        return [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _day_of_year_to_month(day_of_year: int, month_days: List[int]) -> int:
    """Convert 0-based day-of-year to 1-based month number."""
    cumulative = 0
    for month_idx, days in enumerate(month_days):
        if day_of_year < cumulative + days:
            return month_idx + 1
        cumulative += days
    return 12  # fallback


def create_scenario(
    base_loader: BaseDataLoader,
    scenario_name: str,
    seed: int = 42,
) -> ScenarioLoader:
    """Create a ScenarioLoader from a named preset.

    Args:
        base_loader: The underlying data loader.
        scenario_name: Name of the preset scenario (see PRESET_SCENARIOS).
        seed: Random seed.

    Returns:
        A ScenarioLoader instance.

    Raises:
        ValueError: If scenario_name is not a known preset.
    """
    if scenario_name not in PRESET_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available: {list(PRESET_SCENARIOS.keys())}"
        )
    return ScenarioLoader(base_loader, PRESET_SCENARIOS[scenario_name], seed=seed)
