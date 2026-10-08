"""Synthetic data generator — produces PV/WT/load time series via mathematical models.

This is the default data source, preserving backward compatibility with the
original environment's _generate_renewable_and_load method. The model uses:
- PV: sinusoidal diurnal curve + seasonal modulation + Gaussian noise
- WT: baseline + Gaussian noise (with weak diurnal pattern)
- Load: two-peak (morning/evening) diurnal pattern + seasonal + noise
- Price: simple time-of-use (TOU) schedule
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .base_loader import BaseDataLoader, DataPoint


class SyntheticDataLoader(BaseDataLoader):
    """Synthetic microgrid data generator with configurable system sizing.

    Produces deterministic pseudo-realistic time series given a seed.
    The generator is stateless — output depends only on step index + seed.

    Args:
        pv_capacity_kw: Nameplate PV capacity (kW).
        wt_capacity_kw: Nameplate WT capacity (kW).
        base_load_kw: Baseline electrical load (kW).
        steps_per_day: Time steps per day (e.g., 96 = 15-min intervals).
        num_days: Total days to generate.
        seed: Random seed for noise.
        grid_buy_price: Base grid buy price (yuan/kWh).
    """

    def __init__(
        self,
        pv_capacity_kw: float = 2000.0,
        wt_capacity_kw: float = 1500.0,
        base_load_kw: float = 5000.0,
        steps_per_day: int = 96,
        num_days: int = 365,
        seed: int = 42,
        grid_buy_price: float = 0.8,
    ):
        super().__init__(seed=seed)
        self.pv_capacity_kw = pv_capacity_kw
        self.wt_capacity_kw = wt_capacity_kw
        self.base_load_kw = base_load_kw
        self._steps_per_day = steps_per_day
        self._num_days = num_days
        self.grid_buy_price = grid_buy_price
        self._total_steps = steps_per_day * num_days

        # Pre-generate noise arrays for reproducibility (stateless indexing)
        self._pv_noise = self._rng.normal(0, pv_capacity_kw * 0.05, self._total_steps)
        self._wt_noise = self._rng.normal(0, wt_capacity_kw * 0.1, self._total_steps)
        self._load_noise = self._rng.normal(0, base_load_kw * 0.03, self._total_steps)

    def __len__(self) -> int:
        return self._total_steps

    @property
    def steps_per_day(self) -> int:
        return self._steps_per_day

    def __getitem__(self, step_idx: int) -> DataPoint:
        step_idx = step_idx % self._total_steps
        t = (step_idx % self._steps_per_day) / self._steps_per_day  # [0,1)
        day = step_idx // self._steps_per_day

        # --- PV: bell curve around noon + seasonal variation ---
        if 0.25 < t < 0.75:
            pv_factor = max(0.0, np.sin(np.pi * (t - 0.25) / 0.5))
        else:
            pv_factor = 0.0
        seasonal = 0.5 + 0.5 * np.cos(2 * np.pi * (day - 172) / 365)
        pv_base = self.pv_capacity_kw * pv_factor * (0.7 + 0.3 * seasonal)
        pv = max(0.0, pv_base + self._pv_noise[step_idx])

        # --- WT: baseline + noise (with weak diurnal pattern) ---
        base_wt = self.wt_capacity_kw * 0.3
        wt = max(0.0, min(self.wt_capacity_kw, base_wt + self._wt_noise[step_idx]))

        # --- Load: two-peak diurnal pattern + winter heating bump + noise ---
        morning_peak = np.exp(-0.5 * ((t - 0.3) / 0.08) ** 2)
        evening_peak = np.exp(-0.5 * ((t - 0.8) / 0.1) ** 2)
        base_load_factor = 0.6 + 0.3 * (morning_peak + evening_peak)
        winter_factor = 0.1 * np.cos(2 * np.pi * (day - 355) / 365)
        load_factor = base_load_factor + winter_factor
        load = max(0.0, self.base_load_kw * load_factor + self._load_noise[step_idx])

        # --- Price: simple time-of-use (TOU) schedule ---
        price = self._tou_price(t)

        return DataPoint(
            pv_kw=float(pv),
            wt_kw=float(wt),
            load_kw=float(load),
            grid_price=price,
            extra={
                "day_of_year": float(day),
                "time_of_day": float(t),
            },
        )

    def _tou_price(self, t: float) -> float:
        """Time-of-use electricity price schedule.

        Peak: 08:00-11:00, 18:00-21:00 → 1.2x
        Valley: 23:00-06:00 → 0.6x
        Flat: other hours → 1.0x
        """
        hour = t * 24.0
        if (8 <= hour < 11) or (18 <= hour < 21):
            return self.grid_buy_price * 1.2  # peak
        elif 23 <= hour or hour < 6:
            return self.grid_buy_price * 0.6  # valley
        else:
            return self.grid_buy_price  # flat
