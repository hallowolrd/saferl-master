"""Base data loader interface for microgrid time-series data.

All data sources (synthetic, CSV, NREL, etc.) must implement this interface
so that environments can swap data backends transparently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class DataPoint:
    """A single time-step of microgrid input data.

    Attributes:
        pv_kw: Photovoltaic output (kW).
        wt_kw: Wind turbine output (kW).
        load_kw: Electrical load (kW).
        grid_price: Grid electricity price (yuan/kWh) for buy side.
        extra: Optional extra fields (temperature, irradiance, etc.).
    """
    pv_kw: float
    wt_kw: float
    load_kw: float
    grid_price: float = 0.8
    extra: Dict[str, float] = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


class BaseDataLoader(ABC):
    """Abstract base class for microgrid data loaders.

    Subclasses must implement:
        - __len__: total number of time steps
        - __getitem__: return DataPoint for a given step index
        - reset: reset internal state (if any)

    The loader is expected to be deterministic given a seed, so that
    experiments are reproducible.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def __len__(self) -> int:
        """Total number of time steps available."""
        ...

    @abstractmethod
    def __getitem__(self, step_idx: int) -> DataPoint:
        """Return data for a specific step index.

        Args:
            step_idx: Integer step index (0-based).

        Returns:
            DataPoint with PV, WT, load, and price values.
        """
        ...

    def get_batch(self, start_idx: int, length: int) -> Tuple[np.ndarray, ...]:
        """Return a batch of data as numpy arrays.

        Returns:
            Tuple of (pv_array, wt_array, load_array, price_array),
            each of shape (length,).
        """
        pvs, wts, loads, prices = [], [], [], []
        for i in range(start_idx, start_idx + length):
            dp = self[i % len(self)]
            pvs.append(dp.pv_kw)
            wts.append(dp.wt_kw)
            loads.append(dp.load_kw)
            prices.append(dp.grid_price)
        return (
            np.array(pvs, dtype=np.float32),
            np.array(wts, dtype=np.float32),
            np.array(loads, dtype=np.float32),
            np.array(prices, dtype=np.float32),
        )

    def reset(self) -> None:
        """Reset loader state. Override if your loader has state."""
        self._rng = np.random.default_rng(self.seed)

    @property
    @abstractmethod
    def steps_per_day(self) -> int:
        """Number of time steps per day."""
        ...

    @property
    def num_days(self) -> int:
        """Total number of days of data available."""
        return len(self) // self.steps_per_day

    def summary(self) -> Dict[str, float]:
        """Return basic statistics of the dataset."""
        # Sample first 10k steps for efficiency
        n = min(len(self), 10000)
        pvs, wts, loads, _ = self.get_batch(0, n)
        return {
            "total_steps": len(self),
            "steps_per_day": self.steps_per_day,
            "num_days": self.num_days,
            "pv_mean": float(np.mean(pvs)),
            "pv_std": float(np.std(pvs)),
            "pv_max": float(np.max(pvs)),
            "wt_mean": float(np.mean(wts)),
            "wt_std": float(np.std(wts)),
            "wt_max": float(np.max(wts)),
            "load_mean": float(np.mean(loads)),
            "load_std": float(np.std(loads)),
            "load_max": float(np.max(loads)),
        }
