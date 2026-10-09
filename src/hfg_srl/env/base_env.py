"""Base microgrid environment with fuzzy constraints.

Follows Gymnasium API conventions. All specific microgrid environments
(grid-connected, islanded) inherit from this base class.

Data is provided by a BaseDataLoader instance, making the data backend
(synthetic, CSV, NREL, etc.) fully swappable via configuration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..data import BaseDataLoader, create_data_loader
from ..utils.config import EnvConfig, HFGConfig

logger = logging.getLogger(__name__)

# --- Factory / Registry ---

ENV_FACTORY: Dict[str, type] = {}


def register_env(name: str):
    """Decorator to register a microgrid environment class."""
    def decorator(cls: type) -> type:
        ENV_FACTORY[name] = cls
        return cls
    return decorator


def make_env(name: str, cfg: EnvConfig, data_loader: Optional[BaseDataLoader] = None) -> "MicrogridEnv":
    """Create an environment instance by name.

    Args:
        name: Environment name in ENV_FACTORY.
        cfg: Environment configuration.
        data_loader: Optional pre-constructed data loader. If None,
            one is created from cfg.data settings.

    Returns:
        Microgrid environment instance.

    Raises:
        ValueError: If environment name is not registered.
    """
    env_cls = ENV_FACTORY.get(name)
    if env_cls is None:
        raise ValueError(
            f"Environment '{name}' not registered. "
            f"Available: {list(ENV_FACTORY.keys())}"
        )
    return env_cls(cfg, data_loader=data_loader)


# --- Base Class ---

@dataclass
class EnvStepInfo:
    """Information returned by env.step()."""
    cost: float = 0.0
    constraint_violation: float = 0.0
    fcsd: float = 1.0  # Fuzzy Constraint Satisfaction Degree
    soc: float = 0.5
    net_load: float = 0.0
    pv_output: float = 0.0
    wt_output: float = 0.0
    load: float = 0.0
    grid_power: float = 0.0
    ess_power: float = 0.0
    de_power: float = 0.0


class MicrogridEnv(ABC):
    """Abstract base class for microgrid environments.

    Subclasses must implement:
        - _get_observation()
        - _apply_action(action)
        - _compute_reward(info)
        - reset()

    Data provisioning is delegated to a BaseDataLoader.
    """

    def __init__(self, cfg: EnvConfig, data_loader: Optional[BaseDataLoader] = None):
        self.cfg = cfg
        self.current_step: int = 0
        self.current_day: int = 0
        self.total_steps: int = 0
        self.max_steps: int = cfg.num_days * cfg.time_steps_per_day
        self._seed: Optional[int] = None
        self._np_random: Optional[np.random.Generator] = None

        # Create data loader if not provided
        if data_loader is None:
            data_source = getattr(cfg, "data_source", "synthetic")
            data_file = getattr(cfg, "data_file", None)
            scenario = getattr(cfg, "scenario", None)
            self.data_loader = create_data_loader(
                source=data_source,
                cfg=cfg,
                file_path=data_file,
                scenario=scenario,
                seed=getattr(cfg, "seed", 42),
            )
        else:
            self.data_loader = data_loader

    @property
    @abstractmethod
    def observation_space_shape(self) -> Tuple[int, ...]:
        """Shape of observation space."""
        ...

    @property
    @abstractmethod
    def action_space_shape(self) -> Tuple[int, ...]:
        """Shape of action space."""
        ...

    @property
    @abstractmethod
    def action_high(self) -> np.ndarray:
        """Upper bound of action space."""
        ...

    @property
    @abstractmethod
    def action_low(self) -> np.ndarray:
        """Lower bound of action space."""
        ...

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for environment and data loader."""
        self._seed = seed
        self._np_random = np.random.default_rng(seed)
        if self.data_loader is not None:
            self.data_loader.reset()

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Returns:
            observation, info dict.
        """
        ...

    @abstractmethod
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Action array.

        Returns:
            observation, reward, terminated, truncated, info dict.
        """
        ...

    def render(self) -> None:
        """Render environment (optional)."""
        pass

    def close(self) -> None:
        """Close environment and clean up resources."""
        pass

    # --- Helpers ---

    def _time_of_day(self) -> float:
        """Get normalized time of day [0, 1]."""
        spd = self.data_loader.steps_per_day if self.data_loader else self.cfg.time_steps_per_day
        return (self.current_step % spd) / spd

    def _day_of_year(self) -> int:
        """Get current day of year."""
        spd = self.data_loader.steps_per_day if self.data_loader else self.cfg.time_steps_per_day
        return self.current_step // spd

    def _get_next_data_point(self) -> Tuple[float, float, float, float]:
        """Get PV, WT, load, and price for the current step from data loader.

        Returns:
            Tuple of (pv_kw, wt_kw, load_kw, grid_price).
        """
        dp = self.data_loader[self.current_step]
        return dp.pv_kw, dp.wt_kw, dp.load_kw, dp.grid_price

    def _preload_episode_data(self) -> None:
        """Pre-load all data points for this episode into numpy arrays.

        Fix-3: avoids per-step DataPoint allocation + __getitem__ overhead.
        Called from reset(). Uses data_loader.get_batch() which is vectorized.
        """
        if self.data_loader is None:
            self._pv_series = np.zeros(self.max_steps, dtype=np.float32)
            self._wt_series = np.zeros(self.max_steps, dtype=np.float32)
            self._load_series = np.zeros(self.max_steps, dtype=np.float32)
            self._price_series = np.full(
                self.max_steps, self.cfg.grid_buy_price, dtype=np.float32
            )
            return
        pv, wt, load, price = self.data_loader.get_batch(0, self.max_steps)
        self._pv_series = pv
        self._wt_series = wt
        self._load_series = load
        self._price_series = price

    def _get_data_at(self, step: int) -> Tuple[float, float, float, float]:
        """Fast index into pre-loaded episode data."""
        s = step % len(self._pv_series)
        return (
            float(self._pv_series[s]),
            float(self._wt_series[s]),
            float(self._load_series[s]),
            float(self._price_series[s]),
        )

    def get_state_description(self) -> Dict[str, float]:
        """Return human-readable state description (for logging)."""
        return {
            "step": self.current_step,
            "day": self.current_day,
            "time_of_day": self._time_of_day(),
        }
