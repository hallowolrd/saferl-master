"""Evaluation metrics for safe RL and microgrid dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics collected over a single episode."""
    total_reward: float = 0.0
    total_cost: float = 0.0
    total_constraint_violation: float = 0.0
    constraint_violation_count: int = 0
    num_steps: int = 0
    # Fuzzy constraint satisfaction
    mean_fcsd: float = 0.0
    min_fcsd: float = 1.0
    # Energy metrics
    renewable_curtailment_kwh: float = 0.0
    grid_import_kwh: float = 0.0
    grid_export_kwh: float = 0.0
    ess_throughput_kwh: float = 0.0


@dataclass
class EvalResults:
    """Aggregated evaluation results over multiple episodes."""
    episodes: List[EpisodeMetrics] = field(default_factory=list)

    def add_episode(self, metrics: EpisodeMetrics) -> None:
        self.episodes.append(metrics)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def mean(self, attr: str) -> float:
        values = [getattr(ep, attr) for ep in self.episodes]
        return float(np.mean(values)) if values else 0.0

    def std(self, attr: str) -> float:
        values = [getattr(ep, attr) for ep in self.episodes]
        return float(np.std(values)) if values else 0.0

    def summary(self) -> Dict[str, float]:
        """Return summary dictionary of key metrics."""
        return {
            "reward_mean": self.mean("total_reward"),
            "reward_std": self.std("total_reward"),
            "cost_mean": self.mean("total_cost"),
            "cost_std": self.std("total_cost"),
            "violation_mean": self.mean("total_constraint_violation"),
            "violation_rate": self.mean("constraint_violation_count")
                             / max(self.mean("num_steps"), 1),
            "fcsd_mean": self.mean("mean_fcsd"),
            "fcsd_min_mean": self.mean("min_fcsd"),
            "renewable_curtailment_mean": self.mean("renewable_curtailment_kwh"),
            "num_episodes": self.num_episodes,
        }


def compute_fcsd(constraint_values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Compute Fuzzy Constraint Satisfaction Degree.

    Uses a sigmoidal fuzzy membership function around the threshold.

    Args:
        constraint_values: Array of constraint values (e.g., violation magnitude).
        thresholds: Array of constraint thresholds (hard constraint boundary).

    Returns:
        FCSD values in [0, 1].
    """
    # Smooth transition around threshold
    k = 10.0  # steepness
    fcsd = 1.0 / (1.0 + np.exp(k * (constraint_values - thresholds)))
    return np.clip(fcsd, 0.0, 1.0)
