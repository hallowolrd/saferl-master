"""Shared utilities for on-policy algorithms (PPO family, CPO)."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class RolloutStorage:
    """Per-episode on-policy rollout storage."""

    def __init__(self, with_cost: bool):
        self.with_cost = with_cost
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.costs: List[float] = []
        self.cost_values: List[float] = []
        # Bootstrap values at the final observation (0 if terminated).
        self.bootstrap_r: float = 0.0
        self.bootstrap_c: float = 0.0

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        cost: Optional[float] = None,
        cost_value: Optional[float] = None,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(float(done))
        if self.with_cost:
            self.costs.append(cost if cost is not None else 0.0)
            self.cost_values.append(cost_value if cost_value is not None else 0.0)

    def finish(self, bootstrap_r: float, bootstrap_c: float = 0.0) -> None:
        self.bootstrap_r = bootstrap_r
        self.bootstrap_c = bootstrap_c

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.__init__(self.with_cost)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
    bootstrap: float = 0.0,
) -> np.ndarray:
    """Generalized Advantage Estimation.

    ``bootstrap`` is V(s_last) when the episode was truncated (time-limit),
    and 0.0 when it truly terminated.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = bootstrap if t == n - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    return advantages


def normalize_advantage(adv: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Center advantages; optionally rescale to unit std.

    Cost advantages for Lagrangian methods must be centered only
    (normalize=False) to preserve the constraint scale.
    """
    centered = adv - adv.mean()
    if not normalize:
        return centered.astype(np.float32)
    return (centered / (adv.std() + 1e-8)).astype(np.float32)
