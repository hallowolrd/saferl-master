"""Base class for RL agents."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..utils.config import AlgorithmConfig, HFGConfig

logger = logging.getLogger(__name__)

AGENT_FACTORY: Dict[str, type] = {}


def register_agent(name: str):
    """Decorator to register an agent class."""
    def decorator(cls: type) -> type:
        AGENT_FACTORY[name] = cls
        return cls
    return decorator


def make_agent(name: str, cfg: HFGConfig, obs_dim: int, act_dim: int,
               action_space_high: np.ndarray, device: str) -> "BaseAgent":
    """Create an agent instance by name.

    Args:
        name: Agent name in AGENT_FACTORY.
        cfg: Full configuration.
        obs_dim: Observation dimension.
        act_dim: Action dimension.
        action_space_high: Upper bound of action space.
        device: Computation device.

    Returns:
        Agent instance.
    """
    agent_cls = AGENT_FACTORY.get(name)
    if agent_cls is None:
        raise ValueError(
            f"Agent '{name}' not registered. "
            f"Available: {list(AGENT_FACTORY.keys())}"
        )
    return agent_cls(cfg, obs_dim, act_dim, action_space_high, device)


class BaseAgent(ABC):
    """Abstract base class for RL agents.

    Subclasses must implement:
        - select_action(state, deterministic)
        - update(batch, step)
        - save(path) / load(path)
    """

    def __init__(
        self,
        cfg: HFGConfig,
        obs_dim: int,
        act_dim: int,
        action_space_high: np.ndarray,
        device: str,
    ):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_space_high = torch.tensor(
            action_space_high, dtype=torch.float32, device=device
        )
        self.action_space_low = -self.action_space_high  # symmetric default
        self.device = device
        self.total_steps = 0

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select an action given a state.

        Args:
            state: Observation array.
            deterministic: If True, return mean action (no exploration).

        Returns:
            Action array.
        """
        ...

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        """Perform one gradient update step.

        Args:
            batch: Dictionary of tensors (observations, actions, rewards, etc.).
            step: Current training step.

        Returns:
            Dictionary of loss/metric values for logging.
        """
        ...

    def save(self, path: str) -> None:
        """Save model checkpoints.

        Args:
            path: Directory path to save into.
        """
        os.makedirs(path, exist_ok=True)

    def load(self, path: str) -> None:
        """Load model from checkpoint.

        Args:
            path: Directory path to load from.
        """
        pass

    def train(self) -> None:
        """Set agent to training mode."""
        pass

    def eval(self) -> None:
        """Set agent to evaluation mode."""
        pass
