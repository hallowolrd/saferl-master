"""Replay buffer for off-policy RL algorithms.

Standard FIFO replay buffer that stores transitions directly as torch tensors
on the training device (avoids per-update numpy->tensor conversion overhead).
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Optional


class ReplayBuffer:
    """Standard FIFO replay buffer.

    Stores transitions as pre-allocated torch tensors on the target device.
    Sampling uses torch indexing (no numpy round-trip, no H2D transfer per update).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_size: int = int(1e5),
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.device = device

        # Pre-allocate storage directly on device (no numpy round-trip).
        self.observations = torch.zeros(max_size, obs_dim, dtype=torch.float32, device=device)
        self.actions = torch.zeros(max_size, act_dim, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.next_observations = torch.zeros(max_size, obs_dim, dtype=torch.float32, device=device)
        self.dones = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.fcsd = torch.ones(max_size, dtype=torch.float32, device=device)
        self.fcsd_min = torch.ones(max_size, dtype=torch.float32, device=device)
        self.constraint_costs = torch.zeros(max_size, dtype=torch.float32, device=device)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        fcsd: float = 1.0,
        fcsd_min: float = 1.0,
        constraint_cost: float = 0.0,
    ) -> None:
        """Add a transition to the buffer."""
        idx = self.ptr
        self.observations[idx] = torch.as_tensor(obs, dtype=torch.float32)
        self.actions[idx] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[idx] = reward
        self.next_observations[idx] = torch.as_tensor(next_obs, dtype=torch.float32)
        self.dones[idx] = float(done)
        self.fcsd[idx] = fcsd
        self.fcsd_min[idx] = fcsd_min
        self.constraint_costs[idx] = constraint_cost

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions.

        Uses torch indexing directly on device tensors — no numpy round-trip,
        no H2D transfer.  For GPU training this eliminates ~150KB H2D per update.
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
            "fcsd": self.fcsd[indices],
            "fcsd_min": self.fcsd_min[indices],
            "constraint_costs": self.constraint_costs[indices],
        }

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0
