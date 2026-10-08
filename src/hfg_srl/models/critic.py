"""Critic (Q-function / Value function) networks."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor import _build_mlp


class QNetwork(nn.Module):
    """Single Q-network: Q(s, a)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        super().__init__()
        input_dim = obs_dim + act_dim
        self.net = _build_mlp(input_dim, hidden_dims, 1, activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value.

        Args:
            obs: Observation (batch_size, obs_dim).
            action: Action (batch_size, act_dim).

        Returns:
            Q-value (batch_size, 1).
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class TwinQNetwork(nn.Module):
    """Twin Q-networks for clipped double Q-learning (SAC/TD3)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        super().__init__()
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dims, activation)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dims, activation)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both Q-values.

        Returns:
            Tuple of (q1, q2).
        """
        return self.q1(obs, action), self.q2(obs, action)

    def min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get minimum of the two Q-values (for conservative estimation)."""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


class ValueNetwork(nn.Module):
    """State value network V(s)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dims, 1, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
