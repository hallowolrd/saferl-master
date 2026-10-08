"""Actor networks for SAC-style algorithms."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "relu",
) -> nn.Sequential:
    """Build a simple MLP."""
    layers = []
    prev_dim = input_dim
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]

    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(act_fn())
        prev_dim = h

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianActor(nn.Module):
    """Stochastic actor with squashed Gaussian policy (SAC-style).

    Outputs mean and log-std of a Gaussian, then applies tanh squashing
    to map actions to [-1, 1].
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Shared trunk
        self.trunk = _build_mlp(obs_dim, hidden_dims[:-1], hidden_dims[-1], activation)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], act_dim)

        # Initialize output layers with small weights
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action distribution parameters.

        Args:
            obs: Observation tensor (batch_size, obs_dim).

        Returns:
            Tuple of (mean, log_std).
        """
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Args:
            obs: Observation tensor.
            deterministic: If True, return mean (no sampling).

        Returns:
            Tuple of (action, log_prob, mean_action).
            Action is in [-1, 1] due to tanh squashing.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        if deterministic:
            x_t = mean
        else:
            x_t = dist.rsample()  # reparameterization trick

        y_t = torch.tanh(x_t)  # squashed to [-1, 1]

        # Log prob with tanh correction
        log_prob = dist.log_prob(x_t)
        # Enforce action bounds correction (SAC appendix C)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)

        return y_t, log_prob, mean_action


class DeterministicActor(nn.Module):
    """Deterministic actor (DDPG/TD3-style)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dims, act_dim, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic action in [-1, 1]."""
        return torch.tanh(self.net(obs))
