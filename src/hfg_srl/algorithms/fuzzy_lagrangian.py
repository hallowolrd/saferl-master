"""Fuzzy-Lagrangian constraint satisfaction mechanism.

Extends the standard Lagrangian method for constrained RL by:
1. Using Fuzzy Constraint Satisfaction Degree (FCSD) instead of binary violation
2. Adaptive lambda adjustment based on fuzzy constraint level
3. Smoother constraint handling that avoids the on/off penalty behavior

The core idea: instead of penalizing only when constraint > threshold,
we apply a continuous penalty proportional to (1 - FCSD), creating
a smooth gradient signal that guides the policy toward safer behavior
even before the hard constraint is violated.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from ..utils.config import FuzzyConfig

logger = logging.getLogger(__name__)


@dataclass
class FuzzyLagrangianState:
    """Current state of the fuzzy Lagrangian mechanism."""
    lambda_val: float
    fcsd_ma: float  # moving average of FCSD
    target_violation: float


class FuzzyLagrangian(nn.Module):
    """Fuzzy-Lagrangian constraint satisfaction module.

    Computes the constraint penalty term in the Lagrangian:
        L = J_reward + lambda * (target - FCSD)

    Where FCSD (Fuzzy Constraint Satisfaction Degree) is in [0, 1],
    with 1 = fully satisfied, 0 = fully violated.

    Lambda is updated via gradient ascent on the constraint objective:
        lambda <- lambda + lr_lambda * (target_fcsd - current_fcsd)

    This is the fuzzy version of the standard Lagrangian method:
    - Standard: penalty when cost > threshold (binary)
    - Fuzzy: penalty proportional to (1 - FCSD) (continuous)
    """

    def __init__(self, cfg: FuzzyConfig):
        super().__init__()
        self.cfg = cfg

        # Lagrange multiplier (learnable)
        self.log_lambda = nn.Parameter(
            torch.tensor(cfg.lambda_init).log()
        )

        # Moving average of FCSD (for lambda adjustment)
        self.register_buffer(
            "fcsd_ma",
            torch.tensor(1.0, dtype=torch.float32),
        )
        self.ma_alpha = 0.95  # moving average smoothing

        # Log-space is used so the multiplier can span cost-unit magnitudes
        # (1e0 .. 1e5) without applying an additive step-size that is only
        # meaningful at a single scale.

    @property
    def lambda_val(self) -> torch.Tensor:
        """Get current lambda value (constrained to positive)."""
        lam = torch.exp(self.log_lambda)
        lam = torch.clamp(lam, self.cfg.lambda_min, self.cfg.lambda_max)
        return lam

    def compute_penalty(self, fcsd: torch.Tensor) -> torch.Tensor:
        """Compute fuzzy constraint penalty.

        Args:
            fcsd: Fuzzy Constraint Satisfaction Degree (batch_size,).

        Returns:
            Penalty term (batch_size,).
        """
        # Penalty = lambda * max(0, target - fcsd)
        # When fcsd < target: positive penalty
        # When fcsd >= target: zero penalty (constraint satisfied)
        violation = torch.clamp(self.cfg.fcsd_target - fcsd, min=0.0)
        penalty = self.lambda_val * violation
        return penalty

    def update_lambda(self, fcsd_batch: torch.Tensor) -> float:
        """Update Lagrange multiplier based on FCSD.

        Standard Lagrangian update: lambda += lr * (target - actual)

        Args:
            fcsd_batch: Batch of FCSD values.

        Returns:
            New lambda value (float).
        """
        fcsd_mean = fcsd_batch.mean().detach()

        # Update moving average
        self.fcsd_ma = self.ma_alpha * self.fcsd_ma + (1 - self.ma_alpha) * fcsd_mean

        # Gradient ascent on lambda (maximize constraint cost = push toward satisfaction)
        # dL/d(lambda) = max(0, target - fcsd). Multiplicative (log-space) ascent
        # keeps the update scale-free across the cost-unit magnitude of lambda.
        violation = max(0.0, self.cfg.fcsd_target - self.fcsd_ma.item())

        with torch.no_grad():
            current_lam = self.lambda_val.item()
            fcsd_ma_val = self.fcsd_ma.item()
            target = self.cfg.fcsd_target
            # Deadband: when FCSD is within [target - deadband, target + deadband],
            # freeze lambda. Without this, on hard-constraint scenarios (e.g. S4
            # islanded extreme) lambda monotonically climbs to lambda_max within a
            # few episodes; the inflated multiplier then amplifies cost-critic
            # Bellman error into a noisy actor gradient, causing late-stage
            # policy degradation (violation rate drifting from ~37% to ~55%).
            deadband = 0.05
            if fcsd_ma_val < target - deadband:
                new_lam = current_lam * math.exp(
                    self.cfg.lambda_lr * (target - fcsd_ma_val)
                )
            elif fcsd_ma_val > target + deadband:
                margin = fcsd_ma_val - target
                new_lam = current_lam * math.exp(
                    -self.cfg.lambda_lr * 0.5 * margin
                )
            else:
                new_lam = current_lam

            new_lam = max(self.cfg.lambda_min, min(self.cfg.lambda_max, new_lam))
            self.log_lambda.fill_(math.log(new_lam))

        return self.lambda_val.item()

    def get_state(self) -> FuzzyLagrangianState:
        """Get current state for logging."""
        return FuzzyLagrangianState(
            lambda_val=self.lambda_val.item(),
            fcsd_ma=float(self.fcsd_ma.item()),
            target_violation=max(0.0, self.cfg.fcsd_target - self.fcsd_ma.item()),
        )

    def lagrangian_objective(
        self,
        reward: torch.Tensor,
        fcsd: torch.Tensor,
    ) -> torch.Tensor:
        """Compute full Lagrangian objective.

        L = E[reward] - lambda * max(0, target - E[fcsd])

        Note: We subtract the penalty (want to maximize reward, minimize violation).

        Args:
            reward: Reward values (batch_size,).
            fcsd: FCSD values (batch_size,).

        Returns:
            Lagrangian objective (scalar).
        """
        avg_reward = reward.mean()
        penalty = self.compute_penalty(fcsd.mean().unsqueeze(0))
        return avg_reward - penalty.mean()
