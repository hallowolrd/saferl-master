"""Differentiable TSK (Takagi-Sugeno-Kang) Fuzzy System.

Implements a zero-order and first-order TSK fuzzy system using
PyTorch for end-to-end differentiability. Can be used as:
- Fuzzy constraint satisfaction estimator
- Fuzzy reward shaping function
- Fuzzy policy network component
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMembership(nn.Module):
    """Gaussian membership function layer.

    Computes membership degrees for each input dimension and each rule.
    Parameters (means and stds) are learnable.
    """

    def __init__(self, num_inputs: int, num_rules: int, init_std: float = 1.0):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_rules = num_rules

        # Learnable parameters: means and log-stds per rule per input
        # Shape: (num_rules, num_inputs)
        self.means = nn.Parameter(torch.randn(num_rules, num_inputs) * 0.5)
        self.log_stds = nn.Parameter(torch.zeros(num_rules, num_inputs) + math.log(init_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute membership degrees.

        Args:
            x: Input tensor of shape (batch_size, num_inputs).

        Returns:
            Membership degrees of shape (batch_size, num_rules, num_inputs).
        """
        # x: (B, I) -> (B, 1, I)
        x_expanded = x.unsqueeze(1)
        # means, log_stds: (R, I) -> (1, R, I)
        means = self.means.unsqueeze(0)
        stds = torch.exp(self.log_stds).unsqueeze(0)

        # Gaussian membership: exp(-0.5 * ((x - mu) / sigma)^2)
        diff = x_expanded - means
        membership = torch.exp(-0.5 * (diff / stds) ** 2)
        return membership


class TSKFuzzySystem(nn.Module):
    """First-order TSK fuzzy inference system.

    Architecture:
        Input -> Membership (Gaussian) -> Firing Strength ->
        Consequent (linear) -> Weighted Sum -> Output

    Zero-order TSK is a special case where consequent is constant
    (achieved by setting first_order=False).
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_rules: int = 16,
        first_order: bool = True,
        init_std: float = 1.0,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_rules = num_rules
        self.first_order = first_order

        # Antecedent: membership functions
        self.membership = GaussianMembership(num_inputs, num_rules, init_std)

        # Consequent parameters
        if first_order:
            # Each rule has (num_inputs + 1) params per output (linear + bias)
            # Shape: (num_rules, num_outputs, num_inputs + 1)
            self.consequent = nn.Parameter(
                torch.randn(num_rules, num_outputs, num_inputs + 1) * 0.01
            )
        else:
            # Zero-order: each rule has a constant per output
            self.consequent = nn.Parameter(torch.zeros(num_rules, num_outputs))

    def firing_strengths(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rule firing strengths.

        Args:
            x: Input (batch_size, num_inputs).

        Returns:
            Tuple of (firing_strengths, normalized_firing_strengths),
            each of shape (batch_size, num_rules).
        """
        # Membership: (B, R, I)
        mem = self.membership(x)
        # Firing strength: product across input dimensions -> (B, R)
        fire = torch.prod(mem, dim=-1)
        # Normalize
        fire_sum = fire.sum(dim=-1, keepdim=True) + 1e-8
        fire_norm = fire / fire_sum
        return fire, fire_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TSK inference.

        Args:
            x: Input tensor (batch_size, num_inputs).

        Returns:
            Output tensor (batch_size, num_outputs).
        """
        batch_size = x.shape[0]
        _, fire_norm = self.firing_strengths(x)  # (B, R)

        if self.first_order:
            # Consequent: (R, O, I+1)
            # For each rule: y_r = sum(w_ri * x_i) + b_r
            # x with bias: (B, I+1)
            x_with_bias = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=-1)
            # (B, I+1) @ (R, O, I+1)^T  -> need to do per-rule
            # Expand for batch matmul:
            # fire_norm: (B, R, 1)
            # rule_outputs: (B, R, O)
            # x_with_bias: (B, 1, I+1)
            # consequent: (1, R, O, I+1) -> not directly compatible
            # Use einsum: b i, r o i -> b r o
            rule_outputs = torch.einsum("bi,roi->bro", x_with_bias, self.consequent)
        else:
            # Zero-order: consequent is (R, O), expand to (B, R, O)
            rule_outputs = self.consequent.unsqueeze(0).expand(batch_size, -1, -1)

        # Weighted sum: (B, R, 1) * (B, R, O) -> sum over R -> (B, O)
        output = (fire_norm.unsqueeze(-1) * rule_outputs).sum(dim=1)
        return output

    def get_rule_importance(self) -> torch.Tensor:
        """Get average rule importance (for interpretation).

        Returns:
            Importance scores of shape (num_rules,).
        """
        # Approximate by looking at parameter magnitudes
        if self.first_order:
            importance = self.consequent.abs().mean(dim=(1, 2))
        else:
            importance = self.consequent.abs().mean(dim=-1)
        return importance
