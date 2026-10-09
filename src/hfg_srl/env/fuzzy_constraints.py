"""Fuzzy constraint definitions for microgrid safety.

Provides:
- FuzzyConstraint: single fuzzy constraint with membership function
- FuzzyConstraintSet: collection of fuzzy constraints with aggregation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class FuzzyConstraint:
    """A single fuzzy safety constraint.

    Defines a constraint with a fuzzy boundary using a membership function.
    FCSD (Fuzzy Constraint Satisfaction Degree) = 1 means fully satisfied,
    0 means fully violated.

    Attributes:
        name: Constraint identifier.
        threshold: Hard constraint threshold (where FCSD = 0.5 by default).
        width: Fuzzy transition zone width (controls softness).
        direction: 'upper' (value < threshold is safe) or 'lower' (value > threshold is safe).
        weight: Relative importance weight for aggregation.
    """

    name: str
    threshold: float
    width: float
    direction: str = "upper"  # upper | lower
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.direction not in ("upper", "lower"):
            raise ValueError(f"Invalid direction: {self.direction}")
        if self.width <= 0:
            raise ValueError("width must be positive")

    def fcsd(self, value: float) -> float:
        """Compute Fuzzy Constraint Satisfaction Degree.

        Args:
            value: Current constraint value.

        Returns:
            FCSD in [0, 1].
        """
        if self.direction == "upper":
            # value below threshold is safe
            x = (self.threshold - value) / self.width
        else:
            # value above threshold is safe
            x = (value - self.threshold) / self.width

        # Sigmoidal membership function (smooth transition)
        fcsd = 1.0 / (1.0 + np.exp(-5.0 * x))
        return float(np.clip(fcsd, 0.0, 1.0))

    def violation(self, value: float) -> float:
        """Compute soft constraint dissatisfaction magnitude (0 only when FCSD=1).

        Note: this is a soft *satisfaction shortfall* (1 - FCSD) used for
        reward shaping, NOT a hard bound crossing. It is positive for nearly
        every state inside the safe region. For a binary/hard violation flag
        use :meth:`is_violated` or :meth:`hard_violation`.
        """
        return max(0.0, 1.0 - self.fcsd(value))

    def hard_violation(self, value: float) -> float:
        """Hard constraint violation magnitude (0 throughout the safe region).

        Returns the amount by which ``value`` crosses the hard threshold, and
        exactly 0 for any value within bounds.
        """
        if self.direction == "upper":
            return max(0.0, value - self.threshold)
        return max(0.0, self.threshold - value)

    def is_violated(self, value: float) -> bool:
        """Return True when the value crosses the hard safety threshold."""
        if self.direction == "upper":
            return value > self.threshold
        return value < self.threshold

    def __call__(self, value: float) -> float:
        return self.fcsd(value)


class FuzzyConstraintSet:
    """A set of fuzzy constraints with aggregation.

    Aggregates multiple FCSD values into an overall safety score
    using weighted averaging (default) or other operators.
    """

    def __init__(
        self,
        constraints: Optional[List[FuzzyConstraint]] = None,
        agg_method: str = "weighted_average",
    ):
        self.constraints: List[FuzzyConstraint] = constraints or []
        self.agg_method = agg_method  # weighted_average | min | product
        # Pre-compute arrays for vectorized compute_all.
        n = len(self.constraints)
        self._thresholds = np.array([c.threshold for c in self.constraints], dtype=np.float64)
        self._widths = np.array([c.width for c in self.constraints], dtype=np.float64)
        self._directions = np.array([1.0 if c.direction == "upper" else -1.0 for c in self.constraints])
        self._weights = np.array([c.weight for c in self.constraints], dtype=np.float64)
        self._weights_norm = self._weights / self._weights.sum()
        self._n = n

    def add(self, constraint: FuzzyConstraint) -> None:
        """Add a fuzzy constraint to the set."""
        self.constraints.append(constraint)

    def fcsd_vector(self, values: List[float]) -> np.ndarray:
        """Compute per-constraint FCSD values.

        Args:
            values: List of constraint values (one per constraint).

        Returns:
            Array of FCSD values.
        """
        if len(values) != len(self.constraints):
            raise ValueError(
                f"Expected {len(self.constraints)} values, got {len(values)}"
            )
        return np.array([c.fcsd(v) for c, v in zip(self.constraints, values)])

    def aggregate_fcsd(self, values: List[float]) -> float:
        """Compute aggregated FCSD across all constraints.

        Args:
            values: List of constraint values.

        Returns:
            Aggregated FCSD in [0, 1].
        """
        fcsd_vals = self.fcsd_vector(values)
        weights = np.array([c.weight for c in self.constraints])
        weights = weights / weights.sum()  # normalize

        if self.agg_method == "weighted_average":
            return float(np.dot(fcsd_vals, weights))
        elif self.agg_method == "min":
            return float(np.min(fcsd_vals))
        elif self.agg_method == "product":
            return float(np.prod(fcsd_vals ** weights))
        else:
            raise ValueError(f"Unknown aggregation method: {self.agg_method}")

    def total_violation(self, values: List[float]) -> float:
        """Compute total weighted soft violation (1 - FCSD)."""
        violations = np.array([c.violation(v) for c, v in zip(self.constraints, values)])
        weights = np.array([c.weight for c in self.constraints])
        return float(np.dot(violations, weights))

    def violation_mask(self, values: List[float]) -> np.ndarray:
        """Boolean mask of constraints whose hard threshold is crossed."""
        if len(values) != len(self.constraints):
            raise ValueError(
                f"Expected {len(self.constraints)} values, got {len(values)}"
            )
        return np.array(
            [c.is_violated(v) for c, v in zip(self.constraints, values)],
            dtype=bool,
        )

    def num_violated(self, values: List[float]) -> int:
        """Number of constraints whose hard threshold is crossed."""
        return int(np.sum(self.violation_mask(values)))

    def max_hard_violation(self, values: List[float]) -> float:
        """Largest hard violation magnitude across all constraints."""
        if len(values) != len(self.constraints):
            raise ValueError(
                f"Expected {len(self.constraints)} values, got {len(values)}"
            )
        return float(np.max([c.hard_violation(v) for c, v in zip(self.constraints, values)]))

    def __len__(self) -> int:
        return len(self.constraints)

    def __getitem__(self, idx: int) -> FuzzyConstraint:
        return self.constraints[idx]

    def compute_all(
        self, values: List[float]
    ) -> Tuple[float, float, float, bool, np.ndarray]:
        """Vectorized single-pass computation of all per-step safety metrics.

        Returns:
            (fcsd_weighted_avg, fcsd_min, weighted_violation,
             any_hard_violated, fcsd_vector)
        """
        if len(values) != self._n:
            raise ValueError(
                f"Expected {self._n} values, got {len(values)}"
            )
        v = np.asarray(values, dtype=np.float64)

        # Vectorized sigmoid: fcsd = 1 / (1 + exp(-5*x))
        # where x = (threshold - v) / width for upper, (v - threshold) / width for lower.
        # Using direction multiplier: sign=+1 for upper, -1 for lower.
        x = self._directions * (self._thresholds - v) / self._widths
        fcsd_vec = 1.0 / (1.0 + np.exp(-5.0 * x))
        np.clip(fcsd_vec, 0.0, 1.0, out=fcsd_vec)

        # Hard violation: upper -> v > threshold, lower -> v < threshold.
        hard_mask = (self._directions * (v - self._thresholds)) > 0

        if self.agg_method == "weighted_average":
            fcsd_avg = float(np.dot(fcsd_vec, self._weights_norm))
        elif self.agg_method == "min":
            fcsd_avg = float(np.min(fcsd_vec))
        elif self.agg_method == "product":
            fcsd_avg = float(np.prod(fcsd_vec ** self._weights_norm))
        else:
            raise ValueError(f"Unknown aggregation method: {self.agg_method}")

        fcsd_min = float(np.min(fcsd_vec))
        violation = float(np.dot(1.0 - fcsd_vec, self._weights_norm))
        any_hard = bool(hard_mask.any())
        return fcsd_avg, fcsd_min, violation, any_hard, fcsd_vec
