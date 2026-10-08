"""M1 + M2: Fuzzy-rule similarity and membership-parameter transfer.

Implements the mechanism described in Chapter 4.4 of the HFG-SRL paper:

- **M1 — Jaccard fuzzy-set similarity** (Sec 4.4.1):
  sim_k = ∫ min(μ_k^s, μ_k^t) dx / ∫ max(μ_k^s, μ_k^t) dx
  Used to categorize each constraint k into transferable / adaptable / relearn.

- **M2 — Membership parameter transfer** (Sec 4.4.2):
  Transferable: β_k^t = β_k^s,  x_ref_k^t = x_ref_k^s
  Adaptable:    β_k^t = β_k^s · Δ_k,  x_ref_k^t = x_ref_k^s + δ_k
  Relearn:      β_k^t = β_default,  x_ref_k^t = x_ref_default

The implementation is deliberately numeric (fixed-grid quadrature) so it
works with any 1-D membership function supplied as a callable, not just
Gaussians.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Primitive membership function
# ---------------------------------------------------------------------------

def gaussian_membership(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """1-D Gaussian membership function μ(x) = exp(-0.5·((x-μ)/σ)²).

    Args:
        x: Scalar or array.
        mean: Center of the Gaussian (x_ref).
        sigma: Width (related to β boundary width parameter).

    Returns:
        Membership degree in [0, 1].
    """
    x_arr = np.asarray(x, dtype=float)
    sigma = max(sigma, 1e-6)
    return np.exp(-0.5 * ((x_arr - mean) / sigma) ** 2)


# ---------------------------------------------------------------------------
# M1: Jaccard similarity
# ---------------------------------------------------------------------------

def jaccard_similarity(
    mu_source: Callable[[np.ndarray], np.ndarray],
    mu_target: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float] = (-5.0, 5.0),
    n_samples: int = 2001,
) -> float:
    """Numerically compute the Jaccard index of two 1-D fuzzy sets.

        sim = ∫ min(μ_s, μ_t) dx / ∫ max(μ_s, μ_t) dx

    Uses mid-point Riemann quadrature on a uniform grid.

    Args:
        mu_source: Membership function of the source scenario.
        mu_target: Membership function of the target scenario.
        x_range: (x_min, x_max) integration bounds.
        n_samples: Number of grid points (more = more accurate).

    Returns:
        Jaccard index in [0, 1]. Returns 0.0 if the max integral is ~0.
    """
    x = np.linspace(x_range[0], x_range[1], n_samples)
    dx = (x_range[1] - x_range[0]) / (n_samples - 1)

    ms = np.asarray(mu_source(x), dtype=float)
    mt = np.asarray(mu_target(x), dtype=float)
    ms = np.clip(ms, 0.0, 1.0)
    mt = np.clip(mt, 0.0, 1.0)

    min_int = float(np.sum(np.minimum(ms, mt)) * dx)
    max_int = float(np.sum(np.maximum(ms, mt)) * dx)

    if max_int < 1e-9:
        return 0.0
    return min(1.0, max(0.0, min_int / max_int))


# ---------------------------------------------------------------------------
# M1: Categorize constraints
# ---------------------------------------------------------------------------

def categorize_constraints(
    similarities: Dict[str, float],
    tau_high: float = 0.8,
    tau_low: float = 0.3,
) -> Dict[str, str]:
    """Categorize each constraint based on its Jaccard similarity score.

    Args:
        similarities: Mapping constraint_name -> sim_k.
        tau_high: Threshold above which constraint is directly transferable.
        tau_low: Threshold below which constraint must be relearned.

    Returns:
        Mapping constraint_name -> one of {"transferable", "adaptable", "relearn"}.
    """
    out: Dict[str, str] = {}
    for name, sim in similarities.items():
        if sim >= tau_high:
            out[name] = "transferable"
        elif sim >= tau_low:
            out[name] = "adaptable"
        else:
            out[name] = "relearn"
    return out


# ---------------------------------------------------------------------------
# M2: Membership parameter transfer
# ---------------------------------------------------------------------------

def transfer_membership_params(
    source_mu: float,
    source_beta: float,
    category: str,
    delta: float = 0.0,
    Delta: float = 1.0,
    default_mu: float = 0.5,
    default_beta: float = 1.0,
) -> Dict[str, float]:
    """Compute the target-scenario membership parameters per M2.

    Args:
        source_mu: Source center x_ref^s.
        source_beta: Source boundary width β^s.
        category: One of {"transferable", "adaptable", "relearn"}.
        delta: Adaptable shift  δ_k (default 0).
        Delta: Adaptable scale  Δ_k (default 1).
        default_mu: Expert-default center for relearn constraints.
        default_beta: Expert-default width for relearn constraints.

    Returns:
        Dict with keys "x_ref" and "beta".
    """
    if category == "transferable":
        return {"x_ref": float(source_mu), "beta": float(source_beta)}
    if category == "adaptable":
        return {"x_ref": float(source_mu) + float(delta),
                "beta": float(source_beta) * float(Delta)}
    if category == "relearn":
        return {"x_ref": float(default_mu), "beta": float(default_beta)}
    raise ValueError(f"Unknown category: {category!r}")
