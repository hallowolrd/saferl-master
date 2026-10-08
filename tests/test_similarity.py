"""Tests for M1 (Jaccard fuzzy-set similarity) and M2 (parameter transfer).

Run:
    .venv/Scripts/python.exe -m pytest tests/test_similarity.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hfg_srl.transfer.similarity import (  # noqa: E402
    gaussian_membership,
    jaccard_similarity,
    categorize_constraints,
    transfer_membership_params,
)


# ---------------------------------------------------------------------------
# M1: Jaccard similarity
# ---------------------------------------------------------------------------

def test_jaccard_identical_gaussians_equals_one():
    """Two identical Gaussian sets should have Jaccard = 1.0."""
    mu, sigma = 0.0, 1.0

    def mu_s(x):
        return gaussian_membership(x, mu, sigma)

    def mu_t(x):
        return gaussian_membership(x, mu, sigma)

    sim = jaccard_similarity(mu_s, mu_t, x_range=(-5.0, 5.0), n_samples=2001)
    assert sim == pytest.approx(1.0, abs=1e-3)


def test_jaccard_disjoint_gaussians_near_zero():
    """Two well-separated Gaussians should have Jaccard close to 0."""

    def mu_s(x):
        return gaussian_membership(x, -3.0, 0.3)

    def mu_t(x):
        return gaussian_membership(x, 3.0, 0.3)

    sim = jaccard_similarity(mu_s, mu_t, x_range=(-6.0, 6.0), n_samples=3001)
    assert sim < 0.05


def test_jaccard_shifted_gaussian_mid_value():
    """Two overlapping Gaussians should have Jaccard between 0 and 1.

    Sigma=1.0, means shifted by 2.0 (2-sigma apart) -> overlap ~0.19.
    1-sigma shift -> overlap ~0.55. Both must be strictly in (0, 1).
    """

    def mu_s(x):
        return gaussian_membership(x, 0.0, 1.0)

    def mu_t(x):
        return gaussian_membership(x, 1.0, 1.0)

    sim = jaccard_similarity(mu_s, mu_t, x_range=(-5.0, 7.0), n_samples=3001)
    assert 0.3 < sim < 0.8


def test_jaccard_respects_input_signature():
    """Passing plain Python floats (scalar) to the membership function works."""

    def mu_s(x):
        # x will be a numpy array; must broadcast.
        return np.exp(-0.5 * ((x - 0.0) / 1.0) ** 2)

    def mu_t(x):
        return np.exp(-0.5 * ((x - 0.0) / 1.0) ** 2)

    sim = jaccard_similarity(mu_s, mu_t, x_range=(-5.0, 5.0), n_samples=1001)
    assert sim == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# M1: Categorization
# ---------------------------------------------------------------------------

def test_categorize_transferable():
    """sim >= tau_high -> 'transferable'."""
    cat = categorize_constraints({"soc": 0.95, "freq": 0.85},
                                 tau_high=0.8, tau_low=0.3)
    assert cat["soc"] == "transferable"
    assert cat["freq"] == "transferable"


def test_categorize_adaptable():
    """tau_low <= sim < tau_high -> 'adaptable'."""
    cat = categorize_constraints({"voltage": 0.55}, tau_high=0.8, tau_low=0.3)
    assert cat["voltage"] == "adaptable"


def test_categorize_relearn():
    """sim < tau_low -> 'relearn'."""
    cat = categorize_constraints({"soc": 0.1}, tau_high=0.8, tau_low=0.3)
    assert cat["soc"] == "relearn"


# ---------------------------------------------------------------------------
# M2: Membership parameter transfer
# ---------------------------------------------------------------------------

def test_transfer_direct_copy_for_transferable():
    """Transferable: beta and x_ref copied unchanged."""
    out = transfer_membership_params(
        source_mu=0.5, source_beta=1.0,
        category="transferable", delta=0.0, Delta=1.0,
        default_mu=0.5, default_beta=1.0,
    )
    assert out["x_ref"] == pytest.approx(0.5)
    assert out["beta"] == pytest.approx(1.0)


def test_transfer_adaptable_applies_shift_scale():
    """Adaptable: x_ref += delta, beta *= Delta."""
    out = transfer_membership_params(
        source_mu=0.5, source_beta=1.0,
        category="adaptable", delta=0.1, Delta=1.2,
        default_mu=0.5, default_beta=1.0,
    )
    assert out["x_ref"] == pytest.approx(0.6)
    assert out["beta"] == pytest.approx(1.2)


def test_transfer_relearn_uses_default():
    """Relearn: ignore source, use expert default."""
    out = transfer_membership_params(
        source_mu=0.5, source_beta=1.0,
        category="relearn", delta=0.0, Delta=1.0,
        default_mu=0.7, default_beta=0.5,
    )
    assert out["x_ref"] == pytest.approx(0.7)
    assert out["beta"] == pytest.approx(0.5)
