"""Tests for M4: exponential safety relaxation in TransferAgent.

M4 spec (paper 4.4.2):
    alpha_cur(t) = alpha_target + (alpha_init - alpha_target) * exp(-nu * t)

The conservative factor starts high (= alpha_init / alpha_target, conservative)
and decays exponentially toward 1.0 (= alpha_target, no conservative scaling).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hfg_srl.transfer.transfer_agent import TransferAgent  # noqa: E402


def _make_agent(alpha_init: float = 1.5, alpha_target: float = 1.0,
               nu: float = 0.05) -> TransferAgent:
    """Build a minimal TransferAgent bypassing HFGConfig for unit testing."""
    agent = TransferAgent.__new__(TransferAgent)
    agent.conservative_factor = alpha_init  # starts at conservative
    agent.alpha_init = alpha_init
    agent.alpha_target = alpha_target
    agent.nu = nu
    agent._step_count = 0
    return agent


def test_exponential_relaxation_starts_conservative():
    """At t=0, conservative_factor should equal alpha_init (most conservative)."""
    agent = _make_agent(alpha_init=1.5, alpha_target=1.0, nu=0.05)
    assert agent.conservative_factor == pytest.approx(1.5)


def test_exponential_relaxation_converges_to_target():
    """After many steps, factor should approach alpha_target (= 1.0)."""
    agent = _make_agent(alpha_init=1.5, alpha_target=1.0, nu=0.05)
    for _ in range(200):
        agent.step()
    # exp(-0.05 * 200) = exp(-10) ≈ 4.5e-5 → factor ≈ 1.0
    assert agent.conservative_factor == pytest.approx(1.0, abs=0.01)


def test_exponential_relaxation_is_monotone_decreasing():
    """Factor must decrease monotonically (no oscillation)."""
    agent = _make_agent(alpha_init=2.0, alpha_target=1.0, nu=0.05)
    prev = agent.conservative_factor
    for _ in range(50):
        agent.step()
        assert agent.conservative_factor <= prev + 1e-9
        prev = agent.conservative_factor


def test_exponential_relaxation_matches_closed_form():
    """alpha_cur(t) = alpha_target + (alpha_init - alpha_target) * exp(-nu * t)."""
    agent = _make_agent(alpha_init=1.5, alpha_target=1.0, nu=0.1)
    for t in range(1, 11):
        agent.step()
        expected = 1.0 + (1.5 - 1.0) * math.exp(-0.1 * t)
        assert agent.conservative_factor == pytest.approx(expected, abs=1e-6)
