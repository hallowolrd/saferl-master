"""Tests for M1/M2 integration helpers in experiment_runner.

Focus: build_constraint_specs() — pure function that extracts membership
parameters from two EnvConfig objects and formats them for similarity.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hfg_srl.utils.config import EnvConfig  # noqa: E402
from run.experiment_runner import build_constraint_specs  # noqa: E402


def _make_env_cfg(mode: str = "grid_connected",
                 soc_optimal_min: float = 0.3,
                 soc_optimal_max: float = 0.8,
                 frequency_limit_hz: float = 0.5) -> EnvConfig:
    return EnvConfig(
        mode=mode,
        soc_min=0.1, soc_max=0.9,
        soc_optimal_min=soc_optimal_min,
        soc_optimal_max=soc_optimal_max,
        frequency_limit_hz=frequency_limit_hz,
        voltage_limit_pu=1.05,
    )


def test_build_specs_returns_list_of_dicts():
    """Should return a list of spec dicts with required keys."""
    src = _make_env_cfg("grid_connected")
    tgt = _make_env_cfg("grid_connected")
    specs = build_constraint_specs(src, tgt)
    assert isinstance(specs, list)
    assert len(specs) > 0
    for spec in specs:
        assert "name" in spec
        assert "source_mu" in spec and "source_beta" in spec
        assert "target_mu" in spec and "target_beta" in spec
        assert "x_range" in spec


def test_build_specs_soc_constraints_present():
    """SOC upper and lower band constraints should be in the specs."""
    src = _make_env_cfg("grid_connected")
    tgt = _make_env_cfg("grid_connected")
    specs = build_constraint_specs(src, tgt)
    names = [s["name"] for s in specs]
    assert "soc_upper" in names
    assert "soc_lower" in names


def test_build_specs_island_mode_includes_freq():
    """Islanded mode should include frequency deviation constraint."""
    src = _make_env_cfg("islanded", frequency_limit_hz=0.5)
    tgt = _make_env_cfg("islanded", frequency_limit_hz=0.5)
    specs = build_constraint_specs(src, tgt)
    names = [s["name"] for s in specs]
    assert "freq_dev" in names


def test_build_specs_grid_mode_no_freq():
    """Grid-connected mode should NOT include frequency constraint."""
    src = _make_env_cfg("grid_connected")
    tgt = _make_env_cfg("grid_connected")
    specs = build_constraint_specs(src, tgt)
    names = [s["name"] for s in specs]
    assert "freq_dev" not in names


def test_build_specs_soc_mu_centered_on_optimal_band():
    """Membership center for SOC upper should be at soc_optimal_max."""
    src = _make_env_cfg("grid_connected", soc_optimal_max=0.8)
    tgt = _make_env_cfg("grid_connected", soc_optimal_max=0.85)
    specs = build_constraint_specs(src, tgt)
    soc_upper = next(s for s in specs if s["name"] == "soc_upper")
    assert soc_upper["source_mu"] == pytest.approx(0.8)
    assert soc_upper["target_mu"] == pytest.approx(0.85)


def test_build_specs_soc_beta_proportional_to_band_width():
    """Membership width should reflect the distance from optimal to hard bound."""
    src = _make_env_cfg("grid_connected", soc_optimal_min=0.3, soc_optimal_max=0.8)
    tgt = _make_env_cfg("grid_connected")
    specs = build_constraint_specs(src, tgt)
    soc_upper = next(s for s in specs if s["name"] == "soc_upper")
    # beta = soc_max - soc_optimal_max = 0.9 - 0.8 = 0.1
    assert soc_upper["source_beta"] == pytest.approx(0.1)


def test_build_specs_x_range_covers_soc_band():
    """SOC integration range should cover [0, 1]."""
    src = _make_env_cfg("grid_connected")
    tgt = _make_env_cfg("grid_connected")
    specs = build_constraint_specs(src, tgt)
    soc_upper = next(s for s in specs if s["name"] == "soc_upper")
    assert soc_upper["x_range"][0] <= 0.0
    assert soc_upper["x_range"][1] >= 1.0
