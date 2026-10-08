"""Tests for microgrid environment module."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hfg_srl.env import make_env, FuzzyConstraint, FuzzyConstraintSet
from hfg_srl.utils.config import EnvConfig


class TestFuzzyConstraint:
    """Tests for fuzzy constraint classes."""

    def test_fcsd_upper_constraint(self):
        """Upper constraint: value below threshold should have high FCSD."""
        fc = FuzzyConstraint(name="test", threshold=0.8, width=0.1, direction="upper")
        # Well below threshold -> fully satisfied
        assert fc.fcsd(0.5) > 0.99
        # Well above threshold -> fully violated
        assert fc.fcsd(1.0) < 0.01
        # At threshold -> ~0.5
        assert 0.4 < fc.fcsd(0.8) < 0.6

    def test_fcsd_lower_constraint(self):
        """Lower constraint: value above threshold should have high FCSD."""
        fc = FuzzyConstraint(name="test", threshold=0.2, width=0.05, direction="lower")
        # Well above threshold -> fully satisfied
        assert fc.fcsd(0.5) > 0.99
        # Well below threshold -> fully violated
        assert fc.fcsd(0.0) < 0.01

    def test_fcsd_range(self):
        """FCSD should always be in [0, 1]."""
        fc = FuzzyConstraint(name="test", threshold=0.5, width=0.1, direction="upper")
        for val in np.linspace(-10, 10, 100):
            fcsd = fc.fcsd(val)
            assert 0.0 <= fcsd <= 1.0

    def test_constraint_set_aggregation(self):
        """Constraint set aggregation should work correctly."""
        fcs = FuzzyConstraintSet(
            constraints=[
                FuzzyConstraint("a", 0.8, 0.1, "upper", weight=1.0),
                FuzzyConstraint("b", 0.2, 0.05, "lower", weight=1.0),
            ],
            agg_method="weighted_average",
        )
        # Both satisfied -> high aggregate
        agg = fcs.aggregate_fcsd([0.5, 0.5])  # a: high FCSD, b: high FCSD
        assert agg > 0.9

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            FuzzyConstraint(name="test", threshold=0.5, width=0.1, direction="invalid")


class TestGridConnectedEnv:
    """Tests for grid-connected microgrid environment."""

    def _make_env(self):
        cfg = EnvConfig(num_days=2, time_steps_per_day=24)
        return make_env("grid_connected", cfg)

    def test_reset(self):
        env = self._make_env()
        env.seed(42)
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space_shape
        assert "fcsd" in info
        assert "soc" in info

    def test_step(self):
        env = self._make_env()
        env.seed(42)
        obs, _ = env.reset(seed=42)
        action = np.zeros(env.action_space_shape, dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert next_obs.shape == obs.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "cost" in info
        assert "fcsd" in info
        assert 0.0 <= info["fcsd"] <= 1.0

    def test_action_bounds(self):
        env = self._make_env()
        env.seed(42)
        env.reset(seed=42)

        # Test with maximum action
        max_action = env.action_high
        next_obs, reward, _, _, info = env.step(max_action)
        assert not np.isnan(reward)
        assert 0.0 <= info.get("soc", 0.0) <= 1.0

        # Test with minimum action
        min_action = env.action_low
        next_obs, reward, _, _, info = env.step(min_action)
        assert not np.isnan(reward)

    def test_episode_truncation(self):
        """Environment should truncate after max_steps."""
        cfg = EnvConfig(num_days=1, time_steps_per_day=10)
        env = make_env("grid_connected", cfg)
        env.seed(0)
        obs, _ = env.reset(seed=0)
        done = False
        steps = 0
        while not done:
            action = np.zeros(env.action_space_shape, dtype=np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            if steps > 20:
                break
        assert steps <= 11  # 10 steps + last step


class TestIslandedEnv:
    """Tests for islanded microgrid environment."""

    def _make_env(self):
        cfg = EnvConfig(num_days=1, time_steps_per_day=24)
        return make_env("islanded", cfg)

    def test_reset(self):
        env = self._make_env()
        env.seed(42)
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space_shape
        assert "fcsd" in info

    def test_step(self):
        env = self._make_env()
        env.seed(42)
        obs, _ = env.reset(seed=42)
        action = np.zeros(env.action_space_shape, dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == obs.shape
        assert "freq_deviation" in info
        assert "voltage_deviation" in info
        assert 0.0 <= info["fcsd"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
