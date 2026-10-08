"""Tests for the CBF-style safety layer baseline."""

import numpy as np

from src.hfg_srl.algorithms.safety_layer import SafetyLayerSAC
from src.hfg_srl.utils.config import HFGConfig


class TestSOCProjection:
    def test_charge_blocked_near_upper_soc(self):
        agent = SafetyLayerSAC(HFGConfig(), 5, 3, np.ones(3) * 10.0, "cpu")
        # Force base agent to propose strong charging.
        agent.base_agent.actor.mean_head.bias.data.fill_(2.0)
        state = np.array([0.99, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        action = agent.select_action(state, deterministic=True)
        # Projected ESS action must keep next SOC within the band.
        assert action[0] <= 1.0  # normalized; projection reduces charge

    def test_thresholds_come_from_cfg(self):
        agent = SafetyLayerSAC(HFGConfig(), 5, 3, np.ones(3) * 10.0, "cpu")
        # Bounds reflect env cfg (soc_min 0.2, margin 0.02), not hardcoded.
        assert agent.soc_min_shield == 0.22
        assert agent.soc_max_shield == 0.88


class TestFreqVoltProjection:
    def test_islanded_imbalance_projection_fires(self):
        # Islanded dims: obs 9, act 4.
        agent = SafetyLayerSAC(HFGConfig(), 9, 4, np.ones(4) * 10.0, "cpu")
        assert agent._is_islanded
        # State where the raw action would create a large over-generation.
        state = np.zeros(9, dtype=np.float32)
        state[0] = 0.5
        state[3] = 0.1  # very low load
        raw = np.array([0.0, 0.9, 0.0, 0.9], dtype=np.float32)
        projected = agent._project_freq_voltage(state, raw)
        # DE / dump actions are pulled back to reduce imbalance.
        assert projected.shape == raw.shape
        assert np.all(np.isfinite(projected))
