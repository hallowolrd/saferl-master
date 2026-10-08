"""Tests for fuzzy system module."""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hfg_srl.fuzzy_system import TSKFuzzySystem, FuzzyRewardShaper, expert_rule_base
from hfg_srl.utils.config import FuzzyConfig


class TestTSKFuzzySystem:
    """Tests for TSK fuzzy system."""

    def test_forward_shape(self):
        """Output should have correct shape."""
        fsys = TSKFuzzySystem(num_inputs=4, num_outputs=1, num_rules=8, first_order=False)
        x = torch.randn(16, 4)
        out = fsys(x)
        assert out.shape == (16, 1)

    def test_first_order_forward(self):
        """First-order TSK should produce correct output shape."""
        fsys = TSKFuzzySystem(num_inputs=4, num_outputs=2, num_rules=8, first_order=True)
        x = torch.randn(16, 4)
        out = fsys(x)
        assert out.shape == (16, 2)

    def test_firing_strengths_normalize(self):
        """Normalized firing strengths should sum to ~1."""
        fsys = TSKFuzzySystem(num_inputs=3, num_outputs=1, num_rules=5)
        x = torch.randn(8, 3)
        _, fire_norm = fsys.firing_strengths(x)
        # Sum over rules should be ~1
        fire_sum = fire_norm.sum(dim=-1)
        assert torch.allclose(fire_sum, torch.ones_like(fire_sum), atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through the fuzzy system."""
        fsys = TSKFuzzySystem(num_inputs=3, num_outputs=1, num_rules=4, first_order=True)
        x = torch.randn(8, 3, requires_grad=True)
        out = fsys(x)
        loss = out.sum()
        loss.backward()
        # Check that gradients exist for input and params
        assert x.grad is not None
        assert fsys.membership.means.grad is not None
        assert fsys.consequent.grad is not None

    def test_zero_order_special_case(self):
        """Zero-order should be a subset of first-order behavior."""
        fsys0 = TSKFuzzySystem(num_inputs=2, num_outputs=1, num_rules=4, first_order=False)
        # Set all linear coefficients to zero, bias to constant
        fsys1 = TSKFuzzySystem(num_inputs=2, num_outputs=1, num_rules=4, first_order=True)

        # Copy constant parts
        with torch.no_grad():
            # Zero-order consequent shape: (R, O)
            # First-order: (R, O, I+1) -- last dim is linear + bias
            fsys1.consequent[:, :, :-1] = 0.0  # zero linear terms
            fsys1.consequent[:, :, -1] = fsys0.consequent  # same bias

        x = torch.randn(10, 2)
        out0 = fsys0(x)
        out1 = fsys1(x)
        # Should be approximately equal
        assert torch.allclose(out0, out1, atol=1e-5)


class TestFuzzyRewardShaper:
    """Tests for fuzzy reward shaper."""

    def _make_shaper(self):
        cfg = FuzzyConfig(
            num_rules=8,
            num_inputs=4,
            use_expert_rules=True,
            reward_shaping_weight_init=1.0,
            reward_shaping_weight_final=0.1,
            reward_shaping_decay_steps=1000,
        )
        rule_base = expert_rule_base()
        return FuzzyRewardShaper(cfg, num_inputs=4, rule_base=rule_base)

    def test_potential_shape(self):
        shaper = self._make_shaper()
        state = torch.randn(16, 4)
        phi = shaper.potential(state)
        assert phi.shape == (16, 1)

    def test_shaped_reward(self):
        shaper = self._make_shaper()
        state = torch.randn(8, 4)
        next_state = torch.randn(8, 4)
        reward = torch.randn(8, 1)
        shaped = shaper.shape_reward(reward, state, next_state, gamma=0.99)
        assert shaped.shape == reward.shape

    def test_weight_decay(self):
        """Shaping weight should decrease over steps."""
        cfg = FuzzyConfig(
            num_rules=4,
            num_inputs=2,
            use_expert_rules=False,
            reward_shaping_weight_init=1.0,
            reward_shaping_weight_final=0.0,
            reward_shaping_decay_steps=100,
        )
        shaper = FuzzyRewardShaper(cfg, num_inputs=2, rule_base=None)
        initial_w = shaper.current_weight

        for _ in range(50):
            shaper.step()
        mid_w = shaper.current_weight

        for _ in range(50):
            shaper.step()
        final_w = shaper.current_weight

        assert initial_w >= mid_w >= final_w
        assert final_w <= 0.05  # close to zero after full decay


class TestExpertRuleBase:
    """Tests for expert rule base."""

    def test_rule_base_creation(self):
        rb = expert_rule_base()
        assert len(rb) > 0
        assert len(rb.input_sets) > 0
        assert len(rb.output_names) > 0

    def test_rule_base_to_tsk_params(self):
        rb = expert_rule_base()
        input_names = ["soc", "pv_normalized", "load_normalized", "time_of_day"]
        means, stds, conseq = rb.to_tsk_params(input_names)

        assert means.shape[0] == len(rb)
        assert means.shape[1] == len(input_names)
        assert conseq.shape[0] == len(rb)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
