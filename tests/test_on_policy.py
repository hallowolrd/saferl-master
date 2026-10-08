"""Tests for on-policy baselines: PPO, PPO-Lagrangian, CPO."""

import numpy as np
import torch

from src.hfg_srl.algorithms.ppo import PPO
from src.hfg_srl.algorithms.ppo_lagrangian import PPOLagrangian
from src.hfg_srl.algorithms.cpo import (
    CPO,
    conjugate_gradient,
    cpo_direction,
)
from src.hfg_srl.utils.config import HFGConfig, PPOConfig


def make_cfg(ppo_cfg=None):
    return HFGConfig(ppo=ppo_cfg or PPOConfig())


class TestGAE:
    def test_bootstrap_value_used_when_truncated(self):
        # Rewards 0, values 0 everywhere, but truncated with a large
        # bootstrap value -> last advantage must reflect it.
        rewards = np.zeros(4, dtype=np.float32)
        values = np.zeros(4, dtype=np.float32)
        dones = np.zeros(4, dtype=np.float32)
        adv = PPO.compute_gae_static(
            rewards, values, dones, gamma=1.0, lam=1.0, bootstrap=10.0
        )
        assert adv[-1] == 10.0
        assert np.allclose(adv, 10.0)

    def test_no_bootstrap_when_terminated(self):
        rewards = np.zeros(4, dtype=np.float32)
        values = np.zeros(4, dtype=np.float32)
        dones = np.zeros(4, dtype=np.float32)
        dones[-1] = 1.0
        adv = PPO.compute_gae_static(
            rewards, values, dones, gamma=1.0, lam=1.0, bootstrap=10.0
        )
        assert adv[-1] == 0.0


class TestPPO:
    def test_update_runs_and_clears_buffer(self):
        cfg = make_cfg()
        agent = PPO(cfg, 5, 3, np.ones(3) * 10.0, "cpu")
        for _ in range(20):
            s = np.random.randn(5).astype(np.float32)
            agent.select_action(s)
            agent.store_reward_and_done(-1.0, False)
        agent.finish_rollout(last_obs=np.random.randn(5).astype(np.float32),
                             truncated=True)
        info = agent.update(None, 0)
        assert "policy_loss" in info
        assert len(agent.rollout) == 0


class TestPPOLagrangian:
    def test_cost_advantage_not_unit_normalized(self):
        # Constant cost advantage with large magnitude must keep its
        # scale (center only), not be divided down to ~1.
        costs = np.full(50, 5.0, dtype=np.float32)
        values = np.zeros(50, dtype=np.float32)
        dones = np.zeros(50, dtype=np.float32)
        adv = PPOLagrangian.compute_gae_static(
            costs, values, dones, 1.0, 1.0, bootstrap=0.0
        )
        adv_centered = adv - adv.mean()
        # After centering, std remains 0 here; use a non-constant case.
        costs2 = np.array([0.0] * 25 + [10.0] * 25, dtype=np.float32)
        adv2 = PPOLagrangian.compute_gae_static(
            costs2, values, dones, 1.0, 1.0, bootstrap=0.0
        )
        scaled = PPOLagrangian.normalize_advantage(adv2, normalize=False)
        assert scaled.std() == adv2.std()
        assert scaled.std() > 1.5

    def test_lambda_nonneg_and_update_runs(self):
        cfg = make_cfg()
        agent = PPOLagrangian(cfg, 5, 3, np.ones(3) * 10.0, "cpu")
        for _ in range(20):
            s = np.random.randn(5).astype(np.float32)
            agent.select_action(s)
            agent.store_reward_and_done(-1.0, cost=1.0, done=False)
        agent.finish_rollout(last_obs=s, truncated=True)
        info = agent.update(None, 0)
        assert agent.lambda_val.item() >= 0.0
        assert "lambda" in info


class TestCPOMath:
    def test_conjugate_gradient_inverts_known_matrix(self):
        A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
        b = torch.tensor([2.0, 1.0])
        x = conjugate_gradient(lambda v: A @ v, b, iters=20, tol=1e-10)
        assert torch.allclose(A @ x, b, atol=1e-5)

    def test_direction_shape_and_descent(self):
        # Identity metric; reward grad g, cost grad b orthogonal.
        g = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        Hvp = lambda v: v + 0.1 * v
        x, flag = cpo_direction(g, b, Hvp, cost_slack=0.1, target_kl=0.01)
        assert x.shape == g.shape
        assert torch.isfinite(x).all()
        # Direction must have positive reward improvement g.x > 0.
        assert torch.dot(g, x) > 0


class TestCPOAgent:
    def test_select_action_and_update(self):
        cfg = make_cfg()
        agent = CPO(cfg, 5, 3, np.ones(3) * 10.0, "cpu")
        for _ in range(24):
            s = np.random.randn(5).astype(np.float32)
            agent.select_action(s)
            agent.store_reward_and_done(-1.0, cost=1.0, done=False)
        agent.finish_rollout(last_obs=s, truncated=True)
        info = agent.update(None, 0)
        assert "kl" in info
        assert "accepted" in info
        # Trust region after the step.
        assert info["kl"] <= cfg.ppo.target_kl * 1.5 or info["accepted"] == 0.0
