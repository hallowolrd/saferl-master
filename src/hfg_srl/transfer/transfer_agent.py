"""Transfer agent for cross-scenario safe policy adaptation.

Implements the constraint-aware transfer pipeline described in Chapter 4.4:

- **M1**: Jaccard fuzzy-rule similarity (delegated to `similarity.py`)
- **M2**: Membership parameter transfer per category (delegated to `similarity.py`)
- **M3**: Layer-frozen weight transfer (simplified progressive adapter)
- **M4**: Exponential safety relaxation of the conservative factor:
      alpha_cur(t) = alpha_target + (alpha_init - alpha_target) * exp(-nu * t)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..algorithms.base_agent import BaseAgent
from ..utils.config import TransferConfig, HFGConfig
from .similarity import (
    categorize_constraints,
    gaussian_membership,
    jaccard_similarity,
    transfer_membership_params,
)

logger = logging.getLogger(__name__)


class TransferAgent:
    """Constraint-aware cross-scenario transfer agent.

    Methods:
        - Full transfer: copy all weights and fine-tune
        - Fine-tune: copy weights, freeze bottom layers
        - Progressive adapter: copy weights, freeze trunk, train head only

    Safety during transfer:
        - Conservative factor α_cur(t) relaxes exponentially from α_init to α_target
          (paper 4.4.2, Eq. progressive relaxation).
        - Action scale = 1 / α_cur; higher α_cur → smaller actions (more conservative).
    """

    def __init__(
        self,
        cfg: HFGConfig,
        source_agent: BaseAgent,
        target_obs_dim: int,
        target_act_dim: int,
        target_action_high: np.ndarray,
        device: str,
    ):
        self.cfg = cfg
        self.t_cfg: TransferConfig = cfg.transfer
        self.source_agent = source_agent
        self.device = device

        # M4: exponential safety relaxation parameters.
        # alpha_target = 1.0 means "no conservative scaling".
        # alpha_init > 1.0 means "start conservative".
        self.alpha_target: float = 1.0
        self.alpha_init: float = float(self.t_cfg.conservative_init)
        # Derive decay rate ν from decay_steps so that at t = decay_steps,
        # alpha_cur ≈ alpha_target + 0.05 * (alpha_init - alpha_target).
        #   => exp(-nu * decay_steps) = 0.05  =>  nu = -ln(0.05) / decay_steps
        decay_steps = max(self.t_cfg.conservative_decay_steps, 1)
        self.nu: float = float(np.log(20.0) / decay_steps)

        self.conservative_factor: float = self.alpha_init
        self._step_count: int = 0

        # M1/M2: per-constraint similarity + category + transferred params
        self.constraint_categories: Dict[str, str] = {}
        self.constraint_similarities: Dict[str, float] = {}
        self.transferred_params: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_target_policy(self, target_agent: BaseAgent) -> None:
        """Initialize target policy from source policy.

        Transfer method depends on cfg.transfer.method.
        """
        method = self.t_cfg.method

        if method == "full_transfer":
            self._copy_weights(self.source_agent, target_agent)
            logger.info("Full transfer: copied all weights from source to target.")

        elif method == "fine_tune":
            self._copy_weights(self.source_agent, target_agent)
            self._freeze_bottom_layers(target_agent, freeze_frac=0.5)
            logger.info("Fine-tune transfer: copied weights, froze bottom 50%.")

        elif method == "progressive_adapter":
            self._setup_progressive_transfer(target_agent)
            logger.info("Progressive adapter transfer: set up adapter layers.")

        else:
            raise ValueError(f"Unknown transfer method: {method}")

    # ------------------------------------------------------------------
    # M1 + M2: Fuzzy-rule similarity + parameter transfer
    # ------------------------------------------------------------------

    def compute_constraint_similarities(
        self,
        constraint_specs: List[Dict],
    ) -> Dict[str, float]:
        """Compute Jaccard similarity per constraint (M1).

        Args:
            constraint_specs: List of dicts with keys:
                - "name": constraint name (e.g. "soc_upper", "freq_dev")
                - "source_mu", "source_beta": source membership params
                - "target_mu", "target_beta": target membership params
                - "x_range": integration bounds (e.g. (0.0, 1.0) for SOC)

        Returns:
            Mapping constraint_name -> sim_k in [0, 1].
        """
        sims: Dict[str, float] = {}
        for spec in constraint_specs:
            name = spec["name"]
            s_mu, s_beta = spec["source_mu"], spec["source_beta"]
            t_mu, t_beta = spec["target_mu"], spec["target_beta"]
            x_range = spec.get("x_range", (-5.0, 5.0))

            def mu_s(x, _s_mu=s_mu, _s_beta=s_beta):
                return gaussian_membership(x, _s_mu, _s_beta)

            def mu_t(x, _t_mu=t_mu, _t_beta=t_beta):
                return gaussian_membership(x, _t_mu, _t_beta)

            sims[name] = jaccard_similarity(mu_s, mu_t, x_range=x_range)

        self.constraint_similarities = sims
        self.constraint_categories = categorize_constraints(sims)
        logger.info("Constraint similarity: %s", sims)
        logger.info("Constraint categories: %s", self.constraint_categories)
        return sims

    def transfer_constraint_params(
        self,
        constraint_specs: List[Dict],
    ) -> Dict[str, Dict[str, float]]:
        """Apply M2 per-constraint parameter transfer.

        Must be called after compute_constraint_similarities().
        """
        out: Dict[str, Dict[str, float]] = {}
        for spec in constraint_specs:
            name = spec["name"]
            category = self.constraint_categories.get(name, "relearn")
            out[name] = transfer_membership_params(
                source_mu=spec["source_mu"],
                source_beta=spec["source_beta"],
                category=category,
                delta=spec.get("delta", 0.0),
                Delta=spec.get("Delta", 1.0),
                default_mu=spec.get("default_mu", spec["source_mu"]),
                default_beta=spec.get("default_beta", spec["source_beta"]),
            )
        self.transferred_params = out
        logger.info("Transferred constraint params: %s", out)
        return out

    # ------------------------------------------------------------------
    # Weight transfer helpers (M3)
    # ------------------------------------------------------------------

    def _copy_weights(self, source: BaseAgent, target: BaseAgent) -> None:
        if hasattr(source, "actor") and hasattr(target, "actor"):
            try:
                target.actor.load_state_dict(source.actor.state_dict())
            except Exception as e:
                logger.warning(f"Actor weight transfer skipped: {e}")

        if hasattr(source, "critic") and hasattr(target, "critic"):
            try:
                target.critic.load_state_dict(source.critic.state_dict())
                if hasattr(target, "critic_target"):
                    target.critic_target.load_state_dict(source.critic.state_dict())
            except Exception as e:
                logger.warning(f"Critic weight transfer skipped: {e}")

    def _freeze_bottom_layers(self, agent: BaseAgent, freeze_frac: float = 0.5) -> None:
        if not hasattr(agent, "actor") or not hasattr(agent.actor, "trunk"):
            return
        all_params = list(agent.actor.trunk.parameters())
        num_freeze = int(len(all_params) * freeze_frac)
        for i, param in enumerate(all_params):
            if i < num_freeze:
                param.requires_grad = False
        logger.info("Froze %d/%d actor trunk parameter groups.", num_freeze, len(all_params))

    def _setup_progressive_transfer(self, target_agent: BaseAgent) -> None:
        self._copy_weights(self.source_agent, target_agent)
        if hasattr(target_agent, "actor"):
            for param in target_agent.actor.parameters():
                param.requires_grad = False
            if hasattr(target_agent.actor, "mean_head"):
                for param in target_agent.actor.mean_head.parameters():
                    param.requires_grad = True
            if hasattr(target_agent.actor, "log_std_head"):
                for param in target_agent.actor.log_std_head.parameters():
                    param.requires_grad = True

    # ------------------------------------------------------------------
    # M4: Exponential safety relaxation
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one fine-tuning step and relax the conservative factor.

        Implements paper Eq.:
            alpha_cur(t) = alpha_target + (alpha_init - alpha_target) * exp(-nu * t)
        """
        self._step_count += 1
        t = self._step_count
        decay = np.exp(-self.nu * t)
        self.conservative_factor = (
            self.alpha_target
            + (self.alpha_init - self.alpha_target) * decay
        )

    def get_conservative_action_scale(self) -> float:
        """Action scale in (0, 1]. Returns 1 / alpha_cur.

        While conservative (alpha_cur > 1), actions are scaled down.
        """
        return 1.0 / max(self.conservative_factor, 1.0)
