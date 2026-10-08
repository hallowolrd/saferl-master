"""Fuzzy reward shaping engine.

Uses expert knowledge encoded as fuzzy rules to shape the reward signal,
accelerating learning while maintaining convergence guarantees (via
potential-based shaping approach).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .tsk_fuzzy import TSKFuzzySystem
from .fuzzy_rules import FuzzyRuleBase, expert_rule_base
from ..utils.config import FuzzyConfig

logger = logging.getLogger(__name__)


class FuzzyRewardShaper(nn.Module):
    """Fuzzy reward shaping engine.

    Computes a shaping reward based on the current state using a
    TSK fuzzy system initialized from expert rules. The shaping weight
    decays over training to allow the agent to learn from the true reward.

    This follows the potential-based reward shaping principle:
        F(s, a, s') = gamma * Phi(s') - Phi(s)
    where Phi(s) is the fuzzy potential function.

    Using a state-only potential preserves the optimal policy.
    """

    def __init__(
        self,
        cfg: FuzzyConfig,
        num_inputs: int,
        rule_base: Optional[FuzzyRuleBase] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_inputs = num_inputs

        # Fuzzy system for potential function Phi(s)
        self.fuzzy_system = TSKFuzzySystem(
            num_inputs=num_inputs,
            num_outputs=1,  # potential value
            num_rules=cfg.num_rules,
            first_order=False,  # zero-order for interpretability
        )

        # Initialize from expert rules if available
        if cfg.use_expert_rules and rule_base is not None:
            self._init_from_rule_base(rule_base)

        # Shaping weight (will be decayed during training)
        self.register_buffer(
            "shaping_weight",
            torch.tensor(cfg.reward_shaping_weight_init, dtype=torch.float32),
        )
        self._step_count = 0

    def _semantic_input_names(self) -> list:
        """Map each observation index to its expert-rule semantic input name.

        Expert rules use the keys "soc", "pv_normalized", "load_normalized"
        and "time_of_day". Indices without a corresponding expert set are
        None (treated as don't-care by the rule base). Observation layout:

            grid (8):    soc, pv, wt, load, t, sin, cos, price
            island (9):  soc, pv, wt, load, freq, volt, t, sin, cos
        """
        if self.num_inputs >= 9:
            return ["soc", "pv_normalized", None, "load_normalized",
                    None, None, "time_of_day", None, None]
        return ["soc", "pv_normalized", None, "load_normalized",
                "time_of_day", None, None, None][:self.num_inputs]

    def _init_from_rule_base(self, rule_base: FuzzyRuleBase) -> None:
        """Initialize fuzzy system parameters from expert rule base."""
        input_names = self._semantic_input_names()
        means, stds, conseq, confidences = rule_base.to_tsk_params(
            input_names
        )

        num_rules = min(len(rule_base), self.cfg.num_rules)
        num_mapped_inputs = min(means.shape[1], self.num_inputs)

        with torch.no_grad():
            self.fuzzy_system.membership.means[:num_rules, :num_mapped_inputs] = (
                torch.from_numpy(means[:num_rules, :num_mapped_inputs])
            )
            stds_clipped = torch.clamp(
                torch.from_numpy(stds[:num_rules, :num_mapped_inputs]), 1e-2
            )
            self.fuzzy_system.membership.log_stds[:num_rules, :num_mapped_inputs] = (
                torch.log(stds_clipped)
            )
            # Consequents scaled by PER-RULE confidence (not rule[0].confidence
            # for all). This restores the expert's intended relative weighting
            # between strong rules (low-SOC protection, conf=0.9) and weak
            # rules (night standby, conf=0.5).
            if conseq.shape[0] > 0:
                conf_t = torch.from_numpy(confidences[:num_rules]).unsqueeze(-1)
                self.fuzzy_system.consequent[:num_rules, :] = (
                    torch.from_numpy(conseq[:num_rules, :]) * conf_t
                )
            else:
                self.fuzzy_system.consequent[:num_rules, :] = torch.zeros(num_rules, 1)

        logger.info(
            f"Initialized fuzzy reward shaper from {num_rules} expert rules "
            f"({num_mapped_inputs} input dimensions mapped)."
        )

    def potential(self, state: torch.Tensor) -> torch.Tensor:
        """Compute potential function Phi(s).

        Args:
            state: State tensor (batch_size, state_dim).

        Returns:
            Potential values (batch_size, 1).
        """
        return self.fuzzy_system(state)

    def shape_reward(
        self,
        reward: torch.Tensor,
        state: torch.Tensor,
        next_state: torch.Tensor,
        gamma: float,
        dones: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute shaped reward using potential-based shaping.

        F(s,a,s') = gamma * Phi(s') - Phi(s).  To preserve the
        policy-invariance guarantee of Ng et al., the terminal potential must
        be zero: for absorbing/terminal next states we use F = -Phi(s)
        (i.e. Phi(s_terminal) = 0).

        Args:
            reward: Original environment reward (batch_size, 1).
            state: Current state (batch_size, state_dim).
            next_state: Next state (batch_size, state_dim).
            gamma: Discount factor.
            dones: Optional terminal mask (batch_size,) with 1.0 on absorbing
                transitions.  If None, all transitions are treated as
                non-terminal.

        Returns:
            Shaped reward (batch_size, 1).
        """
        phi_s = self.potential(state)
        phi_next = self.potential(next_state)
        if dones is None:
            shaping_term = gamma * phi_next - phi_s
        else:
            # dones shape: (batch,) -> broadcast to (batch, 1)
            d = dones.view(-1, 1).to(reward.dtype)
            # Terminal: Phi(s') = 0  => F = -Phi(s)
            # Non-terminal:           F = gamma*Phi(s') - Phi(s)
            shaping_term = torch.where(
                d > 0.5, -phi_s, gamma * phi_next - phi_s
            )
        shaped = reward + self.shaping_weight * shaping_term
        return shaped

    def step(self) -> None:
        """Advance one training step and decay shaping weight via cosine
        annealing.

        Cosine annealing keeps the weight high early (expert guidance is
        most valuable when Q is inaccurate), then smoothly eases to final
        near the end of decay — avoiding the abrupt linear cutoff that can
        strip guidance before the policy has converged.
        """
        self._step_count += 1
        init_w = self.cfg.reward_shaping_weight_init
        final_w = self.cfg.reward_shaping_weight_final
        decay_steps = self.cfg.reward_shaping_decay_steps

        if self._step_count <= decay_steps and decay_steps > 0:
            # Cosine annealing: w = final + 0.5*(init-final)*(1+cos(pi*t/T))
            t = self._step_count / decay_steps
            cos_factor = 0.5 * (1.0 + np.cos(np.pi * t))
            new_w = final_w + (init_w - final_w) * cos_factor
            self.shaping_weight.fill_(new_w)

    @property
    def current_weight(self) -> float:
        return float(self.shaping_weight.item())

    def apply_m2_init(
        self,
        m2_init: Dict[str, Dict[str, float]],
        mode: str,
        target_frequency_limit_hz: float = 0.5,
    ) -> int:
        """Warm-start antecedent membership from transferred fuzzy priors (M2).

        Overwrites the SoC (and, in islanded mode, frequency) antecedent
        Gaussian centers/widths of the TSK reward shaper with the parameters
        produced by the M1 -> M2 transfer pipeline. This is what gives M1's
        categorization a behavioral effect: rather than random/expert-default
        fuzzy priors, the target shaper starts at the specification implied by
        the source scenario.

        Coordinate conventions:
            - SoC input (dim 0) is physical in [0, 1]; m2_init x_ref/beta are
              already in this scale.
            - Frequency input (islanded dim 4) is normalized deviation
              freq_deviation / frequency_limit_hz, so the m2 beta (given in
              Hz as frequency_limit/3) is divided by frequency_limit to yield
              a dimensionless width of ~1/3.

        Only rules that actually use a dimension (narrow antecedent std) are
        touched; "don't-care" rules with wide stds are left unchanged.
        High-SoC rules (mean >= 0.5) receive soc_upper priors; low-SoC rules
        receive soc_lower priors.

        Args:
            m2_init: Constraint name -> {"x_ref": float, "beta": float}.
            mode: "grid_connected" or "islanded".

        Returns:
            Number of (rule, dimension) antecedent entries overwritten.
        """
        membership = self.fuzzy_system.membership
        num_rules = membership.means.shape[0]
        # Threshold below which an antecedent is considered to "use" the dim.
        care_std = 0.5
        changed = 0

        soc_upper = m2_init.get("soc_upper")
        soc_lower = m2_init.get("soc_lower")
        if (soc_upper or soc_lower):
            means = membership.means.data[:, 0]
            stds = torch.exp(membership.log_stds.data[:, 0])
            for r in range(num_rules):
                if stds[r] > care_std:
                    continue  # don't-care dimension
                if means[r] >= 0.5 and soc_upper:
                    x_ref, beta = soc_upper["x_ref"], soc_upper["beta"]
                elif means[r] < 0.5 and soc_lower:
                    x_ref, beta = soc_lower["x_ref"], soc_lower["beta"]
                else:
                    continue
                membership.means.data[r, 0] = float(x_ref)
                membership.log_stds.data[r, 0] = math.log(max(float(beta), 1e-2))
                changed += 1

        freq = m2_init.get("freq_dev")
        freq_dim = 4  # islanded obs layout: [..., freq_normalized at index 4]
        if freq and mode == "islanded" and self.num_inputs > freq_dim:
            beta_norm = float(freq["beta"]) / max(float(target_frequency_limit_hz), 1e-3)
            means = membership.means.data[:, freq_dim]
            stds = torch.exp(membership.log_stds.data[:, freq_dim])
            for r in range(num_rules):
                if stds[r] > care_std:
                    continue
                membership.means.data[r, freq_dim] = float(freq["x_ref"])
                membership.log_stds.data[r, freq_dim] = math.log(max(beta_norm, 1e-2))
                changed += 1

        if changed:
            logger.info(f"M2: warm-started {changed} reward-shaper antecedent entries.")
        return changed
