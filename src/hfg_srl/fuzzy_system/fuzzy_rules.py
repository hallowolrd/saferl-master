"""Fuzzy rule base for expert knowledge encoding.

Provides structures for defining fuzzy If-Then rules that encode
domain expert knowledge (e.g., microgrid operating guidelines).
Rules can be translated into TSK fuzzy system initial parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FuzzyRule:
    """A single fuzzy If-Then rule.

    Format:
        IF (input_1 is term_1) AND (input_2 is term_2) ...
        THEN output_1 = value_1, output_2 = value_2, ...

    Attributes:
        antecedent: Dict mapping input name -> fuzzy set name.
        consequent: Dict mapping output name -> (float) constant value (zero-order).
        confidence: Rule confidence in [0, 1].
        source: Optional source description (e.g., "expert: operator").
    """

    antecedent: Dict[str, str]
    consequent: Dict[str, float]
    confidence: float = 1.0
    source: str = ""

    def matches_inputs(self, input_names: List[str]) -> bool:
        """Check if rule's antecedent covers all specified inputs."""
        return all(name in self.antecedent for name in input_names)


@dataclass
class FuzzySetDef:
    """Definition of a fuzzy set on a particular input dimension.

    Attributes:
        name: Linguistic term name (e.g., "LOW", "HIGH", "MEDIUM").
        mean: Gaussian mean.
        std: Gaussian std (width).
    """

    name: str
    mean: float
    std: float


class FuzzyRuleBase:
    """A collection of fuzzy rules with shared fuzzy set definitions.

    Provides utilities for:
    - Adding rules
    - Computing rule firing for given input
    - Converting to TSK fuzzy system parameters (for initialization)
    """

    def __init__(self):
        self.rules: List[FuzzyRule] = []
        # input_name -> {fuzzy_set_name: FuzzySetDef}
        self.input_sets: Dict[str, Dict[str, FuzzySetDef]] = {}
        self.output_names: List[str] = []

    def add_input_sets(self, input_name: str, sets: List[FuzzySetDef]) -> None:
        """Register fuzzy set definitions for an input dimension."""
        if input_name not in self.input_sets:
            self.input_sets[input_name] = {}
        for s in sets:
            self.input_sets[input_name][s.name] = s

    def add_rule(self, rule: FuzzyRule) -> None:
        """Add a rule to the rule base."""
        self.rules.append(rule)
        # Track output names
        for out_name in rule.consequent:
            if out_name not in self.output_names:
                self.output_names.append(out_name)

    def add_rules(self, rules: List[FuzzyRule]) -> None:
        """Add multiple rules."""
        for r in rules:
            self.add_rule(r)

    def __len__(self) -> int:
        return len(self.rules)

    def to_tsk_params(
        self,
        input_names: List[str],
        output_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert rule base to TSK fuzzy system parameters.

        Returns:
            Tuple of (means, stds, consequents, confidences):
                - means: (num_rules, num_inputs) array
                - stds: (num_rules, num_inputs) array
                - consequents: (num_rules, num_outputs) array (zero-order,
                  NOT pre-scaled by confidence; caller applies it)
                - confidences: (num_rules,) array in [0, 1] per rule
        """
        output_names = output_names or self.output_names
        num_rules = len(self.rules)
        num_inputs = len(input_names)
        num_outputs = len(output_names)

        means = np.zeros((num_rules, num_inputs), dtype=np.float32)
        stds = np.ones((num_rules, num_inputs), dtype=np.float32)
        consequents = np.zeros((num_rules, num_outputs), dtype=np.float32)
        confidences = np.ones((num_rules,), dtype=np.float32)

        for r_idx, rule in enumerate(self.rules):
            confidences[r_idx] = rule.confidence
            # Antecedent means and stds
            for i_idx, in_name in enumerate(input_names):
                if in_name in rule.antecedent:
                    set_name = rule.antecedent[in_name]
                    if in_name in self.input_sets and set_name in self.input_sets[in_name]:
                        fset = self.input_sets[in_name][set_name]
                        means[r_idx, i_idx] = fset.mean
                        stds[r_idx, i_idx] = fset.std
                    else:
                        means[r_idx, i_idx] = 0.0
                        stds[r_idx, i_idx] = 2.0
                else:
                    # "don't care" -> wide membership
                    means[r_idx, i_idx] = 0.0
                    stds[r_idx, i_idx] = 5.0

            # Consequent (raw value; caller multiplies by per-rule confidence)
            for o_idx, out_name in enumerate(output_names):
                consequents[r_idx, o_idx] = rule.consequent.get(out_name, 0.0)

        return means, stds, consequents, confidences


# --- Predefined expert rule base for microgrid dispatch ---

def expert_rule_base() -> FuzzyRuleBase:
    """Create a pre-populated expert rule base for microgrid ESS dispatch.

    Encodes common operating heuristics from grid operators:
    - SOC low -> reduce discharge / increase charge
    - PV high + SOC high -> reduce charge / increase discharge
    - Peak price hours -> discharge more
    - High load -> discharge to support
    """
    rb = FuzzyRuleBase()

    # Define fuzzy sets for inputs (normalized to [-1, 1] or [0, 1])
    rb.add_input_sets("soc", [
        FuzzySetDef("LOW", 0.25, 0.12),
        FuzzySetDef("MEDIUM", 0.5, 0.15),
        FuzzySetDef("HIGH", 0.75, 0.12),
    ])
    rb.add_input_sets("pv_normalized", [
        FuzzySetDef("LOW", 0.1, 0.1),
        FuzzySetDef("MEDIUM", 0.5, 0.2),
        FuzzySetDef("HIGH", 0.9, 0.1),
    ])
    rb.add_input_sets("load_normalized", [
        FuzzySetDef("LOW", 0.3, 0.12),
        FuzzySetDef("MEDIUM", 0.6, 0.15),
        FuzzySetDef("HIGH", 0.9, 0.1),
    ])
    rb.add_input_sets("time_of_day", [
        FuzzySetDef("NIGHT", 0.1, 0.08),
        FuzzySetDef("MORNING", 0.3, 0.08),
        FuzzySetDef("MIDDAY", 0.5, 0.08),
        FuzzySetDef("EVENING", 0.8, 0.08),
    ])

    # Define rules (output: ess_power_bias in [-1, 1], positive = charge bias)
    rules = [
        # Rule 1: SOC low -> charge more (discharge less)
        FuzzyRule(
            antecedent={"soc": "LOW"},
            consequent={"ess_power_bias": 0.5},
            confidence=0.9,
            source="expert: low_soc_protection",
        ),
        # Rule 2: SOC high -> discharge more
        FuzzyRule(
            antecedent={"soc": "HIGH"},
            consequent={"ess_power_bias": -0.4},
            confidence=0.8,
            source="expert: high_soc_utilization",
        ),
        # Rule 3: High PV + not high SOC -> charge
        FuzzyRule(
            antecedent={"pv_normalized": "HIGH", "soc": "MEDIUM"},
            consequent={"ess_power_bias": 0.3},
            confidence=0.7,
            source="expert: pv_charge",
        ),
        # Rule 4: High load + high SOC -> discharge
        FuzzyRule(
            antecedent={"load_normalized": "HIGH", "soc": "MEDIUM"},
            consequent={"ess_power_bias": -0.3},
            confidence=0.7,
            source="expert: peak_support",
        ),
        # Rule 5: Morning peak + medium SOC -> support load
        FuzzyRule(
            antecedent={"time_of_day": "MORNING", "load_normalized": "HIGH"},
            consequent={"ess_power_bias": -0.25},
            confidence=0.6,
            source="expert: morning_peak",
        ),
        # Rule 6: Evening peak + medium SOC -> support load
        FuzzyRule(
            antecedent={"time_of_day": "EVENING", "soc": "MEDIUM"},
            consequent={"ess_power_bias": -0.3},
            confidence=0.65,
            source="expert: evening_peak",
        ),
        # Rule 7: Midday high PV + low SOC -> charge aggressively
        FuzzyRule(
            antecedent={"time_of_day": "MIDDAY", "pv_normalized": "HIGH", "soc": "LOW"},
            consequent={"ess_power_bias": 0.7},
            confidence=0.85,
            source="expert: midday_charge",
        ),
        # Rule 8: Night + high SOC -> ready for morning
        FuzzyRule(
            antecedent={"time_of_day": "NIGHT", "soc": "HIGH"},
            consequent={"ess_power_bias": -0.1},
            confidence=0.5,
            source="expert: night_standby",
        ),
    ]

    rb.add_rules(rules)
    return rb
