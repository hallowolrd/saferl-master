"""Fuzzy system module.

Implements differentiable TSK fuzzy systems, fuzzy rule-based knowledge
encoding, fuzzy reward shaping, and fuzzy constraint boundary estimation.
"""

from .tsk_fuzzy import TSKFuzzySystem
from .fuzzy_rules import FuzzyRule, FuzzyRuleBase, expert_rule_base
from .reward_shaper import FuzzyRewardShaper

__all__ = [
    "TSKFuzzySystem",
    "FuzzyRule",
    "FuzzyRuleBase",
    "expert_rule_base",
    "FuzzyRewardShaper",
]
