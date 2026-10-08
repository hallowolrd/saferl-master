"""Algorithm module.

Core algorithms and baselines:
- HFG-SAC: Hierarchical Fuzzy-Guided Soft Actor-Critic (main algorithm)
- Safe SAC: Standard Lagrangian SAC baseline
- SAC: Vanilla Soft Actor-Critic
- Fuzzy SAC: SAC + fuzzy reward shaping (no constraint protection)
- PPO: Proximal Policy Optimization
- PPO-Lagrangian: PPO with Lagrangian constraint handling
- CPO: Constrained Policy Optimization (simplified)
- Safety Layer: CBF-inspired safety layer on SAC
- Fuzzy-Lagrangian: Fuzzy constraint satisfaction via Lagrangian relaxation
"""

from .base_agent import BaseAgent, register_agent, AGENT_FACTORY, make_agent
from .hfg_sac import HFGSAC
from .fuzzy_lagrangian import FuzzyLagrangian
from .safe_sac import SafeSAC
from .sac import SAC
from .fuzzy_sac import FuzzySAC
from .ppo import PPO
from .ppo_lagrangian import PPOLagrangian
from .cpo import CPO
from .safety_layer import SafetyLayerSAC

__all__ = [
    "BaseAgent",
    "register_agent",
    "AGENT_FACTORY",
    "make_agent",
    "HFGSAC",
    "FuzzyLagrangian",
    "SafeSAC",
    "SAC",
    "FuzzySAC",
    "PPO",
    "PPOLagrangian",
    "CPO",
    "SafetyLayerSAC",
]
