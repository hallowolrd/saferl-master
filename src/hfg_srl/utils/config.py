"""Configuration management using dataclasses (immutable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EnvConfig:
    """Microgrid environment configuration."""
    mode: str = "grid_connected"  # grid_connected | islanded
    time_steps_per_day: int = 96  # 15-min intervals
    num_days: int = 365
    pv_capacity_kw: float = 2000.0
    wt_capacity_kw: float = 1500.0
    ess_capacity_kwh: float = 5000.0
    ess_power_kw: float = 2000.0
    de_rated_power_kw: float = 3000.0
    base_load_kw: float = 5000.0
    interruptible_load_kw: float = 1000.0
    grid_power_limit_kw: float = 3000.0
    # Data source
    data_source: str = "synthetic"  # synthetic | csv | parquet
    data_file: Optional[str] = None  # Path to data file for csv/parquet
    scenario: Optional[str] = None  # Scenario name for slicing (e.g., "summer")
    seed: int = 42  # Random seed for environment and data loader
    # Fuzzy constraint parameters
    soc_min: float = 0.2
    soc_max: float = 0.9
    soc_shield_margin: float = 0.02  # hard shield margin inside hard bounds
    soc_optimal_min: float = 0.3
    soc_optimal_max: float = 0.7
    # Cost coefficients
    grid_buy_price: float = 0.8  # yuan/kWh
    grid_sell_price: float = 0.4  # yuan/kWh
    de_fuel_cost: float = 0.6  # yuan/kWh
    ess_degradation_cost: float = 0.05  # yuan/kWh per cycle
    # Safety
    voltage_limit_pu: float = 1.05
    frequency_limit_hz: float = 0.5
    # Hinge penalty slope for freq/voltage excursions in islanded env
    # (RMB per Hz of excess beyond 30% of limit).  Higher = earlier learning signal.
    freq_penalty_slope: float = 1000.0
    # Per-kWh cost of interrupting non-critical load (islanded mode).
    load_shed_cost: float = 2.0


@dataclass(frozen=True)
class FuzzyConfig:
    """Fuzzy system configuration."""
    # TSK fuzzy system
    num_rules: int = 16
    num_inputs: int = 4  # SOC, SOC_delta, net_load, time_of_day
    num_outputs: int = 1  # constraint satisfaction degree
    membership_type: str = "gaussian"  # gaussian | trapezoidal
    # Reward shaping
    reward_shaping_weight_init: float = 1.0
    reward_shaping_weight_final: float = 0.1
    reward_shaping_decay_steps: int = 50000
    # Fuzzy-Lagrangian. The multiplier is in *cost units* (¥) so that the
    # penalty term lambda * (fcsd_target - FCSD) is commensurate with the
    # daily operating cost (~1e4 ¥). A lambda ~ O(1) makes the safety term
    # ~1e-4 of the actor objective and thus inert (see run/aggregate_results.py
    # diagnostics). lambda_lr is now a log-space growth rate.
    lambda_init: float = 5000.0
    lambda_lr: float = 1.0
    lambda_min: float = 0.0
    lambda_max: float = 100000.0
    fcsd_target: float = 0.9  # target min-FCSD (worst-constraint satisfaction degree)
    # Shield-activation penalty: mu * relu(|ess_raw| - bound) added to actor
    # loss so the policy learns to stay *inside* the hard shield band instead
    # of permanently relying on the clamp.  0 disables.
    # Raised from 1.0 -> 8.0: on islanded extreme scenarios the old mu=1.0 was
    # dominated by lambda * cost_hat (~O(1) under the scaled Lagrangian) and
    # the actor kept driving raw ESS action into the shield, producing SOC
    # oscillation at the band edge.  mu=8 gives the differentiable hinge enough
    # weight to pull raw action inside the band before the hard clamp fires.
    shield_act_penalty_mu: float = 8.0
    # B1: Frequency/voltage hard shield (islanded only).  When the predicted
    # imbalance_ratio exceeds this threshold, DE / load-shed / dump-load
    # actions are clamped so the next step's freq_dev stays below
    # freq_volt_shield_freq_fraction * freq_limit.  This is the last line of
    # defense for islanded extreme scenarios where the soft reward penalty
    # alone is insufficient.  Set to 0.0 to disable.
    # 0.175 corresponds to freq_dev = 0.35 Hz = 70% of the 0.5 Hz limit.
    freq_volt_shield_enabled: bool = True
    freq_volt_shield_safety_ratio: float = 0.175
    # B2: Cost-critic target separates hard violations (any constraint
    # crossing its hard threshold, fcsd_min < hard_threshold) from soft
    # shortfalls (fcsd_min in [hard_threshold, fcsd_target]).  Without this,
    # a "near miss" (fcsd_min=0.49) and a "deep violation" (fcsd_min=0.1)
    # produce almost identical per-step cost ~0.35, so the actor cannot
    # distinguish the two and never prioritises avoiding hard crossings.
    hard_violation_threshold: float = 0.5
    hard_violation_cost_weight: float = 1.0
    soft_violation_cost_weight: float = 0.1
    # Fuzzy rules
    use_expert_rules: bool = True
    expert_rule_confidence: float = 0.8


@dataclass(frozen=True)
class AlgorithmConfig:
    """Algorithm hyperparameters."""
    name: str = "hfg_sac"
    # SAC basics
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_update_interval: int = 1
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    # Training
    batch_size: int = 256
    buffer_size: int = int(1e6)
    max_steps: int = int(1e6)
    warmup_steps: int = 10000
    updates_per_step: int = 1
    eval_interval: int = 10000
    eval_episodes: int = 10
    # Safety
    use_safety_layer: bool = True
    safety_method: str = "fuzzy_lagrangian"  # fuzzy_lagrangian | lagrangian | cbf
    # Logging
    log_interval: int = 1000


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for on-policy baselines (PPO / PPO-Lag / CPO)."""
    # PPO clipping
    clip_eps: float = 0.2
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10            # PPO value/policy fit epochs
    minibatch_size: int = 64
    # Constraint semantics: binary per-step cost (1.0 on hard violation);
    # cost_limit_rate is the maximum allowed per-step violation rate,
    # matching the 5% safety threshold used in the paper (Fig. 5.1).
    cost_limit_rate: float = 0.05
    normalize_cost_advantage: bool = False  # must stay False for Lagrangian
    # CPO trust region
    target_kl: float = 0.01
    cg_iters: int = 10
    cg_damping: float = 0.1
    backtrack_coef: float = 0.8
    backtrack_iters: int = 10


@dataclass(frozen=True)
class TransferConfig:
    """Transfer learning configuration."""
    enabled: bool = False
    source_scenario: str = "grid_connected_summer"
    target_scenario: str = "islanded_normal"
    # Transfer method
    method: str = "progressive_adapter"  # progressive_adapter | fine_tune | full_transfer
    adapter_hidden_dim: int = 64
    # Safety during transfer
    conservative_init: float = 3.0  # initial safety multiplier
    conservative_decay_steps: int = 20000
    # Scenario metric
    metric_type: str = "fuzzy_rule_similarity"  # fuzzy_rule_similarity | distribution_distance
    # Few-shot
    few_shot_steps: int = 5000


@dataclass(frozen=True)
class HFGConfig:
    """Top-level configuration for HFG-SRL."""
    env: EnvConfig = field(default_factory=EnvConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda
    output_dir: str = "outputs"
    experiment_name: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to nested dictionary."""
        return _dataclass_to_dict(self)


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass to dict."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj
