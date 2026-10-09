"""Experiment runner for HFG-SRL paper experiments.

Runs all algorithms across all scenarios and saves results.

Experiments:
- RQ1: Performance comparison (all algorithms x 4 scenarios)
- RQ2: Robustness under extreme conditions
- RQ3: Transfer learning
- RQ4: Ablation study
- RQ5: Sensitivity analysis

Usage:
    python run/experiment_runner.py --output_dir outputs/experiments
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hfg_srl.env import make_env
from hfg_srl.algorithms import make_agent, AGENT_FACTORY
from hfg_srl.algorithms.model_based import MILPOracle, MPC
from hfg_srl.buffers import ReplayBuffer
from hfg_srl.data import SyntheticDataLoader, create_scenario, ScenarioConfig
from hfg_srl.utils.config import HFGConfig, EnvConfig, FuzzyConfig, AlgorithmConfig
from hfg_srl.utils.seed import set_seed, get_device


# --- Scenario definitions ---

SCENARIOS = {
    "S1_grid_normal": {
        "env_mode": "grid_connected",
        "scenario_name": "typical_week_summer",
        "extreme_multiplier": None,
        "description": "Grid-connected, normal conditions",
    },
    "S2_grid_extreme": {
        "env_mode": "grid_connected",
        "scenario_name": "summer_extreme",
        "extreme_multiplier": {"pv": 1.2, "load": 1.3, "wt": 0.6},
        "description": "Grid-connected, extreme conditions",
    },
    "S3_island_normal": {
        "env_mode": "islanded",
        "scenario_name": "typical_week_summer",
        "extreme_multiplier": None,
        "description": "Islanded, normal conditions",
        # S3 and S4 share identical safety resources (same interruptible load,
        # same shed cost, same shield margin). Only weather differs: S3 normal
        # summer, S4 extreme summer (PV x0.5, load x1.25, WT x0.8).
        "interruptible_load_kw": 600.0,
        "freq_penalty_slope": 1500.0,
        "freq_volt_shield_safety_ratio": 0.15,
        "load_shed_cost": 1.0,
    },
    "S4_island_extreme": {
        "env_mode": "islanded",
        "scenario_name": "summer_extreme",
        "extreme_multiplier": {"pv": 0.5, "load": 1.25, "wt": 0.8},
        "description": "Islanded, extreme conditions",
        # Plan D: relax shield from 0.10 -> 0.15 (allow freq closer to 0.5 Hz),
        # reduce freq penalty 3000 -> 1500, and halve load-shed unit cost
        # (2.0 -> 1.0 RMB/kWh) so the policy uses shedding as a last resort
        # rather than a continuous buffer.  interruptible_load stays at 600 kW.
        "interruptible_load_kw": 600.0,
        "freq_penalty_slope": 1500.0,
        "freq_volt_shield_safety_ratio": 0.15,
        "load_shed_cost": 1.0,
    },
}

# Algorithms to run for main comparison
MAIN_ALGORITHMS = [
    "sac",
    "ppo",
    "ppo_lagrangian",
    "cpo",
    "safety_layer",
    "fuzzy_sac",
    "hfg_sac",
]

# Ablation variants (all based on HFG-SAC)
ABLATION_VARIANTS = {
    "hfg_sac_full": "Full HFG-SAC (both layers)",
    "hfg_sac_no_fuzzycon": "HFG-SAC without fuzzy constraint layer",
    "hfg_sac_no_fuzzyknow": "HFG-SAC without fuzzy knowledge layer",
    "hfg_sac_baseline": "Safe SAC baseline (no fuzzy layers)",
}


@dataclass
class RunResult:
    """Result of a single training run."""
    algorithm: str
    scenario: str
    seed: int
    total_cost: float = 0.0
    cost_per_day: float = 0.0
    violation_rate: float = 0.0
    avg_fcsd: float = 0.0
    min_fcsd: float = 1.0
    max_violation: float = 0.0
    convergence_episode: int = 0
    final_reward: float = 0.0
    training_time_s: float = 0.0
    # Training curves
    episode_rewards: List[float] = field(default_factory=list)
    episode_costs: List[float] = field(default_factory=list)
    episode_violation_rates: List[float] = field(default_factory=list)
    episode_fcsds: List[float] = field(default_factory=list)
    # Eval results (final)
    eval_costs: List[float] = field(default_factory=list)
    eval_violations: List[float] = field(default_factory=list)
    eval_fcsds: List[float] = field(default_factory=list)
    # Extra metadata (e.g., transfer similarity table)
    metadata: Dict = field(default_factory=dict)


def create_env_config(
    env_mode: str,
    scenario_cfg: dict,
    seed: int = 42,
    num_days: int = 7,
) -> Tuple[EnvConfig, SyntheticDataLoader]:
    """Create environment config and data loader for a scenario."""
    base_loader = SyntheticDataLoader(
        pv_capacity_kw=500.0,  # 500 kW PV
        wt_capacity_kw=300.0,  # 300 kW WT
        base_load_kw=1200.0,  # 1.2 MW base load
        steps_per_day=96,  # 15-min intervals
        num_days=365,
        seed=seed,
        grid_buy_price=0.8,
    )

    # Create scenario loader
    if scenario_cfg.get("extreme_multiplier"):
        scenario = ScenarioConfig(
            name=scenario_cfg["scenario_name"],
            months=[6, 7, 8] if "summer" in scenario_cfg["scenario_name"] else None,
            extreme_multiplier=scenario_cfg["extreme_multiplier"],
        )
        from hfg_srl.data.scenarios import ScenarioLoader
        data_loader = ScenarioLoader(base_loader, scenario, seed=seed)
    else:
        data_loader = create_scenario(base_loader, scenario_cfg["scenario_name"], seed=seed)

    # Adjust num_days
    env_cfg = EnvConfig(
        mode=env_mode,
        time_steps_per_day=96,
        num_days=num_days,
        pv_capacity_kw=500.0,
        wt_capacity_kw=300.0,
        ess_capacity_kwh=1000.0,  # 1 MWh
        ess_power_kw=300.0,  # 300 kW
        de_rated_power_kw=400.0,  # 400 kW
        base_load_kw=1200.0,
        interruptible_load_kw=scenario_cfg.get("interruptible_load_kw", 200.0),
        grid_power_limit_kw=500.0,
        soc_min=scenario_cfg.get("soc_min", 0.1),
        soc_max=scenario_cfg.get("soc_max", 0.9),
        soc_optimal_min=scenario_cfg.get("soc_optimal_min", 0.3),
        soc_optimal_max=scenario_cfg.get("soc_optimal_max", 0.8),
        grid_buy_price=0.8,
        grid_sell_price=0.4,
        de_fuel_cost=0.6,
        ess_degradation_cost=0.05,
        voltage_limit_pu=1.05,
        frequency_limit_hz=scenario_cfg.get("frequency_limit_hz", 0.5),
        freq_penalty_slope=scenario_cfg.get("freq_penalty_slope", 1000.0),
        load_shed_cost=scenario_cfg.get("load_shed_cost", 2.0),
        seed=seed,
    )
    return env_cfg, data_loader


def train_agent(
    algorithm_name: str,
    scenario_name: str,
    scenario_cfg: dict,
    seed: int,
    n_episodes: int = 200,
    device: str = "cpu",
    output_dir: Optional[Path] = None,
) -> RunResult:
    """Train a single agent on a single scenario."""
    set_seed(seed)
    # Per-worker thread budget. When running under ProcessPoolExecutor with N
    # workers, oversubscribing (e.g. 10 threads x 6 workers on 20 cores)
    # causes ~3x slowdown due to contention. The parallel runner sets
    # HFG_NUM_THREADS; fall back to cpu_count//2 for single-process runs.
    if device == "cpu":
        import torch
        env_threads = os.environ.get("HFG_NUM_THREADS")
        if env_threads and env_threads.isdigit():
            n_threads = max(1, int(env_threads))
        else:
            n_threads = max(1, (os.cpu_count() or 4) // 2)
        torch.set_num_threads(n_threads)

    env_cfg, data_loader = create_env_config(
        scenario_cfg["env_mode"], scenario_cfg, seed=seed, num_days=7,
    )

    # Build HFG config
    algo_cfg = AlgorithmConfig(
        name=algorithm_name,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        hidden_dims=[256, 256],
        activation="relu",
        batch_size=256,
        buffer_size=int(1e5),
        max_steps=int(1e6),
        warmup_steps=1000,
        updates_per_step=1,
        eval_interval=2000,
        eval_episodes=3,
        log_interval=500,
    )

    # Scenario-specific FCSD target. S4 (islanded + PV=0.5x, load=1.25x,
    # wind=0.8x) is physically unable to sustain FCSD=0.85 for long horizons:
    # net_load ~= 1.25 * 1200 kW = 1500 kW vs. DE 400 kW + ESS 300 kW + weak
    # PV/WT, so freq/voltage deviations are unavoidable.  Lowering the target
    # to 0.75 stops the Lagrangian from fighting an unattainable constraint,
    # which was the root cause of late-stage violation-rate climbing
    # (ep 4-18: ~37%, ep 50+: ~55%).
    is_s4_extreme = (
        scenario_name == "S4_island_extreme"
        or (
            scenario_cfg["env_mode"] == "islanded"
            and (scenario_cfg.get("extreme_multiplier") or {}).get("pv", 1.0) < 0.8
        )
    )
    fcsd_target = scenario_cfg.get("fcsd_target") or (0.75 if is_s4_extreme else 0.85)

    # S4: fire B1 freq/voltage shield earlier (at freq>0.20 Hz instead of
    # 0.35 Hz) so it activates while DE/shed still have headroom.
    shield_safety_ratio = scenario_cfg.get("freq_volt_shield_safety_ratio", 0.175)

    fuzzy_cfg = FuzzyConfig(
        num_rules=16,
        num_inputs=9 if scenario_cfg["env_mode"] == "islanded" else 8,
        fcsd_target=fcsd_target,
        use_expert_rules=("no_fuzzyknow" not in algorithm_name),
        reward_shaping_weight_init=0.3 if "no_fuzzyknow" not in algorithm_name else 0.0,
        reward_shaping_weight_final=0.01,
        reward_shaping_decay_steps=10000,
        # Cost-return Lagrangian scale: C_hat ~ discounted deficit (~10-30),
        # Q ~ return-scale cost (~1000-2000). lambda ~ O(1) balances them.
        lambda_init=2.0,
        lambda_lr=0.05,
        lambda_max=50.0,
        freq_volt_shield_safety_ratio=shield_safety_ratio,
    )

    hfg_cfg = HFGConfig(
        env=env_cfg,
        algorithm=algo_cfg,
        fuzzy=fuzzy_cfg,
        seed=seed,
        device=device,
    )

    # Create env
    env = make_env(scenario_cfg["env_mode"], env_cfg, data_loader=data_loader)
    env.seed(seed)

    obs_dim = env.observation_space_shape[0]
    act_dim = env.action_space_shape[0]
    action_high = env.action_high
    action_low = env.action_low

    # Adjust algorithm name for ablation variants. The -FuzzyCon ablation
    # uses the dedicated fuzzy_sac agent (lower layer only, no cost critic).
    actual_algo = algorithm_name
    if algorithm_name == "hfg_sac_no_fuzzycon":
        actual_algo = "fuzzy_sac"
    elif algorithm_name.startswith("hfg_sac_no_") or algorithm_name == "hfg_sac_baseline":
        actual_algo = "hfg_sac" if "no_fuzzyknow" in algorithm_name else "safe_sac"
        if algorithm_name == "hfg_sac_baseline":
            actual_algo = "safe_sac"
    elif algorithm_name == "hfg_sac_full":
        actual_algo = "hfg_sac"

    # Create agent
    agent = make_agent(actual_algo, hfg_cfg, obs_dim, act_dim, action_high, device)

    # Note: ablation switches (use_expert_rules /
    # reward_shaping_weight_init) are passed via FuzzyConfig above — no
    # post-construction mutation of the frozen dataclass is needed.

    agent.train()

    # Replay buffer (for off-policy algorithms)
    is_on_policy = algorithm_name in ("ppo", "ppo_lagrangian", "cpo")
    if not is_on_policy:
        buffer = ReplayBuffer(obs_dim, act_dim, algo_cfg.buffer_size, device)

    result = RunResult(
        algorithm=algorithm_name,
        scenario=scenario_name,
        seed=seed,
    )

    steps_per_episode = 96 * 7  # match the 7-day eval horizon
    total_steps = 0
    start_time = time.time()

    best_eval_cost = float("inf")
    # Checkpoint score: cost plus a huge penalty for violations, so the
    # saved policy is safe rather than merely cheap.
    def _ckpt_score(cost: float, viol_pct: float) -> float:
        return cost + 1.0e6 * viol_pct

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_cost = 0.0
        episode_violations = 0
        episode_fcsd_sum = 0.0
        episode_min_fcsd = 1.0
        episode_max_viol = 0.0

        for step in range(steps_per_episode):
            # Action selection
            if not is_on_policy and total_steps < algo_cfg.warmup_steps:
                action = np.random.uniform(
                    low=action_low, high=action_high, size=act_dim
                ).astype(np.float32)
            else:
                action = agent.select_action(obs, deterministic=False)

            # Clip action to valid range
            action = np.clip(action, action_low, action_high)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            if not is_on_policy:
                buffer.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                    fcsd=info.get("fcsd", 1.0),
                    fcsd_min=info.get("fcsd_min", 1.0),
                    constraint_cost=info.get("constraint_violation", 0.0),
                )

                # Update every 4 steps (update_skip=4).  SAC with batch_size=256
                # already gives a low-variance gradient estimate; skipping to 4
                # cuts gradient updates from 67K to 33K with minimal policy
                # quality impact (verified on S1/S2 cost & violation rate).
                if total_steps >= algo_cfg.warmup_steps and total_steps % 4 == 0:
                    batch = buffer.sample(algo_cfg.batch_size)
                    agent.update(batch, total_steps)
            else:
                # On-policy: store in agent's rollout buffer
                if hasattr(agent, "store_reward_and_done"):
                    if "ppo_lagrangian" in algorithm_name or "cpo" in algorithm_name:
                        # Binary per-step cost: 1.0 on a hard constraint violation.
                        agent.store_reward_and_done(
                            reward, float(info.get("violated", False)), done
                        )
                    else:
                        agent.store_reward_and_done(reward, done)

            # Track metrics
            episode_reward += reward
            episode_cost += info.get("cost", 0.0)
            fcsd = info.get("fcsd", 1.0)
            episode_fcsd_sum += fcsd
            episode_min_fcsd = min(episode_min_fcsd, fcsd)
            viol = info.get("constraint_violation", 0.0)
            episode_max_viol = max(episode_max_viol, viol)
            if info.get("violated", False):
                episode_violations += 1

            obs = next_obs
            total_steps += 1

            if done:
                break

        # On-policy update at end of episode. Bootstrap V(s_last) for
        # time-limit truncation before performing the update.
        if is_on_policy and total_steps >= algo_cfg.warmup_steps:
            if hasattr(agent, "finish_rollout"):
                agent.finish_rollout(obs, truncated=truncated)
            agent.update(None, episode)
        elif is_on_policy:
            # Discard pre-warmup rollouts so they never contaminate an update.
            agent.rollout = type(agent.rollout)(with_cost=agent.rollout.with_cost)

        # Record episode metrics
        result.episode_rewards.append(episode_reward)
        result.episode_costs.append(episode_cost)
        result.episode_violation_rates.append(episode_violations / max(step + 1, 1) * 100.0)
        result.episode_fcsds.append(episode_fcsd_sum / max(step + 1, 1))

        # Periodic evaluation.
        # Fix-8: eval every 50 episodes with 1 eval episode (was every 20 with
        # 2 episodes).  Mid-training eval is only used for checkpoint
        # selection, not for reported metrics — the final 10-episode eval
        # below remains the authoritative number.  This cuts mid-training
        # eval step count by ~80%.
        if (episode + 1) % 50 == 0 or episode == n_episodes - 1:
            eval_metrics = evaluate_agent(agent, env, n_episodes=1)
            result.eval_costs.append(eval_metrics["cost_mean"])
            result.eval_violations.append(eval_metrics["violation_rate_mean"])
            result.eval_fcsds.append(eval_metrics["fcsd_mean"])

            ckpt_score = _ckpt_score(eval_metrics["cost_mean"], eval_metrics["violation_rate_mean"])
            if ckpt_score < best_eval_cost:
                best_eval_cost = ckpt_score
                if output_dir:
                    ckpt_dir = output_dir / "checkpoints" / f"{algorithm_name}_{scenario_name}_s{seed}"
                    agent.save(str(ckpt_dir))

    result.training_time_s = time.time() - start_time

    # Final evaluation
    final_eval = evaluate_agent(agent, env, n_episodes=10)
    result.total_cost = final_eval["cost_mean"]
    result.cost_per_day = final_eval["cost_mean"] / 7.0
    result.violation_rate = final_eval["violation_rate_mean"]
    result.avg_fcsd = final_eval["fcsd_mean"]
    result.min_fcsd = final_eval["fcsd_min"]
    result.final_reward = final_eval["reward_mean"]

    # Estimate convergence (episode where cost reaches 95% of best)
    if len(result.episode_costs) > 10:
        window = 10
        smoothed = np.convolve(
            result.episode_costs, np.ones(window) / window, mode="valid"
        )
        best_cost = np.min(smoothed[len(smoothed) // 2:]) if len(smoothed) > 20 else np.min(smoothed)
        target = best_cost * 1.05
        for i, c in enumerate(smoothed):
            if c <= target:
                result.convergence_episode = i + window
                break

    # Save result
    if output_dir:
        result_dir = output_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{algorithm_name}_{scenario_name}_s{seed}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)

    print(f"  [{algorithm_name} on {scenario_name} seed={seed}] "
          f"cost={result.total_cost:.1f}  viol={result.violation_rate:.2f}%  "
          f"fcsd={result.avg_fcsd:.4f}  time={result.training_time_s:.1f}s")

    return result


def evaluate_agent(agent, env, n_episodes: int = 5) -> Dict[str, float]:
    """Evaluate agent and return metrics."""
    agent.eval()
    costs = []
    rewards = []
    violation_rates = []
    fcsds = []
    fcsd_mins = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_cost = 0.0
        ep_reward = 0.0
        ep_violations = 0
        ep_fcsd_sum = 0.0
        ep_min_fcsd = 1.0
        steps = 0
        done = False

        while not done and steps < 96 * 7:
            action = agent.select_action(obs, deterministic=True)
            action = np.clip(action, env.action_low, env.action_high)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_cost += info.get("cost", 0.0)
            ep_reward += reward
            fcsd = info.get("fcsd", 1.0)
            ep_fcsd_sum += fcsd
            ep_min_fcsd = min(ep_min_fcsd, fcsd)
            if info.get("violated", False):
                ep_violations += 1
            steps += 1

        costs.append(ep_cost)
        rewards.append(ep_reward)
        violation_rates.append(ep_violations / max(steps, 1) * 100.0)
        fcsds.append(ep_fcsd_sum / max(steps, 1))
        fcsd_mins.append(ep_min_fcsd)

    agent.train()

    return {
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "violation_rate_mean": float(np.mean(violation_rates)),
        "violation_rate_std": float(np.std(violation_rates)),
        "fcsd_mean": float(np.mean(fcsds)),
        "fcsd_std": float(np.std(fcsds)),
        "fcsd_min": float(np.min(fcsd_mins)),
    }


def run_model_baselines(
    scenario_name: str,
    scenario_cfg: dict,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict]:
    """Run MILP and MPC baselines for a scenario."""
    env_cfg, data_loader = create_env_config(
        scenario_cfg["env_mode"], scenario_cfg, seed=42, num_days=7,
    )

    results = {}

    # MILP
    milp = MILPOracle(env_cfg)
    if scenario_cfg["env_mode"] == "grid_connected":
        milp_result = milp.solve_grid_connected(data_loader, num_days=7)
    else:
        milp_result = milp.solve_islanded(data_loader, num_days=7)

    results["MILP"] = {
        "cost": milp_result.cost,
        "violation_rate": milp_result.violation_rate,
        "avg_fcsd": milp_result.avg_fcsd,
        "max_violation": milp_result.max_violation,
    }

    # MPC
    mpc = MPC(env_cfg, horizon_hours=24.0)
    if scenario_cfg["env_mode"] == "grid_connected":
        mpc_result = mpc.run_grid_connected(data_loader, num_days=7)
    else:
        mpc_result = mpc.run_islanded(data_loader, num_days=7)

    results["MPC"] = {
        "cost": mpc_result.cost,
        "violation_rate": mpc_result.violation_rate,
        "avg_fcsd": mpc_result.avg_fcsd,
        "max_violation": mpc_result.max_violation,
    }

    print(f"  [MILP on {scenario_name}] cost={milp_result.cost:.1f}  viol={milp_result.violation_rate:.2f}%")
    print(f"  [MPC on {scenario_name}]  cost={mpc_result.cost:.1f}  viol={mpc_result.violation_rate:.2f}%")

    if output_dir:
        result_dir = output_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        with open(result_dir / f"model_baselines_{scenario_name}.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def run_main_comparison(
    output_dir: Path,
    seeds: List[int],
    n_episodes: int,
    device: str,
    algorithms: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Run main performance comparison (RQ1)."""
    print("\n" + "=" * 60)
    print("RQ1: Main Performance Comparison")
    print("=" * 60)

    if algorithms is None:
        algorithms = MAIN_ALGORITHMS

    all_results = {}

    for scenario_name, scenario_cfg in SCENARIOS.items():
        print(f"\nScenario: {scenario_name} ({scenario_cfg['description']})")

        # Model-based baselines (deterministic, single run)
        model_results = run_model_baselines(scenario_name, scenario_cfg, output_dir)
        all_results[scenario_name] = {"model_baselines": model_results}

        # RL algorithms
        rl_results = {}
        for algo in algorithms:
            algo_results = []
            for seed in seeds:
                result = train_agent(
                    algo, scenario_name, scenario_cfg, seed,
                    n_episodes=n_episodes, device=device, output_dir=output_dir,
                )
                algo_results.append(result)
            rl_results[algo] = algo_results
        all_results[scenario_name]["rl_algorithms"] = rl_results

    # Save summary
    summary = {}
    for scenario_name, scenario_data in all_results.items():
        summary[scenario_name] = {}
        # Model baselines
        for name, res in scenario_data.get("model_baselines", {}).items():
            summary[scenario_name][name] = {
                "cost": res["cost"],
                "cost_std": 0.0,
                "violation_rate": res["violation_rate"],
                "violation_rate_std": 0.0,
                "avg_fcsd": res["avg_fcsd"],
            }
        # RL algorithms
        for algo_name, results_list in scenario_data.get("rl_algorithms", {}).items():
            costs = [r.total_cost for r in results_list]
            viols = [r.violation_rate for r in results_list]
            fcsds = [r.avg_fcsd for r in results_list]
            summary[scenario_name][algo_name] = {
                "cost_mean": float(np.mean(costs)),
                "cost_std": float(np.std(costs)),
                "violation_rate_mean": float(np.mean(viols)),
                "violation_rate_std": float(np.std(viols)),
                "fcsd_mean": float(np.mean(fcsds)),
                "fcsd_std": float(np.std(fcsds)),
            }

    with open(output_dir / "results" / "main_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return all_results


def run_ablation_study(
    output_dir: Path,
    seeds: List[int],
    n_episodes: int,
    device: str,
) -> Dict:
    """Run ablation study (RQ4) on S4 scenario."""
    print("\n" + "=" * 60)
    print("RQ4: Ablation Study (S4 Islanded Extreme)")
    print("=" * 60)

    scenario_name = "S4_island_extreme"
    scenario_cfg = SCENARIOS[scenario_name]

    ablation_results = {}
    for variant_name, description in ABLATION_VARIANTS.items():
        print(f"\nVariant: {variant_name} ({description})")
        variant_results = []
        for seed in seeds:
            result = train_agent(
                variant_name, scenario_name, scenario_cfg, seed,
                n_episodes=n_episodes, device=device, output_dir=output_dir,
            )
            variant_results.append(result)
        ablation_results[variant_name] = variant_results

    # Summary
    summary = {}
    for name, results_list in ablation_results.items():
        costs = [r.total_cost for r in results_list]
        viols = [r.violation_rate for r in results_list]
        fcsds = [r.avg_fcsd for r in results_list]
        convs = [r.convergence_episode for r in results_list]
        summary[name] = {
            "cost_mean": float(np.mean(costs)),
            "cost_std": float(np.std(costs)),
            "violation_rate_mean": float(np.mean(viols)),
            "violation_rate_std": float(np.std(viols)),
            "fcsd_mean": float(np.mean(fcsds)),
            "fcsd_std": float(np.std(fcsds)),
            "convergence_mean": float(np.mean(convs)),
            "convergence_std": float(np.std(convs)),
        }

    with open(output_dir / "results" / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return ablation_results


def run_sensitivity_analysis(
    output_dir: Path,
    seeds: List[int],
    n_episodes: int,
    device: str,
) -> Dict:
    """Run sensitivity analysis (β, α_target, κ₀) on S2 scenario."""
    print("\n" + "=" * 60)
    print("RQ5: Sensitivity Analysis (S2 Grid-Connected Extreme)")
    print("=" * 60)

    scenario_name = "S2_grid_extreme"
    scenario_cfg = SCENARIOS[scenario_name]

    sensitivity_results = {}

    # 1. Fuzzy boundary width (β equivalent — mapped via fuzzy constraint width)
    print("\n1. Fuzzy boundary width sensitivity")
    beta_values = [0.02, 0.05, 0.1, 0.2, 0.4]  # constraint width values
    beta_results = {}
    for width in beta_values:
        # Temporarily modify config
        results = []
        for seed in seeds:
            # We'll use a modified training with custom width
            set_seed(seed)
            env_cfg, data_loader = create_env_config(
                scenario_cfg["env_mode"], scenario_cfg, seed=seed, num_days=7,
            )
            # Modify fuzzy constraint width — we'll run via train_agent with override
            result = _train_with_param(
                "hfg_sac", scenario_name, scenario_cfg, seed,
                n_episodes=n_episodes, device=device, output_dir=output_dir,
                constraint_width=width,
            )
            results.append(result)
        beta_results[str(width)] = results
    sensitivity_results["beta"] = beta_results

    # 2. Target satisfaction degree (α_target)
    print("\n2. Target FCSD sensitivity")
    alpha_targets = [0.7, 0.8, 0.85, 0.9, 0.95]
    alpha_results = {}
    for alpha in alpha_targets:
        results = []
        for seed in seeds:
            result = _train_with_param(
                "hfg_sac", scenario_name, scenario_cfg, seed,
                n_episodes=n_episodes, device=device, output_dir=output_dir,
                fcsd_target=alpha,
            )
            results.append(result)
        alpha_results[str(alpha)] = results
    sensitivity_results["alpha_target"] = alpha_results

    # 3. Knowledge shaping weight (κ₀)
    print("\n3. Knowledge shaping weight sensitivity")
    kappa_values = [0.0, 0.1, 0.3, 0.5, 1.0]
    kappa_results = {}
    for kappa in kappa_values:
        results = []
        for seed in seeds:
            result = _train_with_param(
                "hfg_sac", scenario_name, scenario_cfg, seed,
                n_episodes=n_episodes, device=device, output_dir=output_dir,
                shaping_weight=kappa,
            )
            results.append(result)
        kappa_results[str(kappa)] = results
    sensitivity_results["kappa"] = kappa_results

    # Save summary
    summary = {}
    for param_name, param_results in sensitivity_results.items():
        summary[param_name] = {}
        for value, results_list in param_results.items():
            costs = [r.total_cost for r in results_list]
            viols = [r.violation_rate for r in results_list]
            fcsds = [r.avg_fcsd for r in results_list]
            summary[param_name][value] = {
                "cost_mean": float(np.mean(costs)),
                "cost_std": float(np.std(costs)),
                "violation_rate_mean": float(np.mean(viols)),
                "violation_rate_std": float(np.std(viols)),
                "fcsd_mean": float(np.mean(fcsds)),
                "fcsd_std": float(np.std(fcsds)),
            }

    with open(output_dir / "results" / "sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return sensitivity_results


def _train_with_param(
    algorithm_name: str,
    scenario_name: str,
    scenario_cfg: dict,
    seed: int,
    n_episodes: int,
    device: str,
    output_dir: Path,
    constraint_width: Optional[float] = None,
    fcsd_target: Optional[float] = None,
    shaping_weight: Optional[float] = None,
) -> RunResult:
    """Train with a specific parameter override."""
    # Use same training as train_agent but with modified config
    set_seed(seed)

    env_cfg, data_loader = create_env_config(
        scenario_cfg["env_mode"], scenario_cfg, seed=seed, num_days=7,
    )

    # Modify constraint width in env config if needed
    # (This is approximate — the fuzzy constraint width is set at env level)

    algo_cfg = AlgorithmConfig(
        name=algorithm_name,
        hidden_dims=[256, 256],
        batch_size=128,
        buffer_size=int(1e5),
        warmup_steps=500,
        eval_interval=2000,
        log_interval=500,
    )

    fcsd_t = fcsd_target if fcsd_target is not None else 0.85
    sw_init = shaping_weight if shaping_weight is not None else 0.3
    # Map constraint_width (β sweep) to the real fuzzy boundary knob:
    # hard_violation_threshold — the FCSD boundary between hard and soft
    # violation regions in the cost-critic target.
    hard_thresh = constraint_width if constraint_width is not None else 0.5

    fuzzy_cfg = FuzzyConfig(
        num_rules=16,
        num_inputs=8,
        fcsd_target=fcsd_t,
        use_expert_rules=sw_init > 0,
        reward_shaping_weight_init=sw_init,
        reward_shaping_weight_final=0.01,
        reward_shaping_decay_steps=10000,
        hard_violation_threshold=hard_thresh,
    )

    hfg_cfg = HFGConfig(
        env=env_cfg,
        algorithm=algo_cfg,
        fuzzy=fuzzy_cfg,
        seed=seed,
        device=device,
    )

    env = make_env(scenario_cfg["env_mode"], env_cfg, data_loader=data_loader)
    env.seed(seed)

    obs_dim = env.observation_space_shape[0]
    act_dim = env.action_space_shape[0]
    action_high = env.action_high
    action_low = env.action_low

    agent = make_agent("hfg_sac", hfg_cfg, obs_dim, act_dim, action_high, device)
    agent.train()

    buffer = ReplayBuffer(obs_dim, act_dim, algo_cfg.buffer_size, device)

    result = RunResult(
        algorithm=f"hfg_sac_param",
        scenario=scenario_name,
        seed=seed,
    )

    steps_per_episode = 96 * 7
    total_steps = 0
    start_time = time.time()

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_cost = 0.0
        episode_violations = 0
        episode_fcsd_sum = 0.0

        for step in range(steps_per_episode):
            if total_steps < algo_cfg.warmup_steps:
                action = np.random.uniform(
                    low=action_low, high=action_high, size=act_dim
                ).astype(np.float32)
            else:
                action = agent.select_action(obs, deterministic=False)
            action = np.clip(action, action_low, action_high)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(
                obs=obs, action=action, reward=reward, next_obs=next_obs,
                done=done, fcsd=info.get("fcsd", 1.0),
                fcsd_min=info.get("fcsd_min", 1.0),
                constraint_cost=info.get("constraint_violation", 0.0),
            )

            if total_steps >= algo_cfg.warmup_steps:
                batch = buffer.sample(algo_cfg.batch_size)
                agent.update(batch, total_steps)

            episode_reward += reward
            episode_cost += info.get("cost", 0.0)
            episode_fcsd_sum += info.get("fcsd", 1.0)
            if info.get("violated", False):
                episode_violations += 1

            obs = next_obs
            total_steps += 1
            if done:
                break

        result.episode_rewards.append(episode_reward)
        result.episode_costs.append(episode_cost)
        result.episode_violation_rates.append(episode_violations / max(step + 1, 1) * 100.0)
        result.episode_fcsds.append(episode_fcsd_sum / max(step + 1, 1))

    result.training_time_s = time.time() - start_time
    final_eval = evaluate_agent(agent, env, n_episodes=10)
    result.total_cost = final_eval["cost_mean"]
    result.cost_per_day = final_eval["cost_mean"] / 7.0
    result.violation_rate = final_eval["violation_rate_mean"]
    result.avg_fcsd = final_eval["fcsd_mean"]
    result.min_fcsd = final_eval["fcsd_min"]
    result.final_reward = final_eval["reward_mean"]

    return result


def run_extremity_analysis(
    output_dir: Path,
    seeds: List[int],
    n_episodes: int,
    device: str,
) -> Dict:
    """Run robustness under varying extremity (RQ2) on islanded mode."""
    print("\n" + "=" * 60)
    print("RQ2: Robustness under Varying Extremity")
    print("=" * 60)

    extremity_levels = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    # 5 algorithms: SAC (no safety), PPO-Lagrangian (crisp), CPO (constrained
    # policy opt), FuzzySAC (fuzzy reward shaping only, no Lagrangian), and
    # HFG-SAC (full two-layer).  FuzzySAC is the key ablation control that
    # isolates "fuzzy boundary" vs "fuzzy + Lagrangian synergy".
    algorithms = ["sac", "ppo_lagrangian", "cpo", "fuzzy_sac", "hfg_sac"]

    results = {}
    for algo in algorithms:
        algo_results = {}
        for level in extremity_levels:
            # Create scenario with this extremity level
            scenario_cfg = {
                "env_mode": "islanded",
                "scenario_name": "extremity_test",
                "extreme_multiplier": {"pv": max(0.2, 1.0 - level * 0.3), "load": 1.0 + level * 0.2, "wt": 1.0 - level * 0.15},
                "description": f"Islanded with extremity level {level}",
            }
            level_results = []
            for seed in seeds:
                result = train_agent(
                    algo, f"extremity_{level}", scenario_cfg, seed,
                    n_episodes=n_episodes, device=device, output_dir=output_dir,
                )
                level_results.append(result)
            algo_results[str(level)] = level_results
        results[algo] = algo_results

    # Summary
    summary = {}
    for algo, algo_data in results.items():
        summary[algo] = {}
        for level, results_list in algo_data.items():
            costs = [r.total_cost for r in results_list]
            viols = [r.violation_rate for r in results_list]
            summary[algo][level] = {
                "cost_mean": float(np.mean(costs)),
                "cost_std": float(np.std(costs)),
                "violation_rate_mean": float(np.mean(viols)),
                "violation_rate_std": float(np.std(viols)),
            }

    with open(output_dir / "results" / "extremity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results


def run_transfer_experiments(
    output_dir: Path,
    seeds: List[int],
    n_episodes: int,
    device: str,
) -> Dict:
    """Run transfer learning experiments (RQ3)."""
    print("\n" + "=" * 60)
    print("RQ3: Transfer Learning")
    print("=" * 60)

    # T1: Summer -> Winter (grid-connected)
    # T2: Grid -> Island
    # T3: Normal -> Extreme (islanded)

    transfer_tasks = {
        # T1: grid normal -> grid extreme, same battery band [0.3, 0.8]
        "T1_grid_same": {
            "source": {
                "env_mode": "grid_connected",
                "scenario_name": "typical_week_summer",
                "extreme_multiplier": None,
            },
            "target": {
                "env_mode": "grid_connected",
                "scenario_name": "summer_extreme",
                "extreme_multiplier": {"pv": 1.2, "load": 1.3, "wt": 0.6},
            },
        },
        # T2: grid normal -> grid extreme, target battery band narrowed to [0.4, 0.7]
        "T2_grid_shifted": {
            "source": {
                "env_mode": "grid_connected",
                "scenario_name": "typical_week_summer",
                "extreme_multiplier": None,
            },
            "target": {
                "env_mode": "grid_connected",
                "scenario_name": "summer_extreme",
                "extreme_multiplier": {"pv": 1.2, "load": 1.3, "wt": 0.6},
                "soc_optimal_min": 0.4,
                "soc_optimal_max": 0.7,
            },
        },
        # T3: island normal -> island extreme, same battery band [0.3, 0.8]
        "T3_island_same": {
            "source": {
                "env_mode": "islanded",
                "scenario_name": "typical_week_summer",
                "extreme_multiplier": None,
                "interruptible_load_kw": 600.0,
                "freq_penalty_slope": 1500.0,
                "freq_volt_shield_safety_ratio": 0.15,
                "load_shed_cost": 1.0,
            },
            "target": {
                "env_mode": "islanded",
                "scenario_name": "summer_extreme",
                "extreme_multiplier": {"pv": 0.5, "load": 1.25, "wt": 0.8},
                "interruptible_load_kw": 600.0,
                "freq_penalty_slope": 1500.0,
                "freq_volt_shield_safety_ratio": 0.15,
                "load_shed_cost": 1.0,
            },
        },
        # T4: island normal -> island extreme, target battery band narrowed to [0.4, 0.7]
        "T4_island_shifted": {
            "source": {
                "env_mode": "islanded",
                "scenario_name": "typical_week_summer",
                "extreme_multiplier": None,
                "interruptible_load_kw": 600.0,
                "freq_penalty_slope": 1500.0,
                "freq_volt_shield_safety_ratio": 0.15,
                "load_shed_cost": 1.0,
            },
            "target": {
                "env_mode": "islanded",
                "scenario_name": "summer_extreme",
                "extreme_multiplier": {"pv": 0.5, "load": 1.25, "wt": 0.8},
                "interruptible_load_kw": 600.0,
                "freq_penalty_slope": 1500.0,
                "freq_volt_shield_safety_ratio": 0.15,
                "load_shed_cost": 1.0,
                "soc_optimal_min": 0.4,
                "soc_optimal_max": 0.7,
            },
        },
    }

    results = {}

    for task_name, task_cfg in transfer_tasks.items():
        print(f"\nTransfer Task: {task_name}")
        task_results = {}

        # 1. From-scratch training on target (HFG-SAC)
        print("  From scratch (HFG-SAC)...")
        scratch_results = []
        for seed in seeds:
            result = train_agent(
                "hfg_sac", f"{task_name}_scratch", task_cfg["target"], seed,
                n_episodes=n_episodes, device=device, output_dir=output_dir,
            )
            scratch_results.append(result)
        task_results["from_scratch"] = scratch_results

        # 2. External anchor: PPO-Lagrangian from scratch on target.
        # This is the "no transfer at all" baseline for a non-fuzzy safe RL
        # algorithm.  Comparing HFG conservative transfer's sample-efficiency
        # against this anchor lets us claim "transfer reduces sample cost by
        # >= X%" relative to a state-of-the-art safe RL baseline.
        print("  From scratch (PPO-Lagrangian anchor)...")
        anchor_results = []
        for seed in seeds:
            result = train_agent(
                "ppo_lagrangian", f"{task_name}_anchor", task_cfg["target"],
                seed, n_episodes=n_episodes, device=device, output_dir=output_dir,
            )
            anchor_results.append(result)
        task_results["ppo_lagrangian_scratch"] = anchor_results

        # 3. Naive transfer (direct copy + fine-tune)
        print("  Naive transfer...")
        naive_results = []
        for seed in seeds:
            result = _run_transfer(
                task_cfg["source"], task_cfg["target"], seed,
                n_episodes=n_episodes, device=device, output_dir=output_dir,
                conservative=False,
            )
            naive_results.append(result)
        task_results["naive_transfer"] = naive_results

        # 4. HFG-SAC transfer (conservative initialization)
        print("  HFG-SAC transfer...")
        hfg_results = []
        for seed in seeds:
            result = _run_transfer(
                task_cfg["source"], task_cfg["target"], seed,
                n_episodes=n_episodes, device=device, output_dir=output_dir,
                conservative=True,
            )
            hfg_results.append(result)
        task_results["hfg_transfer"] = hfg_results

        results[task_name] = task_results

    # Summary
    summary = {}
    for task_name, task_data in results.items():
        summary[task_name] = {}
        for method, results_list in task_data.items():
            costs = [r.total_cost for r in results_list]
            viols = [r.violation_rate for r in results_list]
            convs = [r.convergence_episode for r in results_list]
            summary[task_name][method] = {
                "cost_mean": float(np.mean(costs)),
                "cost_std": float(np.std(costs)),
                "violation_rate_mean": float(np.mean(viols)),
                "violation_rate_std": float(np.std(viols)),
                "convergence_mean": float(np.mean(convs)),
                "convergence_std": float(np.std(convs)),
            }

    with open(output_dir / "results" / "transfer_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results


def build_constraint_specs(
    source_env_cfg,
    target_env_cfg,
) -> List[Dict]:
    """Build per-constraint membership specs for M1 (Jaccard) and M2 (param transfer).

    Maps EnvConfig fields onto the Gaussian membership model used by
    similarity.py:

    - **soc_upper**: μ = soc_optimal_max, β = soc_max - soc_optimal_max
    - **soc_lower**: μ = soc_optimal_min, β = soc_optimal_min - soc_min
    - **freq_dev** (islanded only): μ = 0, β = frequency_limit_hz / 3
    - **voltage_dev** (islanded only): μ = 0, β = voltage_limit_pu / 3

    Args:
        source_env_cfg: Source EnvConfig.
        target_env_cfg: Target EnvConfig.

    Returns:
        List of spec dicts with keys: name, source_mu, source_beta,
        target_mu, target_beta, x_range.
    """
    specs: List[Dict] = []

    # SOC upper band (both modes)
    specs.append({
        "name": "soc_upper",
        "source_mu": float(source_env_cfg.soc_optimal_max),
        "source_beta": max(float(source_env_cfg.soc_max - source_env_cfg.soc_optimal_max), 1e-3),
        "target_mu": float(target_env_cfg.soc_optimal_max),
        "target_beta": max(float(target_env_cfg.soc_max - target_env_cfg.soc_optimal_max), 1e-3),
        "x_range": (0.0, 1.0),
    })

    # SOC lower band (both modes)
    specs.append({
        "name": "soc_lower",
        "source_mu": float(source_env_cfg.soc_optimal_min),
        "source_beta": max(float(source_env_cfg.soc_optimal_min - source_env_cfg.soc_min), 1e-3),
        "target_mu": float(target_env_cfg.soc_optimal_min),
        "target_beta": max(float(target_env_cfg.soc_optimal_min - target_env_cfg.soc_min), 1e-3),
        "x_range": (0.0, 1.0),
    })

    # Frequency deviation (islanded only)
    if source_env_cfg.mode == "islanded" or target_env_cfg.mode == "islanded":
        specs.append({
            "name": "freq_dev",
            "source_mu": 0.0,
            "source_beta": max(float(source_env_cfg.frequency_limit_hz) / 3.0, 1e-3),
            "target_mu": 0.0,
            "target_beta": max(float(target_env_cfg.frequency_limit_hz) / 3.0, 1e-3),
            "x_range": (-1.0, 1.0),
        })

    return specs


def compute_m2_adjustments(spec: Dict) -> Tuple[float, float]:
    """Derive M2 adaptable-constraint adjustments from the safety specs.

    The target safety specification is known at deployment time (battery
    operating band, grid-code frequency limit), so the adjustment of an
    *adaptable* constraint is computed directly from the measured spec
    difference rather than treated as a tuned hyperparameter:

        δ_k = μ^t − μ^s   (required center shift)
        Δ_k = β^t / β^s   (required width scaling)

    With these values, M2 initialization (x_ref^s + δ_k, β^s·Δ_k) lands
    exactly on the target membership parameters, warm-starting the target
    fuzzy layer at its correct specification while the actor trunk is
    copied from the source.  Transferable constraints copy source params
    unchanged (δ=0, Δ=1); relearn constraints ignore them and use expert
    defaults.

    Args:
        spec: One entry produced by :func:`build_constraint_specs`.

    Returns:
        (delta, Delta) as Python floats.
    """
    delta = float(spec["target_mu"]) - float(spec["source_mu"])
    source_beta = max(float(spec["source_beta"]), 1e-6)
    Delta = float(spec["target_beta"]) / source_beta
    return delta, Delta


def _run_transfer(
    source_cfg: dict,
    target_cfg: dict,
    seed: int,
    n_episodes: int,
    device: str,
    output_dir: Path,
    conservative: bool = True,
    use_M1: bool = True,
    use_M3: bool = True,
    use_M4: bool = True,
) -> RunResult:
    """Run a single transfer experiment."""
    set_seed(seed)

    # Step 1: Train source policy
    source_env_cfg, source_loader = create_env_config(
        source_cfg["env_mode"], source_cfg, seed=seed, num_days=7,
    )
    source_hfg_cfg = HFGConfig(
        env=source_env_cfg,
        algorithm=AlgorithmConfig(
            name="hfg_sac",
            hidden_dims=[256, 256],
            batch_size=128,
            buffer_size=int(1e5),
            warmup_steps=500,
            eval_interval=2000,
            log_interval=500,
        ),
        fuzzy=FuzzyConfig(
            num_rules=16,
            num_inputs=8 if source_cfg["env_mode"] == "grid_connected" else 9,
            fcsd_target=0.85,
            use_expert_rules=True,
            reward_shaping_weight_init=0.3,
            reward_shaping_weight_final=0.01,
            reward_shaping_decay_steps=10000,
        ),
        seed=seed,
        device=device,
    )

    source_env = make_env(source_cfg["env_mode"], source_env_cfg, data_loader=source_loader)
    source_env.seed(seed)
    source_obs_dim = source_env.observation_space_shape[0]
    source_act_dim = source_env.action_space_shape[0]
    source_action_high = source_env.action_high

    source_agent = make_agent("hfg_sac", source_hfg_cfg, source_obs_dim, source_act_dim, source_action_high, device)
    source_agent.train()

    # Quick source training (fewer episodes for transfer)
    source_buffer = ReplayBuffer(source_obs_dim, source_act_dim, int(1e5), device)
    steps_per_episode = 96 * 7
    total_steps = 0
    source_episodes = min(100, n_episodes // 2)

    for ep in range(source_episodes):
        obs, info = source_env.reset()
        for _ in range(steps_per_episode):
            if total_steps < 500:
                action = np.random.uniform(
                    low=source_env.action_low, high=source_env.action_high, size=source_act_dim
                ).astype(np.float32)
            else:
                action = source_agent.select_action(obs, deterministic=False)
            action = np.clip(action, source_env.action_low, source_env.action_high)
            next_obs, reward, terminated, truncated, info = source_env.step(action)
            done = terminated or truncated
            source_buffer.add(
                obs=obs, action=action, reward=reward, next_obs=next_obs,
                done=done, fcsd=info.get("fcsd", 1.0),
                fcsd_min=info.get("fcsd_min", 1.0),
                constraint_cost=info.get("constraint_violation", 0.0),
            )
            if total_steps >= 500:
                batch = source_buffer.sample(128)
                source_agent.update(batch, total_steps)
            obs = next_obs
            total_steps += 1
            if done:
                break

    # Step 2: Create target env and transfer
    target_env_cfg, target_loader = create_env_config(
        target_cfg["env_mode"], target_cfg, seed=seed, num_days=7,
    )

    # M1: Compute per-constraint Jaccard similarity (paper 4.4.1)
    constraint_specs = build_constraint_specs(source_env_cfg, target_env_cfg)
    constraint_similarities: Dict[str, float] = {}
    constraint_categories: Dict[str, str] = {}
    if conservative and use_M1:
        from hfg_srl.transfer.similarity import (
            jaccard_similarity, categorize_constraints, gaussian_membership,
        )
        for spec in constraint_specs:
            name = spec["name"]
            def mu_s(x, _s_mu=spec["source_mu"], _s_b=spec["source_beta"]):
                return gaussian_membership(x, _s_mu, _s_b)
            def mu_t(x, _t_mu=spec["target_mu"], _t_b=spec["target_beta"]):
                return gaussian_membership(x, _t_mu, _t_b)
            sim = jaccard_similarity(mu_s, mu_t, x_range=spec["x_range"])
            constraint_similarities[name] = sim
        constraint_categories = categorize_constraints(constraint_similarities)
        print(f"    M1 constraint similarity: {constraint_similarities}")
        print(f"    M1 categories: {constraint_categories}")
    elif conservative and not use_M1:
        # Ablation: skip Jaccard, treat every constraint as "adaptable"
        # (i.e., use source params with identity adjustment, no categorization).
        for spec in constraint_specs:
            constraint_categories[spec["name"]] = "adaptable"
        print(f"    M1 ablation: all constraints forced to 'adaptable'")

    # M2: derive per-constraint adjustments from the safety specs and compute
    # the target fuzzy-layer initialization (paper 4.4.2 Step 2).
    m2_init: Dict[str, Dict[str, float]] = {}
    if conservative:
        from hfg_srl.transfer.similarity import transfer_membership_params
        for spec in constraint_specs:
            name = spec["name"]
            category = constraint_categories.get(name, "adaptable")
            delta, Delta = compute_m2_adjustments(spec)
            init = transfer_membership_params(
                source_mu=spec["source_mu"],
                source_beta=spec["source_beta"],
                category=category,
                delta=delta,
                Delta=Delta,
                default_mu=spec["target_mu"],
                default_beta=spec["target_beta"],
            )
            m2_init[name] = init
        print(f"    M2 fuzzy-layer init: {m2_init}")

    target_hfg_cfg = HFGConfig(
        env=target_env_cfg,
        algorithm=AlgorithmConfig(
            name="hfg_sac",
            hidden_dims=[256, 256],
            batch_size=128,
            buffer_size=int(1e5),
            warmup_steps=500,
            eval_interval=2000,
            log_interval=500,
        ),
        fuzzy=FuzzyConfig(
            num_rules=16,
            num_inputs=8 if target_cfg["env_mode"] == "grid_connected" else 9,
            fcsd_target=0.85,
            use_expert_rules=True,
            reward_shaping_weight_init=0.3,
            reward_shaping_weight_final=0.01,
            reward_shaping_decay_steps=10000,
        ),
        seed=seed,
        device=device,
    )

    target_env = make_env(target_cfg["env_mode"], target_env_cfg, data_loader=target_loader)
    target_env.seed(seed)
    target_obs_dim = target_env.observation_space_shape[0]
    target_act_dim = target_env.action_space_shape[0]
    target_action_high = target_env.action_high
    target_action_low = target_env.action_low

    target_agent = make_agent("hfg_sac", target_hfg_cfg, target_obs_dim, target_act_dim, target_action_high, device)

    # Transfer weights (only if dimensions match)
    dims_match = source_obs_dim == target_obs_dim and source_act_dim == target_act_dim
    if dims_match:
        if hasattr(source_agent, "actor") and hasattr(target_agent, "actor"):
            try:
                target_agent.actor.load_state_dict(source_agent.actor.state_dict())
                target_agent.critic.load_state_dict(source_agent.critic.state_dict())
                target_agent.critic_target.load_state_dict(source_agent.critic.state_dict())
                print(f"    Transferred weights (obs={source_obs_dim}, act={source_act_dim})")
            except Exception as e:
                print(f"    Weight transfer skipped: {e}")

    # M3: Partial freezing (paper §4.4.2 Step 3).
    # Freeze bottom 50% of actor trunk (general temporal/component features);
    # leave upper trunk + mean/log_std heads trainable.  Only applies when
    # weights were actually copied (dims_match).
    m3_active = False
    if conservative and use_M3 and dims_match:
        actor = target_agent.actor
        if hasattr(actor, "trunk") and isinstance(actor.trunk, nn.Sequential):
            linear_idx = [i for i, m in enumerate(actor.trunk) if isinstance(m, nn.Linear)]
            n_linear = len(linear_idx)
            if n_linear >= 2:
                freeze_count = n_linear // 2  # bottom 50%
                for i in linear_idx[:freeze_count]:
                    for p in actor.trunk[i].parameters():
                        p.requires_grad = False
                m3_active = True
                print(f"    M3: froze bottom {freeze_count}/{n_linear} actor trunk Linear layer(s)")

    # Conservative initialization
    if conservative:
        # Reduce action std for more conservative exploration
        if hasattr(target_agent, "actor") and hasattr(target_agent.actor, "log_std_head"):
            import torch
            with torch.no_grad():
                target_agent.actor.log_std_head.bias.fill_(-3.0)  # more conservative
        # M2: warm-start the reward shaper antecedent from transferred priors.
        m2_entries = 0
        if hasattr(target_agent, "fuzzy_reward_shaper") and m2_init:
            m2_entries = target_agent.fuzzy_reward_shaper.apply_m2_init(
                m2_init, target_cfg["env_mode"],
                target_frequency_limit_hz=float(target_env_cfg.frequency_limit_hz),
            )
            print(f"    M2: applied {m2_entries} transferred fuzzy-prior entries")

    target_agent.train()

    # M4: Initialize TransferAgent for exponential safety relaxation
    from hfg_srl.transfer.transfer_agent import TransferAgent
    transfer_agent = None
    if conservative and use_M4:
        transfer_agent = TransferAgent(
            cfg=target_hfg_cfg,
            source_agent=source_agent,
            target_obs_dim=target_obs_dim,
            target_act_dim=target_act_dim,
            target_action_high=target_action_high,
            device=device,
        )
        # Override decay steps based on n_episodes
        transfer_agent.nu = float(np.log(20.0) / max(n_episodes * steps_per_episode * 0.3, 1))
        transfer_agent.alpha_init = 2.5  # start at 2.5x conservative
    elif conservative and not use_M4:
        print(f"    M4 ablation: conservative action scaling disabled (alpha=1 throughout)")

    # Fine-tuning
    target_buffer = ReplayBuffer(target_obs_dim, target_act_dim, int(1e5), device)

    result = RunResult(
        algorithm="hfg_sac_transfer",
        scenario="transfer_target",
        seed=seed,
    )

    total_steps = 0
    start_time = time.time()

    for episode in range(n_episodes):
        obs, info = target_env.reset()
        episode_reward = 0.0
        episode_cost = 0.0
        episode_violations = 0
        episode_fcsd_sum = 0.0

        for step in range(steps_per_episode):
            if total_steps < 200:
                action = np.random.uniform(
                    low=target_action_low, high=target_action_high, size=target_act_dim
                ).astype(np.float32)
            else:
                action = target_agent.select_action(obs, deterministic=False)

            # M4: Exponential conservative scaling (replaces linear decay)
            if transfer_agent is not None:
                transfer_agent.step()
                scale = transfer_agent.get_conservative_action_scale()
                action = action * scale

            action = np.clip(action, target_action_low, target_action_high)
            next_obs, reward, terminated, truncated, info = target_env.step(action)
            done = terminated or truncated

            target_buffer.add(
                obs=obs, action=action, reward=reward, next_obs=next_obs,
                done=done, fcsd=info.get("fcsd", 1.0),
                fcsd_min=info.get("fcsd_min", 1.0),
                constraint_cost=info.get("constraint_violation", 0.0),
            )

            if total_steps >= 200:
                batch = target_buffer.sample(128)
                target_agent.update(batch, total_steps)

            episode_reward += reward
            episode_cost += info.get("cost", 0.0)
            episode_fcsd_sum += info.get("fcsd", 1.0)
            if info.get("violated", False):
                episode_violations += 1

            obs = next_obs
            total_steps += 1
            if done:
                break

        result.episode_rewards.append(episode_reward)
        result.episode_costs.append(episode_cost)
        result.episode_violation_rates.append(episode_violations / max(step + 1, 1) * 100.0)
        result.episode_fcsds.append(episode_fcsd_sum / max(step + 1, 1))

    result.training_time_s = time.time() - start_time

    # M1/M3: Store similarity table and freezing status for paper Table 5.x
    result.metadata = {
        "m1_similarities": constraint_similarities,
        "m1_categories": constraint_categories,
        "m2_init": m2_init,
        "m3_frozen": m3_active,
        "dims_match": dims_match,
    }

    final_eval = evaluate_agent(target_agent, target_env, n_episodes=10)
    result.total_cost = final_eval["cost_mean"]
    result.cost_per_day = final_eval["cost_mean"] / 7.0
    result.violation_rate = final_eval["violation_rate_mean"]
    result.avg_fcsd = final_eval["fcsd_mean"]
    result.min_fcsd = final_eval["fcsd_min"]
    result.final_reward = final_eval["reward_mean"]

    # Estimate convergence
    if len(result.episode_costs) > 10:
        window = 10
        smoothed = np.convolve(
            result.episode_costs, np.ones(window) / window, mode="valid"
        )
        best = np.min(smoothed)
        target = best * 1.05
        for i, c in enumerate(smoothed):
            if c <= target:
                result.convergence_episode = i + window
                break

    return result


def main():
    parser = argparse.ArgumentParser(description="HFG-SRL Experiment Runner")
    parser.add_argument("--output_dir", type=str, default="outputs/experiments",
                        help="Output directory")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 2024],
                        help="Random seeds (5 seeds for statistical robustness)")
    parser.add_argument("--n_episodes", type=int, default=200,
                        help="Number of training episodes")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cpu/cuda)")
    parser.add_argument("--experiments", type=str, nargs="+",
                        default=["main", "ablation", "sensitivity", "extremity", "transfer"],
                        help="Which experiments to run")
    parser.add_argument("--algorithms", type=str, nargs="+", default=None,
                        help="Specific algorithms to run (main comparison)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Episodes per run: {args.n_episodes}")
    print(f"Experiments: {args.experiments}")

    results = {}

    start_all = time.time()

    if "main" in args.experiments:
        results["main"] = run_main_comparison(
            output_dir, args.seeds, args.n_episodes, device, args.algorithms,
        )

    if "ablation" in args.experiments:
        results["ablation"] = run_ablation_study(
            output_dir, args.seeds, args.n_episodes, device,
        )

    if "sensitivity" in args.experiments:
        results["sensitivity"] = run_sensitivity_analysis(
            output_dir, args.seeds, args.n_episodes, device,
        )

    if "extremity" in args.experiments:
        results["extremity"] = run_extremity_analysis(
            output_dir, args.seeds, args.n_episodes, device,
        )

    if "transfer" in args.experiments:
        results["transfer"] = run_transfer_experiments(
            output_dir, args.seeds, args.n_episodes, device,
        )

    total_time = time.time() - start_all
    print(f"\n\n{'=' * 60}")
    print(f"All experiments complete. Total time: {total_time / 3600:.2f} hours")
    print(f"Results saved to: {output_dir / 'results'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
