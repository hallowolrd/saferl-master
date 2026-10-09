"""Diagnose WHICH constraint is being violated in S3 vs S4.

Loads a trained HFG-SAC checkpoint and runs deterministic evaluation,
logging per-step violation events: which constraint, magnitude, timing.
"""
import sys, os
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import torch
from run.experiment_runner import create_env_config
from hfg_srl.env.islanded import IslandedEnv
from hfg_srl.data.synthetic_loader import SyntheticDataLoader
from hfg_srl.data.scenarios import ScenarioLoader, ScenarioConfig, create_scenario
from hfg_srl.algorithms.hfg_sac import HFG_SAC
from hfg_srl.utils.config import HFGConfig, AlgorithmConfig, EnvConfig, FuzzyConfig
from hfg_srl.utils.seed import set_seed

def make_env(scenario_cfg, seed=42):
    base = SyntheticDataLoader(
        pv_capacity_kw=500.0, wt_capacity_kw=300.0, base_load_kw=1200.0,
        steps_per_day=96, num_days=365, seed=seed, grid_buy_price=0.8,
    )
    if scenario_cfg.get("extreme_multiplier"):
        sc = ScenarioConfig(
            name=scenario_cfg["scenario_name"],
            months=[6,7,8] if "summer" in scenario_cfg["scenario_name"] else None,
            extreme_multiplier=scenario_cfg["extreme_multiplier"],
        )
        loader = ScenarioLoader(base, sc, seed=seed)
    else:
        loader = create_scenario(base, scenario_cfg["scenario_name"], seed=seed)
    env_cfg = EnvConfig(
        mode="islanded", time_steps_per_day=96, num_days=7,
        pv_capacity_kw=500.0, wt_capacity_kw=300.0,
        ess_capacity_kwh=1000.0, ess_power_kw=300.0,
        de_rated_power_kw=400.0, base_load_kw=1200.0,
        interruptible_load_kw=scenario_cfg.get("interruptible_load_kw", 200.0),
        grid_power_limit_kw=500.0,
        soc_min=0.1, soc_max=0.9, soc_optimal_min=0.3, soc_optimal_max=0.8,
        grid_buy_price=0.8, grid_sell_price=0.4, de_fuel_cost=0.6,
        ess_degradation_cost=0.05, voltage_limit_pu=1.05,
        frequency_limit_hz=0.5,
        freq_penalty_slope=scenario_cfg.get("freq_penalty_slope", 1000.0),
        load_shed_cost=scenario_cfg.get("load_shed_cost", 2.0),
        seed=seed,
    )
    return IslandedEnv(env_cfg, data_loader=loader), env_cfg, loader

def evaluate_with_diagnostics(env, agent, n_episodes=3):
    agent.eval()
    viol_events = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            action = np.clip(action, env.action_low, env.action_high)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info.get("violated", False):
                # Determine which constraint
                soc = info["soc"]
                freq = info["freq_deviation"]
                volt = info["voltage_deviation"]
                reasons = []
                if soc > 0.9: reasons.append(f"SOC_high={soc:.3f}")
                if soc < 0.1: reasons.append(f"SOC_low={soc:.3f}")
                if freq > 0.5: reasons.append(f"freq={freq:.3f}Hz")
                if volt > 0.05: reasons.append(f"volt={volt:.4f}")
                viol_events.append({
                    "ep": ep, "step": step, "hour": step/4,
                    "soc": soc, "freq": freq, "volt": volt,
                    "imbalance_ratio": info["imbalance_ratio"],
                    "load_shed_kw": info["load_shed"],
                    "reasons": reasons,
                })
            step += 1
    agent.train()
    return viol_events

S3_CFG = {
    "env_mode": "islanded", "scenario_name": "typical_week_summer",
    "extreme_multiplier": None,
    "freq_penalty_slope": 1500.0, "freq_volt_shield_safety_ratio": 0.13,
    "interruptible_load_kw": 200.0, "load_shed_cost": 2.0,
}
S4_CFG = {
    "env_mode": "islanded", "scenario_name": "summer_extreme",
    "extreme_multiplier": {"pv": 0.5, "load": 1.25, "wt": 0.8},
    "freq_penalty_slope": 1500.0, "freq_volt_shield_safety_ratio": 0.15,
    "interruptible_load_kw": 600.0, "load_shed_cost": 1.0,
}

for scen_name, scen_cfg in [("S3", S3_CFG), ("S4", S4_CFG)]:
    print(f"\n{'='*60}")
    print(f"  {scen_name} diagnostics")
    print(f"{'='*60}")
    env, env_cfg, loader = make_env(scen_cfg, seed=42)

    # Build agent matching training config
    algo_cfg = AlgorithmConfig(name="hfg_sac")
    fuzzy_cfg = FuzzyConfig(
        use_expert_rules=True,
        freq_volt_shield_enabled=True,
        freq_volt_shield_safety_ratio=scen_cfg.get("freq_volt_shield_safety_ratio", 0.175),
    )
    cfg = HFGConfig(algorithm=algo_cfg, env=env_cfg, fuzzy=fuzzy_cfg)
    obs_dim = env.observation_space_shape[0]
    act_dim = env.action_space_shape[0]
    agent = HFG_SAC(cfg, obs_dim, act_dim, env.action_high, device="cpu")

    # Load checkpoint
    ckpt_dir = f"outputs/experiments/checkpoints/hfg_sac_{'S3_island_normal' if scen_name=='S3' else 'S4_island_extreme'}_s42"
    agent.load(ckpt_dir)

    events = evaluate_with_diagnostics(env, agent, n_episodes=3)
    print(f"Total violation events in 3 eval episodes: {len(events)}")
    # Categorize
    freq_viol = sum(1 for e in events if any("freq" in r for r in e["reasons"]))
    soc_hi = sum(1 for e in events if any("SOC_high" in r for r in e["reasons"]))
    soc_lo = sum(1 for e in events if any("SOC_low" in r for r in e["reasons"]))
    print(f"  Frequency violations: {freq_viol}")
    print(f"  SOC-high violations:  {soc_hi}")
    print(f"  SOC-low violations:  {soc_lo}")
    if events:
        print(f"\n  First 5 events:")
        for e in events[:5]:
            print(f"    ep={e['ep']} step={e['step']} hour={e['hour']:.1f} "
                  f"soc={e['soc']:.3f} freq={e['freq']:.3f} "
                  f"imb={e['imbalance_ratio']:.3f} shed={e['load_shed_kw']:.0f}kW "
                  f"reasons={e['reasons']}")
        print(f"\n  Last 5 events:")
        for e in events[-5:]:
            print(f"    ep={e['ep']} step={e['step']} hour={e['hour']:.1f} "
                  f"soc={e['soc']:.3f} freq={e['freq']:.3f} "
                  f"imb={e['imbalance_ratio']:.3f} shed={e['load_shed_kw']:.0f}kW "
                  f"reasons={e['reasons']}")
