"""Minimal violation diagnostic — runs trained agent, logs WHY violations happen."""
import sys, os
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import torch
from run.experiment_runner import create_env_config, train_agent, evaluate_agent
from hfg_srl.algorithms.base_agent import make_agent
from hfg_srl.utils.config import HFGConfig, AlgorithmConfig, EnvConfig, FuzzyConfig
from hfg_srl.utils.seed import set_seed

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
    print(f"  {scen_name}: interruptible={scen_cfg['interruptible_load_kw']}kW, "
          f"shield_ratio={scen_cfg['freq_volt_shield_safety_ratio']}")
    print(f"{'='*60}")

    set_seed(42)
    env_cfg, data_loader = create_env_config(
        scen_cfg["env_mode"], scen_cfg, seed=42, num_days=7,
    )
    from hfg_srl.env.islanded import IslandedEnv
    env = IslandedEnv(env_cfg, data_loader=data_loader)

    # Build agent via registry
    algo_cfg = AlgorithmConfig(name="hfg_sac")
    fuzzy_cfg = FuzzyConfig(
        use_expert_rules=True,
        freq_volt_shield_enabled=True,
        freq_volt_shield_safety_ratio=scen_cfg["freq_volt_shield_safety_ratio"],
    )
    cfg = HFGConfig(algorithm=algo_cfg, env=env_cfg, fuzzy=fuzzy_cfg)
    agent = make_agent(
        "hfg_sac", cfg,
        env.observation_space_shape[0], env.action_space_shape[0],
        env.action_high, device="cpu",
    )
    ckpt_dir = f"outputs/experiments/checkpoints/hfg_sac_{'S3_island_normal' if scen_name=='S3' else 'S4_island_extreme'}_s42"
    agent.load(ckpt_dir)
    agent.eval()

    # Run 3 eval episodes, log every violation event
    freq_viol = soc_hi = soc_lo = total_steps = 0
    viol_details = []
    for ep in range(3):
        obs, info = env.reset()
        done = False
        step = 0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            action = np.clip(action, env.action_low, env.action_high)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1
            if info.get("violated", False):
                soc = info["soc"]
                freq = info["freq_deviation"]
                reasons = []
                if soc > 0.9: reasons.append(f"SOC_hi={soc:.3f}"); soc_hi += 1
                if soc < 0.1: reasons.append(f"SOC_lo={soc:.3f}"); soc_lo += 1
                if freq > 0.5: reasons.append(f"freq={freq:.3f}"); freq_viol += 1
                if len(viol_details) < 10:
                    viol_details.append(
                        f"  ep={ep} step={step} hour={step/4:.1f}h "
                        f"soc={soc:.3f} freq={freq:.3f} "
                        f"imb={info['imbalance_ratio']:.3f} "
                        f"shed={info['load_shed']:.0f}kW {reasons}"
                    )
            step += 1

    print(f"Total steps: {total_steps}, violation events: {freq_viol+soc_hi+soc_lo}")
    print(f"  Frequency violations (>0.5 Hz): {freq_viol}")
    print(f"  SOC-high violations (>0.9):     {soc_hi}")
    print(f"  SOC-low violations (<0.1):      {soc_lo}")
    if viol_details:
        print(f"  First events:")
        for line in viol_details:
            print(line)
