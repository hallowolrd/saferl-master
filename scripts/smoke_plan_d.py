"""Smoke test Plan D: evaluate existing trained actor with NEW shield params.

Uses the S3 and S4 actors trained under the OLD config, but evaluates them
in envs built with the NEW Plan D shield/slope/shed-cost parameters.

This previews what retraining might achieve (shield dominates anyway).

Usage:
    .venv/Scripts/python.exe scripts/smoke_plan_d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run.experiment_runner import SCENARIOS, create_env_config  # noqa: E402
from hfg_srl.env.base_env import make_env  # noqa: E402
from hfg_srl.algorithms import make_agent  # noqa: E402
from hfg_srl.utils.config import AlgorithmConfig, FuzzyConfig, HFGConfig  # noqa: E402
from hfg_srl.utils.seed import set_seed  # noqa: E402


def build_agent_and_env(scen_name, seed=123):
    set_seed(seed)
    cfg = SCENARIOS[scen_name]
    env_cfg, dl = create_env_config(cfg["env_mode"], cfg, seed=seed, num_days=7)

    algo = AlgorithmConfig(
        name="hfg_sac", hidden_dims=[128, 128], batch_size=256,
        buffer_size=int(1e5), warmup_steps=100, eval_interval=1000,
        eval_episodes=1, log_interval=1000,
    )
    shield = cfg.get("freq_volt_shield_safety_ratio", 0.175)
    fcsd_t = 0.75 if scen_name == "S4_island_extreme" else 0.85
    fuzz = FuzzyConfig(
        num_rules=16, num_inputs=9, fcsd_target=fcsd_t,
        use_expert_rules=True,
        reward_shaping_weight_init=0.0, reward_shaping_weight_final=0.0,
        lambda_init=2.0, lambda_lr=0.05, lambda_max=50.0,
        freq_volt_shield_safety_ratio=shield,
    )
    hfg = HFGConfig(env=env_cfg, algorithm=algo, fuzzy=fuzz,
                    seed=seed, device="cpu")
    env = make_env(cfg["env_mode"], env_cfg, data_loader=dl)
    env.seed(seed)
    ag = make_agent("hfg_sac", hfg, env.observation_space_shape[0],
                    env.action_space_shape[0], env.action_high, "cpu")
    ag.eval()

    # Load existing trained actor (trained under OLD config).
    ckpt = ROOT / f"outputs/experiments/checkpoints/hfg_sac_{scen_name}_s{seed}/hfg_sac.pt"
    if ckpt.exists():
        c = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        ag.actor.load_state_dict(c["actor"])
        if "log_alpha" in c:
            ag.log_alpha.data = c["log_alpha"].data
        print(f"  loaded actor from {ckpt.name} (steps={c.get('total_steps','?')})")
    else:
        print(f"  WARNING: no checkpoint at {ckpt}, using random init")

    return ag, env, env_cfg


def rollout(ag, env, env_cfg, episodes=5):
    stats = {"steps": 0, "freq": [], "volt": [], "soc": [],
             "de": [], "shed": [], "ess": [], "cost": [],
             "freq_hard": 0, "shed_active": 0}
    for ep in range(episodes):
        obs, _ = env.reset(seed=300 + ep)
        done = False
        while not done:
            a = ag.select_action(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            stats["steps"] += 1
            stats["freq"].append(info["freq_deviation"])
            stats["volt"].append(info["voltage_deviation"])
            stats["soc"].append(info["soc"])
            stats["de"].append(getattr(env, "de_power", 0))
            stats["shed"].append(info["load_shed"])
            stats["ess"].append(getattr(env, "ess_power", 0))
            stats["cost"].append(info["cost"])
            if info["freq_deviation"] > 0.5:
                stats["freq_hard"] += 1
            if info["load_shed"] > 1.0:
                stats["shed_active"] += 1
    return stats


def main():
    for scen in ["S3_island_normal", "S4_island_extreme"]:
        print(f"\n=== {scen} (Plan D config) ===")
        ag, env, ecfg = build_agent_and_env(scen)
        print(f"  interruptible={ecfg.interruptible_load_kw:.0f} kW")
        print(f"  freq_slope   ={ecfg.freq_penalty_slope:.0f}")
        print(f"  shed_cost    ={ecfg.load_shed_cost:.2f} RMB/kWh")
        shield = SCENARIOS[scen].get("freq_volt_shield_safety_ratio", 0.175)
        print(f"  shield_ratio ={shield} (freq trigger = {shield*0.5:.2f} Hz)")

        s = rollout(ag, env, ecfg)
        n = s["steps"]
        f = np.array(s["freq"]); sh = np.array(s["shed"])
        de = np.array(s["de"]); cost = np.array(s["cost"])

        print(f"\n  --- Results (trained actor + NEW shield) ---")
        print(f"  hard freq viol:  {s['freq_hard']}/{n} = {100*s['freq_hard']/n:.1f}%")
        print(f"  freq_dev mean:   {f.mean():.3f} Hz  p95={np.percentile(f,95):.3f}  max={f.max():.3f}")
        print(f"  shed mean:       {sh.mean():.0f} kW  max={sh.max():.0f}")
        print(f"  shed active:     {100*s['shed_active']/n:.1f}% of steps")
        print(f"  DE mean:         {de.mean():.0f} kW (rated 400)")
        # 96 steps per day x 0.25h = 24h; cost is per-step
        daily_cost = cost.mean() * 96  # 96 steps/day
        print(f"  est cost/day:    RMB {daily_cost:,.0f}")


if __name__ == "__main__":
    main()
