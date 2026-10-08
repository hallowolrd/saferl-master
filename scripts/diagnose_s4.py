"""Diagnose S4 violation type distribution.

Loads a trained HFG-SAC checkpoint, runs an eval rollout on S4, and
breaks down hard violations by constraint: SOC upper / SOC lower /
frequency / voltage.

Usage:
    .venv/Scripts/python.exe scripts/diagnose_s4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run.experiment_runner import SCENARIOS, create_env_config, train_agent  # noqa: E402
from hfg_srl.env.base_env import make_env  # noqa: E402
from hfg_srl.algorithms import make_agent  # noqa: E402
from hfg_srl.utils.config import AlgorithmConfig, FuzzyConfig, HFGConfig, EnvConfig  # noqa: E402
from hfg_srl.utils.seed import set_seed  # noqa: E402


def build_agent(env_cfg, data_loader, device, seed=123):
    """Reconstruct the HFG-SAC agent exactly as in train_agent()."""
    scenario_cfg = SCENARIOS["S4_island_extreme"]
    algo_cfg = AlgorithmConfig(
        name="hfg_sac",
        lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        gamma=0.99, tau=0.005, alpha=0.2, auto_alpha=True,
        hidden_dims=[128, 128], activation="relu",
        batch_size=256, buffer_size=int(1e5), max_steps=int(1e6),
        warmup_steps=1000, updates_per_step=1,
        eval_interval=2000, eval_episodes=3, log_interval=500,
    )
    fuzzy_cfg = FuzzyConfig(
        num_rules=16, num_inputs=9, fcsd_target=0.75,
        use_expert_rules=True,
        reward_shaping_weight_init=0.3,
        reward_shaping_weight_final=0.01,
        reward_shaping_decay_steps=10000,
        lambda_init=2.0, lambda_lr=0.05, lambda_max=50.0,
    )
    hfg_cfg = HFGConfig(env=env_cfg, algorithm=algo_cfg, fuzzy=fuzzy_cfg,
                        seed=seed, device=device)
    env = make_env("islanded", env_cfg, data_loader=data_loader)
    env.seed(seed)
    obs_dim = env.observation_space_shape[0]
    act_dim = env.action_space_shape[0]
    action_high = env.action_high
    agent = make_agent("hfg_sac", hfg_cfg, obs_dim, act_dim, action_high, device)
    return agent, env


def run_rollout(agent, env, episodes=5, freq_limit=0.5, volt_limit=0.05,
                soc_min=0.1, soc_max=0.9):
    """Run eval episodes and collect per-constraint violation stats."""
    totals = {
        "steps": 0,
        "any_hard": 0,
        "freq_hard": 0,
        "volt_hard": 0,
        "soc_upper_hard": 0,
        "soc_lower_hard": 0,
        "freq_excess_sum": 0.0,
        "volt_excess_sum": 0.0,
        "soc_upper_excess_sum": 0.0,
        "soc_lower_excess_sum": 0.0,
        # Magnitude distributions
        "freq_dev_hist": [],
        "volt_dev_hist": [],
        "soc_hist": [],
        "imbalance_hist": [],
        "de_power_hist": [],
        "shed_power_hist": [],
        "ess_power_hist": [],
    }

    agent.eval()
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            soc = info["soc"]
            fdev = info["freq_deviation"]
            vdev = info["voltage_deviation"]

            totals["steps"] += 1
            totals["freq_dev_hist"].append(fdev)
            totals["volt_dev_hist"].append(vdev)
            totals["soc_hist"].append(soc)
            totals["imbalance_hist"].append(info.get("imbalance_ratio", 0.0))

            # action magnitudes (read from env state)
            totals["de_power_hist"].append(getattr(env, "de_power", 0.0))
            totals["shed_power_hist"].append(info.get("load_shed", 0.0))
            totals["ess_power_hist"].append(getattr(env, "ess_power", 0.0))

            f_hard = fdev > freq_limit
            v_hard = vdev > volt_limit
            su_hard = soc > soc_max
            sl_hard = soc < soc_min

            if f_hard:
                totals["freq_hard"] += 1
                totals["freq_excess_sum"] += fdev - freq_limit
            if v_hard:
                totals["volt_hard"] += 1
                totals["volt_excess_sum"] += vdev - volt_limit
            if su_hard:
                totals["soc_upper_hard"] += 1
                totals["soc_upper_excess_sum"] += soc - soc_max
            if sl_hard:
                totals["soc_lower_hard"] += 1
                totals["soc_lower_excess_sum"] += soc_min - soc
            if f_hard or v_hard or su_hard or sl_hard:
                totals["any_hard"] += 1

    return totals


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    env_cfg, data_loader = create_env_config(
        "islanded", SCENARIOS["S4_island_extreme"], seed=123, num_days=7,
    )
    agent, env = build_agent(env_cfg, data_loader, device, seed=123)

    ckpt_dir = ROOT / "outputs/experiments/checkpoints/hfg_sac_S4_island_extreme_s123"
    print(f"Loading checkpoint dir: {ckpt_dir}")
    # Old checkpoint predates cost_critic_target; load actor only.
    ckpt = torch.load(str(ckpt_dir / "hfg_sac.pt"), map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor"])
    if "log_alpha" in ckpt:
        agent.log_alpha.data = ckpt["log_alpha"].data
    print(f"Loaded actor (steps={ckpt.get('total_steps', '?')})")

    stats = run_rollout(agent, env, episodes=5,
                        freq_limit=env_cfg.frequency_limit_hz,
                        volt_limit=0.05,
                        soc_min=env_cfg.soc_min, soc_max=env_cfg.soc_max)

    n = stats["steps"]
    print(f"\n=== S4 Per-Constraint Violation Breakdown ({n} steps, 5 episodes) ===")
    print(f"Any hard violation:  {stats['any_hard']:>5d} / {n}  = {100*stats['any_hard']/n:.1f}%")
    print(f"  Frequency > {env_cfg.frequency_limit_hz} Hz:  {stats['freq_hard']:>5d}  = {100*stats['freq_hard']/n:.1f}%")
    print(f"  Voltage   > 5%:           {stats['volt_hard']:>5d}  = {100*stats['volt_hard']/n:.1f}%")
    print(f"  SOC upper > {env_cfg.soc_max}:           {stats['soc_upper_hard']:>5d}  = {100*stats['soc_upper_hard']/n:.1f}%")
    print(f"  SOC lower < {env_cfg.soc_min}:           {stats['soc_lower_hard']:>5d}  = {100*stats['soc_lower_hard']/n:.1f}%")

    fdev = np.array(stats["freq_dev_hist"])
    vdev = np.array(stats["volt_dev_hist"])
    soc = np.array(stats["soc_hist"])
    imb = np.array(stats["imbalance_hist"])
    de = np.array(stats["de_power_hist"])
    shed = np.array(stats["shed_power_hist"])
    ess = np.array(stats["ess_power_hist"])

    print(f"\n=== Magnitude distributions ===")
    print(f"freq_dev:  mean={fdev.mean():.3f} Hz  max={fdev.max():.3f}  "
          f"p50={np.percentile(fdev,50):.3f}  p95={np.percentile(fdev,95):.3f}")
    print(f"volt_dev:  mean={vdev.mean():.4f}   max={vdev.max():.4f}  "
          f"p50={np.percentile(vdev,50):.4f}  p95={np.percentile(vdev,95):.4f}")
    print(f"SOC:       mean={soc.mean():.3f}     min={soc.min():.3f}  "
          f"p05={np.percentile(soc,5):.3f}  p50={np.percentile(soc,50):.3f}")
    print(f"imbalance_ratio: mean={imb.mean():.3f}  min={imb.min():.3f}  max={imb.max():.3f}")
    print(f"DE power:  mean={de.mean():.0f} kW  max={de.max():.0f} kW (rated {env_cfg.de_rated_power_kw:.0f})")
    print(f"Shed:      mean={shed.mean():.0f} kW  max={shed.max():.0f} kW (interruptible {env_cfg.interruptible_load_kw:.0f})")
    print(f"ESS power: mean={ess.mean():.0f} kW  max={ess.max():.0f} kW (rated {env_cfg.ess_power_kw:.0f})")

    # How often is DE at max?
    de_at_max = (de > 0.95 * env_cfg.de_rated_power_kw).mean()
    shed_at_max = (shed > 0.95 * env_cfg.interruptible_load_kw).mean()
    print(f"\nDE at rated (>95%):    {100*de_at_max:.1f}% of steps")
    print(f"Shed at max (>95%):    {100*shed_at_max:.1f}% of steps")

    # Under-generation vs over-generation
    under = (imb < -0.1).mean()
    over = (imb > 0.1).mean()
    print(f"Under-generation (imb < -0.1): {100*under:.1f}% of steps")
    print(f"Over-generation  (imb >  0.1): {100*over:.1f}% of steps")


if __name__ == "__main__":
    main()
