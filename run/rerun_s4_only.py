"""Retrain HFG-SAC on S4 only, 3 seeds in parallel.

Uses the strengthened S4 config (interruptible_load=600, freq_slope=3000,
shield_safety_ratio=0.10).  Output overwrites the old S4 JSON files.

Usage:
    .venv/Scripts/python.exe run/rerun_s4_only.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch

if sys.platform == "win32":
    mp.set_start_method("spawn", force=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "outputs" / "experiments" / "results"
SEEDS = [42, 123, 456]
SCENARIO = "S4_island_extreme"


def run_one(seed: int) -> dict:
    from run.experiment_runner import train_agent, SCENARIOS

    torch.cuda.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scenario_cfg = SCENARIOS[SCENARIO]
    t0 = time.time()
    r = train_agent(
        "hfg_sac", SCENARIO, scenario_cfg, seed,
        n_episodes=200, device=device, output_dir=None,
    )
    elapsed = time.time() - t0

    result = {
        "algorithm": "hfg_sac",
        "scenario": SCENARIO,
        "seed": seed,
        "total_cost": r.total_cost,
        "cost_per_day": r.cost_per_day,
        "violation_rate": r.violation_rate,
        "avg_fcsd": r.avg_fcsd,
        "min_fcsd": r.min_fcsd,
        "convergence_episode": r.convergence_episode,
        "training_time_s": r.training_time_s,
        "final_reward": r.final_reward,
        "episode_rewards": r.episode_rewards,
        "episode_costs": r.episode_costs,
        "episode_violation_rates": r.episode_violation_rates,
        "episode_fcsds": r.episode_fcsds,
        "eval_costs": r.eval_costs,
        "eval_violations": r.eval_violations,
        "eval_fcsds": r.eval_fcsds,
    }

    out_path = OUTPUT_DIR / f"hfg_sac_{SCENARIO}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[done] S4 seed={seed}: "
          f"cost/day={r.cost_per_day:,.0f} viol={r.violation_rate:.1f}% "
          f"fcsd={r.avg_fcsd:.3f} ({elapsed:.0f}s)", flush=True)
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    print(f"Retraining HFG-SAC on S4 with {len(SEEDS)} parallel seeds")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    with mp.Pool(processes=len(SEEDS)) as pool:
        all_results = pool.map(run_one, SEEDS)

    total_elapsed = time.time() - total_start
    print(f"\nS4 retrain completed in {total_elapsed/60:.1f} min")

    print(f"\n{'Seed':<6} {'Cost RMB/d':>14} {'Viol %':>10} {'Avg FCSD':>10}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['seed']:<6} {r['cost_per_day']:>14,.0f} "
              f"{r['violation_rate']:>10.2f} {r['avg_fcsd']:>10.4f}")


if __name__ == "__main__":
    main()
