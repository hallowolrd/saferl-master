"""Retrain HFG-SAC on S3 + S4 with Plan D config, 3 seeds each.

Plan D:
- S3: shield=0.13, slope=1500, shed_cost=2.0
- S4: shield=0.15, slope=1500, shed_cost=1.0, interruptible=600

Output overwrites existing S3/S4 JSON files.
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
SCENARIOS = ["S3_island_normal", "S4_island_extreme"]


def run_one(job):
    scen, seed = job
    from run.experiment_runner import train_agent, SCENARIOS as SC

    torch.cuda.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = SC[scen]
    t0 = time.time()
    r = train_agent(
        "hfg_sac", scen, cfg, seed,
        n_episodes=200, device=device, output_dir=None,
    )
    elapsed = time.time() - t0

    result = {
        "algorithm": "hfg_sac", "scenario": scen, "seed": seed,
        "total_cost": r.total_cost, "cost_per_day": r.cost_per_day,
        "violation_rate": r.violation_rate, "avg_fcsd": r.avg_fcsd,
        "min_fcsd": r.min_fcsd, "convergence_episode": r.convergence_episode,
        "training_time_s": r.training_time_s, "final_reward": r.final_reward,
        "episode_rewards": r.episode_rewards, "episode_costs": r.episode_costs,
        "episode_violation_rates": r.episode_violation_rates,
        "episode_fcsds": r.episode_fcsds,
        "eval_costs": r.eval_costs, "eval_violations": r.eval_violations,
        "eval_fcsds": r.eval_fcsds,
    }
    out_path = OUTPUT_DIR / f"hfg_sac_{scen}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[done] {scen} seed={seed}: cost/d={r.cost_per_day:,.0f} "
          f"viol={r.violation_rate:.1f}% fcsd={r.avg_fcsd:.3f} ({elapsed:.0f}s)",
          flush=True)
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(scen, seed) for scen in SCENARIOS for seed in SEEDS]
    print(f"Retraining {len(jobs)} runs: S3+S4 x 3 seeds each")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    t0 = time.time()
    with mp.Pool(processes=6) as pool:
        results = pool.map(run_one, jobs)
    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed/60:.1f} min")

    print(f"\n{'Scenario':<22} {'Seed':<6} {'Cost RMB/d':>14} {'Viol%':>8} {'FCSD':>7}")
    print("-" * 60)
    for r in results:
        print(f"{r['scenario']:<22} {r['seed']:<6} "
              f"{r['cost_per_day']:>14,.0f} {r['violation_rate']:>8.2f} "
              f"{r['avg_fcsd']:>7.4f}")


if __name__ == "__main__":
    main()
