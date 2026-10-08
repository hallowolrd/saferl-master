"""Parallel HFG-SAC rerun: one worker per scenario, 3 seeds each.

Why this is faster:
- Env stepping is CPU-bound (Python microgrid sim); GPU is idle most of the
  time waiting for the next action.
- With 20 CPU cores and an 8 GB GPU, we can run 4 scenarios in parallel
  without contention. Each worker uses ~0.5-1 GB of GPU memory; 4 workers
  fit comfortably.
- Speedup ≈ 4x vs the sequential script.

Usage:
    .venv/Scripts/python.exe run/rerun_hfg_sac_parallel.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import torch

# Must set spawn context on Windows; 'fork' is unavailable.
if sys.platform == "win32":
    mp.set_start_method("spawn", force=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "outputs" / "experiments" / "results"
N_EPISODES = 200
SEEDS = [42, 123, 456]
SCENARIO_NAMES = [
    "S1_grid_normal",
    "S2_grid_extreme",
    "S3_island_normal",
    "S4_island_extreme",
]


def run_one(scenario_name: str, seed: int) -> dict:
    """Train HFG-SAC on one (scenario, seed) pair. Runs in a worker process."""
    # Import inside worker so each process re-initializes CUDA context cleanly.
    from run.experiment_runner import train_agent, SCENARIOS

    torch.cuda.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scenario_cfg = SCENARIOS[scenario_name]
    t0 = time.time()
    r = train_agent(
        "hfg_sac", scenario_name, scenario_cfg, seed,
        n_episodes=N_EPISODES, device=device, output_dir=None,
    )
    elapsed = time.time() - t0

    result = {
        "algorithm": "hfg_sac",
        "scenario": scenario_name,
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

    out_path = OUTPUT_DIR / f"hfg_sac_{scenario_name}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[done] {scenario_name} seed={seed}: "
          f"cost/day={r.cost_per_day:,.0f} viol={r.violation_rate:.1f}% "
          f"fcsd={r.avg_fcsd:.3f} ({elapsed:.0f}s)", flush=True)
    return result


def run_scenario_worker(scenario_name: str) -> list:
    """Worker: run all 3 seeds for one scenario sequentially."""
    print(f"[worker start] {scenario_name}", flush=True)
    results = []
    for seed in SEEDS:
        results.append(run_one(scenario_name, seed))
    print(f"[worker done]  {scenario_name}", flush=True)
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    n_workers = min(len(SCENARIO_NAMES), os.cpu_count() or 4)
    print(f"Launching {n_workers} parallel workers (one per scenario)")
    print(f"GPU available: {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    with mp.Pool(processes=n_workers) as pool:
        all_results = pool.map(run_scenario_worker, SCENARIO_NAMES)

    total_elapsed = time.time() - total_start
    print(f"\nAll 12 runs completed in {total_elapsed/60:.1f} min "
          f"(vs ~4x sequential time)")

    # Summary table
    print(f"\n{'Scenario':<22} {'Seed':<6} {'Cost RMB/d':>14} {'Viol %':>10} "
          f"{'Avg FCSD':>10}")
    print("-" * 70)
    for worker_results in all_results:
        for r in worker_results:
            print(f"{r['scenario']:<22} {r['seed']:<6} "
                  f"{r['cost_per_day']:>14,.0f} "
                  f"{r['violation_rate']:>10.2f} "
                  f"{r['avg_fcsd']:>10.4f}")


if __name__ == "__main__":
    main()
