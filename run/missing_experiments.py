"""Run the missing RQ1 main-comparison experiments (skip completed runs).

Parallelizable across (algorithm, scenario, seed). Skips any combination whose
result JSON already exists, so it is safe to re-run after interruption.

Usage:
    .venv/Scripts/python.exe run/missing_experiments.py --dry-run
    .venv/Scripts/python.exe run/missing_experiments.py --workers 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = [42, 123, 456, 789, 2024]
SCENARIOS = [
    "S1_grid_normal",
    "S2_grid_extreme",
    "S3_island_normal",
    "S4_island_extreme",
]
ALGORITHMS = [
    "sac",
    "ppo",
    "ppo_lagrangian",
    "cpo",
    "safety_layer",
    "fuzzy_sac",
    "hfg_sac",
]
N_EPISODES = 200
DEVICE = os.environ.get("HFG_DEVICE", "cpu")
OUTPUT_DIR = Path(
    os.environ.get("HFG_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "experiments")
)


def result_path(algo: str, scenario: str, seed: int) -> Path:
    return OUTPUT_DIR / "results" / f"{algo}_{scenario}_s{seed}.json"


def model_baseline_path(scenario: str) -> Path:
    return OUTPUT_DIR / "results" / f"model_baselines_{scenario}.json"


def missing_jobs() -> list:
    """Return list of job tuples: ("rl", algo, scenario, seed) or ("model", scenario)."""
    jobs = []
    for scenario in SCENARIOS:
        for algo in ALGORITHMS:
            for seed in SEEDS:
                if not result_path(algo, scenario, seed).exists():
                    jobs.append(("rl", algo, scenario, seed))
        if not model_baseline_path(scenario).exists():
            jobs.append(("model", scenario, ""))
    return jobs


def _run_job(job: tuple) -> tuple:
    """Worker entrypoint (must be importable under multiprocessing spawn)."""
    from run.experiment_runner import (
        run_model_baselines,
        train_agent,
        SCENARIOS as SC,
    )

    if job[0] == "rl":
        _, algo, scenario, seed = job
        t0 = time.time()
        n_episodes = int(os.environ.get("HFG_N_EPISODES", N_EPISODES))
        r = train_agent(
            algo, scenario, SC[scenario], seed,
            n_episodes=n_episodes, device=DEVICE, output_dir=OUTPUT_DIR,
        )
        return (algo, scenario, seed, "ok", r.total_cost,
                r.violation_rate, r.avg_fcsd, time.time() - t0)
    else:
        _, scenario, _ = job
        t0 = time.time()
        run_model_baselines(scenario, SC[scenario], OUTPUT_DIR)
        return ("model", scenario, "", "ok", 0.0, 0.0, 0.0, time.time() - t0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="run at most N jobs (for validation)")
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                        help="training episodes per run (for validation)")
    args = parser.parse_args()

    os.environ["HFG_N_EPISODES"] = str(args.episodes)

    jobs = missing_jobs()
    if args.limit is not None:
        jobs = jobs[: args.limit]

    print(f"Missing jobs: {len(jobs)} (n_episodes={N_EPISODES}, "
          f"device={DEVICE}, seeds={SEEDS})")
    for j in jobs:
        print("  ", j)

    if args.dry_run:
        return

    workers = args.workers or min(4, os.cpu_count() or 4)
    print(f"\nLaunching {len(jobs)} jobs with {workers} workers...")
    log = OUTPUT_DIR / "progress.log"

    done = ok = failed = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_job, j): j for j in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            done += 1
            try:
                res = fut.result()
                ok += 1
                elapsed = time.time() - t_start
                with open(log, "a", encoding="utf-8") as f:
                    f.write(f"[{done}/{len(jobs)} +{res[-1]:.0f}s] {res}\n")
                print(f"[{done}/{len(jobs)}] OK  {res[:-1]} "
                      f"({time.time() - t_start:.0f}s elapsed)")
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback
                with open(log, "a", encoding="utf-8") as f:
                    f.write(f"[{done}/{len(jobs)}] FAIL {job}: {e}\n")
                print(f"[{done}/{len(jobs)}] FAIL {job}: {e}")
                traceback.print_exc()

    print(f"\nDone. ok={ok} failed={failed} total={len(jobs)} "
          f"wall={time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()