"""Unified parallel runner for all HFG-SRL experiments (RQ1-RQ5).

All configuration is at the top of this file. Defaults:
  - 5 seeds, 200 episodes per run
  - Full retrain (no skip)
  - All 5 RQs enabled

Usage:
    .venv/Scripts/python.exe run/run_all_parallel.py
    .venv/Scripts/python.exe run/run_all_parallel.py --dry-run
    .venv/Scripts/python.exe run/run_all_parallel.py --workers 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

# Must set spawn context on Windows; 'fork' is unavailable.
if sys.platform == "win32":
    mp.set_start_method("spawn", force=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ============================================================================
# CONFIGURATION — edit these values to control the experiment
# ============================================================================

# Random seeds (5 for statistical robustness)
SEEDS = [42, 123, 456, 789, 2024]

# Training episodes per run
N_EPISODES = 200

# Which RQs to run (comment out / remove to skip)
ENABLED_RQS = {
    "rq1": False,  # Main comparison: 7 algos x 4 scenarios (already completed)
    "rq2": True,   # Robustness: 6 algos x 7 extremity levels x 5 seeds = 210 runs
    "rq3": False,  # Transfer: 3 tasks x 4 methods
    "rq4": False,  # Ablation: 4 variants on S4
    "rq5": False,  # Sensitivity: 3 axes x 5 values on S2
}

# If True, skip runs whose result JSON already exists (good for resuming).
# If False, retrain everything from scratch (overwrites existing JSONs).
SKIP_EXISTING = True

# Parallel workers (each worker is one process; tune to your CPU/GPU)
N_WORKERS = 6

# Output directory
OUTPUT_DIR = ROOT / "outputs" / "experiments"

# Device: "auto" | "cpu" | "cuda"
# CPU is faster for these tiny models (256x256 MLP, 8-9 dim input) —
# GPU kernel launch overhead dominates compute.
DEVICE = "cpu"

# ============================================================================
# Experiment matrices (do not edit unless you know what you're doing)
# ============================================================================

RQ1_ALGORITHMS = [
    "sac", "ppo", "ppo_lagrangian", "cpo",
    "safety_layer", "fuzzy_sac", "hfg_sac",
]
RQ1_SCENARIOS = [
    "S1_grid_normal", "S2_grid_extreme",
    "S3_island_normal", "S4_island_extreme",
]

RQ2_ALGORITHMS = ["sac", "ppo_lagrangian", "cpo", "safety_layer", "fuzzy_sac", "hfg_sac"]
RQ2_LEVELS = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

RQ3_TASKS = {
    # T1: grid normal -> grid extreme, same battery band
    "T1_grid_same": {
        "source": {"env_mode": "grid_connected",
                   "scenario_name": "typical_week_summer",
                   "extreme_multiplier": None},
        "target": {"env_mode": "grid_connected",
                   "scenario_name": "summer_extreme",
                   "extreme_multiplier": {"pv": 1.2, "load": 1.3, "wt": 0.6}},
    },
    # T2: grid normal -> grid extreme, target battery band narrowed to [0.4, 0.7]
    "T2_grid_shifted": {
        "source": {"env_mode": "grid_connected",
                   "scenario_name": "typical_week_summer",
                   "extreme_multiplier": None},
        "target": {"env_mode": "grid_connected",
                   "scenario_name": "summer_extreme",
                   "extreme_multiplier": {"pv": 1.2, "load": 1.3, "wt": 0.6},
                   "soc_optimal_min": 0.4, "soc_optimal_max": 0.7},
    },
    # T3: island normal -> island extreme, same battery band
    "T3_island_same": {
        "source": {"env_mode": "islanded",
                   "scenario_name": "typical_week_summer",
                   "extreme_multiplier": None,
                   "interruptible_load_kw": 600.0,
                   "freq_penalty_slope": 1500.0,
                   "freq_volt_shield_safety_ratio": 0.15,
                   "load_shed_cost": 1.0},
        "target": {"env_mode": "islanded",
                   "scenario_name": "summer_extreme",
                   "extreme_multiplier": {"pv": 0.5, "load": 1.25, "wt": 0.8},
                   "interruptible_load_kw": 600.0,
                   "freq_penalty_slope": 1500.0,
                   "freq_volt_shield_safety_ratio": 0.15,
                   "load_shed_cost": 1.0},
    },
    # T4: island normal -> island extreme, target battery band narrowed to [0.4, 0.7]
    "T4_island_shifted": {
        "source": {"env_mode": "islanded",
                   "scenario_name": "typical_week_summer",
                   "extreme_multiplier": None,
                   "interruptible_load_kw": 600.0,
                   "freq_penalty_slope": 1500.0,
                   "freq_volt_shield_safety_ratio": 0.15,
                   "load_shed_cost": 1.0},
        "target": {"env_mode": "islanded",
                   "scenario_name": "summer_extreme",
                   "extreme_multiplier": {"pv": 0.5, "load": 1.25, "wt": 0.8},
                   "interruptible_load_kw": 600.0,
                   "freq_penalty_slope": 1500.0,
                   "freq_volt_shield_safety_ratio": 0.15,
                   "load_shed_cost": 1.0,
                   "soc_optimal_min": 0.4, "soc_optimal_max": 0.7},
    },
}

RQ4_VARIANTS = [
    "hfg_sac_full", "hfg_sac_no_fuzzycon",
    "hfg_sac_no_fuzzyknow", "hfg_sac_baseline",
]

RQ5_SWEEPS = {
    # Plan A: keep only β sweep (Theorem 4.1 FC-MDP→CMDP limit).
    # α_target and κ sweeps removed — RQ4 ablation already covers κ=0,
    # and α_target choice is routine hyperparameter tuning.
    "beta":  [0.02, 0.05, 0.1, 0.2, 0.4],   # maps to hard_violation_threshold
}

# ============================================================================
# Job construction
# ============================================================================

def _extremity_cfg(level: float) -> dict:
    """Build islanded scenario cfg for a given extremity level.

    Mirrors S4_island_extreme's Plan D safety resources so the shield /
    interruptible-load logic is active; only the weather multiplier varies.
    """
    return {
        "env_mode": "islanded",
        "scenario_name": "summer_extreme",
        "extreme_multiplier": {
            "pv": max(0.2, 1.0 - level * 0.3),
            "load": 1.0 + level * 0.2,
            "wt": 1.0 - level * 0.15,
        },
        # Plan D islanded safety resources (same as S4_island_extreme)
        "interruptible_load_kw": 500.0,  # SANITY: 500 kW midpoint between 400 and 600
        "freq_penalty_slope": 1500.0,
        "freq_volt_shield_safety_ratio": 0.15,
        "load_shed_cost": 1.0,
        "fcsd_target": 0.85,  # SANITY: tighten fuzzy constraint from 0.75 to 0.85
    }


def build_jobs() -> list:
    """Return a list of job dicts. Each job is picklable for spawn."""
    jobs = []

    # --- RQ1: main comparison ---
    if ENABLED_RQS["rq1"]:
        from run.experiment_runner import SCENARIOS
        for algo in RQ1_ALGORITHMS:
            for scenario in RQ1_SCENARIOS:
                for seed in SEEDS:
                    jobs.append({
                        "type": "rq1",
                        "algo": algo,
                        "scenario": scenario,
                        "scenario_cfg": SCENARIOS[scenario],
                        "seed": seed,
                    })

    # --- RQ2: robustness under extremity ---
    if ENABLED_RQS["rq2"]:
        for algo in RQ2_ALGORITHMS:
            for level in RQ2_LEVELS:
                for seed in SEEDS:
                    jobs.append({
                        "type": "rq2",
                        "algo": algo,
                        "level": level,
                        "scenario_cfg": _extremity_cfg(level),
                        "seed": seed,
                    })

    # --- RQ3: transfer ---
    if ENABLED_RQS["rq3"]:
        for task_name, task_cfg in RQ3_TASKS.items():
            target = task_cfg["target"]
            source = task_cfg["source"]
            for seed in SEEDS:
                # from_scratch (hfg_sac on target)
                jobs.append({
                    "type": "rq3_scratch",
                    "task": task_name,
                    "method": "from_scratch",
                    "algo": "hfg_sac",
                    "scenario_cfg": target,
                    "seed": seed,
                })
                # anchor (ppo_lagrangian on target, no transfer)
                jobs.append({
                    "type": "rq3_scratch",
                    "task": task_name,
                    "method": "ppo_lagrangian_scratch",
                    "algo": "ppo_lagrangian",
                    "scenario_cfg": target,
                    "seed": seed,
                })
                # naive transfer
                jobs.append({
                    "type": "rq3_transfer",
                    "task": task_name,
                    "method": "naive_transfer",
                    "source_cfg": source,
                    "target_cfg": target,
                    "conservative": False,
                    "seed": seed,
                })
                # hfg conservative transfer
                jobs.append({
                    "type": "rq3_transfer",
                    "task": task_name,
                    "method": "hfg_transfer",
                    "source_cfg": source,
                    "target_cfg": target,
                    "conservative": True,
                    "use_M1": True,
                    "use_M4": True,
                    "seed": seed,
                })
                # ablation: no M1 (skip Jaccard, all constraints forced to adaptable)
                jobs.append({
                    "type": "rq3_transfer",
                    "task": task_name,
                    "method": "hfg_transfer_no_M1",
                    "source_cfg": source,
                    "target_cfg": target,
                    "conservative": True,
                    "use_M1": False,
                    "use_M4": True,
                    "seed": seed,
                })
                # ablation: no M4 (no exponential conservative scaling)
                jobs.append({
                    "type": "rq3_transfer",
                    "task": task_name,
                    "method": "hfg_transfer_no_M4",
                    "source_cfg": source,
                    "target_cfg": target,
                    "conservative": True,
                    "use_M1": True,
                    "use_M4": False,
                    "seed": seed,
                })

    # --- RQ4: ablation on S4 ---
    if ENABLED_RQS["rq4"]:
        from run.experiment_runner import SCENARIOS
        s4_cfg = SCENARIOS["S4_island_extreme"]
        for variant in RQ4_VARIANTS:
            for seed in SEEDS:
                jobs.append({
                    "type": "rq4",
                    "variant": variant,
                    "scenario": "S4_island_extreme",
                    "scenario_cfg": s4_cfg,
                    "seed": seed,
                })

    # --- RQ5: sensitivity on S2 ---
    if ENABLED_RQS["rq5"]:
        from run.experiment_runner import SCENARIOS
        s2_cfg = SCENARIOS["S2_grid_extreme"]
        for axis, values in RQ5_SWEEPS.items():
            for value in values:
                for seed in SEEDS:
                    jobs.append({
                        "type": "rq5",
                        "axis": axis,
                        "value": value,
                        "scenario_cfg": s2_cfg,
                        "seed": seed,
                    })

    return jobs


# ============================================================================
# Result file naming
# ============================================================================

def result_path(job: dict) -> Path:
    """Return the JSON path for a job (used for skip-existing + saving)."""
    results_dir = OUTPUT_DIR / "results"
    seed = job["seed"]

    if job["type"] == "rq1":
        fname = f"{job['algo']}_{job['scenario']}_s{seed}.json"
    elif job["type"] == "rq2":
        fname = f"{job['algo']}_extremity_{job['level']}_s{seed}.json"
    elif job["type"] == "rq3_scratch":
        tag = job["method"]  # from_scratch or ppo_lagrangian_scratch
        fname = f"{job['algo']}_{job['task']}_{tag}_s{seed}.json"
    elif job["type"] == "rq3_transfer":
        fname = f"hfg_sac_{job['task']}_{job['method']}_s{seed}.json"
    elif job["type"] == "rq4":
        fname = f"{job['variant']}_S4_island_extreme_s{seed}.json"
    elif job["type"] == "rq5":
        fname = f"hfg_sac_sens_{job['axis']}_{job['value']}_s{seed}.json"
    else:
        raise ValueError(f"Unknown job type: {job['type']}")

    return results_dir / fname


# ============================================================================
# Worker function (runs in a separate process)
# ============================================================================

def run_job(job: dict) -> dict:
    """Execute one experiment job. Must be top-level for spawn pickling."""
    # Per-worker CPU thread budget — must be set BEFORE importing torch so
    # that intra-op parallelism picks up the value.
    if DEVICE == "cpu":
        n_cpus = os.cpu_count() or 4
        os.environ["HFG_NUM_THREADS"] = str(max(1, n_cpus // max(N_WORKERS, 1)))

    import torch
    from run.experiment_runner import (
        train_agent, _train_with_param, _run_transfer, SCENARIOS,
    )

    # Resolve device
    device = DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.init()

    seed = job["seed"]
    t0 = time.time()

    if job["type"] in ("rq1", "rq3_scratch"):
        algo = job["algo"]
        scenario_cfg = job["scenario_cfg"]
        if job["type"] == "rq1":
            scenario_name = job["scenario"]
        else:
            scenario_name = f"{job['task']}_{job['method']}"
        r = train_agent(
            algo, scenario_name, scenario_cfg, seed,
            n_episodes=N_EPISODES, device=device, output_dir=OUTPUT_DIR,
        )

    elif job["type"] == "rq2":
        r = train_agent(
            job["algo"], f"extremity_{job['level']}", job["scenario_cfg"],
            seed, n_episodes=N_EPISODES, device=device, output_dir=OUTPUT_DIR,
        )

    elif job["type"] == "rq3_transfer":
        r = _run_transfer(
            job["source_cfg"], job["target_cfg"], seed,
            n_episodes=N_EPISODES, device=device, output_dir=OUTPUT_DIR,
            conservative=job["conservative"],
            use_M1=job.get("use_M1", True),
            use_M4=job.get("use_M4", True),
        )

    elif job["type"] == "rq4":
        r = train_agent(
            job["variant"], job["scenario"], job["scenario_cfg"], seed,
            n_episodes=N_EPISODES, device=device, output_dir=OUTPUT_DIR,
        )

    elif job["type"] == "rq5":
        # Map axis to the _train_with_param override
        kwargs = {}
        if job["axis"] == "beta":
            kwargs["constraint_width"] = job["value"]
        elif job["axis"] == "alpha":
            kwargs["fcsd_target"] = job["value"]
        elif job["axis"] == "kappa":
            kwargs["shaping_weight"] = job["value"]
        r = _train_with_param(
            "hfg_sac", f"sens_{job['axis']}_{job['value']}",
            job["scenario_cfg"], seed,
            n_episodes=N_EPISODES, device=device, output_dir=OUTPUT_DIR,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown job type: {job['type']}")

    elapsed = time.time() - t0

    # Save result JSON (train_agent already saves for rq1/rq2/rq4,
    # but we also save for rq3_transfer and rq5 which don't auto-save).
    out_path = result_path(job)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(r), f, indent=2, default=str)

    return {
        "job": job,
        "cost_per_day": r.cost_per_day,
        "violation_rate": r.violation_rate,
        "avg_fcsd": r.avg_fcsd,
        "elapsed_s": elapsed,
        "output": str(out_path),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all HFG-SRL experiments in parallel")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print job count and exit without running")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"Parallel workers (default {N_WORKERS})")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results").mkdir(exist_ok=True)

    jobs = build_jobs()

    # Filter: skip existing if configured
    if SKIP_EXISTING:
        jobs = [j for j in jobs if not result_path(j).exists()]

    total = len(jobs)
    print(f"Total jobs: {total}")
    print(f"Seeds: {SEEDS}")
    print(f"Episodes per run: {N_EPISODES}")
    print(f"Workers: {args.workers}")
    print(f"Skip existing: {SKIP_EXISTING}")
    print(f"Output: {OUTPUT_DIR / 'results'}")

    # Count by RQ
    from collections import Counter
    counts = Counter(j["type"] for j in jobs)
    for t, c in sorted(counts.items()):
        print(f"  {t}: {c} jobs")

    if args.dry_run:
        print("\nDry run — no jobs executed.")
        return

    if total == 0:
        print("Nothing to do (all results exist).")
        return

    log_path = OUTPUT_DIR / "run_all_parallel.log"
    done = ok = failed = 0
    t_start = time.time()

    print(f"\nLaunching {total} jobs with {args.workers} workers...\n")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            done += 1
            try:
                res = fut.result()
                ok += 1
                msg = (f"[{done}/{total}] OK  {res['job']['type']:18s} "
                       f"seed={job['seed']}  "
                       f"cost/d={res['cost_per_day']:>10,.0f}  "
                       f"viol={res['violation_rate']:>6.1f}%  "
                       f"fcsd={res['avg_fcsd']:.3f}  "
                       f"({res['elapsed_s']:.0f}s)")
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback
                msg = (f"[{done}/{total}] FAIL {job['type']} "
                       f"seed={job['seed']}: {e}")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(traceback.format_exc() + "\n")

            print(msg, flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Done. ok={ok}  failed={failed}  total={total}")
    print(f"Wall time: {total_elapsed/3600:.2f} hours")
    print(f"Log: {log_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
