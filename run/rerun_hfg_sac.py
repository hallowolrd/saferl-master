"""Re-run all HFG-SAC experiments with the fixed min-FCSD safety mechanism.

Runs 4 scenarios x 3 seeds = 12 experiments sequentially.
Old results are backed up to results_backup_pre_fix/.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from run.experiment_runner import train_agent, SCENARIOS, RunResult
import json


ALGO = "hfg_sac"
SEEDS = [42, 123, 456]
SCENARIO_NAMES = [
    "S1_grid_normal",
    "S2_grid_extreme",
    "S3_island_normal",
    "S4_island_extreme",
]
N_EPISODES = 200
# Auto-detect GPU; fall back to CPU if CUDA unavailable.
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/experiments")


def result_to_dict(r: RunResult) -> dict:
    """Convert TrainResult to JSON-serializable dict."""
    return {
        "algorithm": ALGO,
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


def main() -> None:
    total_start = time.time()
    results = []

    for scenario_name in SCENARIO_NAMES:
        scenario_cfg = SCENARIOS[scenario_name]
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"Running {ALGO} on {scenario_name} seed={seed}")
            print(f"{'='*60}")
            t0 = time.time()

            r = train_agent(
                ALGO, scenario_name, scenario_cfg, seed,
                n_episodes=N_EPISODES, device=DEVICE,
                output_dir=None,  # we save manually
            )

            elapsed = time.time() - t0
            print(f"  => cost={r.total_cost:,.0f}  viol={r.violation_rate:.2f}%  "
                  f"fcsd={r.avg_fcsd:.4f}  time={elapsed:.1f}s")

            # Save result
            result_path = OUTPUT_DIR / "results" / f"{ALGO}_{scenario_name}_s{seed}.json"
            with open(result_path, "w") as f:
                json.dump(result_to_dict(r), f, indent=2, default=str)
            print(f"  saved to {result_path}")

            results.append((scenario_name, seed, r))

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All {len(results)} experiments completed in {total_elapsed/60:.1f} min")
    print(f"{'='*60}")

    # Summary table
    print(f"\n{'Scenario':<20} {'Seed':<6} {'Cost (¥/d)':<14} {'Viol. (%)':<10} {'Avg FCSD':<10} {'Min FCSD':<10}")
    print("-" * 70)
    for scenario_name, seed, r in results:
        print(f"{scenario_name:<20} {seed:<6} {r.total_cost/7:<14,.0f} "
              f"{r.violation_rate:<10.2f} {r.avg_fcsd:<10.4f} {r.min_fcsd:<10.4f}")


if __name__ == "__main__":
    main()
