"""RQ3 pilot smoke test.

Single-seed, short-episode run on T3 (islanded normal -> islanded extreme)
to validate the transfer pipeline before committing to the full 3-task x
4-method x 5-seed sweep.

Compares three arms on the T3 target:
  (a) HFG-SAC from scratch
  (b) Naive transfer (direct weight copy, no fuzzy prior, no M4 scaling)
  (c) HFG-SAC conservative transfer (M1 Jaccard prior + M4 action relaxation)

Run: uv run python run/pilot_rq3.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Make repo root importable when invoked as `uv run python run/pilot_rq3.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "run"))

os.environ.setdefault("HFG_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))

from experiment_runner import train_agent, _run_transfer  # noqa: E402

# T3: islanded normal (source) -> islanded extreme (target)
SOURCE = {
    "env_mode": "islanded",
    "scenario_name": "typical_week_summer",
    "extreme_multiplier": None,
}
TARGET = {
    "env_mode": "islanded",
    "scenario_name": "summer_extreme",
    "extreme_multiplier": {"pv": 0.5, "load": 1.25, "wt": 0.8},
}

SEED = 42
N_EPISODES = 60  # short pilot; source = min(100, 30) = 30 episodes
DEVICE = "cpu"
OUT_DIR = REPO_ROOT / "outputs" / "experiments" / "pilot_rq3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def summarise(tag: str, result) -> dict:
    row = {
        "arm": tag,
        "cost_per_day": round(float(result.cost_per_day), 2),
        "violation_pct": round(float(result.violation_rate), 2),
        "min_fcsd": round(float(result.min_fcsd), 3),
        "convergence_ep": int(result.convergence_episode),
        "train_time_s": round(float(result.training_time_s), 1),
    }
    if getattr(result, "metadata", None):
        row["m1_similarities"] = result.metadata.get("constraint_similarities")
        row["m1_categories"] = result.metadata.get("constraint_categories")
    return row


def main() -> None:
    rows = []

    print("\n[A] HFG-SAC from scratch on T3 target...")
    t0 = time.time()
    r_scratch = train_agent(
        "hfg_sac", "T3_pilot_scratch", TARGET, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
    )
    rows.append(summarise("from_scratch", r_scratch))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n[B] Naive transfer (conservative=False)...")
    t0 = time.time()
    r_naive = _run_transfer(
        SOURCE, TARGET, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
        conservative=False,
    )
    rows.append(summarise("naive_transfer", r_naive))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n[C] HFG-SAC conservative transfer (M1+M4)...")
    t0 = time.time()
    r_hfg = _run_transfer(
        SOURCE, TARGET, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
        conservative=True,
    )
    rows.append(summarise("conservative_transfer", r_hfg))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n" + "=" * 64)
    print("PILOT RQ3 SUMMARY (T3, seed=42, n_episodes=60)")
    print("=" * 64)
    for r in rows:
        print(json.dumps(r, indent=2))

    out_json = OUT_DIR / "pilot_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "config": {
            "task": "T3_island_normal_extreme",
            "seed": SEED, "n_episodes": N_EPISODES,
        }}, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
