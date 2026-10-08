"""RQ3 T2 diagnostic pilot.

T2 = grid-connected summer -> islanded summer.
This is the critical transfer task: obs dims differ (grid=8, island=9),
so weight transfer is SKIPPED and M3 freezing is inactive.  We test whether
M1 (Jaccard fuzzy prior) + M4 (conservative action scaling) provide any
benefit over naive training when no weights can be copied.

Arms (single seed, 200 episodes target):
  A. from_scratch HFG on islanded target
  B. naive transfer (conservative=False) -- same as A since dims differ
  C. M4-only (conservative=True, use_M1=False, use_M3=False)
  D. M1+M4 (conservative=True, use_M1=True, use_M3=False)

Run: uv run python run/pilot_rq3_t2.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "run"))

os.environ.setdefault("HFG_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))

from experiment_runner import train_agent, _run_transfer  # noqa: E402

# T2: grid summer -> island summer
SOURCE = {
    "env_mode": "grid_connected",
    "scenario_name": "typical_week_summer",
    "extreme_multiplier": None,
}
TARGET = {
    "env_mode": "islanded",
    "scenario_name": "typical_week_summer",
    "extreme_multiplier": None,
}

SEED = 42
N_EPISODES = 200
DEVICE = "cpu"
OUT_DIR = REPO_ROOT / "outputs" / "experiments" / "pilot_rq3_t2"
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
        for k, v in result.metadata.items():
            if isinstance(v, dict):
                row[k] = {kk: round(vv, 3) if isinstance(vv, float) else vv
                          for kk, vv in v.items()}
            else:
                row[k] = v
    return row


def main() -> None:
    rows = []

    print("\n[A] HFG-SAC from scratch on T2 target (islanded)...")
    t0 = time.time()
    r_a = train_agent(
        "hfg_sac", "T2_pilot_scratch", TARGET, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
    )
    rows.append(summarise("from_scratch", r_a))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n[B] Naive transfer (conservative=False)...")
    t0 = time.time()
    r_b = _run_transfer(
        SOURCE, TARGET, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
        conservative=False,
    )
    rows.append(summarise("naive_transfer", r_b))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n[C] M4-only (conservative, no M1, no M3)...")
    t0 = time.time()
    r_c = _run_transfer(
        SOURCE, TARGET, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
        conservative=True, use_M1=False, use_M3=False, use_M4=True,
    )
    rows.append(summarise("M4_only", r_c))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n[D] M1+M4 (full conservative, no M3 since dims differ)...")
    t0 = time.time()
    r_d = _run_transfer(
        SOURCE, TARGET, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
        conservative=True, use_M1=True, use_M3=False, use_M4=True,
    )
    rows.append(summarise("M1_M4", r_d))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n" + "=" * 64)
    print("PILOT RQ3 T2 SUMMARY (grid->island, seed=42, n_episodes=200)")
    print("=" * 64)
    for r in rows:
        print(json.dumps(r, indent=2))

    out_json = OUT_DIR / "pilot_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "config": {
            "task": "T2_grid_island", "seed": SEED,
            "n_episodes": N_EPISODES,
        }}, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
