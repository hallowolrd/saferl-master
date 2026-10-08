"""Smoke test for RQ2 (5 algos) and RQ3 (4 methods) code paths.

Minimal: 1 seed, 3 episodes, 1 level (RQ2) and T3 only (RQ3).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("HFG_DEVICE", "cpu")

OUT = PROJECT_ROOT / "outputs" / "smoke_rq2_rq3_v2"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results").mkdir(exist_ok=True)

SMOKE_EPISODES = 3
SMOKE_SEED = 42


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"SMOKE: {title}")
    print("=" * 70)


def safe(tag: str, fn):
    t0 = time.time()
    try:
        result = fn()
        dt = time.time() - t0
        print(f"[{tag}] OK in {dt:.1f}s")
        return {"tag": tag, "ok": True, "elapsed_s": dt, "result": result}
    except Exception as e:  # noqa: BLE001
        dt = time.time() - t0
        tb = traceback.format_exc()
        print(f"[{tag}] FAIL in {dt:.1f}s: {type(e).__name__}: {e}")
        print(tb)
        return {"tag": tag, "ok": False, "elapsed_s": dt,
                "error": f"{type(e).__name__}: {e}", "traceback": tb}


def summarize_runresult(r) -> dict:
    return {
        "algorithm": getattr(r, "algorithm", "?"),
        "scenario": getattr(r, "scenario", "?"),
        "total_cost": getattr(r, "total_cost", None),
        "violation_rate": getattr(r, "violation_rate", None),
        "avg_fcsd": getattr(r, "avg_fcsd", None),
    }


def smoke_rq2():
    """RQ2: 5 algorithms, level=1.0 only."""
    from run.experiment_runner import train_agent
    banner("RQ2 extremity level=1.0 (all 5 algorithms)")
    level = 1.0
    scenario_cfg = {
        "env_mode": "islanded",
        "scenario_name": "extremity_test",
        "extreme_multiplier": {
            "pv": max(0.2, 1.0 - level * 0.3),
            "load": 1.0 + level * 0.2,
            "wt": 1.0 - level * 0.15,
        },
    }
    out = {}
    for algo in ["sac", "ppo_lagrangian", "cpo", "fuzzy_sac", "hfg_sac"]:
        def _fn(a=algo):
            return train_agent(
                a, f"extremity_{level}_smoke", scenario_cfg, SMOKE_SEED,
                n_episodes=SMOKE_EPISODES, device="cpu", output_dir=OUT,
            )
        res = safe(f"RQ2:{algo}", _fn)
        if res["ok"]:
            res["metrics"] = summarize_runresult(res["result"])
        out[algo] = res
    return out


def smoke_rq3():
    """RQ3: T3 transfer, 4 methods, 1 seed."""
    from run.experiment_runner import train_agent, _run_transfer
    banner("RQ3 transfer T3 (4 methods)")
    source = {
        "env_mode": "islanded",
        "scenario_name": "typical_week_summer",
        "extreme_multiplier": None,
    }
    target = {
        "env_mode": "islanded",
        "scenario_name": "summer_extreme",
        "extreme_multiplier": {"pv": 0.5, "load": 1.25, "wt": 0.8},
    }
    out = {}

    # from_scratch (hfg_sac)
    res = safe(
        "RQ3:from_scratch_hfg",
        lambda: train_agent(
            "hfg_sac", "T3_smoke_scratch", target, SMOKE_SEED,
            n_episodes=SMOKE_EPISODES, device="cpu", output_dir=OUT,
        ),
    )
    if res["ok"]:
        res["metrics"] = summarize_runresult(res["result"])
    out["from_scratch_hfg"] = res

    # ppo_lagrangian anchor
    res = safe(
        "RQ3:anchor_ppo_lagrangian",
        lambda: train_agent(
            "ppo_lagrangian", "T3_smoke_anchor", target, SMOKE_SEED,
            n_episodes=SMOKE_EPISODES, device="cpu", output_dir=OUT,
        ),
    )
    if res["ok"]:
        res["metrics"] = summarize_runresult(res["result"])
    out["anchor_ppo_lagrangian"] = res

    # naive transfer
    res = safe(
        "RQ3:naive_transfer",
        lambda: _run_transfer(
            source, target, SMOKE_SEED,
            n_episodes=SMOKE_EPISODES, device="cpu", output_dir=OUT,
            conservative=False,
        ),
    )
    if res["ok"]:
        res["metrics"] = summarize_runresult(res["result"])
    out["naive_transfer"] = res

    # hfg transfer
    res = safe(
        "RQ3:hfg_transfer",
        lambda: _run_transfer(
            source, target, SMOKE_SEED,
            n_episodes=SMOKE_EPISODES, device="cpu", output_dir=OUT,
            conservative=True,
        ),
    )
    if res["ok"]:
        res["metrics"] = summarize_runresult(res["result"])
    out["hfg_transfer"] = res

    return out


def main():
    report = {}
    report["RQ2_extremity"] = smoke_rq2()
    report["RQ3_transfer"] = smoke_rq3()

    with open(OUT / "smoke_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("SMOKE SUMMARY")
    print("=" * 70)
    for rq, cases in report.items():
        for tag, res in cases.items():
            mark = "OK  " if res["ok"] else "FAIL"
            extra = ""
            if res["ok"] and "metrics" in res:
                m = res["metrics"]
                extra = (f" cost={m['total_cost']:.0f} "
                         f"viol={m['violation_rate']:.1f}% "
                         f"fcsd={m['avg_fcsd']:.3f}")
            elif not res["ok"]:
                extra = f"  -> {res.get('error','?')}"
            print(f"  [{mark}] {rq:18s} {tag:30s}{extra}")


if __name__ == "__main__":
    main()
