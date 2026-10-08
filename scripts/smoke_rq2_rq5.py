"""Smoke test for RQ2/RQ3/RQ4/RQ5 code paths.

Runs each pipeline with 1 seed, 3 episodes, minimal sweep values to verify:
  - code path executes without crashing
  - result JSON is produced
  - metrics look sane (cost > 0, violation in [0, 100], fcsd in [0, 1])

Usage:
    .venv/Scripts/python.exe scripts/smoke_rq2_rq5.py
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

OUT = PROJECT_ROOT / "outputs" / "smoke_rq2_rq5"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results").mkdir(exist_ok=True)

SMOKE_EPISODES = 3
SMOKE_SEED = 42


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"SMOKE: {title}")
    print("=" * 70)


def safe(tag: str, fn):
    """Run fn() and capture outcome."""
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
    """Extract key metrics from a RunResult."""
    return {
        "algorithm": getattr(r, "algorithm", "?"),
        "scenario": getattr(r, "scenario", "?"),
        "total_cost": getattr(r, "total_cost", None),
        "violation_rate": getattr(r, "violation_rate", None),
        "avg_fcsd": getattr(r, "avg_fcsd", None),
        "training_time_s": round(getattr(r, "training_time_s", 0.0), 1),
    }


# ---------- RQ4: ablation variants ----------
def smoke_rq4():
    from run.experiment_runner import (
        ABLATION_VARIANTS, SCENARIOS, train_agent,
    )
    banner(f"RQ4 ablation variants: {list(ABLATION_VARIANTS)}")
    cfg = SCENARIOS["S4_island_extreme"]
    out = {}
    for variant in ABLATION_VARIANTS:
        def _fn(v=variant):
            return train_agent(
                v, "S4_smoke", cfg, SMOKE_SEED,
                n_episodes=SMOKE_EPISODES, device="cpu", output_dir=OUT,
            )
        res = safe(f"RQ4:{variant}", _fn)
        if res["ok"]:
            res["metrics"] = summarize_runresult(res["result"])
        out[variant] = res
    return out


# ---------- RQ2: extremity (2 levels, 1 algo) ----------
def smoke_rq2():
    from run.experiment_runner import train_agent
    banner("RQ2 extremity sweep (levels 0.8 / 1.5, hfg_sac)")
    out = {}
    for level in [0.8, 1.5]:
        scenario_cfg = {
            "env_mode": "islanded",
            "scenario_name": "extremity_test",
            "extreme_multiplier": {
                "pv": max(0.2, 1.0 - level * 0.3),
                "load": 1.0 + level * 0.2,
                "wt": 1.0 - level * 0.15,
            },
        }
        def _fn(l=level, c=scenario_cfg):
            return train_agent(
                "hfg_sac", f"extremity_{l}_smoke", c, SMOKE_SEED,
                n_episodes=SMOKE_EPISODES, device="cpu", output_dir=OUT,
            )
        res = safe(f"RQ2:level={level}", _fn)
        if res["ok"]:
            res["metrics"] = summarize_runresult(res["result"])
        out[f"level_{level}"] = res
    return out


# ---------- RQ3: transfer (T3 only, 1 seed) ----------
def smoke_rq3():
    from run.experiment_runner import _run_transfer
    banner("RQ3 transfer T3 (island normal -> island extreme), conservative=True")
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
    res = safe(
        "RQ3:T3_conservative",
        lambda: _run_transfer(
            source, target, SMOKE_SEED,
            n_episodes=SMOKE_EPISODES, device="cpu", output_dir=OUT,
            conservative=True,
        ),
    )
    if res["ok"]:
        res["metrics"] = summarize_runresult(res["result"])
    return {"T3_conservative": res}


# ---------- RQ5: sensitivity (1 value per axis) ----------
def smoke_rq5():
    from run.experiment_runner import (
        SCENARIOS, _train_with_param,
    )
    banner("RQ5 sensitivity (one value per axis)")
    cfg = SCENARIOS["S2_grid_extreme"]
    out = {}
    cases = {
        "beta_width=0.05": dict(constraint_width=0.05),
        "alpha=0.85":      dict(fcsd_target=0.85),
        "kappa=0.3":       dict(shaping_weight=0.3),
    }
    for tag, kw in cases.items():
        def _fn(kw=kw):
            return _train_with_param(
                "hfg_sac", "S2_smoke", cfg, SMOKE_SEED,
                n_episodes=SMOKE_EPISODES, device="cpu", output_dir=OUT,
                **kw,
            )
        res = safe(f"RQ5:{tag}", _fn)
        if res["ok"]:
            res["metrics"] = summarize_runresult(res["result"])
        out[tag] = res
    return out


def main():
    report = {}
    report["RQ4_ablation"] = smoke_rq4()
    report["RQ2_extremity"] = smoke_rq2()
    report["RQ3_transfer"] = smoke_rq3()
    report["RQ5_sensitivity"] = smoke_rq5()

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

    print(f"\nFull report: {OUT / 'smoke_report.json'}")


if __name__ == "__main__":
    main()
