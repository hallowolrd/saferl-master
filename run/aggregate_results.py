"""Aggregate per-seed result JSONs into summary tables.

The parallel driver (run/missing_experiments.py) writes one JSON per
(algorithm, scenario, seed) but does not emit the summary files that
run/analyze_results.py expects. This script reconstructs those summaries
from the per-seed files so the downstream analysis pipeline works.

Outputs (under <output_dir>/results/):
    main_comparison_summary.json   — per-scenario, per-algo mean/std
    model_baselines_*.json         — already written by the runner (read-only)
And (under <output_dir>/tables/):
    main_comparison.md / .csv      — human-readable table (daily cost + viol.)

Usage:
    .venv/Scripts/python.exe run/aggregate_results.py --input_dir outputs/experiments
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCENARIOS = ["S1_grid_normal", "S2_grid_extreme", "S3_island_normal", "S4_island_extreme"]
ALGORITHMS = ["sac", "ppo", "ppo_lagrangian", "cpo", "safety_layer", "fuzzy_sac", "hfg_sac"]

DISPLAY = {
    "sac": "SAC",
    "ppo": "PPO",
    "ppo_lagrangian": "PPO-Lagrangian",
    "cpo": "CPO",
    "safety_layer": "Safety Layer (CBF)",
    "fuzzy_sac": "Fuzzy SAC",
    "hfg_sac": "HFG-SAC (ours)",
    "MILP": "MILP (oracle)",
    "MPC": "MPC",
}


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def load_seed_results(results_dir: Path) -> Dict[Tuple[str, str], List[dict]]:
    """Group per-seed dicts by (scenario, algorithm)."""
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for f in sorted(results_dir.glob("*_s*.json")):
        if "summary" in f.name or "baselines" in f.name:
            continue
        parts = f.stem.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        algo_scenario, _seed = parts
        for scenario in SCENARIOS:
            if algo_scenario.endswith(scenario):
                algo = algo_scenario[: -len(scenario) - 1]
                break
        else:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        grouped.setdefault((scenario, algo), []).append(data)
    return grouped


def load_model_baselines(results_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for f in sorted(results_dir.glob("model_baselines_*.json")):
        scenario = f.stem.replace("model_baselines_", "")
        try:
            out[scenario] = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def build_summary(
    grouped: Dict[Tuple[str, str], List[dict]],
    baselines: Dict[str, dict],
) -> dict:
    summary: Dict[str, dict] = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for model_name in ("MILP", "MPC"):
            if scenario in baselines and model_name in baselines[scenario]:
                b = baselines[scenario][model_name]
                summary[scenario][model_name] = {
                    "cost_mean": float(b.get("cost", 0.0)),
                    "cost_std": 0.0,
                    "violation_rate_mean": float(b.get("violation_rate", 0.0)),
                    "violation_rate_std": 0.0,
                    "fcsd_mean": float(b.get("avg_fcsd", 1.0)),
                    "fcsd_std": 0.0,
                }
        for algo in ALGORITHMS:
            seeds = grouped.get((scenario, algo))
            if not seeds:
                continue
            costs = [r["total_cost"] for r in seeds]
            daily = [r.get("cost_per_day", c / 7.0) for r, c in zip(seeds, costs)]
            viols = [r["violation_rate"] for r in seeds]
            fcsds = [r["avg_fcsd"] for r in seeds]
            convs = [r["convergence_episode"] for r in seeds if r.get("convergence_episode", 0) > 0]
            cm, cs = _mean_std(costs)
            dm, ds = _mean_std(daily)
            vm, vs = _mean_std(viols)
            fm, fs = _mean_std(fcsds)
            summary[scenario][algo] = {
                "cost_mean": cm,
                "cost_std": cs,
                "cost_per_day_mean": dm,
                "cost_per_day_std": ds,
                "violation_rate_mean": vm,
                "violation_rate_std": vs,
                "fcsd_mean": fm,
                "fcsd_std": fs,
                "convergence_mean": float(np.mean(convs)) if convs else float("nan"),
                "n_seeds": len(seeds),
            }
    return summary


def write_markdown_table(summary: dict, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = [["algorithm", "scenario", "cost_mean", "cost_std",
             "violation_mean", "violation_std", "fcsd_mean"]]
    lines = ["# Main Performance Comparison (daily cost, mean ± std over seeds)", ""]
    lines.append("| Algorithm | S1: Grid Normal | S2: Grid Extreme | S3: Island Normal | S4: Island Extreme |")
    lines.append("|---|---|---|---|---|")
    lines.append("| | Cost (¥/d) / Viol. (%) | Cost (¥/d) / Viol. (%) | Cost (¥/d) / Viol. (%) | Cost (¥/d) / Viol. (%) |")

    all_names = ["MILP", "MPC"] + ALGORITHMS
    for name in all_names:
        label = DISPLAY.get(name, name)
        row = f"| **{label}** "
        for scenario in SCENARIOS:
            d = summary[scenario].get(name)
            if d is None:
                row += "| — "
                continue
            dm = d.get("cost_per_day_mean", d.get("cost_mean", 0.0) / 7.0)
            ds = d.get("cost_per_day_std", 0.0)
            vm = d["violation_rate_mean"]
            vs = d["violation_rate_std"]
            if name in ("MILP", "MPC"):
                cell = f"| {dm:,.0f} / {vm:.1f} "
            else:
                cell = f"| {dm:,.0f}±{ds:,.0f} / {vm:.1f}±{vs:.1f} "
            row += cell
            rows.append([name, scenario, d["cost_mean"], d["cost_std"],
                         vm, vs, d["fcsd_mean"]])
        row += "|"
        lines.append(row)

    (tables_dir / "main_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    with open(tables_dir / "main_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs/experiments")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    results_dir = input_dir / "results"
    tables_dir = input_dir / "tables"

    grouped = load_seed_results(results_dir)
    baselines = load_model_baselines(results_dir)

    if not grouped:
        print("No per-seed result files found.")
        return

    summary = build_summary(grouped, baselines)

    (results_dir / "main_comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown_table(summary, tables_dir)

    n_runs = sum(len(v) for v in grouped.values())
    n_cond = len(grouped)
    print(f"Aggregated {n_runs} runs across {n_cond} (scenario, algo) conditions")
    print(f"  summary -> {results_dir / 'main_comparison_summary.json'}")
    print(f"  table   -> {tables_dir / 'main_comparison.md'}")
    for scenario in SCENARIOS:
        algos = [a for a in ["MILP", "MPC"] + ALGORITHMS if a in summary[scenario]]
        print(f"  {scenario}: {len(algos)} conditions ({', '.join(algos)})")


if __name__ == "__main__":
    main()