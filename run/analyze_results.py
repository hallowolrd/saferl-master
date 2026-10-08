"""Unified experiment analysis: aggregate per-seed JSONs + generate figures/tables.

This single script replaces both `aggregate_results.py` and the old
`analyze_results.py`. It reads per-(algo, scenario, seed) JSONs from
`outputs/experiments/results/`, aggregates them into the five summary tables
that the figures expect, and then renders all publication-quality plots,
markdown/CSV tables, and a short analysis report.

Usage:
    .venv/Scripts/python.exe run/analyze_results.py
    .venv/Scripts/python.exe run/analyze_results.py \\
        --input_dir outputs/experiments --output_dir outputs/analysis
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa: F401  (kept for downstream convenience)

# ============================================================================
# Plot style
# ============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

PALETTE = {
    "hfg_sac": "#E69F00", "HFG-SAC (ours)": "#E69F00", "HFG-SAC": "#E69F00",
    "sac": "#56B4E9", "SAC": "#56B4E9",
    "ppo": "#009E73", "PPO": "#009E73",
    "ppo_lagrangian": "#F0E442", "PPO-Lagrangian": "#F0E442",
    "cpo": "#0072B2", "CPO": "#0072B2",
    "safety_layer": "#D55E00", "Safety Layer": "#D55E00",
    "CBF Safety Layer": "#D55E00",
    "fuzzy_sac": "#CC79A7", "Fuzzy SAC": "#CC79A7",
    "safe_sac": "#999999", "Safe SAC": "#999999",
    "MILP": "#000000", "MPC": "#333333",
    "hfg_sac_full": "#E69F00", "Full HFG-SAC": "#E69F00",
    "hfg_sac_no_fuzzycon": "#56B4E9", "-FuzzyCon": "#56B4E9",
    "hfg_sac_no_fuzzyknow": "#009E73", "-FuzzyKnow": "#009E73",
    "hfg_sac_baseline": "#999999", "-Both (Safe SAC)": "#999999",
}

DISPLAY_NAMES = {
    "hfg_sac": "HFG-SAC (ours)",
    "sac": "SAC", "ppo": "PPO",
    "ppo_lagrangian": "PPO-Lagrangian",
    "cpo": "CPO",
    "safety_layer": "CBF Safety Layer",
    "fuzzy_sac": "Fuzzy SAC",
    "safe_sac": "Safe SAC",
    "MILP": "MILP (oracle)", "MPC": "MPC",
    "hfg_sac_full": "Full HFG-SAC",
    "hfg_sac_no_fuzzycon": "-FuzzyCon",
    "hfg_sac_no_fuzzyknow": "-FuzzyKnow",
    "hfg_sac_baseline": "-Both (Safe SAC)",
    "from_scratch": "From scratch",
    "naive_transfer": "Naive transfer",
    "hfg_transfer": "HFG-SAC transfer",
}

# ============================================================================
# Aggregation
# ============================================================================

RQ1_SCENARIOS = ["S1_grid_normal", "S2_grid_extreme",
                "S3_island_normal", "S4_island_extreme"]
RQ1_ALGOS = ["sac", "ppo", "ppo_lagrangian", "cpo",
             "safety_layer", "fuzzy_sac", "hfg_sac"]
RQ4_VARIANTS = ["hfg_sac_full", "hfg_sac_no_fuzzycon",
                "hfg_sac_no_fuzzyknow", "hfg_sac_baseline"]
RQ3_TASKS = ["T1_summer_winter", "T2_grid_island", "T3_normal_extreme"]
RQ3_METHODS = ["from_scratch", "naive_transfer", "hfg_transfer"]


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return float("nan"), 0.0
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _summarize_runs(runs: List[dict]) -> dict:
    """Collapse a list of per-seed result dicts into mean/std metrics."""
    costs = [r.get("total_cost", 0.0) for r in runs]
    daily = [r.get("cost_per_day", c / 7.0) for r, c in zip(runs, costs)]
    viols = [r.get("violation_rate", 0.0) for r in runs]
    fcsds = [r.get("avg_fcsd", 0.0) for r in runs]
    convs = [r["convergence_episode"] for r in runs
             if r.get("convergence_episode", 0) and r["convergence_episode"] > 0]
    cm, cs = _mean_std(costs)
    dm, ds = _mean_std(daily)
    vm, vs = _mean_std(viols)
    fm, fs = _mean_std(fcsds)
    conv_m = float(np.mean(convs)) if convs else float("nan")
    return {
        "cost_mean": cm, "cost_std": cs,
        "cost_per_day_mean": dm, "cost_per_day_std": ds,
        "violation_rate_mean": vm, "violation_rate_std": vs,
        "fcsd_mean": fm, "fcsd_std": fs,
        "convergence_mean": conv_m,
        "n_seeds": len(runs),
    }


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def aggregate(results_dir: Path) -> dict:
    """Scan results_dir and build every summary the figures need."""
    if not results_dir.exists():
        raise SystemExit(f"Results dir not found: {results_dir}")

    individual: Dict[str, dict] = {}
    rq1: Dict[Tuple[str, str], List[dict]] = {}
    rq2: Dict[Tuple[str, float], List[dict]] = {}
    rq3: Dict[Tuple[str, str], List[dict]] = {}
    rq4: Dict[str, List[dict]] = {}
    rq5: Dict[Tuple[str, str, float], List[dict]] = {}

    # Regexes for the naming conventions emitted by run_all_parallel.py.
    re_rq1 = re.compile(
        r"^(?P<algo>[A-Za-z_]+)_(?P<scen>S[1-4]_(grid|island)_(normal|extreme))_s(?P<seed>\d+)$")
    re_rq2 = re.compile(
        r"^(?P<algo>[A-Za-z_]+)_extremity_(?P<level>\d+\.?\d*)_s(?P<seed>\d+)$")
    re_rq3_scratch = re.compile(
        r"^(?P<algo>[A-Za-z_]+)_(?P<task>T[1-3]_[a-z_]+)_(?P<method>from_scratch|ppo_lagrangian_scratch)_s(?P<seed>\d+)$")
    re_rq3_transfer = re.compile(
        r"^hfg_sac_(?P<task>T[1-3]_[a-z_]+)_(?P<method>naive_transfer|hfg_transfer)_s(?P<seed>\d+)$")
    re_rq4 = re.compile(
        r"^(?P<variant>hfg_sac_(full|no_fuzzycon|no_fuzzyknow|baseline))_S4_island_extreme_s(?P<seed>\d+)$")
    re_rq5 = re.compile(
        r"^hfg_sac_sens_(?P<axis>beta|alpha|kappa)_(?P<value>\d+\.?\d*)_s(?P<seed>\d+)$")

    for f in sorted(results_dir.glob("*.json")):
        if "summary" in f.name or "baselines" in f.name:
            continue
        stem = f.stem
        data = _load_json(f)
        if data is None:
            continue
        individual[stem] = data

        if m := re_rq1.match(stem):
            rq1.setdefault((m["scen"], m["algo"]), []).append(data)
        elif m := re_rq2.match(stem):
            rq2.setdefault((m["algo"], float(m["level"])), []).append(data)
        elif m := re_rq3_scratch.match(stem):
            method = "from_scratch" if m["method"] == "from_scratch" else m["method"]
            rq3.setdefault((m["task"], method), []).append(data)
        elif m := re_rq3_transfer.match(stem):
            rq3.setdefault((m["task"], m["method"]), []).append(data)
        elif m := re_rq4.match(stem):
            rq4.setdefault(m["variant"], []).append(data)
        elif m := re_rq5.match(stem):
            rq5.setdefault((m["axis"], m["value"], float(m["value"])), []).append(data)

    # ---- main_comparison ----
    main: Dict[str, Dict[str, dict]] = {s: {} for s in RQ1_SCENARIOS}
    for (scen, algo), runs in rq1.items():
        if algo in RQ1_ALGOS:
            main[scen][algo] = _summarize_runs(runs)

    # Inject MILP/MPC model baselines if present.
    for scen in RQ1_SCENARIOS:
        bl_path = results_dir / f"model_baselines_{scen}.json"
        bl = _load_json(bl_path)
        if not bl:
            continue
        for model_name in ("MILP", "MPC"):
            if model_name in bl:
                b = bl[model_name]
                main[scen][model_name] = {
                    "cost_mean": float(b.get("cost", 0.0)),
                    "cost_std": 0.0,
                    "cost_per_day_mean": float(b.get("cost_per_day", b.get("cost", 0.0) / 7.0)),
                    "cost_per_day_std": 0.0,
                    "violation_rate_mean": float(b.get("violation_rate", 0.0)),
                    "violation_rate_std": 0.0,
                    "fcsd_mean": float(b.get("avg_fcsd", 1.0)),
                    "fcsd_std": 0.0,
                    "convergence_mean": float("nan"),
                    "n_seeds": 1,
                }

    # ---- extremity (RQ2) ----
    extremity: Dict[str, Dict[str, dict]] = {}
    for (algo, level), runs in rq2.items():
        extremity.setdefault(algo, {})[str(level)] = _summarize_runs(runs)

    # ---- transfer (RQ3) ----
    transfer: Dict[str, Dict[str, dict]] = {}
    for (task, method), runs in rq3.items():
        # Normalize ppo_lagrangian_scratch -> anchor, keep from_scratch as-is.
        key = "from_scratch" if method in ("from_scratch", "ppo_lagrangian_scratch") else method
        transfer.setdefault(task, {})[key] = _summarize_runs(runs)

    # ---- ablation (RQ4) ----
    ablation: Dict[str, dict] = {}
    for variant, runs in rq4.items():
        ablation[variant] = _summarize_runs(runs)

    # ---- sensitivity (RQ5) ----
    # Figure 5 expects keys: "beta", "alpha_target", "kappa".
    sensitivity: Dict[str, Dict[str, dict]] = {}
    for (axis, value_str, value), runs in rq5.items():
        key = "alpha_target" if axis == "alpha" else axis
        sensitivity.setdefault(key, {})[value_str] = _summarize_runs(runs)

    return {
        "main_comparison": main,
        "ablation": ablation,
        "sensitivity": sensitivity,
        "extremity": extremity,
        "transfer": transfer,
        "individual_runs": individual,
    }


# ============================================================================
# Figures (unchanged logic from original analyze_results.py)
# ============================================================================

def generate_training_curves(results: dict, output_dir: Path) -> None:
    print("Generating training curves figure...")
    individual = results.get("individual_runs", {})
    algos = ["sac", "ppo", "ppo_lagrangian", "cpo",
             "safety_layer", "fuzzy_sac", "hfg_sac"]
    scenario = "S1_grid_normal"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for algo in algos:
        rewards_list, violations_list, fcsds_list = [], [], []
        min_len = None
        for seed in [42, 123, 456, 789, 2024]:
            key = f"{algo}_{scenario}_s{seed}"
            if key not in individual:
                continue
            run = individual[key]
            r = run.get("episode_rewards", [])
            v = run.get("episode_violation_rates", [])
            f = run.get("episode_fcsds", [])
            rewards_list.append(r)
            violations_list.append(v)
            fcsds_list.append(f)
            L = min(len(r), len(v), len(f))
            if min_len is None or L < min_len:
                min_len = L
        if not rewards_list or (min_len or 0) < 10:
            continue

        r_arr = np.array([r[:min_len] for r in rewards_list])
        v_arr = np.array([v[:min_len] for v in violations_list])
        f_arr = np.array([f[:min_len] for f in fcsds_list])

        window = 10
        def smooth(x):
            if x.shape[1] <= window:
                return x
            kernel = np.ones(window) / window
            return np.array([np.convolve(row, kernel, mode="valid") for row in x])

        r_s, v_s, f_s = smooth(r_arr), smooth(v_arr), smooth(f_arr)
        r_m, r_sd = r_s.mean(0), r_s.std(0)
        v_m, v_sd = v_s.mean(0), v_s.std(0)
        f_m, f_sd = f_s.mean(0), f_s.std(0)
        x_vals = np.arange(window, window + len(r_m))

        label = DISPLAY_NAMES.get(algo, algo)
        color = PALETTE.get(algo, "#333333")
        lw = 2.5 if algo == "hfg_sac" else 1.5
        zo = 10 if algo == "hfg_sac" else 5

        axes[0].plot(x_vals, r_m, label=label, color=color, linewidth=lw, zorder=zo)
        axes[0].fill_between(x_vals, r_m - r_sd, r_m + r_sd, color=color, alpha=0.15)
        axes[1].plot(x_vals, v_m, label=label, color=color, linewidth=lw, zorder=zo)
        axes[1].fill_between(x_vals, np.maximum(0, v_m - v_sd), v_m + v_sd, color=color, alpha=0.15)
        axes[2].plot(x_vals, f_m, label=label, color=color, linewidth=lw, zorder=zo)
        axes[2].fill_between(x_vals, f_m - f_sd, np.minimum(1, f_m + f_sd), color=color, alpha=0.15)

    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("(a) Training Reward"); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Violation Rate (%)")
    axes[1].set_title("(b) Constraint Violation Rate"); axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log"); axes[1].set_ylim(bottom=0.1)
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Average FCSD")
    axes[2].set_title("(c) Fuzzy Constraint Satisfaction Degree"); axes[2].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    plt.tight_layout()
    fig.savefig(output_dir / "fig1_training_curves.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig1_training_curves.png", bbox_inches="tight")
    plt.close(fig)


def generate_performance_comparison(results: dict, output_dir: Path) -> None:
    print("Generating performance comparison figure...")
    main = results.get("main_comparison", {})
    if not main:
        print("  Skipping: no main comparison data"); return

    scenarios = ["S1_grid_normal", "S2_grid_extreme",
                 "S3_island_normal", "S4_island_extreme"]
    s_labels = ["S1: Grid\nNormal", "S2: Grid\nExtreme",
                "S3: Island\nNormal", "S4: Island\nExtreme"]
    algos = ["sac", "ppo", "ppo_lagrangian", "cpo",
             "safety_layer", "fuzzy_sac", "hfg_sac"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_scen = len(scenarios); n_algo = len(algos); width = 0.10

    for s_idx, (scen, s_lab) in enumerate(zip(scenarios, s_labels)):
        costs, costs_e, viols, viols_e = [], [], [], []
        for algo in algos:
            d = main.get(scen, {}).get(algo, {})
            costs.append(d.get("cost_per_day_mean", d.get("cost_mean", 0)))
            costs_e.append(d.get("cost_per_day_std", d.get("cost_std", 0)))
            viols.append(d.get("violation_rate_mean", 0))
            viols_e.append(d.get("violation_rate_std", 0))
        x = np.arange(n_algo) + (s_idx - n_scen / 2 + 0.5) * width
        axes[0].bar(x, costs, width, yerr=costs_e, label=s_lab, capsize=3, alpha=0.85)
        axes[1].bar(x, viols, width, yerr=viols_e, label=s_lab, capsize=3, alpha=0.85)

    axes[0].set_xticks(np.arange(n_algo))
    axes[0].set_xticklabels([DISPLAY_NAMES.get(a, a) for a in algos], rotation=30, ha="right")
    axes[0].set_ylabel("Daily Operating Cost (¥)")
    axes[0].set_title("(a) Operating Cost")
    axes[0].legend(title="Scenario", fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].set_xticks(np.arange(n_algo))
    axes[1].set_xticklabels([DISPLAY_NAMES.get(a, a) for a in algos], rotation=30, ha="right")
    axes[1].set_ylabel("Constraint Violation Rate (%)")
    axes[1].set_title("(b) Constraint Violation Rate")
    axes[1].legend(title="Scenario", fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_performance_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig2_performance_comparison.png", bbox_inches="tight")
    plt.close(fig)


def generate_pareto_frontier(results: dict, output_dir: Path) -> None:
    print("Generating Pareto frontier figure...")
    main = results.get("main_comparison", {})
    if not main:
        print("  Skipping: no data"); return

    scenarios = ["S2_grid_extreme", "S4_island_extreme"]
    s_labels = ["Grid-connected Extreme", "Islanded Extreme"]
    algos = ["sac", "ppo", "ppo_lagrangian", "cpo",
             "safety_layer", "fuzzy_sac", "hfg_sac"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_i, (scen, s_lab) in enumerate(zip(scenarios, s_labels)):
        ax = axes[ax_i]
        for algo in algos:
            d = main.get(scen, {}).get(algo)
            if not d:
                continue
            cost = d.get("cost_per_day_mean", d.get("cost_mean", 0))
            viol = d.get("violation_rate_mean", 0)
            cost_e = d.get("cost_per_day_std", d.get("cost_std", 0))
            viol_e = d.get("violation_rate_std", 0)
            label = DISPLAY_NAMES.get(algo, algo)
            color = PALETTE.get(algo, "#333333")
            marker = "*" if algo == "hfg_sac" else "o"
            size = 150 if algo == "hfg_sac" else 80
            ax.errorbar(viol, cost, xerr=viol_e, yerr=cost_e,
                       fmt="none", ecolor=color, alpha=0.4, capsize=3)
            ax.scatter(viol, cost, c=color, s=size, marker=marker,
                      label=label, zorder=10 if algo == "hfg_sac" else 5,
                      edgecolors="white", linewidth=0.5)
        ax.set_xlabel("Constraint Violation Rate (%)")
        ax.set_ylabel("Daily Operating Cost (¥)")
        ax.set_title(s_lab); ax.grid(True, alpha=0.3)
        if ax_i == 0:
            ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(output_dir / "fig3_pareto_frontier.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig3_pareto_frontier.png", bbox_inches="tight")
    plt.close(fig)


def generate_ablation(results: dict, output_dir: Path) -> None:
    print("Generating ablation study figure...")
    ablation = results.get("ablation", {})
    if not ablation:
        print("  Skipping: no ablation data"); return

    variants = RQ4_VARIANTS
    v_labels = ["Full\nHFG-SAC", "−FuzzyCon", "−FuzzyKnow", "−Both\n(Safe SAC)"]
    metrics = [("cost", "Daily Cost (¥)"),
               ("violation", "Violation Rate (%)"),
               ("fcsd", "Avg. FCSD"),
               ("convergence", "Convergence (ep.)")]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for m_i, (metric, m_lab) in enumerate(metrics):
        ax = axes[m_i]
        values, errors, colors = [], [], []
        for v in variants:
            d = ablation.get(v, {})
            if metric == "cost":
                values.append(d.get("cost_per_day_mean", d.get("cost_mean", 0)))
                errors.append(d.get("cost_per_day_std", d.get("cost_std", 0)))
            elif metric == "violation":
                values.append(d.get("violation_rate_mean", 0))
                errors.append(d.get("violation_rate_std", 0))
            elif metric == "fcsd":
                values.append(d.get("fcsd_mean", 0))
                errors.append(d.get("fcsd_std", 0))
            else:
                values.append(d.get("convergence_mean", 0))
                errors.append(0)
            colors.append(PALETTE.get(v, "#888888"))
        x = np.arange(len(variants))
        ax.bar(x, values, yerr=errors, capsize=4, color=colors, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(v_labels, fontsize=8)
        ax.set_ylabel(m_lab); ax.set_title(f"({chr(97+m_i)}) {m_lab}")
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_dir / "fig4_ablation_study.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig4_ablation_study.png", bbox_inches="tight")
    plt.close(fig)


def generate_sensitivity(results: dict, output_dir: Path) -> None:
    print("Generating sensitivity analysis figure...")
    sens = results.get("sensitivity", {})
    if not sens:
        print("  Skipping: no sensitivity data"); return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    params = [
        ("beta", "Fuzzy Boundary Width (β)", "β"),
        ("alpha_target", r"Target FCSD ($\alpha_{target}$)", r"$\alpha_{target}$"),
        ("kappa", r"Knowledge Weight ($\kappa_0$)", r"$\kappa_0$"),
    ]
    for p_i, (key, x_lab, p_lab) in enumerate(params):
        ax = axes[p_i]
        if key not in sens:
            continue
        pd = sens[key]
        x_vals = sorted(float(k) for k in pd.keys())
        costs, costs_e, viols, viols_e = [], [], [], []
        for xv in x_vals:
            d = pd[str(xv)]
            costs.append(d.get("cost_per_day_mean", d.get("cost_mean", 0)))
            costs_e.append(d.get("cost_per_day_std", d.get("cost_std", 0)))
            viols.append(d.get("violation_rate_mean", 0))
            viols_e.append(d.get("violation_rate_std", 0))
        ax2 = ax.twinx()
        l1 = ax.errorbar(x_vals, costs, yerr=costs_e, fmt="o-",
                         color="#E69F00", linewidth=2, markersize=6,
                         capsize=3, label="Cost")
        l2 = ax2.errorbar(x_vals, viols, yerr=viols_e, fmt="s--",
                          color="#56B4E9", linewidth=2, markersize=6,
                          capsize=3, label="Violation Rate")
        ax.set_xlabel(x_lab); ax.set_ylabel("Daily Cost (¥)", color="#E69F00")
        ax2.set_ylabel("Violation Rate (%)", color="#56B4E9")
        ax.tick_params(axis="y", labelcolor="#E69F00")
        ax2.tick_params(axis="y", labelcolor="#56B4E9")
        ax.set_title(f"({chr(97+p_i)}) {p_lab} sensitivity")
        ax.grid(True, alpha=0.3)
        ax.legend([l1, l2], ["Cost", "Violation Rate"], fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(output_dir / "fig5_sensitivity_analysis.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig5_sensitivity_analysis.png", bbox_inches="tight")
    plt.close(fig)


def generate_extremity(results: dict, output_dir: Path) -> None:
    print("Generating extremity robustness figure...")
    ext = results.get("extremity", {})
    if not ext:
        print("  Skipping: no extremity data"); return

    algos = ["sac", "ppo_lagrangian", "cpo", "hfg_sac"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for algo in algos:
        if algo not in ext:
            continue
        ad = ext[algo]
        levels = sorted(float(k) for k in ad.keys())
        costs, costs_e, viols, viols_e = [], [], [], []
        for lv in levels:
            d = ad[str(lv)]
            costs.append(d.get("cost_per_day_mean", d.get("cost_mean", 0)))
            costs_e.append(d.get("cost_per_day_std", d.get("cost_std", 0)))
            viols.append(d.get("violation_rate_mean", 0))
            viols_e.append(d.get("violation_rate_std", 0))
        label = DISPLAY_NAMES.get(algo, algo)
        color = PALETTE.get(algo, "#333333")
        lw = 2.5 if algo == "hfg_sac" else 1.5
        zo = 10 if algo == "hfg_sac" else 5
        axes[0].errorbar(levels, costs, yerr=costs_e, fmt="o-",
                         color=color, label=label, linewidth=lw,
                         markersize=5, capsize=3, zorder=zo)
        axes[1].errorbar(levels, viols, yerr=viols_e, fmt="o-",
                         color=color, label=label, linewidth=lw,
                         markersize=5, capsize=3, zorder=zo)
    axes[0].set_xlabel("Extremity Level"); axes[0].set_ylabel("Daily Operating Cost (¥)")
    axes[0].set_title("(a) Cost under Increasing Extremity")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Extremity Level"); axes[1].set_ylabel("Constraint Violation Rate (%)")
    axes[1].set_title("(b) Safety Degradation under Extremity")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8)
    axes[1].set_yscale("log")
    plt.tight_layout()
    fig.savefig(output_dir / "fig6_extremity_robustness.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig6_extremity_robustness.png", bbox_inches="tight")
    plt.close(fig)


def generate_transfer(results: dict, output_dir: Path) -> None:
    print("Generating transfer learning figure...")
    tr = results.get("transfer", {})
    if not tr:
        print("  Skipping: no transfer data"); return

    tasks = RQ3_TASKS
    t_labels = ["T1: Seasonal\n(Summer→Winter)",
                "T2: Mode\n(Grid→Island)",
                "T3: Severity\n(Normal→Extreme)"]
    methods = ["from_scratch", "naive_transfer", "hfg_transfer"]
    m_labels = ["From scratch", "Naive transfer", "HFG-SAC transfer"]
    m_colors = ["#999999", "#D55E00", "#E69F00"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [("cost", "Final Cost (¥)", "(a) Final Performance"),
               ("violation", "Violation Rate (%)", "(b) Transfer Safety"),
               ("convergence", "Fine-tune Episodes", "(c) Transfer Speed")]
    for m_i, (metric, y_lab, title) in enumerate(metrics):
        ax = axes[m_i]
        width = 0.25
        x = np.arange(len(tasks))
        for meth_i, (method, m_lab, color) in enumerate(zip(methods, m_labels, m_colors)):
            values, errors = [], []
            for task in tasks:
                d = tr.get(task, {}).get(method, {})
                if metric == "cost":
                    values.append(d.get("cost_per_day_mean", d.get("cost_mean", 0)))
                    errors.append(d.get("cost_per_day_std", d.get("cost_std", 0)))
                elif metric == "violation":
                    values.append(d.get("violation_rate_mean", 0))
                    errors.append(d.get("violation_rate_std", 0))
                else:
                    values.append(d.get("convergence_mean", 0))
                    errors.append(0)
            ax.bar(x + (meth_i - 1) * width, values, width, yerr=errors,
                   label=m_lab, color=color, capsize=3, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(t_labels, fontsize=8)
        ax.set_ylabel(y_lab); ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "fig7_transfer_learning.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig7_transfer_learning.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Tables + report
# ============================================================================

def generate_tables(results: dict, output_dir: Path) -> None:
    import csv
    print("Generating result tables...")
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    main = results.get("main_comparison", {})
    ablation = results.get("ablation", {})
    transfer = results.get("transfer", {})

    if main:
        rows = [["algorithm", "scenario", "cost_per_day_mean", "cost_per_day_std",
                 "violation_rate_mean", "violation_rate_std", "fcsd_mean"]]
        md = ["# Main Performance Comparison (mean ± std over seeds)", "",
              "| Algorithm | S1: Grid Normal | S2: Grid Extreme | S3: Island Normal | S4: Island Extreme |",
              "|---|---|---|---|---|",
              "| | ¥/d / Viol. % | ¥/d / Viol. % | ¥/d / Viol. % | ¥/d / Viol. % |"]
        for algo in ["MILP", "MPC"] + RQ1_ALGOS:
            row = f"| **{DISPLAY_NAMES.get(algo, algo)}** "
            for scen in RQ1_SCENARIOS:
                d = main.get(scen, {}).get(algo)
                if not d:
                    row += "| — "; continue
                dm = d.get("cost_per_day_mean", d.get("cost_mean", 0) / 7.0)
                ds = d.get("cost_per_day_std", 0.0)
                vm = d["violation_rate_mean"]; vs = d["violation_rate_std"]
                if algo in ("MILP", "MPC"):
                    row += f"| {dm:,.0f} / {vm:.1f} "
                else:
                    row += f"| {dm:,.0f}±{ds:,.0f} / {vm:.1f}±{vs:.1f} "
                rows.append([algo, scen, dm, ds, vm, vs, d["fcsd_mean"]])
            row += "|"; md.append(row)
        (tables_dir / "table1_main_comparison.md").write_text("\n".join(md), encoding="utf-8")
        with open(tables_dir / "table1_main_comparison.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)

    if ablation:
        md = ["# Ablation Study (S4: Islanded Extreme)", "",
              "| Variant | Daily Cost (¥) | Violation Rate (%) | Avg. FCSD | Convergence (ep.) |",
              "|---|---|---|---|---|"]
        rows = [["variant", "cost_mean", "cost_std", "violation_mean",
                 "violation_std", "fcsd_mean", "fcsd_std", "convergence_mean"]]
        for v in RQ4_VARIANTS:
            d = ablation.get(v, {})
            if not d: continue
            md.append(
                f"| {DISPLAY_NAMES.get(v, v)} "
                f"| {d.get('cost_per_day_mean', 0):,.0f}±{d.get('cost_per_day_std', 0):,.0f} "
                f"| {d.get('violation_rate_mean', 0):.1f}±{d.get('violation_rate_std', 0):.1f} "
                f"| {d.get('fcsd_mean', 0):.3f} "
                f"| {d.get('convergence_mean', 0):.0f} |"
            )
            rows.append([v, d.get("cost_mean", 0), d.get("cost_std", 0),
                         d.get("violation_rate_mean", 0), d.get("violation_rate_std", 0),
                         d.get("fcsd_mean", 0), d.get("fcsd_std", 0),
                         d.get("convergence_mean", 0)])
        (tables_dir / "table2_ablation.md").write_text("\n".join(md), encoding="utf-8")
        with open(tables_dir / "table2_ablation.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)

    if transfer:
        md = ["# Transfer Learning Results", "",
              "| Task | Method | Final Cost (¥/d) | Violation Rate (%) | Fine-tune Episodes | Speed-up |",
              "|---|---|---|---|---|---|"]
        rows = [["task", "method", "cost_mean", "cost_std",
                 "violation_mean", "violation_std", "convergence_mean"]]
        for task in RQ3_TASKS:
            methods_d = transfer.get(task, {})
            scratch_conv = methods_d.get("from_scratch", {}).get("convergence_mean", 1)
            for method in RQ3_METHODS:
                d = methods_d.get(method, {})
                if not d: continue
                speed = scratch_conv / max(d.get("convergence_mean", 1), 1)
                md.append(
                    f"| {task} | {DISPLAY_NAMES.get(method, method)} "
                    f"| {d.get('cost_per_day_mean', 0):,.0f}±{d.get('cost_per_day_std', 0):,.0f} "
                    f"| {d.get('violation_rate_mean', 0):.1f}±{d.get('violation_rate_std', 0):.1f} "
                    f"| {d.get('convergence_mean', 0):.0f} | {speed:.1f}× |"
                )
                rows.append([task, method, d.get("cost_mean", 0), d.get("cost_std", 0),
                             d.get("violation_rate_mean", 0), d.get("violation_rate_std", 0),
                             d.get("convergence_mean", 0)])
        (tables_dir / "table3_transfer.md").write_text("\n".join(md), encoding="utf-8")
        with open(tables_dir / "table3_transfer.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)


def generate_analysis_report(results: dict, output_dir: Path) -> None:
    print("Generating analysis report...")
    main = results.get("main_comparison", {})
    ablation = results.get("ablation", {})
    transfer = results.get("transfer", {})

    lines = ["# Experiment Analysis Report", "", "## Key Findings", ""]
    lines.append("### RQ1: Overall Effectiveness")
    if main.get("S4_island_extreme"):
        hfg = main["S4_island_extreme"].get("hfg_sac", {})
        sac = main["S4_island_extreme"].get("sac", {})
        if hfg and sac:
            hc = hfg.get("cost_per_day_mean", hfg.get("cost_mean", 0))
            sc = sac.get("cost_per_day_mean", sac.get("cost_mean", 0))
            hv = hfg.get("violation_rate_mean", 0)
            sv = sac.get("violation_rate_mean", 0)
            lines.append(f"- HFG-SAC on S4: cost ¥{hc:,.0f}/d, violation {hv:.1f}%")
            if sc > 0:
                lines.append(f"- Cost improvement vs SAC: {(1-hc/sc)*100:.1f}%")
            if sv > 0:
                lines.append(f"- Violation reduction vs SAC: {(1-hv/sv)*100:.1f}%")
    lines.append("")

    lines.append("### RQ4: Component Contributions")
    if ablation:
        full = ablation.get("hfg_sac_full", {})
        base = ablation.get("hfg_sac_baseline", {})
        fc = full.get("cost_per_day_mean", full.get("cost_mean", 0))
        bc = base.get("cost_per_day_mean", base.get("cost_mean", 0))
        fv = full.get("violation_rate_mean", 0)
        bv = base.get("violation_rate_mean", 0)
        if bc > 0:
            lines.append(f"- Full vs baseline: cost {(1-fc/bc)*100:.1f}% lower, "
                         f"violation {(1-fv/max(bv,1e-6))*100:.1f}% lower")
    lines.append("")

    lines.append("### RQ3: Transfer Speed-up")
    if transfer:
        for task, d in transfer.items():
            h = d.get("hfg_transfer", {})
            s = d.get("from_scratch", {})
            hc = h.get("convergence_mean", 0)
            sc = s.get("convergence_mean", 0)
            if hc > 0 and sc > 0:
                lines.append(f"- {task}: {sc/hc:.1f}× speed-up; "
                             f"violation {h.get('violation_rate_mean',0):.1f}% "
                             f"(vs {s.get('violation_rate_mean',0):.1f}% from scratch)")

    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate + analyze HFG-SRL results")
    parser.add_argument("--input_dir", type=str, default="outputs/experiments")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis")
    parser.add_argument("--save_summaries", action="store_true",
                        help="Also write the 5 summary JSONs into <input_dir>/results/")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = input_dir / "results"

    print(f"Scanning per-seed results in: {results_dir}")
    results = aggregate(results_dir)

    print(f"  main_comparison : "
          f"{sum(len(v) for v in results['main_comparison'].values())} (scenario, algo) cells")
    print(f"  extremity       : "
          f"{sum(len(v) for v in results['extremity'].values())} (algo, level) cells")
    print(f"  transfer        : "
          f"{sum(len(v) for v in results['transfer'].values())} (task, method) cells")
    print(f"  ablation        : {len(results['ablation'])} variants")
    print(f"  sensitivity     : "
          f"{sum(len(v) for v in results['sensitivity'].values())} (axis, value) cells")
    print(f"  individual runs : {len(results['individual_runs'])} files")

    if args.save_summaries:
        for name in ("main_comparison", "ablation", "sensitivity", "extremity", "transfer"):
            out = results_dir / f"{name}_summary.json"
            out.write_text(json.dumps(results[name], indent=2, default=str), encoding="utf-8")
            print(f"  wrote {out.name}")

    generate_training_curves(results, output_dir)
    generate_performance_comparison(results, output_dir)
    generate_pareto_frontier(results, output_dir)
    generate_ablation(results, output_dir)
    generate_sensitivity(results, output_dir)
    generate_extremity(results, output_dir)
    generate_transfer(results, output_dir)
    generate_tables(results, output_dir)
    generate_analysis_report(results, output_dir)

    print(f"\nAll analysis complete. Output in: {output_dir}")


if __name__ == "__main__":
    main()
