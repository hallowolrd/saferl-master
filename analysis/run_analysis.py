"""Strict experiment analysis: statistics + publication figures.

Generates:
- analysis-output/stats-appendix.md
- analysis-output/figure-catalog.md
- analysis-output/analysis-report.md
- analysis-output/figures/*.pdf
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------- matplotlib style (publication-ready) ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "experiments" / "results"
OUT = ROOT / "analysis" / "analysis-output"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ALGOS = ["ppo", "sac", "ppo_lagrangian", "cpo", "safety_layer", "fuzzy_sac", "hfg_sac"]
ALGO_LABELS = {
    "ppo": "PPO",
    "sac": "SAC",
    "ppo_lagrangian": "PPO-Lag",
    "cpo": "CPO",
    "safety_layer": "SL",
    "fuzzy_sac": "Fuzzy-SAC",
    "hfg_sac": "HFG-SAC (ours)",
}
ALGO_COLORS = {
    "ppo": "#9e9e9e",
    "sac": "#64b5f6",
    "ppo_lagrangian": "#ffb74d",
    "cpo": "#e57373",
    "safety_layer": "#ba68c8",
    "fuzzy_sac": "#4db6ac",
    "hfg_sac": "#d32f2f",
}
SCENARIOS = ["S1_grid_normal", "S2_grid_extreme", "S3_island_normal", "S4_island_extreme"]
SCENARIO_LABELS = {
    "S1_grid_normal": "S1\nGrid normal",
    "S2_grid_extreme": "S2\nGrid extreme",
    "S3_island_normal": "S3\nIsland normal",
    "S4_island_extreme": "S4\nIsland extreme",
}


# ---------- load data ----------
def load_all() -> dict:
    data = defaultdict(lambda: defaultdict(list))  # data[algo][scenario] = [runs]
    for algo in ALGOS:
        for sc in SCENARIOS:
            for seed in [42, 123, 456]:
                p = RESULTS / f"{algo}_{sc}_s{seed}.json"
                if not p.exists():
                    continue
                with open(p) as f:
                    d = json.load(f)
                data[algo][sc].append(d)
    return data


def ci95(xs: list[float]) -> tuple[float, float, float]:
    """Return (mean, half_width, std) with 95% CI via t distribution."""
    a = np.array(xs, dtype=float)
    n = len(a)
    m = a.mean()
    s = a.std(ddof=1) if n > 1 else 0.0
    if n > 1:
        h = stats.t.ppf(0.975, n - 1) * s / math.sqrt(n)
    else:
        h = 0.0
    return float(m), float(h), float(s)


# ---------- stats: HFG-SAC vs each baseline per scenario ----------
def run_statistics(data: dict) -> str:
    lines = ["# Stats Appendix", ""]
    lines.append("## Descriptive statistics (mean ± 95% CI, n=3 seeds)\n")
    lines.append("| Algorithm | Scenario | Cost (¥/d) | Violation (%) | Avg FCSD | Min FCSD |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for algo in ALGOS:
        for sc in SCENARIOS:
            runs = data[algo][sc]
            if not runs:
                continue
            cost_m, cost_h, _ = ci95([r["cost_per_day"] for r in runs])
            viol_m, viol_h, _ = ci95([r["violation_rate"] for r in runs])
            fcsd_m, fcsd_h, _ = ci95([r["avg_fcsd"] for r in runs])
            minfcsd_m, minfcsd_h, _ = ci95([r["min_fcsd"] for r in runs])
            lines.append(
                f"| {ALGO_LABELS[algo]} | {sc} | "
                f"{cost_m:,.0f} ± {cost_h:,.0f} | "
                f"{viol_m:.2f} ± {viol_h:.2f} | "
                f"{fcsd_m:.3f} ± {fcsd_h:.3f} | "
                f"{minfcsd_m:.3f} ± {minfcsd_h:.3f} |"
            )

    # Inferential: HFG-SAC vs each baseline, per scenario, on cost_per_day
    lines.append("")
    lines.append("## Inferential tests: HFG-SAC vs baselines")
    lines.append("")
    lines.append("Welch's t-test (two-sided) on `cost_per_day` across 3 seeds.")
    lines.append("Effect size: Cohen's d (pooled). n=3 per arm — tests are underpowered; report as exploratory.")
    lines.append("")
    lines.append("| Scenario | Comparison | Δ cost (¥/d) | t | p | Cohen's d |")
    lines.append("|---|---|---:|---:|---:|---:|")
    hfg_key = "hfg_sac"
    for sc in SCENARIOS:
        hfg_vals = np.array([r["cost_per_day"] for r in data[hfg_key][sc]])
        for algo in ALGOS:
            if algo == hfg_key:
                continue
            base_vals = np.array([r["cost_per_day"] for r in data[algo][sc]])
            t, p = stats.ttest_ind(hfg_vals, base_vals, equal_var=False)
            # Cohen's d (pooled)
            n1, n2 = len(hfg_vals), len(base_vals)
            s_pool = math.sqrt(((n1 - 1) * hfg_vals.var(ddof=1) + (n2 - 1) * base_vals.var(ddof=1)) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 0
            d = (hfg_vals.mean() - base_vals.mean()) / s_pool if s_pool > 0 else float("nan")
            lines.append(
                f"| {sc} | HFG-SAC vs {ALGO_LABELS[algo]} | "
                f"{hfg_vals.mean() - base_vals.mean():+,.0f} | "
                f"{t:.2f} | {p:.3f} | {d:+.2f} |"
            )

    # Violation rate comparison (primary safety metric)
    lines.append("")
    lines.append("## Safety: violation rate comparison")
    lines.append("")
    lines.append("| Scenario | HFG-SAC viol% | Best baseline viol% | Δ |")
    lines.append("|---|---:|---:|---:|")
    for sc in SCENARIOS:
        hfg_v = np.mean([r["violation_rate"] for r in data[hfg_key][sc]])
        best_base_v = min(
            np.mean([r["violation_rate"] for r in data[a][sc]])
            for a in ALGOS if a != hfg_key
        )
        lines.append(f"| {sc} | {hfg_v:.2f} | {best_base_v:.2f} | {hfg_v - best_base_v:+.2f} |")

    text = "\n".join(lines) + "\n"
    (OUT / "stats-appendix.md").write_text(text, encoding="utf-8")
    return text


# ---------- figures ----------
def fig_main_comparison(data: dict) -> None:
    """Fig 1: grouped bar chart of cost_per_day and violation rate across scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # --- left: cost_per_day ---
    ax = axes[0]
    n_algos = len(ALGOS)
    n_sc = len(SCENARIOS)
    width = 0.11
    x = np.arange(n_sc)
    for i, algo in enumerate(ALGOS):
        means = []
        errs = []
        for sc in SCENARIOS:
            vals = [r["cost_per_day"] for r in data[algo][sc]]
            m, h, _ = ci95(vals)
            means.append(m)
            errs.append(h)
        offset = (i - n_algos / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=errs,
               label=ALGO_LABELS[algo], color=ALGO_COLORS[algo],
               edgecolor="white", linewidth=0.5, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.set_ylabel("Operational cost (¥/day)")
    ax.set_title("(a) Cost per day (mean ± 95% CI, n=3)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(ncol=2, frameon=False, loc="upper right")

    # --- right: violation rate ---
    ax = axes[1]
    for i, algo in enumerate(ALGOS):
        means = []
        errs = []
        for sc in SCENARIOS:
            vals = [r["violation_rate"] for r in data[algo][sc]]
            m, h, _ = ci95(vals)
            means.append(m)
            errs.append(h)
        offset = (i - n_algos / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=errs,
               label=ALGO_LABELS[algo], color=ALGO_COLORS[algo],
               edgecolor="white", linewidth=0.5, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.set_ylabel("Hard constraint violation (%)")
    ax.set_title("(b) Violation rate (mean ± 95% CI, n=3)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(5.0, color="red", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(n_sc - 0.5, 5.5, "5% safety threshold", color="red", fontsize=8, ha="right")

    fig.tight_layout()
    fig.savefig(FIG / "figure-01-main-comparison.pdf")
    fig.savefig(FIG / "figure-01-main-comparison.png")
    plt.close(fig)


def fig_training_curves(data: dict) -> None:
    """Fig 2: training curves — episode reward and cost for HFG-SAC vs key baselines."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    key_algos = ["ppo", "safety_layer", "fuzzy_sac", "hfg_sac"]
    for j, sc in enumerate(SCENARIOS):
        ax = axes[j // 2, j % 2]
        for algo in key_algos:
            runs = data[algo][sc]
            if not runs:
                continue
            # align by episode index (all 200 eps)
            curves = np.array([r["episode_costs"] for r in runs])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
            eps = np.arange(1, len(mean) + 1)
            ax.plot(eps, mean, label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], linewidth=1.2)
            ax.fill_between(eps, mean - std, mean + std, color=ALGO_COLORS[algo], alpha=0.15)
        ax.set_title(SCENARIO_LABELS[sc].replace("\n", " "))
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode cost (¥)")
        ax.grid(alpha=0.3, linestyle="--")
        if j == 0:
            ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Training dynamics: episode cost across scenarios", y=1.00)
    fig.tight_layout()
    fig.savefig(FIG / "figure-02-training-curves.pdf")
    fig.savefig(FIG / "figure-02-training-curves.png")
    plt.close(fig)


def fig_violation_trajectory(data: dict) -> None:
    """Fig 3: episode-level violation rate over training, S3/S4 focus."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for j, sc in enumerate(["S3_island_normal", "S4_island_extreme"]):
        ax = axes[j]
        for algo in ["ppo", "ppo_lagrangian", "safety_layer", "fuzzy_sac", "hfg_sac"]:
            runs = data[algo][sc]
            if not runs:
                continue
            curves = np.array([r["episode_violation_rates"] for r in runs])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
            eps = np.arange(1, len(mean) + 1)
            ax.plot(eps, mean, label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], linewidth=1.2)
            ax.fill_between(eps, mean - std, mean + std, color=ALGO_COLORS[algo], alpha=0.15)
        ax.axhline(5.0, color="red", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(SCENARIO_LABELS[sc].replace("\n", " "))
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode violation rate (%)")
        ax.grid(alpha=0.3, linestyle="--")
        if j == 0:
            ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "figure-03-violation-trajectory.pdf")
    fig.savefig(FIG / "figure-03-violation-trajectory.png")
    plt.close(fig)


def fig_fcsd_heatmap(data: dict) -> None:
    """Fig 4: heatmap of avg FCSD across algorithm x scenario."""
    matrix = np.zeros((len(ALGOS), len(SCENARIOS)))
    for i, algo in enumerate(ALGOS):
        for j, sc in enumerate(SCENARIOS):
            runs = data[algo][sc]
            matrix[i, j] = np.mean([r["avg_fcsd"] for r in runs])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(SCENARIOS)))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.set_yticks(np.arange(len(ALGOS)))
    ax.set_yticklabels([ALGO_LABELS[a] for a in ALGOS])
    for i in range(len(ALGOS)):
        for j in range(len(SCENARIOS)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Avg FCSD (higher = safer)")
    ax.set_title("Fuzzy Constraint Satisfaction Degree (mean, n=3)")
    fig.tight_layout()
    fig.savefig(FIG / "figure-04-fcsd-heatmap.pdf")
    fig.savefig(FIG / "figure-04-fcsd-heatmap.png")
    plt.close(fig)


def fig_pareto(data: dict) -> None:
    """Fig 5: Pareto scatter — cost vs violation rate per algorithm-scenario."""
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"S1_grid_normal": "o", "S2_grid_extreme": "s",
                "S3_island_normal": "^", "S4_island_extreme": "D"}
    for algo in ALGOS:
        for sc in SCENARIOS:
            runs = data[algo][sc]
            if not runs:
                continue
            cost_m = np.mean([r["cost_per_day"] for r in runs])
            viol_m = np.mean([r["violation_rate"] for r in runs])
            ax.scatter(cost_m, viol_m, s=80, color=ALGO_COLORS[algo],
                       marker=markers[sc], edgecolor="black", linewidth=0.6,
                       label=ALGO_LABELS[algo] if sc == SCENARIOS[0] else None,
                       zorder=3)
    ax.axhline(5.0, color="red", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Cost per day (¥)")
    ax.set_ylabel("Violation rate (%)")
    ax.set_title("Cost-safety Pareto frontier (mean of 3 seeds)")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=False, loc="upper right")
    # annotate HFG-SAC points
    for sc in SCENARIOS:
        runs = data["hfg_sac"][sc]
        cost_m = np.mean([r["cost_per_day"] for r in runs])
        viol_m = np.mean([r["violation_rate"] * 100 for r in runs])
        ax.annotate(sc.split("_")[0], (cost_m, viol_m),
                    textcoords="offset points", xytext=(6, 6), fontsize=8,
                    color="#d32f2f")
    fig.tight_layout()
    fig.savefig(FIG / "figure-05-pareto.pdf")
    fig.savefig(FIG / "figure-05-pareto.png")
    plt.close(fig)


# ---------- main ----------
def main() -> None:
    data = load_all()
    print(f"Loaded {sum(len(data[a][s]) for a in ALGOS for s in SCENARIOS)} runs")

    run_statistics(data)
    fig_main_comparison(data)
    fig_training_curves(data)
    fig_violation_trajectory(data)
    fig_fcsd_heatmap(data)
    fig_pareto(data)
    print(f"Figures saved to {FIG}")


if __name__ == "__main__":
    main()
