"""Strict RQ1 analysis: descriptive stats, inferential tests, figures.

Reads outputs/experiments/results/*.json (140 files) and produces:
  - stats_summary.csv
  - stats-appendix.md
  - figures/figure-01-{cost,violation,fcsd}.png
  - figures/figure-04-training-curves.png
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path("E:/研究/saferl")
RESULTS = ROOT / "outputs" / "experiments" / "results"
OUT = ROOT / "outputs" / "analysis" / "rq1"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ALGOS = ["sac", "ppo", "ppo_lagrangian", "cpo", "safety_layer", "fuzzy_sac", "hfg_sac"]
ALGO_LABEL = {
    "sac": "SAC",
    "ppo": "PPO",
    "ppo_lagrangian": "PPO-Lag",
    "cpo": "CPO",
    "safety_layer": "Safe-Layer",
    "fuzzy_sac": "Fuzzy-SAC",
    "hfg_sac": "HFG-SAC (ours)",
}
SCENARIOS = ["S1_grid_normal", "S2_grid_extreme", "S3_island_normal", "S4_island_extreme"]
SCEN_LABEL = {
    "S1_grid_normal": "S1: Grid normal",
    "S2_grid_extreme": "S2: Grid extreme",
    "S3_island_normal": "S3: Island normal",
    "S4_island_extreme": "S4: Island extreme",
}
PAT = re.compile(
    r"^(sac|ppo|ppo_lagrangian|cpo|safety_layer|fuzzy_sac|hfg_sac)"
    r"_(S[1-4]_(?:grid|island)_(?:normal|extreme))_s(\d+)\.json$"
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
rows = []
ep_curves = defaultdict(list)  # (algo, scen) -> list of 5 arrays (200 episodes)
for f in sorted(RESULTS.glob("*.json")):
    m = PAT.match(f.name)
    if not m:
        continue
    algo, scen, seed = m.group(1), m.group(2), int(m.group(3))
    d = json.loads(f.read_text(encoding="utf-8"))
    rows.append({
        "algo": algo, "scenario": scen, "seed": seed,
        "cost_per_day": d["cost_per_day"],
        "violation_rate": d["violation_rate"],
        "avg_fcsd": d["avg_fcsd"],
        "min_fcsd": d["min_fcsd"],
        "training_time_s": d["training_time_s"],
    })
    ep_curves[(algo, scen)].append(np.array(d["episode_costs"]))

df = pd.DataFrame(rows)
assert len(df) == 140, f"Expected 140, got {len(df)}"

# ---------------------------------------------------------------------------
# 2. Descriptive stats: mean +/- std, 95% CI over 5 seeds
# ---------------------------------------------------------------------------
def ci95(x: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    se = x.std(ddof=1) / np.sqrt(n)
    return 1.96 * se  # large-sample approx; with n=5 t(0.975,4)=2.776 but we use normal for readability

desc = (
    df.groupby(["scenario", "algo"])
      .agg(
          cost_mean=("cost_per_day", "mean"),
          cost_std=("cost_per_day", "std"),
          cost_ci=("cost_per_day", lambda x: ci95(x.values)),
          viol_mean=("violation_rate", "mean"),
          viol_std=("violation_rate", "std"),
          fcsd_mean=("avg_fcsd", "mean"),
          fcsd_std=("avg_fcsd", "std"),
          time_mean=("training_time_s", "mean"),
          n=("seed", "nunique"),
      )
      .reset_index()
)
desc.to_csv(OUT / "stats_summary.csv", index=False, float_format="%.3f")
print("=== Descriptive stats (cost/day mean ± CI) ===")
piv = desc.pivot(index="algo", columns="scenario", values="cost_mean").reindex(ALGOS)
print(piv.round(0))
print("\n=== Violation rate % ===")
piv_v = desc.pivot(index="algo", columns="scenario", values="viol_mean").reindex(ALGOS)
print(piv_v.round(2))
print("\n=== FCSD ===")
piv_f = desc.pivot(index="algo", columns="scenario", values="fcsd_mean").reindex(ALGOS)
print(piv_f.round(4))

# ---------------------------------------------------------------------------
# 3. Inferential tests: HFG-SAC vs each baseline, per scenario
#    Mann-Whitney U (non-parametric, n=5 too small for normality checks)
#    Holm correction across the 6 baselines within each scenario/metric
# ---------------------------------------------------------------------------
baselines = [a for a in ALGOS if a != "hfg_sac"]
test_rows = []
for scen in SCENARIOS:
    for metric in ["cost_per_day", "violation_rate", "avg_fcsd"]:
        hfg = df[(df.scenario == scen) & (df.algo == "hfg_sac")][metric].values
        raw = []
        for base in baselines:
            b = df[(df.scenario == scen) & (df.algo == base)][metric].values
            u, p = stats.mannwhitneyu(hfg, b, alternative="two-sided")
            # Effect size: rank-biserial r = 1 - 2U/(n1*n2)
            r_rb = 1 - (2 * u) / (len(hfg) * len(b))
            raw.append((base, p, r_rb, hfg.mean(), b.mean()))
        # Holm correction across 6 baselines
        order = sorted(range(len(raw)), key=lambda i: raw[i][1])
        adj_p = [0.0] * len(raw)
        for rank, idx in enumerate(order):
            adj_p[idx] = min(1.0, raw[idx][1] * (len(raw) - rank))
        for i, (base, p, r_rb, hm, bm) in enumerate(raw):
            test_rows.append({
                "scenario": scen, "metric": metric, "baseline": base,
                "hfg_mean": hm, "baseline_mean": bm,
                "p_raw": p, "p_holm": adj_p[i], "effect_r": r_rb,
            })
tests_df = pd.DataFrame(test_rows)
tests_df.to_csv(OUT / "inferential_tests.csv", index=False, float_format="%.4f")

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "figure.dpi": 130, "savefig.dpi": 200,
})
COLORS = {
    "sac": "#888888", "ppo": "#9ecae1", "ppo_lagrangian": "#4292c6",
    "cpo": "#08519c", "safety_layer": "#a1d99b", "fuzzy_sac": "#fdae6b",
    "hfg_sac": "#d7301f",
}

def grouped_bar(metric_key: str, ylabel: str, title: str, fname: str,
                higher_better: bool = False):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n_algo = len(ALGOS)
    n_scen = len(SCENARIOS)
    width = 0.11
    x = np.arange(n_scen)
    for i, algo in enumerate(ALGOS):
        means, cis = [], []
        for scen in SCENARIOS:
            sub = df[(df.algo == algo) & (df.scenario == scen)][metric_key].values
            means.append(sub.mean())
            cis.append(ci95(sub))
        offset = (i - n_algo / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=ALGO_LABEL[algo],
               color=COLORS[algo], edgecolor="black", linewidth=0.4,
               yerr=cis, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCENARIOS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)

grouped_bar("cost_per_day", "Operating cost (¥/day, lower better)",
            "RQ1: Operating cost across algorithms and scenarios",
            "figure-01-cost.png")
grouped_bar("violation_rate", "Constraint violation rate (%)",
            "RQ1: Safety violation rate (lower better)",
            "figure-02-violation.png")
grouped_bar("avg_fcsd", "Average FCSD (higher = safer)",
            "RQ1: Fuzzy Constraint Satisfaction Degree (higher better)",
            "figure-03-fcsd.png", higher_better=True)

# Training curves: per scenario, mean ± std band over 5 seeds
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, scen in zip(axes.flat, SCENARIOS):
    for algo in ALGOS:
        arrs = ep_curves.get((algo, scen), [])
        if not arrs:
            continue
        A = np.array(arrs)  # (5, 200)
        mean = A.mean(0)
        std = A.std(0)
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=ALGO_LABEL[algo], color=COLORS[algo], lw=1.2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=COLORS[algo])
    ax.set_title(SCEN_LABEL[scen])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode cost (¥)")
    ax.grid(alpha=0.3)
axes[0, 0].legend(ncol=2, fontsize=7)
fig.suptitle("Training dynamics: episode cost (mean ± std over 5 seeds)")
fig.tight_layout()
fig.savefig(FIG / "figure-04-training-curves.png", bbox_inches="tight")
plt.close(fig)

# Cost vs violation scatter (safety-efficiency frontier)
fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
for ax, scen in zip(axes, SCENARIOS):
    for algo in ALGOS:
        sub = df[(df.algo == algo) & (df.scenario == scen)]
        ax.scatter(sub.cost_per_day, sub.violation_rate,
                   s=40, color=COLORS[algo], edgecolor="black",
                   linewidth=0.5, label=ALGO_LABEL[algo])
    ax.set_title(SCEN_LABEL[scen])
    ax.set_xlabel("Cost (¥/day)")
    ax.axhline(5, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Violation rate (%)")
axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Safety–efficiency frontier (dashed line = 5% safety target)")
fig.tight_layout()
fig.savefig(FIG / "figure-05-frontier.png", bbox_inches="tight")
plt.close(fig)

print(f"\nFigures saved to {FIG}")
print("Done.")
