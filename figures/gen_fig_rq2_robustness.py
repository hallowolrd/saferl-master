"""Generate RQ2 robustness curve: violation rate vs extremity level.

Reads from outputs/experiments/results/*extremity_*.json and produces
a two-panel figure:
  (a) Violation rate (%) vs extremity multiplier (mean +/- SEM, 5 seeds)
  (b) Cost per day (CNY) vs extremity multiplier

Saves to outputs/figures/fig_rq2_robustness.{pdf,png}.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

RESULT_GLOB = "outputs/experiments/results/*extremity_*.json"

ALGO_ORDER = [
    "sac",
    "ppo_lagrangian",
    "cpo",
    "safety_layer",
    "fuzzy_sac",
    "hfg_sac",
]

ALGO_LABELS = {
    "sac": "SAC (unconstrained)",
    "ppo_lagrangian": "PPO-Lagrangian",
    "cpo": "CPO",
    "safety_layer": "Safety Layer (CBF)",
    "fuzzy_sac": "Fuzzy-SAC (lower layer)",
    "hfg_sac": "HFG-SAC (ours)",
}

# Okabe-Ito colorblind-safe palette; HFG-SAC highlighted in coral.
ALGO_COLORS = {
    "sac": "#8C8C8C",            # cool gray
    "ppo_lagrangian": "#D55E00", # vermillion (crisp on-policy, crashes)
    "cpo": "#E69F00",            # orange (over-conservative)
    "safety_layer": "#0072B2",   # blue (hard guardrail)
    "fuzzy_sac": "#56B4E9",      # sky blue (ablation)
    "hfg_sac": "#E76F51",        # coral (ours)
}

ALGO_STYLES = {
    "sac": {"linestyle": ":", "marker": "x"},
    "ppo_lagrangian": {"linestyle": "--", "marker": "v"},
    "cpo": {"linestyle": "--", "marker": "s"},
    "safety_layer": {"linestyle": "-.", "marker": "^"},
    "fuzzy_sac": {"linestyle": "-", "marker": "o"},
    "hfg_sac": {"linestyle": "-", "marker": "D"},
}


def load_results() -> Dict[str, Dict[float, List[dict]]]:
    """Return data[algo][level] = list of per-seed metrics."""
    data: Dict[str, Dict[float, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for fp in sorted(glob.glob(RESULT_GLOB)):
        m = re.search(r"([a-z_]+)_extremity_([0-9.]+)_s(\d+)\.json", os.path.basename(fp))
        if not m:
            continue
        algo, level, seed = m.group(1), float(m.group(2)), int(m.group(3))
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        # eval_violations already in percent (0-100); use the final eval point.
        final_viol_pct = d["eval_violations"][-1]
        data[algo][level].append({
            "viol_pct": final_viol_pct,
            "cost": d["cost_per_day"],
            "min_fcsd": d["min_fcsd"],
            "seed": seed,
        })
    return data


def summarize(
    runs: List[dict], key: str
) -> Tuple[float, float]:
    vals = [r[key] for r in runs]
    mean = st.mean(vals)
    sem = st.stdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
    return mean, sem


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
})


def main() -> None:
    data = load_results()
    levels = sorted({lv for a in data.values() for lv in a.keys()})
    x = np.array(levels)

    fig, (ax_viol, ax_cost) = plt.subplots(
        1, 2, figsize=(6.75, 2.8), sharex=True
    )

    for algo in ALGO_ORDER:
        if algo not in data:
            continue
        color = ALGO_COLORS[algo]
        style = ALGO_STYLES[algo]
        is_ours = algo == "hfg_sac"
        lw = 2.4 if is_ours else 1.6
        zorder = 5 if is_ours else 3

        viol_means, viol_sems = [], []
        cost_means, cost_sems = [], []
        for lv in levels:
            runs = data[algo].get(lv, [])
            if not runs:
                viol_means.append(np.nan)
                viol_sems.append(0)
                cost_means.append(np.nan)
                cost_sems.append(0)
                continue
            vm, vse = summarize(runs, "viol_pct")
            cm, cse = summarize(runs, "cost")
            viol_means.append(vm)
            viol_sems.append(vse)
            cost_means.append(cm)
            cost_sems.append(cse)

        viol_means = np.array(viol_means)
        viol_sems = np.array(viol_sems)
        cost_means = np.array(cost_means)
        cost_sems = np.array(cost_sems)

        label = ALGO_LABELS[algo]
        ax_viol.plot(
            x, viol_means,
            color=color, label=label, linewidth=lw, zorder=zorder,
            marker=style["marker"], linestyle=style["linestyle"],
            markeredgewidth=1.2,
        )
        ax_viol.fill_between(
            x, viol_means - viol_sems, viol_means + viol_sems,
            color=color, alpha=0.15, zorder=zorder - 1,
        )

        ax_cost.plot(
            x, cost_means,
            color=color, linewidth=lw, zorder=zorder,
            marker=style["marker"], linestyle=style["linestyle"],
            markeredgewidth=1.2,
        )
        ax_cost.fill_between(
            x, cost_means - cost_sems, cost_means + cost_sems,
            color=color, alpha=0.12, zorder=zorder - 1,
        )

    # Panel (a): violation rate
    ax_viol.set_ylabel("Constraint violation rate (%)")
    ax_viol.set_xlabel("Extremity multiplier $m$")
    ax_viol.set_ylim(-3, 105)
    ax_viol.set_xticks(x)
    ax_viol.set_xticklabels([f"{lv:.1f}×" for lv in levels])
    ax_viol.axhline(5, color="#888", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_viol.text(0.82, 6.5, "5% safety target", fontsize=7, color="#666")
    ax_viol.set_title("(a) Robustness under increasing extremity")

    # Panel (b): cost per day
    ax_cost.set_ylabel("Operating cost (CNY / day)")
    ax_cost.set_xlabel("Extremity multiplier $m$")
    ax_cost.set_title("(b) Operating cost vs extremity")
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels([f"{lv:.1f}×" for lv in levels])

    # Single legend spanning both panels, anchored to the right of (a).
    handles, labels = ax_viol.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    out_dir = "outputs/figures"
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "fig_rq2_robustness.pdf")
    png_path = os.path.join(out_dir, "fig_rq2_robustness.png")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
