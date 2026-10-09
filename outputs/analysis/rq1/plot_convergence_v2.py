"""Regenerate 3 convergence figures with clear visual distinction.

Improvements:
- Unique line style per algorithm (solid, dashed, dash-dot, dotted)
- Wider line width (2.0)
- Colorblind-friendly palette
- External legend (below plot)
- Larger figure size
"""
import json, glob, re, os
from collections import defaultdict
from scipy.ndimage import uniform_filter1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAT = re.compile(
    r"^(sac|ppo|ppo_lagrangian|cpo|safety_layer|fuzzy_sac|hfg_sac)"
    r"_(S[1-4]_(?:grid|island)_(?:normal|extreme))_s(\d+)\.json$"
)

# 7 algorithms with distinct visual signatures:
# - color (colorblind-friendly palette)
# - line style (solid, dashed, dash-dot, dotted)
# - line width (HFG-SAC thickest to highlight)
ALGO_STYLE = {
    "sac":            {"color": "#7f7f7f", "ls": ":",  "lw": 1.5, "label": "SAC (unsafe)"},
    "ppo":            {"color": "#1f77b4", "ls": ":",  "lw": 1.5, "label": "PPO (unsafe)"},
    "ppo_lagrangian": {"color": "#ff7f0e", "ls": "--", "lw": 1.5, "label": "PPO-Lagrangian"},
    "cpo":            {"color": "#2ca02c", "ls": "--", "lw": 1.5, "label": "CPO"},
    "safety_layer":   {"color": "#9467bd", "ls": "-.", "lw": 1.5, "label": "Safety-Layer"},
    "fuzzy_sac":      {"color": "#8c564b", "ls": "-.", "lw": 1.5, "label": "Fuzzy-SAC (w/o upper layer)"},
    "hfg_sac":        {"color": "#d62728", "ls": "-",  "lw": 2.5, "label": "HFG-SAC (ours)"},
}
ALGOS = list(ALGO_STYLE.keys())

SCENS = ["S1_grid_normal", "S2_grid_extreme", "S3_island_normal", "S4_island_extreme"]
SCEN_LAB = {"S1_grid_normal": "S1: Grid normal", "S2_grid_extreme": "S2: Grid extreme",
            "S3_island_normal": "S3: Island normal", "S4_island_extreme": "S4: Island extreme"}

# Load all data
costs = defaultdict(list)
viol = defaultdict(list)
fcsd = defaultdict(list)
for f in glob.glob("outputs/experiments/results/*.json"):
    m = PAT.match(os.path.basename(f))
    if not m:
        continue
    algo, scen = m.group(1), m.group(2)
    d = json.load(open(f, encoding="utf-8"))
    costs[(algo,scen)].append(np.array(d["episode_costs"]))
    viol[(algo,scen)].append(np.array(d["episode_violation_rates"]))
    fcsd[(algo,scen)].append(np.array(d["episode_fcsds"]))


def plot_metric(data, ylabel, fname, ylim, target_line=None, legend_loc="outside"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, scen in zip(axes.flat, SCENS):
        for algo in ALGOS:
            arrs = data.get((algo, scen), [])
            if not arrs:
                continue
            A = np.array(arrs)
            st = ALGO_STYLE[algo]
            mean = uniform_filter1d(A.mean(0), size=10)
            std = uniform_filter1d(A.std(0), size=10)
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=st["color"], ls=st["ls"], lw=st["lw"],
                    label=st["label"], alpha=0.9)
            # Light fill only for HFG-SAC to reduce visual clutter
            if algo == "hfg_sac":
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=st["color"])
        ax.set_title(SCEN_LAB[scen], fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if target_line is not None:
            ax.axhline(target_line, color="red", ls="--", lw=1.0, alpha=0.7)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=9)

    # External legend
    handles = [Line2D([0], [0], color=ALGO_STYLE[a]["color"],
                      ls=ALGO_STYLE[a]["ls"], lw=ALGO_STYLE[a]["lw"],
                      label=ALGO_STYLE[a]["label"]) for a in ALGOS]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Training dynamics: {ylabel}\n(10-episode moving average, mean over 5 seeds; shaded band = HFG-SAC ± std)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f"outputs/analysis/rq1/figures/{fname}", bbox_inches="tight", dpi=200)
    plt.close(fig)


# 3 convergence figures
plot_metric(costs, "Episode cost (¥)", "figure-04-training-curves.png",
            ylim=(0, None))
plot_metric(viol, "Episode violation rate (%)", "figure-06-violation-convergence.png",
            ylim=(-5, 105), target_line=5)
plot_metric(fcsd, "Episode avg FCSD", "figure-07-fcsd-convergence.png",
            ylim=(0.4, 1.05))

print("Done — 3 figures regenerated with distinct line styles.")
