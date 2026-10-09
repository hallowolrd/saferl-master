"""Generate violation-rate and FCSD training-convergence figures."""
import json, glob, re, os
from collections import defaultdict
from scipy.ndimage import uniform_filter1d
import numpy as np
import matplotlib.pyplot as plt

PAT = re.compile(
    r"^(sac|ppo|ppo_lagrangian|cpo|safety_layer|fuzzy_sac|hfg_sac)"
    r"_(S[1-4]_(?:grid|island)_(?:normal|extreme))_s(\d+)\.json$"
)
ALGOS = ["sac","ppo","ppo_lagrangian","cpo","safety_layer","fuzzy_sac","hfg_sac"]
LABEL = {"sac":"SAC","ppo":"PPO","ppo_lagrangian":"PPO-Lag","cpo":"CPO",
         "safety_layer":"Safe-Layer","fuzzy_sac":"Fuzzy-SAC","hfg_sac":"HFG-SAC (ours)"}
COLORS = {"sac":"#888888","ppo":"#9ecae1","ppo_lagrangian":"#4292c6","cpo":"#08519c",
          "safety_layer":"#a1d99b","fuzzy_sac":"#fdae6b","hfg_sac":"#d7301f"}
SCENS = ["S1_grid_normal","S2_grid_extreme","S3_island_normal","S4_island_extreme"]
SCEN_LAB = {"S1_grid_normal":"S1: Grid normal","S2_grid_extreme":"S2: Grid extreme",
            "S3_island_normal":"S3: Island normal","S4_island_extreme":"S4: Island extreme"}

viol = defaultdict(list)
fcsd = defaultdict(list)
for f in glob.glob("outputs/experiments/results/*.json"):
    m = PAT.match(os.path.basename(f))
    if not m:
        continue
    algo, scen = m.group(1), m.group(2)
    d = json.load(open(f, encoding="utf-8"))
    viol[(algo,scen)].append(np.array(d["episode_violation_rates"]))
    fcsd[(algo,scen)].append(np.array(d["episode_fcsds"]))

def plot_metric(data, ylabel, fname, ylim, target_line=None, legend_loc="best"):
    fig, axes = plt.subplots(2, 2, figsize=(11,7), sharex=True)
    for ax, scen in zip(axes.flat, SCENS):
        for algo in ALGOS:
            arrs = data.get((algo,scen), [])
            if not arrs:
                continue
            A = np.array(arrs)
            mean = uniform_filter1d(A.mean(0), size=10)
            std = uniform_filter1d(A.std(0), size=10)
            x = np.arange(1, len(mean)+1)
            ax.plot(x, mean, label=LABEL[algo], color=COLORS[algo], lw=1.4)
            ax.fill_between(x, mean-std, mean+std, alpha=0.10, color=COLORS[algo])
        ax.set_title(SCEN_LAB[scen])
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        if target_line is not None:
            ax.axhline(target_line, color="red", ls="--", lw=0.8, alpha=0.6)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
    axes[0,0].legend(ncol=2, fontsize=7, loc=legend_loc)
    fig.suptitle(f"Training dynamics: {ylabel} (10-ep moving avg, mean ± std over 5 seeds)")
    fig.tight_layout()
    fig.savefig(f"outputs/analysis/rq1/figures/{fname}", bbox_inches="tight", dpi=200)
    plt.close(fig)

plot_metric(viol, "Episode violation rate (%)", "figure-06-violation-convergence.png",
            ylim=(-5, 105), target_line=5, legend_loc="upper right")
plot_metric(fcsd, "Episode avg FCSD", "figure-07-fcsd-convergence.png",
            ylim=(0.4, 1.05), legend_loc="lower right")
print("done")
