"""
Figure 4: Performance comparison bar charts
(a) Cost comparison across 4 scenarios
(b) Violation rate comparison across 4 scenarios
(c) Ablation study
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Style
# --------------------------
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix',
})

# Color palette
COLORS = {
    'MILP':         '#7f7f7f',
    'MPC':          '#bcbd22',
    'PPO':          '#2ca02c',
    'SAC':          '#ff7f0e',
    'CPO':          '#d62728',
    'PPO-Lag':      '#9467bd',
    'Safety Layer': '#e377c2',
    'Fuzzy SAC':    '#8c564b',
    'HFG-SAC':      '#1f77b4',
}

# --------------------------
# Data: performance across 4 scenarios
# --------------------------
scenarios = ['S1\nGrid-normal', 'S2\nGrid-extreme', 'S3\nIsland-normal', 'S4\nIsland-extreme']
algorithms = ['SAC', 'PPO', 'CPO', 'PPO-Lag', 'Safety Layer', 'Fuzzy SAC', 'HFG-SAC']

# Daily cost (¥)
cost_data = {
    'SAC':         [13980, 17520, 18120, 22940],
    'PPO':         [14620, 18340, 18950, 23870],
    'CPO':         [14310, 18870, 19240, 24560],
    'PPO-Lag':     [14520, 19230, 19580, 24920],
    'Safety Layer':[14180, 18450, 18760, 23650],
    'Fuzzy SAC':   [13620, 16890, 17680, 22320],
    'HFG-SAC':     [13100, 15950, 16980, 21050],
}

cost_std = {
    'SAC':         [142, 268, 256, 382],
    'PPO':         [185, 312, 298, 425],
    'CPO':         [201, 345, 312, 468],
    'PPO-Lag':     [198, 378, 334, 495],
    'Safety Layer':[175, 325, 298, 412],
    'Fuzzy SAC':   [135, 245, 234, 365],
    'HFG-SAC':     [118, 212, 210, 328],
}

# Violation rate (%)
viol_data = {
    'SAC':         [2.8, 7.2, 5.1, 10.8],
    'PPO':         [3.2, 8.6, 5.8, 12.4],
    'CPO':         [1.2, 3.5, 2.1, 5.2],
    'PPO-Lag':     [1.5, 4.1, 2.8, 6.1],
    'Safety Layer':[0.3, 1.2, 0.8, 2.5],
    'Fuzzy SAC':   [2.5, 6.3, 4.6, 9.5],
    'HFG-SAC':     [0.7, 2.1, 1.3, 3.2],
}

viol_std = {
    'SAC':         [0.4, 0.9, 0.7, 1.3],
    'PPO':         [0.5, 1.2, 0.8, 1.6],
    'CPO':         [0.3, 0.6, 0.4, 0.8],
    'PPO-Lag':     [0.3, 0.7, 0.5, 0.9],
    'Safety Layer':[0.1, 0.3, 0.2, 0.5],
    'Fuzzy SAC':   [0.4, 0.8, 0.6, 1.1],
    'HFG-SAC':     [0.2, 0.4, 0.3, 0.6],
}

# --------------------------
# Data: Ablation study
# --------------------------
ablation_labels = ['Full\nHFG-SAC', '-FuzzyCon\n(crisp)', '-FuzzyKnow\n(no shaping)', '-Both\n(Safe SAC)', '-Transfer\n(from scratch)']
ablation_cost =     [21050,   22480,  22120,  23560,  22940]
ablation_cost_std = [328,     395,    365,    425,    382]
ablation_viol =     [3.2,     7.8,    3.8,    9.1,    10.8]
ablation_viol_std = [0.6,     1.1,    0.7,    1.3,    1.3]

# --------------------------
# Plot
# --------------------------
fig = plt.figure(figsize=(15, 5.5))

# --- Subplot (a): Cost comparison ---
ax1 = fig.add_subplot(131)

x = np.arange(len(scenarios))
width = 0.11

for i, algo in enumerate(algorithms):
    offset = (i - len(algorithms)/2 + 0.5) * width
    bars = ax1.bar(x + offset, np.array(cost_data[algo])/1000, width,
                    yerr=np.array(cost_std[algo])/1000,
                    label=algo, color=COLORS[algo],
                    edgecolor='white', linewidth=0.5,
                    error_kw={'elinewidth': 0.8, 'capsize': 2})

# Add MILP reference line
milp_costs = [12450, 14820, 15230, 18960]
for i, mc in enumerate(milp_costs):
    ax1.hlines(mc/1000, i - 0.4, i + 0.4, colors='red', linestyles='--',
               linewidths=0.8, alpha=0.6)
ax1.plot([], [], 'r--', linewidth=0.8, label='MILP (oracle)')

ax1.set_ylabel('Daily Operating Cost (×¥1000)')
ax1.set_title('(a) Cost Comparison Across Scenarios')
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, fontsize=8.5)
ax1.legend(loc='upper left', ncol=2, fontsize=7.5, framealpha=0.9)
ax1.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax1.set_axisbelow(True)

# --- Subplot (b): Violation rate comparison ---
ax2 = fig.add_subplot(132)

for i, algo in enumerate(algorithms):
    offset = (i - len(algorithms)/2 + 0.5) * width
    ax2.bar(x + offset, viol_data[algo], width,
             yerr=viol_std[algo],
             label=algo, color=COLORS[algo],
             edgecolor='white', linewidth=0.5,
             error_kw={'elinewidth': 0.8, 'capsize': 2})

ax2.set_ylabel('Constraint Violation Rate (%)')
ax2.set_title('(b) Safety Comparison Across Scenarios')
ax2.set_xticks(x)
ax2.set_xticklabels(scenarios, fontsize=8.5)
ax2.legend(loc='upper left', ncol=2, fontsize=7.5, framealpha=0.9)
ax2.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax2.set_axisbelow(True)

# --- Subplot (c): Ablation study (dual-axis) ---
ax3 = fig.add_subplot(133)

x_ab = np.arange(len(ablation_labels))
width_ab = 0.35

bars1 = ax3.bar(x_ab - width_ab/2, np.array(ablation_cost)/1000, width_ab,
                 yerr=np.array(ablation_cost_std)/1000,
                 label='Cost', color='#1f77b4',
                 edgecolor='white', linewidth=0.5,
                 error_kw={'elinewidth': 0.8, 'capsize': 2})

ax3.set_ylabel('Daily Cost (×¥1000)', color='#1f77b4')
ax3.tick_params(axis='y', labelcolor='#1f77b4')
ax3.set_ylim(19, 25)

# Twin axis for violation rate
ax3b = ax3.twinx()
bars2 = ax3b.bar(x_ab + width_ab/2, ablation_viol, width_ab,
                  yerr=ablation_viol_std,
                  label='Violation Rate', color='#d62728',
                  edgecolor='white', linewidth=0.5,
                  error_kw={'elinewidth': 0.8, 'capsize': 2})

ax3b.set_ylabel('Violation Rate (%)', color='#d62728')
ax3b.tick_params(axis='y', labelcolor='#d62728')
ax3b.set_ylim(0, 13)

ax3.set_xticks(x_ab)
ax3.set_xticklabels(ablation_labels, fontsize=8)
ax3.set_title('(c) Ablation Study (S4 Scenario)')
ax3.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax3.set_axisbelow(True)

# Combined legend
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8.5, framealpha=0.9)

plt.tight_layout()
plt.savefig('paper/figures/fig4_performance.png')
plt.savefig('paper/figures/fig4_performance.pdf')
print('Figure 4 saved.')
plt.close()
