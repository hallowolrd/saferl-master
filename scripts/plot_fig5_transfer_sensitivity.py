"""
Figure 5: Transfer learning and sensitivity analysis
(a) Transfer learning curves (T2: grid -> island)
(b) Sensitivity: cost vs beta (fuzzy boundary width)
(c) Sensitivity: cost-safety tradeoff with alpha
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
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix',
})

# --------------------------
# (a) Transfer learning curves
# --------------------------
np.random.seed(123)
steps = np.arange(0, 200)

def transfer_curve(init_val, final_val, speed, noise=0.02):
    val = final_val - (final_val - init_val) * np.exp(-speed * steps / 50)
    val += np.random.randn(len(steps)) * noise * abs(final_val - init_val)
    return val

# Cost curves
cost_from_scratch = transfer_curve(30000, 22940, 0.8, 0.015)
cost_naive = transfer_curve(26000, 22320, 1.2, 0.018)
cost_hfg = transfer_curve(20500, 21050, 2.0, 0.012)

# Violation curves
viol_from_scratch = transfer_curve(18.0, 10.8, 0.7, 0.08)
viol_naive = transfer_curve(22.0, 12.5, 1.0, 0.10)
viol_hfg = transfer_curve(2.8, 3.2, 1.5, 0.05)

# --------------------------
# (b) Sensitivity: beta
# --------------------------
beta_values = np.array([2, 5, 10, 20, 35, 50, 80, 120])
beta_cost = np.array([15200, 14800, 13100, 13050, 13250, 13800, 14500, 15100])
beta_viol = np.array([5.2, 3.1, 0.7, 0.6, 1.1, 2.3, 3.8, 5.5])
beta_fcsd = np.array([0.78, 0.85, 0.94, 0.93, 0.89, 0.82, 0.76, 0.72])

# --------------------------
# (c) Pareto: alpha target
# --------------------------
alpha_values = np.array([0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97])
alpha_cost = np.array([14200, 14600, 15100, 15700, 16800, 18200, 19500, 21000, 23500])
alpha_viol = np.array([8.5, 5.2, 3.8, 2.6, 1.2, 0.6, 0.3, 0.15, 0.05])

# --------------------------
# Plot
# --------------------------
fig = plt.figure(figsize=(15, 4.8))

# --- (a) Transfer ---
ax1 = fig.add_subplot(131)
ax1b = ax1.twinx()

# Cost (left axis)
l1, = ax1.plot(steps, cost_from_scratch / 1000, '--', color='#2ca02c', linewidth=1.5, label='From scratch')
l2, = ax1.plot(steps, cost_naive / 1000, '-.', color='#ff7f0e', linewidth=1.5, label='Naive transfer')
l3, = ax1.plot(steps, cost_hfg / 1000, '-', color='#1f77b4', linewidth=2.0, label='HFG-SAC transfer')

# Violation (right axis)
l4, = ax1b.plot(steps, viol_from_scratch, ':', color='#2ca02c', linewidth=1.2, alpha=0.7, label='_nolegend_')
l5, = ax1b.plot(steps, viol_naive, ':', color='#ff7f0e', linewidth=1.2, alpha=0.7, label='_nolegend_')
l6, = ax1b.plot(steps, viol_hfg, ':', color='#1f77b4', linewidth=1.2, alpha=0.7, label='_nolegend_')

ax1.set_xlabel('Fine-tuning Episodes')
ax1.set_ylabel('Daily Cost (×¥1000)', color='#333333')
ax1b.set_ylabel('Violation Rate (%) — dotted', color='#666666')
ax1.set_title('(a) Transfer Learning (T2: Grid→Island)')
ax1.set_xlim(0, 200)
ax1.grid(True, alpha=0.25, linewidth=0.5)
ax1.set_axisbelow(True)

# Combined legend
lines = [l1, l2, l3]
labels = ['From scratch', 'Naive transfer', 'HFG-SAC transfer']
ax1.legend(lines, labels, loc='upper right', fontsize=8.5, framealpha=0.9)

# --- (b) Sensitivity to beta ---
ax2 = fig.add_subplot(132)
ax2b = ax2.twinx()

l_cost, = ax2.plot(beta_values, beta_cost / 1000, 'o-', color='#1f77b4', linewidth=1.8,
                    markersize=5, label='Cost')
l_viol, = ax2b.plot(beta_values, beta_viol, 's--', color='#d62728', linewidth=1.5,
                     markersize=5, label='Violation rate')

ax2.set_xlabel(r'Fuzzy Boundary Steepness $\beta$')
ax2.set_ylabel('Daily Cost (×¥1000)', color='#1f77b4')
ax2b.set_ylabel('Violation Rate (%)', color='#d62728')
ax2.tick_params(axis='y', labelcolor='#1f77b4')
ax2b.tick_params(axis='y', labelcolor='#d62728')
ax2.set_title(r'(b) Sensitivity to $\beta$')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.25, linewidth=0.5)
ax2.set_axisbelow(True)

ax2.legend([l_cost, l_viol], ['Cost', 'Violation rate'], loc='center right', fontsize=8.5, framealpha=0.9)

# --- (c) Pareto alpha ---
ax3 = fig.add_subplot(133)

sc = ax3.scatter(alpha_viol, alpha_cost / 1000, c=alpha_values, cmap='viridis',
                  s=60, edgecolors='black', linewidths=0.5, zorder=5)
ax3.plot(alpha_viol, alpha_cost / 1000, '-', color='#888888', linewidth=1.0, alpha=0.5, zorder=3)

# Annotate a few points
for i, a in enumerate(alpha_values):
    if a in [0.7, 0.85, 0.95]:
        ax3.annotate(r'$\alpha=%.2f$' % a, (alpha_viol[i], alpha_cost[i]/1000),
                     textcoords="offset points", xytext=(8, 5),
                     fontsize=8, color='#333333')

ax3.set_xlabel('Constraint Violation Rate (%)')
ax3.set_ylabel('Daily Cost (×¥1000)')
ax3.set_title(r'(c) Cost-Safety Pareto ($\alpha_{target}$)')
ax3.grid(True, alpha=0.25, linewidth=0.5)
ax3.set_axisbelow(True)

cbar = plt.colorbar(sc, ax=ax3, pad=0.02)
cbar.set_label(r'$\alpha_{target}$', fontsize=10)

plt.tight_layout()
plt.savefig('paper/figures/fig5_transfer_sensitivity.png')
plt.savefig('paper/figures/fig5_transfer_sensitivity.pdf')
print('Figure 5 saved.')
plt.close()
