"""
Figure 1: Fuzzy Constraint Satisfaction Degree (FCSD) Illustration
Shows sigmoidal membership functions for SOC upper bound,
comparing crisp (hard) constraint vs fuzzy (soft) constraint.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# --------------------------
# Style Configuration (IEEE-style)
# --------------------------
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.linewidth': 1.0,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

# Color palette (colorblind-friendly)
C_CRISP = '#d62728'     # red
C_FUZZY = '#1f77b4'     # blue
C_FUZZY_2 = '#2ca02c'   # green
C_SHADE = '#aec7e8'     # light blue
C_TEXT = '#333333'

# --------------------------
# Data: Sigmoidal membership for SOC upper bound
# --------------------------
soc = np.linspace(0.5, 1.05, 500)
soc_hard = 0.9  # hard limit

# Crisp (step function)
crisp = np.where(soc <= soc_hard, 1.0, 0.0)

# Fuzzy - different beta values
beta_medium = 25.0
ref_medium = 0.88
fuzzy_medium = 1.0 / (1.0 + np.exp(-beta_medium * (ref_medium - soc)))

beta_sharp = 50.0
fuzzy_sharp = 1.0 / (1.0 + np.exp(-beta_sharp * (ref_medium - soc)))

beta_wide = 10.0
fuzzy_wide = 1.0 / (1.0 + np.exp(-beta_wide * (ref_medium - soc)))

# --------------------------
# Plot
# --------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))

# Shaded zones
ax.axvspan(0.5, 0.8, alpha=0.12, color='#2ca02c', label='_nolegend_')
ax.axvspan(0.8, 0.9, alpha=0.10, color='#ff7f0e', label='_nolegend_')
ax.axvspan(0.9, 1.05, alpha=0.10, color='#d62728', label='_nolegend_')

# Zone labels
ax.text(0.65, 1.03, 'Safe zone\n(recommended)', ha='center', va='bottom',
        fontsize=9, color='#2ca02c', fontweight='bold')
ax.text(0.85, 1.03, 'Transition\nzone', ha='center', va='bottom',
        fontsize=9, color='#ff7f0e', fontweight='bold')
ax.text(0.975, 1.03, 'Violation\nzone', ha='center', va='bottom',
        fontsize=9, color='#d62728', fontweight='bold')

# Crisp constraint
ax.step(soc, crisp, color=C_CRISP, linewidth=2.2, where='post',
        label='Crisp constraint (hard boundary)', zorder=5)

# Fuzzy constraints
ax.plot(soc, fuzzy_wide, color=C_FUZZY_2, linewidth=1.8, linestyle='--',
        label=r'Fuzzy (wide, $\beta=10$)', zorder=4)
ax.plot(soc, fuzzy_medium, color=C_FUZZY, linewidth=2.5,
        label=r'Fuzzy (medium, $\beta=25$)', zorder=5)
ax.plot(soc, fuzzy_sharp, color='#9467bd', linewidth=1.8, linestyle=':',
        label=r'Fuzzy (sharp, $\beta=50$)', zorder=4)

# Reference point marker
ax.plot([ref_medium, ref_medium], [0, 0.5], 'k--', linewidth=0.8, alpha=0.6)
ax.plot([0, ref_medium], [0.5, 0.5], 'k--', linewidth=0.8, alpha=0.6)
ax.text(ref_medium - 0.005, 0.52, r'$\mu=0.5$', fontsize=9, ha='right', color=C_TEXT)

# Hard limit marker
ax.axvline(x=soc_hard, color=C_CRISP, linestyle='-.', linewidth=1.0, alpha=0.5)
ax.text(soc_hard + 0.005, 0.15, 'Hard limit\n(SOC=0.9)', fontsize=8.5,
        color=C_CRISP, va='center')

# Axes
ax.set_xlabel('State of Charge (SOC) [p.u.]')
ax.set_ylabel('Constraint Satisfaction Degree $\mu(s,a)$')
ax.set_xlim(0.5, 1.02)
ax.set_ylim(-0.05, 1.15)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])

# Legend
ax.legend(loc='lower left', frameon=True, framealpha=0.95, edgecolor='#cccccc')

# Grid
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('paper/figures/fig1_fuzzy_constraint.png')
plt.savefig('paper/figures/fig1_fuzzy_constraint.pdf')
print('Figure 1 saved.')
plt.close()
