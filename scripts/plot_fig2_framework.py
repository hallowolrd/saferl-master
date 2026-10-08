"""
Figure 2: HFG-SAC Framework Architecture Diagram
Shows the hierarchical fuzzy-guided SAC architecture with:
- Upper layer: Fuzzy Constraint Protection
- Lower layer: Fuzzy Knowledge Reward Shaping
- Base SAC network
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.patches import ConnectionPatch

# --------------------------
# Style
# --------------------------
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix',
})

# Colors
C_STATE = '#4c72b0'
C_ACTION = '#dd8452'
C_UPPER = '#c44e52'     # fuzzy constraint (upper)
C_LOWER = '#55a868'     # fuzzy knowledge (lower)
C_SAC = '#8172b2'
C_REWARD = '#ccb974'
C_BG_UPPER = '#fadbd8'
C_BG_LOWER = '#d5f5e3'
C_BG_SAC = '#e8daef'

fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis('off')

# ===========================
# Helper functions
# ===========================
def draw_box(ax, x, y, w, h, text, color='white', edgecolor='black',
             fontsize=10, fontweight='normal', text_color='black',
             boxstyle='round,pad=0.3'):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle=boxstyle,
                          facecolor=color, edgecolor=edgecolor,
                          linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=text_color)
    return box

def arrow(ax, x1, y1, x2, y2, color='black', style='->', lw=1.5):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle=style, color=color,
                         linewidth=lw, mutation_scale=15)
    ax.add_patch(a)
    return a

# ===========================
# Title
# ===========================
ax.text(6, 8.7, 'Hierarchical Fuzzy-Guided SAC (HFG-SAC) Framework',
        ha='center', va='center', fontsize=14, fontweight='bold')

# ===========================
# Environment (left)
# ===========================
draw_box(ax, 0.3, 3.5, 1.8, 1.5, 'Microgrid\nEnvironment', color='#d6eaf8',
         edgecolor='#2874a6', fontsize=10, fontweight='bold')

# State output from env
draw_box(ax, 2.5, 4.3, 1.2, 0.7, 'State $s_t$', color='#aed6f1',
         edgecolor='#2874a6', fontsize=10)
arrow(ax, 2.1, 4.65, 2.5, 4.65, color='#2874a6')

# Action input to env
draw_box(ax, 2.5, 3.2, 1.2, 0.7, 'Action $a_t$', color='#fad7a0',
         edgecolor='#d68910', fontsize=10)
arrow(ax, 2.5, 3.55, 2.1, 3.55, color='#d68910')

# Reward output
draw_box(ax, 2.5, 5.4, 1.2, 0.6, 'Reward $r_t$', color='#f9e79f',
         edgecolor='#d4ac0d', fontsize=9)
arrow(ax, 2.1, 5.7, 2.5, 5.7, color='#d4ac0d')

# ===========================
# Base SAC Actor-Critic (center)
# ===========================
# SAC background
sac_bg = FancyBboxPatch((4.0, 2.0), 3.5, 4.5,
                         boxstyle='round,pad=0.2',
                         facecolor=C_BG_SAC, edgecolor=C_SAC,
                         linewidth=1.2, linestyle='--')
ax.add_patch(sac_bg)
ax.text(5.75, 6.2, 'Base SAC Network', ha='center', va='center',
        fontsize=11, fontweight='bold', color=C_SAC)

# Actor network
draw_box(ax, 4.3, 4.6, 1.5, 1.2, 'Actor\n$\pi_\phi(a|s)$', color='white',
         edgecolor=C_SAC, fontsize=10)

# Critic network
draw_box(ax, 6.0, 4.6, 1.2, 1.2, 'Critic\n$Q_\\theta(s,a)$', color='white',
         edgecolor=C_SAC, fontsize=10)

# State -> Actor
arrow(ax, 3.7, 5.2, 4.3, 5.2, color='#2874a6')

# State -> Critic (dashed, comes from state)
arrow(ax, 4.8, 4.3, 5.4, 4.3, color='gray', style='->', lw=1)
ax.text(5.1, 4.05, '$s_t, a_t$', ha='center', va='center', fontsize=8, color='gray')

# Action output from Actor
arrow(ax, 5.8, 5.2, 6.0, 5.2, color='#d68910')
# Action goes to env side
arrow(ax, 5.05, 4.6, 3.7, 3.55, color='#d68910', style='->')

# ===========================
# Upper Layer: Fuzzy Constraint Protection
# ===========================
upper_bg = FancyBboxPatch((8.2, 5.2), 3.4, 2.8,
                           boxstyle='round,pad=0.2',
                           facecolor=C_BG_UPPER, edgecolor=C_UPPER,
                           linewidth=1.5)
ax.add_patch(upper_bg)
ax.text(9.9, 7.7, 'Upper Layer: Fuzzy Constraint Protection', ha='center', va='center',
        fontsize=10.5, fontweight='bold', color=C_UPPER)

# TSK Fuzzy System
draw_box(ax, 8.5, 6.4, 2.8, 0.9, 'TSK Fuzzy System\n(Multi-constraint FCSD)', color='white',
         edgecolor=C_UPPER, fontsize=9.5)

# Fuzzy-Lagrangian
draw_box(ax, 8.5, 5.3, 2.8, 0.8, 'Fuzzy-Lagrangian\n$\\lambda \\cdot (\\alpha - \\bar{\\mu})$', color='white',
         edgecolor=C_UPPER, fontsize=9.5)

# Arrows within upper layer
arrow(ax, 9.9, 6.4, 9.9, 6.1, color=C_UPPER)

# State -> Upper layer
arrow(ax, 3.7, 5.7, 8.2, 7.0, color=C_UPPER, style='->')
ax.text(6.0, 7.0, 'state $s_t$, action $a_t$', ha='center', va='bottom',
        fontsize=8, color=C_UPPER)

# Constraint loss -> Actor
arrow(ax, 8.2, 5.7, 5.05, 5.8, color=C_UPPER, style='->')
ax.text(6.8, 5.9, 'constraint loss', ha='center', va='bottom',
        fontsize=8, color=C_UPPER)

# Dual variable update
draw_box(ax, 9.5, 7.0, 0.8, 0.5, '$\\lambda$', color='#f1948a',
         edgecolor=C_UPPER, fontsize=10, fontweight='bold')
arrow(ax, 9.9, 7.0, 9.9, 7.3, color=C_UPPER, style='<->')

# ===========================
# Lower Layer: Fuzzy Knowledge Reward Shaping
# ===========================
lower_bg = FancyBboxPatch((8.2, 1.8), 3.4, 2.4,
                           boxstyle='round,pad=0.2',
                           facecolor=C_BG_LOWER, edgecolor=C_LOWER,
                           linewidth=1.5)
ax.add_patch(lower_bg)
ax.text(9.9, 3.9, 'Lower Layer: Fuzzy Knowledge Shaping', ha='center', va='center',
        fontsize=10.5, fontweight='bold', color=C_LOWER)

# Expert rules
draw_box(ax, 8.5, 3.0, 2.8, 0.7, 'Expert Fuzzy Rules\n(If-Then knowledge)', color='white',
         edgecolor=C_LOWER, fontsize=9)

# Potential-based shaping
draw_box(ax, 8.5, 2.1, 2.8, 0.7, 'Potential-based Shaping\n$F(s,s\') = \\gamma\\Phi(s\') - \\Phi(s)$', color='white',
         edgecolor=C_LOWER, fontsize=9)

arrow(ax, 9.9, 3.0, 9.9, 2.8, color=C_LOWER)

# State + reward -> Lower layer
arrow(ax, 3.7, 5.7, 8.2, 3.5, color=C_LOWER, style='->')
ax.text(5.5, 4.5, '$s_t, r_t$', ha='center', va='bottom',
        fontsize=8, color=C_LOWER)

# Shaped reward -> Critic
arrow(ax, 8.2, 2.5, 6.6, 4.3, color=C_LOWER, style='->')
ax.text(7.5, 3.5, 'shaped\nreward', ha='center', va='center',
        fontsize=8, color=C_LOWER)

# Weight decay
draw_box(ax, 10.5, 1.3, 1.0, 0.4, '$\\kappa_t \\downarrow$', color='#abebc6',
         edgecolor=C_LOWER, fontsize=9)
arrow(ax, 11.0, 1.7, 11.0, 2.1, color=C_LOWER)

# ===========================
# Replay Buffer
# ===========================
draw_box(ax, 4.5, 2.4, 2.5, 0.8, 'Replay Buffer $\\mathcal{D}$', color='#fdebd0',
         edgecolor='#af601a', fontsize=10)
arrow(ax, 5.75, 3.2, 5.75, 4.3, color='#af601a', style='<->')

# ===========================
# Transfer Module (bottom)
# ===========================
transfer_bg = FancyBboxPatch((3.5, 0.3), 5.0, 1.2,
                              boxstyle='round,pad=0.2',
                              facecolor='#ebdef0', edgecolor='#6c3483',
                              linewidth=1.2)
ax.add_patch(transfer_bg)
ax.text(6.0, 1.25, 'Constraint-Aware Cross-Scenario Transfer', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#6c3483')
ax.text(6.0, 0.75, 'Fuzzy Similarity Metric  $\cdot$  Progressive Adaptation  $\cdot$  Conservative Init.',
        ha='center', va='center', fontsize=8.5, color='#6c3483')

arrow(ax, 5.75, 2.4, 5.75, 1.5, color='#6c3483', style='<->')

# ===========================
# Annotations / Legend
# ===========================
# Key
ax.text(0.3, 0.6, 'Key:', fontsize=9, fontweight='bold')
ax.text(0.3, 0.25, 'Information flow', fontsize=8.5, color='#2874a6')
ax.plot([1.7, 2.3], [0.27, 0.27], color=C_UPPER, linewidth=1.5)
ax.text(2.4, 0.18, 'Constraint signal', fontsize=8.5, color=C_UPPER)
ax.plot([3.8, 4.4], [0.27, 0.27], color=C_LOWER, linewidth=1.5)
ax.text(4.5, 0.18, 'Knowledge signal', fontsize=8.5, color=C_LOWER)

plt.tight_layout()
plt.savefig('paper/figures/fig2_framework.png')
plt.savefig('paper/figures/fig2_framework.pdf')
print('Figure 2 saved.')
plt.close()
