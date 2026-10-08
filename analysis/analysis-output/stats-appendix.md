# Stats Appendix

## Descriptive statistics (mean ± 95% CI, n=3 seeds)

| Algorithm | Scenario | Cost (¥/d) | Violation (%) | Avg FCSD | Min FCSD |
|---|---|---:|---:|---:|---:|
| PPO | S1_grid_normal | 16,982 ± 11,552 | 1.74 ± 7.47 | 0.952 ± 0.044 | 0.782 ± 0.464 |
| PPO | S2_grid_extreme | 16,681 ± 4,188 | 42.16 ± 45.27 | 0.792 ± 0.165 | 0.528 ± 0.012 |
| PPO | S3_island_normal | 9,617 ± 3,797 | 4.17 ± 14.80 | 0.989 ± 0.029 | 0.721 ± 0.195 |
| PPO | S4_island_extreme | 13,231 ± 4,644 | 43.70 ± 13.76 | 0.861 ± 0.045 | 0.601 ± 0.219 |
| SAC | S1_grid_normal | 15,516 ± 9,300 | 82.19 ± 72.78 | 0.675 ± 0.292 | 0.552 ± 0.082 |
| SAC | S2_grid_extreme | 25,819 ± 6,260 | 99.26 ± 0.00 | 0.583 ± 0.072 | 0.528 ± 0.041 |
| SAC | S3_island_normal | 4,559 ± 4,155 | 98.96 ± 0.00 | 0.753 ± 0.070 | 0.481 ± 0.034 |
| SAC | S4_island_extreme | 11,151 ± 7,893 | 98.96 ± 0.00 | 0.750 ± 0.037 | 0.474 ± 0.000 |
| PPO-Lag | S1_grid_normal | 22,205 ± 5,077 | 0.00 ± 0.00 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| PPO-Lag | S2_grid_extreme | 44,168 ± 14,113 | 0.00 ± 0.00 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| PPO-Lag | S3_island_normal | 10,922 ± 805 | 0.60 ± 1.11 | 0.997 ± 0.005 | 0.778 ± 0.198 |
| PPO-Lag | S4_island_extreme | 12,222 ± 4,941 | 48.66 ± 13.14 | 0.806 ± 0.211 | 0.474 ± 0.588 |
| CPO | S1_grid_normal | 17,240 ± 4,588 | 0.00 ± 0.00 | 1.000 ± 0.000 | 0.996 ± 0.006 |
| CPO | S2_grid_extreme | 31,675 ± 7,174 | 0.00 ± 0.00 | 0.999 ± 0.001 | 0.996 ± 0.007 |
| CPO | S3_island_normal | 11,725 ± 1,749 | 1.24 ± 2.80 | 0.995 ± 0.009 | 0.743 ± 0.171 |
| CPO | S4_island_extreme | 13,068 ± 911 | 48.96 ± 3.29 | 0.835 ± 0.044 | 0.511 ± 0.269 |
| SL | S1_grid_normal | 30,578 ± 675 | 0.00 ± 0.00 | 0.956 ± 0.053 | 0.927 ± 0.037 |
| SL | S2_grid_extreme | 40,546 ± 25,720 | 0.00 ± 0.00 | 0.959 ± 0.074 | 0.906 ± 0.029 |
| SL | S3_island_normal | 3,712 ± 942 | 56.45 ± 33.87 | 0.822 ± 0.101 | 0.679 ± 0.022 |
| SL | S4_island_extreme | 9,077 ± 2,226 | 57.44 ± 29.38 | 0.811 ± 0.076 | 0.534 ± 0.319 |
| Fuzzy-SAC | S1_grid_normal | 15,140 ± 5,101 | 52.68 ± 123.83 | 0.781 ± 0.508 | 0.671 ± 0.612 |
| Fuzzy-SAC | S2_grid_extreme | 28,032 ± 4,667 | 89.78 ± 40.76 | 0.614 ± 0.145 | 0.527 ± 0.025 |
| Fuzzy-SAC | S3_island_normal | 7,432 ± 8,410 | 60.76 ± 98.81 | 0.854 ± 0.188 | 0.646 ± 0.386 |
| Fuzzy-SAC | S4_island_extreme | 9,451 ± 8,477 | 98.81 ± 0.64 | 0.725 ± 0.159 | 0.449 ± 0.106 |
| HFG-SAC (ours) | S1_grid_normal | 13,959 ± 1,020 | 0.00 ± 0.00 | 0.968 ± 0.036 | 0.904 ± 0.032 |
| HFG-SAC (ours) | S2_grid_extreme | 23,769 ± 2,316 | 0.00 ± 0.00 | 0.925 ± 0.031 | 0.840 ± 0.003 |
| HFG-SAC (ours) | S3_island_normal | 8,363 ± 1,411 | 5.21 ± 3.22 | 0.979 ± 0.006 | 0.679 ± 0.010 |
| HFG-SAC (ours) | S4_island_extreme | 24,004 ± 3,319 | 0.64 ± 2.13 | 0.994 ± 0.009 | 0.736 ± 0.124 |

## Inferential tests: HFG-SAC vs baselines

Welch's t-test (two-sided) on `cost_per_day` across 3 seeds.
Effect size: Cohen's d (pooled). n=3 per arm — tests are underpowered; report as exploratory.

| Scenario | Comparison | Δ cost (¥/d) | t | p | Cohen's d |
|---|---|---:|---:|---:|---:|
| S1_grid_normal | HFG-SAC vs PPO | -3,023 | -1.12 | 0.377 | -0.92 |
| S1_grid_normal | HFG-SAC vs SAC | -1,557 | -0.72 | 0.547 | -0.58 |
| S1_grid_normal | HFG-SAC vs PPO-Lag | -8,246 | -6.85 | 0.017 | -5.59 |
| S1_grid_normal | HFG-SAC vs CPO | -3,282 | -3.00 | 0.085 | -2.45 |
| S1_grid_normal | HFG-SAC vs SL | -16,619 | -58.48 | 0.000 | -47.75 |
| S1_grid_normal | HFG-SAC vs Fuzzy-SAC | -1,181 | -0.98 | 0.425 | -0.80 |
| S2_grid_extreme | HFG-SAC vs PPO | +7,087 | 6.37 | 0.007 | +5.20 |
| S2_grid_extreme | HFG-SAC vs SAC | -2,051 | -1.32 | 0.293 | -1.08 |
| S2_grid_extreme | HFG-SAC vs PPO-Lag | -20,399 | -6.14 | 0.022 | -5.01 |
| S2_grid_extreme | HFG-SAC vs CPO | -7,907 | -4.51 | 0.032 | -3.68 |
| S2_grid_extreme | HFG-SAC vs SL | -16,777 | -2.80 | 0.106 | -2.28 |
| S2_grid_extreme | HFG-SAC vs Fuzzy-SAC | -4,264 | -3.52 | 0.040 | -2.87 |
| S3_island_normal | HFG-SAC vs PPO | -1,254 | -1.33 | 0.290 | -1.09 |
| S3_island_normal | HFG-SAC vs SAC | +3,805 | 3.73 | 0.047 | +3.05 |
| S3_island_normal | HFG-SAC vs PPO-Lag | -2,559 | -6.78 | 0.005 | -5.53 |
| S3_island_normal | HFG-SAC vs CPO | -3,362 | -6.44 | 0.003 | -5.26 |
| S3_island_normal | HFG-SAC vs SL | +4,651 | 11.80 | 0.001 | +9.63 |
| S3_island_normal | HFG-SAC vs Fuzzy-SAC | +931 | 0.47 | 0.683 | +0.38 |
| S4_island_extreme | HFG-SAC vs PPO | +10,773 | 8.12 | 0.002 | +6.63 |
| S4_island_extreme | HFG-SAC vs SAC | +12,853 | 6.46 | 0.010 | +5.27 |
| S4_island_extreme | HFG-SAC vs PPO-Lag | +11,782 | 8.52 | 0.002 | +6.95 |
| S4_island_extreme | HFG-SAC vs CPO | +10,936 | 13.67 | 0.003 | +11.16 |
| S4_island_extreme | HFG-SAC vs SL | +14,927 | 16.07 | 0.000 | +13.12 |
| S4_island_extreme | HFG-SAC vs Fuzzy-SAC | +14,553 | 6.88 | 0.010 | +5.62 |

## Safety: violation rate comparison

| Scenario | HFG-SAC viol% | Best baseline viol% | Δ |
|---|---:|---:|---:|
| S1_grid_normal | 0.00 | 0.00 | +0.00 |
| S2_grid_extreme | 0.00 | 0.00 | +0.00 |
| S3_island_normal | 5.21 | 0.60 | +4.61 |
| S4_island_extreme | 0.64 | 43.70 | -43.06 |
