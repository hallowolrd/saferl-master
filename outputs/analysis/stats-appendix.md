# Statistical Appendix — HFG-SAC Algorithm Comparison

_Units of analysis: independent training runs (seeds). Reported effect size = Hedges' g (small-sample corrected Cohen's d). Significance = two-sided permutation test on the difference of means._

## 1. Descriptive Statistics (mean ± std, 95% CI)

### S1 Grid Normal

| Metric | Algorithm | n | Mean ± Std | 95% CI |
|---|---|---|---|---|
| Daily Operating Cost (¥) | **HFG-SAC (ours)** | 3 | 84964.34 ± 20080.40 [35077.83, 134850.85] |
| Daily Operating Cost (¥) | SAC | 3 | 108610.41 ± 26205.55 [43506.95, 173713.86] |
| Daily Operating Cost (¥) | PPO | 3 | 118874.05 ± 32551.40 [38005.36, 199742.75] |
| Daily Operating Cost (¥) | PPO-Lagrangian | 3 | 155435.08 ± 14305.48 [119895.44, 190974.73] |
| Daily Operating Cost (¥) | CPO | 3 | 120682.09 ± 12927.71 [88565.29, 152798.89] |
| Daily Operating Cost (¥) | CBF Safety Layer | 3 | 214043.72 ± 1901.09 [209320.76, 218766.68] |
| Daily Operating Cost (¥) | Fuzzy SAC | 3 | 105980.78 ± 14374.51 [70269.65, 141691.92] |
| Constraint Violation Rate (%) | **HFG-SAC (ours)** | 3 | 57.99 ± 51.49 [-69.94, 185.91] |
| Constraint Violation Rate (%) | SAC | 3 | 82.19 ± 29.30 [9.41, 154.98] |
| Constraint Violation Rate (%) | PPO | 3 | 1.74 ± 3.01 [-5.73, 9.21] |
| Constraint Violation Rate (%) | PPO-Lagrangian | 3 | 0.00 ± 0.00 [0.00, 0.00] |
| Constraint Violation Rate (%) | CPO | 3 | 0.00 ± 0.00 [0.00, 0.00] |
| Constraint Violation Rate (%) | CBF Safety Layer | 3 | 0.00 ± 0.00 [0.00, 0.00] |
| Constraint Violation Rate (%) | Fuzzy SAC | 3 | 52.68 ± 49.85 [-71.16, 176.52] |
| Avg. FCSD | **HFG-SAC (ours)** | 3 | 0.77 ± 0.19 [0.31, 1.24] |
| Avg. FCSD | SAC | 3 | 0.67 ± 0.12 [0.38, 0.97] |
| Avg. FCSD | PPO | 3 | 0.95 ± 0.02 [0.91, 1.00] |
| Avg. FCSD | PPO-Lagrangian | 3 | 1.00 ± 0.00 [1.00, 1.00] |
| Avg. FCSD | CPO | 3 | 1.00 ± 0.00 [1.00, 1.00] |
| Avg. FCSD | CBF Safety Layer | 3 | 0.96 ± 0.02 [0.90, 1.01] |
| Avg. FCSD | Fuzzy SAC | 3 | 0.78 ± 0.20 [0.27, 1.29] |
| Convergence Episode | **HFG-SAC (ours)** | 3 | 81.33 ± 102.00 [-172.07, 334.74] |
| Convergence Episode | SAC | 3 | 142.33 ± 29.37 [69.38, 215.29] |
| Convergence Episode | PPO | 3 | 163.00 ± 21.38 [109.89, 216.11] |
| Convergence Episode | PPO-Lagrangian | 3 | 173.67 ± 20.84 [121.89, 225.44] |
| Convergence Episode | CPO | 3 | 180.33 ± 3.51 [171.61, 189.06] |
| Convergence Episode | CBF Safety Layer | 3 | 11.00 ± 0.00 [11.00, 11.00] |
| Convergence Episode | Fuzzy SAC | 3 | 103.67 ± 38.14 [8.92, 198.41] |

### S2 Grid Extreme

| Metric | Algorithm | n | Mean ± Std | 95% CI |
|---|---|---|---|---|
| Daily Operating Cost (¥) | **HFG-SAC (ours)** | 3 | 180444.54 ± 23889.71 [121094.42, 239794.65] |
| Daily Operating Cost (¥) | SAC | 3 | 180734.70 ± 17638.83 [136913.89, 224555.52] |
| Daily Operating Cost (¥) | PPO | 3 | 116768.19 ± 11801.78 [87448.59, 146087.80] |
| Daily Operating Cost (¥) | PPO-Lagrangian | 3 | 309175.95 ± 39767.33 [210380.46, 407971.45] |
| Daily Operating Cost (¥) | CPO | 3 | 221726.53 ± 20215.94 [171503.30, 271949.76] |
| Daily Operating Cost (¥) | CBF Safety Layer | 3 | 283822.25 ± 72475.55 [103768.47, 463876.03] |
| Daily Operating Cost (¥) | Fuzzy SAC | 3 | 196227.01 ± 13151.17 [163555.05, 228898.97] |
| Constraint Violation Rate (%) | **HFG-SAC (ours)** | 3 | 98.02 ± 2.15 [92.68, 103.35] |
| Constraint Violation Rate (%) | SAC | 3 | 99.26 ± 0.00 [99.26, 99.26] |
| Constraint Violation Rate (%) | PPO | 3 | 42.16 ± 18.22 [-3.11, 87.44] |
| Constraint Violation Rate (%) | PPO-Lagrangian | 3 | 0.00 ± 0.00 [0.00, 0.00] |
| Constraint Violation Rate (%) | CPO | 3 | 0.00 ± 0.00 [0.00, 0.00] |
| Constraint Violation Rate (%) | CBF Safety Layer | 3 | 0.00 ± 0.00 [0.00, 0.00] |
| Constraint Violation Rate (%) | Fuzzy SAC | 3 | 89.78 ± 16.41 [49.01, 130.55] |
| Avg. FCSD | **HFG-SAC (ours)** | 3 | 0.58 ± 0.01 [0.55, 0.61] |
| Avg. FCSD | SAC | 3 | 0.58 ± 0.03 [0.51, 0.66] |
| Avg. FCSD | PPO | 3 | 0.79 ± 0.07 [0.63, 0.96] |
| Avg. FCSD | PPO-Lagrangian | 3 | 1.00 ± 0.00 [1.00, 1.00] |
| Avg. FCSD | CPO | 3 | 1.00 ± 0.00 [1.00, 1.00] |
| Avg. FCSD | CBF Safety Layer | 3 | 0.96 ± 0.03 [0.88, 1.03] |
| Avg. FCSD | Fuzzy SAC | 3 | 0.61 ± 0.06 [0.47, 0.76] |
| Convergence Episode | **HFG-SAC (ours)** | 3 | 123.00 ± 20.52 [72.03, 173.97] |
| Convergence Episode | SAC | 3 | 156.00 ± 14.11 [120.95, 191.05] |
| Convergence Episode | PPO | 3 | 150.67 ± 6.81 [133.76, 167.58] |
| Convergence Episode | PPO-Lagrangian | 3 | 85.67 ± 14.47 [49.72, 121.61] |
| Convergence Episode | CPO | 3 | 154.67 ± 33.01 [72.67, 236.66] |
| Convergence Episode | CBF Safety Layer | 3 | 36.67 ± 24.19 [-23.44, 96.77] |
| Convergence Episode | Fuzzy SAC | 3 | 115.00 ± 66.12 [-49.27, 279.27] |

### S3 Island Normal

| Metric | Algorithm | n | Mean ± Std | 95% CI |
|---|---|---|---|---|
| Daily Operating Cost (¥) | **HFG-SAC (ours)** | 3 | 39483.63 ± 8228.92 [19040.21, 59927.04] |
| Daily Operating Cost (¥) | SAC | 3 | 31911.23 ± 11707.41 [2826.06, 60996.41] |
| Daily Operating Cost (¥) | PPO | 3 | 67319.40 ± 10698.51 [40740.68, 93898.12] |
| Daily Operating Cost (¥) | PPO-Lagrangian | 3 | 76456.00 ± 2268.06 [70821.37, 82090.63] |
| Daily Operating Cost (¥) | CPO | 3 | 82075.34 ± 4927.08 [69834.80, 94315.89] |
| Daily Operating Cost (¥) | CBF Safety Layer | 3 | 25983.78 ± 2653.55 [19391.47, 32576.10] |
| Daily Operating Cost (¥) | Fuzzy SAC | 3 | 52026.88 ± 23697.59 [-6845.94, 110899.70] |
| Constraint Violation Rate (%) | **HFG-SAC (ours)** | 3 | 76.93 ± 38.15 [-17.83, 171.70] |
| Constraint Violation Rate (%) | SAC | 3 | 98.96 ± 0.00 [98.96, 98.96] |
| Constraint Violation Rate (%) | PPO | 3 | 4.17 ± 5.96 [-10.63, 18.97] |
| Constraint Violation Rate (%) | PPO-Lagrangian | 3 | 0.60 ± 0.45 [-0.51, 1.70] |
| Constraint Violation Rate (%) | CPO | 3 | 1.24 ± 1.13 [-1.56, 4.04] |
| Constraint Violation Rate (%) | CBF Safety Layer | 3 | 56.45 ± 13.63 [22.58, 90.32] |
| Constraint Violation Rate (%) | Fuzzy SAC | 3 | 60.76 ± 39.77 [-38.05, 159.58] |
| Avg. FCSD | **HFG-SAC (ours)** | 3 | 0.82 ± 0.07 [0.66, 0.99] |
| Avg. FCSD | SAC | 3 | 0.75 ± 0.03 [0.68, 0.82] |
| Avg. FCSD | PPO | 3 | 0.99 ± 0.01 [0.96, 1.02] |
| Avg. FCSD | PPO-Lagrangian | 3 | 1.00 ± 0.00 [0.99, 1.00] |
| Avg. FCSD | CPO | 3 | 1.00 ± 0.00 [0.99, 1.00] |
| Avg. FCSD | CBF Safety Layer | 3 | 0.82 ± 0.04 [0.72, 0.92] |
| Avg. FCSD | Fuzzy SAC | 3 | 0.85 ± 0.08 [0.67, 1.04] |
| Convergence Episode | **HFG-SAC (ours)** | 3 | 18.00 ± 13.86 [-16.42, 52.42] |
| Convergence Episode | SAC | 3 | 19.00 ± 7.81 [-0.40, 38.40] |
| Convergence Episode | PPO | 3 | 10.00 ± 0.00 [10.00, 10.00] |
| Convergence Episode | PPO-Lagrangian | 3 | 10.00 ± 0.00 [10.00, 10.00] |
| Convergence Episode | CPO | 3 | 10.00 ± 0.00 [10.00, 10.00] |
| Convergence Episode | CBF Safety Layer | 3 | 68.00 ± 91.10 [-158.32, 294.32] |
| Convergence Episode | Fuzzy SAC | 3 | 15.00 ± 7.81 [-4.40, 34.40] |

### S4 Island Extreme

| Metric | Algorithm | n | Mean ± Std | 95% CI |
|---|---|---|---|---|
| Daily Operating Cost (¥) | **HFG-SAC (ours)** | 3 | 66655.34 ± 2122.82 [61381.53, 71929.14] |
| Daily Operating Cost (¥) | SAC | 3 | 78054.91 ± 22241.94 [22798.41, 133311.41] |
| Daily Operating Cost (¥) | PPO | 3 | 92613.74 ± 13085.52 [60104.89, 125122.59] |
| Daily Operating Cost (¥) | PPO-Lagrangian | 3 | 85552.80 ± 13922.74 [50964.02, 120141.59] |
| Daily Operating Cost (¥) | CPO | 3 | 91474.46 ± 2566.42 [85098.59, 97850.32] |
| Daily Operating Cost (¥) | CBF Safety Layer | 3 | 63536.05 ± 6272.48 [47953.09, 79119.01] |
| Daily Operating Cost (¥) | Fuzzy SAC | 3 | 66154.78 ± 23888.00 [6808.90, 125500.65] |
| Constraint Violation Rate (%) | **HFG-SAC (ours)** | 3 | 99.06 ± 0.17 [98.63, 99.48] |
| Constraint Violation Rate (%) | SAC | 3 | 98.96 ± 0.00 [98.96, 98.96] |
| Constraint Violation Rate (%) | PPO | 3 | 43.70 ± 5.54 [29.94, 57.47] |
| Constraint Violation Rate (%) | PPO-Lagrangian | 3 | 48.66 ± 5.29 [35.52, 61.80] |
| Constraint Violation Rate (%) | CPO | 3 | 48.96 ± 1.32 [45.67, 52.24] |
| Constraint Violation Rate (%) | CBF Safety Layer | 3 | 57.44 ± 11.83 [28.06, 86.82] |
| Constraint Violation Rate (%) | Fuzzy SAC | 3 | 98.81 ± 0.26 [98.17, 99.45] |
| Avg. FCSD | **HFG-SAC (ours)** | 3 | 0.74 ± 0.03 [0.67, 0.81] |
| Avg. FCSD | SAC | 3 | 0.75 ± 0.01 [0.71, 0.79] |
| Avg. FCSD | PPO | 3 | 0.86 ± 0.02 [0.82, 0.91] |
| Avg. FCSD | PPO-Lagrangian | 3 | 0.81 ± 0.08 [0.59, 1.02] |
| Avg. FCSD | CPO | 3 | 0.83 ± 0.02 [0.79, 0.88] |
| Avg. FCSD | CBF Safety Layer | 3 | 0.81 ± 0.03 [0.74, 0.89] |
| Avg. FCSD | Fuzzy SAC | 3 | 0.72 ± 0.06 [0.57, 0.88] |
| Convergence Episode | **HFG-SAC (ours)** | 3 | 12.67 ± 4.62 [1.19, 24.14] |
| Convergence Episode | SAC | 3 | 10.00 ± 0.00 [10.00, 10.00] |
| Convergence Episode | PPO | 3 | 10.00 ± 0.00 [10.00, 10.00] |
| Convergence Episode | PPO-Lagrangian | 3 | 10.00 ± 0.00 [10.00, 10.00] |
| Convergence Episode | CPO | 3 | 10.00 ± 0.00 [10.00, 10.00] |
| Convergence Episode | CBF Safety Layer | 3 | 10.00 ± 0.00 [10.00, 10.00] |
| Convergence Episode | Fuzzy SAC | 3 | 10.00 ± 0.00 [10.00, 10.00] |

## 2. Effect Sizes — HFG-SAC vs Each Baseline (Hedges' g)

Sign convention: negative g means HFG-SAC is _better_ for lower-is-better metrics (cost, violation, convergence) and _worse_ for higher-is-better metrics (FCSD).

### S1 Grid Normal

| Metric | Baseline | g (HFG-SAC − baseline) | Magnitude | Rel. improvement |
|---|---|---|---|---|
| Daily Operating Cost (¥) | SAC | -0.81 | large | +21.8% |
| Daily Operating Cost (¥) | PPO | -1.00 | large | +28.5% |
| Daily Operating Cost (¥) | PPO-Lagrangian | -3.23 | large | +45.3% |
| Daily Operating Cost (¥) | CPO | -1.69 | large | +29.6% |
| Daily Operating Cost (¥) | CBF Safety Layer | -7.24 | large | +60.3% |
| Daily Operating Cost (¥) | Fuzzy SAC | -0.96 | large | +19.8% |
| Constraint Violation Rate (%) | SAC | -0.46 | small | +29.5% |
| Constraint Violation Rate (%) | PPO | +1.23 | large | -3240.0% |
| Constraint Violation Rate (%) | PPO-Lagrangian | +1.27 | large | — |
| Constraint Violation Rate (%) | CPO | +1.27 | large | — |
| Constraint Violation Rate (%) | CBF Safety Layer | +1.27 | large | — |
| Constraint Violation Rate (%) | Fuzzy SAC | +0.08 | negligible | -10.1% |
| Avg. FCSD | SAC | +0.50 | medium | +14.5% |
| Avg. FCSD | PPO | -1.07 | large | -18.8% |
| Avg. FCSD | PPO-Lagrangian | -1.36 | large | -22.7% |
| Avg. FCSD | CPO | -1.36 | large | -22.7% |
| Avg. FCSD | CBF Safety Layer | -1.09 | large | -19.1% |
| Avg. FCSD | Fuzzy SAC | -0.03 | negligible | -1.0% |
| Convergence Episode | SAC | -0.65 | medium | +42.9% |
| Convergence Episode | PPO | -0.89 | large | +50.1% |
| Convergence Episode | PPO-Lagrangian | -1.00 | large | +53.2% |
| Convergence Episode | CPO | -1.10 | large | +54.9% |
| Convergence Episode | CBF Safety Layer | +0.78 | medium | -639.4% |
| Convergence Episode | Fuzzy SAC | -0.23 | small | +21.5% |

### S2 Grid Extreme

| Metric | Baseline | g (HFG-SAC − baseline) | Magnitude | Rel. improvement |
|---|---|---|---|---|
| Daily Operating Cost (¥) | SAC | -0.01 | negligible | +0.2% |
| Daily Operating Cost (¥) | PPO | +2.70 | large | -54.5% |
| Daily Operating Cost (¥) | PPO-Lagrangian | -3.14 | large | +41.6% |
| Daily Operating Cost (¥) | CPO | -1.49 | large | +18.6% |
| Daily Operating Cost (¥) | CBF Safety Layer | -1.53 | large | +36.4% |
| Daily Operating Cost (¥) | Fuzzy SAC | -0.65 | medium | +8.0% |
| Constraint Violation Rate (%) | SAC | -0.65 | medium | +1.2% |
| Constraint Violation Rate (%) | PPO | +3.44 | large | -132.5% |
| Constraint Violation Rate (%) | PPO-Lagrangian | +51.63 | large | — |
| Constraint Violation Rate (%) | CPO | +51.63 | large | — |
| Constraint Violation Rate (%) | CBF Safety Layer | +51.63 | large | — |
| Constraint Violation Rate (%) | Fuzzy SAC | +0.56 | medium | -9.2% |
| Avg. FCSD | SAC | -0.09 | negligible | -0.4% |
| Avg. FCSD | PPO | -3.54 | large | -26.6% |
| Avg. FCSD | PPO-Lagrangian | -39.99 | large | -41.9% |
| Avg. FCSD | CPO | -39.91 | large | -41.9% |
| Avg. FCSD | CBF Safety Layer | -13.27 | large | -39.4% |
| Avg. FCSD | Fuzzy SAC | -0.63 | medium | -5.4% |
| Convergence Episode | SAC | -1.50 | large | +21.2% |
| Convergence Episode | PPO | -1.45 | large | +18.4% |
| Convergence Episode | PPO-Lagrangian | +1.68 | large | -43.6% |
| Convergence Episode | CPO | -0.92 | large | +20.5% |
| Convergence Episode | CBF Safety Layer | +3.08 | large | -235.5% |
| Convergence Episode | Fuzzy SAC | +0.13 | negligible | -7.0% |

### S3 Island Normal

| Metric | Baseline | g (HFG-SAC − baseline) | Magnitude | Rel. improvement |
|---|---|---|---|---|
| Daily Operating Cost (¥) | SAC | +0.60 | medium | -23.7% |
| Daily Operating Cost (¥) | PPO | -2.33 | large | +41.3% |
| Daily Operating Cost (¥) | PPO-Lagrangian | -4.90 | large | +48.4% |
| Daily Operating Cost (¥) | CPO | -5.02 | large | +51.9% |
| Daily Operating Cost (¥) | CBF Safety Layer | +1.77 | large | -52.0% |
| Daily Operating Cost (¥) | Fuzzy SAC | -0.57 | medium | +24.1% |
| Constraint Violation Rate (%) | SAC | -0.65 | medium | +22.3% |
| Constraint Violation Rate (%) | PPO | +2.13 | large | -1746.4% |
| Constraint Violation Rate (%) | PPO-Lagrangian | +2.26 | large | -12825.0% |
| Constraint Violation Rate (%) | CPO | +2.24 | large | -6104.0% |
| Constraint Violation Rate (%) | CBF Safety Layer | +0.57 | medium | -36.3% |
| Constraint Violation Rate (%) | Fuzzy SAC | +0.33 | small | -26.6% |
| Avg. FCSD | SAC | +1.07 | large | +9.0% |
| Avg. FCSD | PPO | -2.82 | large | -16.9% |
| Avg. FCSD | PPO-Lagrangian | -3.00 | large | -17.6% |
| Avg. FCSD | CPO | -2.97 | large | -17.5% |
| Avg. FCSD | CBF Safety Layer | -0.01 | negligible | -0.1% |
| Avg. FCSD | Fuzzy SAC | -0.37 | small | -3.8% |
| Convergence Episode | SAC | -0.07 | negligible | +5.3% |
| Convergence Episode | PPO | +0.65 | medium | -80.0% |
| Convergence Episode | PPO-Lagrangian | +0.65 | medium | -80.0% |
| Convergence Episode | CPO | +0.65 | medium | -80.0% |
| Convergence Episode | CBF Safety Layer | -0.61 | medium | +73.5% |
| Convergence Episode | Fuzzy SAC | +0.21 | small | -20.0% |

### S4 Island Extreme

| Metric | Baseline | g (HFG-SAC − baseline) | Magnitude | Rel. improvement |
|---|---|---|---|---|
| Daily Operating Cost (¥) | SAC | -0.58 | medium | +14.6% |
| Daily Operating Cost (¥) | PPO | -2.22 | large | +28.0% |
| Daily Operating Cost (¥) | PPO-Lagrangian | -1.52 | large | +22.1% |
| Daily Operating Cost (¥) | CPO | -8.43 | large | +27.1% |
| Daily Operating Cost (¥) | CBF Safety Layer | +0.53 | medium | -4.9% |
| Daily Operating Cost (¥) | Fuzzy SAC | +0.02 | negligible | -0.8% |
| Constraint Violation Rate (%) | SAC | +0.65 | medium | -0.1% |
| Constraint Violation Rate (%) | PPO | +11.30 | large | -126.7% |
| Constraint Violation Rate (%) | PPO-Lagrangian | +10.78 | large | -103.6% |
| Constraint Violation Rate (%) | CPO | +42.50 | large | -102.3% |
| Constraint Violation Rate (%) | CBF Safety Layer | +3.98 | large | -72.5% |
| Constraint Violation Rate (%) | Fuzzy SAC | +0.91 | large | -0.3% |
| Avg. FCSD | SAC | -0.19 | negligible | -0.7% |
| Avg. FCSD | PPO | -3.95 | large | -13.6% |
| Avg. FCSD | PPO-Lagrangian | -0.78 | medium | -7.6% |
| Avg. FCSD | CPO | -3.07 | large | -10.8% |
| Avg. FCSD | CBF Safety Layer | -1.82 | large | -8.2% |
| Avg. FCSD | Fuzzy SAC | +0.32 | small | +2.7% |
| Convergence Episode | SAC | +0.65 | medium | -26.7% |
| Convergence Episode | PPO | +0.65 | medium | -26.7% |
| Convergence Episode | PPO-Lagrangian | +0.65 | medium | -26.7% |
| Convergence Episode | CPO | +0.65 | medium | -26.7% |
| Convergence Episode | CBF Safety Layer | +0.65 | medium | -26.7% |
| Convergence Episode | Fuzzy SAC | +0.65 | medium | -26.7% |

## 3. Significance Tests (two-sided permutation test)

_Note: with n=3 seeds per condition, the minimum achievable two-sided p-value is 0.1, so no contrast can reach the conventional 0.05 threshold regardless of effect magnitude. Reported p-values are informational; effect sizes (Section 2) are the primary evidence at this seed count._

### S1 Grid Normal

| Metric | Baseline | p (raw) | Holm-adj. α | Sig. at 0.05? |
|---|---|---|---|---|
| Daily Operating Cost (¥) | SAC | 0.300 | 0.0500 | no |
| Daily Operating Cost (¥) | PPO | 0.200 | 0.0167 | no |
| Daily Operating Cost (¥) | PPO-Lagrangian | 0.100 | 0.0083 | no |
| Daily Operating Cost (¥) | CPO | 0.100 | 0.0100 | no |
| Daily Operating Cost (¥) | CBF Safety Layer | 0.100 | 0.0125 | no |
| Daily Operating Cost (¥) | Fuzzy SAC | 0.200 | 0.0250 | no |
| Constraint Violation Rate (%) | SAC | 0.500 | 0.0250 | no |
| Constraint Violation Rate (%) | PPO | 0.400 | 0.0083 | no |
| Constraint Violation Rate (%) | PPO-Lagrangian | 0.400 | 0.0100 | no |
| Constraint Violation Rate (%) | CPO | 0.400 | 0.0125 | no |
| Constraint Violation Rate (%) | CBF Safety Layer | 0.400 | 0.0167 | no |
| Constraint Violation Rate (%) | Fuzzy SAC | 1.000 | 0.0500 | no |
| Avg. FCSD | SAC | 0.600 | 0.0250 | no |
| Avg. FCSD | PPO | 0.400 | 0.0125 | no |
| Avg. FCSD | PPO-Lagrangian | 0.100 | 0.0083 | no |
| Avg. FCSD | CPO | 0.100 | 0.0100 | no |
| Avg. FCSD | CBF Safety Layer | 0.400 | 0.0167 | no |
| Avg. FCSD | Fuzzy SAC | 0.900 | 0.0500 | no |
| Convergence Episode | SAC | 0.400 | 0.0100 | no |
| Convergence Episode | PPO | 0.400 | 0.0125 | no |
| Convergence Episode | PPO-Lagrangian | 0.400 | 0.0167 | no |
| Convergence Episode | CPO | 0.400 | 0.0250 | no |
| Convergence Episode | CBF Safety Layer | 0.100 | 0.0083 | no |
| Convergence Episode | Fuzzy SAC | 0.700 | 0.0500 | no |

### S2 Grid Extreme

| Metric | Baseline | p (raw) | Holm-adj. α | Sig. at 0.05? |
|---|---|---|---|---|
| Daily Operating Cost (¥) | SAC | 1.000 | 0.0500 | no |
| Daily Operating Cost (¥) | PPO | 0.100 | 0.0083 | no |
| Daily Operating Cost (¥) | PPO-Lagrangian | 0.100 | 0.0100 | no |
| Daily Operating Cost (¥) | CPO | 0.100 | 0.0125 | no |
| Daily Operating Cost (¥) | CBF Safety Layer | 0.100 | 0.0167 | no |
| Daily Operating Cost (¥) | Fuzzy SAC | 0.500 | 0.0250 | no |
| Constraint Violation Rate (%) | SAC | 1.000 | 0.0250 | no |
| Constraint Violation Rate (%) | PPO | 0.100 | 0.0083 | no |
| Constraint Violation Rate (%) | PPO-Lagrangian | 0.100 | 0.0100 | no |
| Constraint Violation Rate (%) | CPO | 0.100 | 0.0125 | no |
| Constraint Violation Rate (%) | CBF Safety Layer | 0.100 | 0.0167 | no |
| Constraint Violation Rate (%) | Fuzzy SAC | 1.000 | 0.0500 | no |
| Avg. FCSD | SAC | 1.000 | 0.0500 | no |
| Avg. FCSD | PPO | 0.100 | 0.0083 | no |
| Avg. FCSD | PPO-Lagrangian | 0.100 | 0.0100 | no |
| Avg. FCSD | CPO | 0.100 | 0.0125 | no |
| Avg. FCSD | CBF Safety Layer | 0.100 | 0.0167 | no |
| Avg. FCSD | Fuzzy SAC | 0.500 | 0.0250 | no |
| Convergence Episode | SAC | 0.200 | 0.0125 | no |
| Convergence Episode | PPO | 0.200 | 0.0167 | no |
| Convergence Episode | PPO-Lagrangian | 0.100 | 0.0083 | no |
| Convergence Episode | CPO | 0.300 | 0.0250 | no |
| Convergence Episode | CBF Safety Layer | 0.100 | 0.0100 | no |
| Convergence Episode | Fuzzy SAC | 0.900 | 0.0500 | no |

### S3 Island Normal

| Metric | Baseline | p (raw) | Holm-adj. α | Sig. at 0.05? |
|---|---|---|---|---|
| Daily Operating Cost (¥) | SAC | 0.500 | 0.0250 | no |
| Daily Operating Cost (¥) | PPO | 0.100 | 0.0083 | no |
| Daily Operating Cost (¥) | PPO-Lagrangian | 0.100 | 0.0100 | no |
| Daily Operating Cost (¥) | CPO | 0.100 | 0.0125 | no |
| Daily Operating Cost (¥) | CBF Safety Layer | 0.100 | 0.0167 | no |
| Daily Operating Cost (¥) | Fuzzy SAC | 0.600 | 0.0500 | no |
| Constraint Violation Rate (%) | SAC | 1.000 | 0.0500 | no |
| Constraint Violation Rate (%) | PPO | 0.100 | 0.0083 | no |
| Constraint Violation Rate (%) | PPO-Lagrangian | 0.100 | 0.0100 | no |
| Constraint Violation Rate (%) | CPO | 0.100 | 0.0125 | no |
| Constraint Violation Rate (%) | CBF Safety Layer | 0.400 | 0.0167 | no |
| Constraint Violation Rate (%) | Fuzzy SAC | 0.500 | 0.0250 | no |
| Avg. FCSD | SAC | 0.200 | 0.0167 | no |
| Avg. FCSD | PPO | 0.100 | 0.0083 | no |
| Avg. FCSD | PPO-Lagrangian | 0.100 | 0.0100 | no |
| Avg. FCSD | CPO | 0.100 | 0.0125 | no |
| Avg. FCSD | CBF Safety Layer | 1.000 | 0.0500 | no |
| Avg. FCSD | Fuzzy SAC | 0.500 | 0.0250 | no |
| Convergence Episode | SAC | 1.000 | 0.0100 | no |
| Convergence Episode | PPO | 1.000 | 0.0125 | no |
| Convergence Episode | PPO-Lagrangian | 1.000 | 0.0167 | no |
| Convergence Episode | CPO | 1.000 | 0.0250 | no |
| Convergence Episode | CBF Safety Layer | 0.700 | 0.0083 | no |
| Convergence Episode | Fuzzy SAC | 1.000 | 0.0500 | no |

### S4 Island Extreme

| Metric | Baseline | p (raw) | Holm-adj. α | Sig. at 0.05? |
|---|---|---|---|---|
| Daily Operating Cost (¥) | SAC | 0.400 | 0.0167 | no |
| Daily Operating Cost (¥) | PPO | 0.100 | 0.0083 | no |
| Daily Operating Cost (¥) | PPO-Lagrangian | 0.100 | 0.0100 | no |
| Daily Operating Cost (¥) | CPO | 0.100 | 0.0125 | no |
| Daily Operating Cost (¥) | CBF Safety Layer | 0.700 | 0.0250 | no |
| Daily Operating Cost (¥) | Fuzzy SAC | 1.000 | 0.0500 | no |
| Constraint Violation Rate (%) | SAC | 1.000 | 0.0500 | no |
| Constraint Violation Rate (%) | PPO | 0.100 | 0.0083 | no |
| Constraint Violation Rate (%) | PPO-Lagrangian | 0.100 | 0.0100 | no |
| Constraint Violation Rate (%) | CPO | 0.100 | 0.0125 | no |
| Constraint Violation Rate (%) | CBF Safety Layer | 0.100 | 0.0167 | no |
| Constraint Violation Rate (%) | Fuzzy SAC | 0.600 | 0.0250 | no |
| Avg. FCSD | SAC | 0.900 | 0.0500 | no |
| Avg. FCSD | PPO | 0.100 | 0.0083 | no |
| Avg. FCSD | PPO-Lagrangian | 0.400 | 0.0167 | no |
| Avg. FCSD | CPO | 0.100 | 0.0100 | no |
| Avg. FCSD | CBF Safety Layer | 0.100 | 0.0125 | no |
| Avg. FCSD | Fuzzy SAC | 0.800 | 0.0250 | no |
| Convergence Episode | SAC | 1.000 | 0.0083 | no |
| Convergence Episode | PPO | 1.000 | 0.0100 | no |
| Convergence Episode | PPO-Lagrangian | 1.000 | 0.0125 | no |
| Convergence Episode | CPO | 1.000 | 0.0167 | no |
| Convergence Episode | CBF Safety Layer | 1.000 | 0.0250 | no |
| Convergence Episode | Fuzzy SAC | 1.000 | 0.0500 | no |

## 4. Limitations and Recommendations

- **Seed count (n=3) is insufficient for inferential power.** The exact permutation test cannot yield p < 0.1 with three independent seeds per condition. Effect sizes are therefore unstable; a single outlier seed can flip a Hedges' g sign. Treat all cross-method claims as directional until ≥5 seeds (ideally 10) are run.
- **Pooled-variance assumption** in Hedges' g is coarse at n=3; report alongside raw means to avoid over-interpretation.
- **Multiple comparisons** were controlled via Holm-Bonferroni (6 baselines × 4 metrics × 4 scenarios), which is conservative. Given the n=3 floor on p, this mainly guards the (rare) case where larger seed counts are added later.
- **Model baselines (MILP/MPC)** are deterministic single runs and are excluded from inferential tests; they serve as a cost lower-bound / safety reference only.
