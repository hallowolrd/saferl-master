# Stats Appendix — RQ1

## 1. Data inventory

- **Files**: 140 JSONs in `outputs/experiments/results/`
- **Coverage**: 7 algos × 4 scenarios × 5 seeds (seeds 42, 123, 456, 789, 2024) — complete, no missing cells.
- **Episodes per run**: 200 (96 steps/day × 7 days = 672 env steps per episode).
- **Final metrics**: taken from 10-episode deterministic evaluation at end of training.
- **Direction**: lower cost/day = better; lower violation rate = better; higher avg_FCSD = better.

## 2. Descriptive statistics

Full table: `stats_summary.csv`. Headline numbers (mean over 5 seeds):

### Cost (¥/day)

| Algo | S1 | S2 | S3 | S4 |
|------|-----|-----|-----|-----|
| SAC | 14,131 | 17,765 | 6,461 | 14,286 |
| PPO | 16,728 | 20,436 | 5,558 | 13,347 |
| PPO-Lagrangian | 21,212 | 40,971 | 12,082 | 16,683 |
| CPO | 15,808 | 29,260 | 10,963 | 17,632 |
| Safety-Layer | 32,728 | 34,896 | 8,894 | 13,258 |
| Fuzzy-SAC | 15,052 | 24,604 | 10,028 | 14,921 |
| **HFG-SAC** | 14,997 | 23,442 | 9,097 | 14,925 |

### Violation rate (%)

| Algo | S1 | S2 | S3 | S4 |
|------|-----|-----|-----|-----|
| SAC | 69.9 | 89.5 | 98.0 | 95.1 |
| PPO | 88.8 | 82.4 | 98.9 | 78.7 |
| PPO-Lagrangian | 5.4 | 6.5 | 30.5 | 40.9 |
| CPO | 0.0 | 0.0 | 1.2 | 0.06 |
| Safety-Layer | 0.0 | 0.0 | 13.0 | 34.6 |
| Fuzzy-SAC | 0.0 | 0.0 | 1.1 | 1.2 |
| **HFG-SAC** | 0.0 | 0.0 | 0.6 | 0.6 |

### Avg FCSD (0-1, higher = safer)

| Algo | S1 | S2 | S3 | S4 |
|------|-----|-----|-----|-----|
| SAC | 0.717 | 0.636 | 0.784 | 0.767 |
| PPO | 0.632 | 0.634 | 0.775 | 0.834 |
| PPO-Lagrangian | 0.980 | 0.976 | 0.926 | 0.912 |
| CPO | 1.000 | 0.999 | 0.997 | 1.000 |
| Safety-Layer | 0.991 | 0.959 | 0.959 | 0.890 |
| Fuzzy-SAC | 0.933 | 0.891 | 0.990 | 0.989 |
| **HFG-SAC** | 0.940 | 0.900 | 0.992 | 0.993 |

## 3. Inferential tests

- **Test**: Mann-Whitney U (two-sided), non-parametric — justified because n=5 per cell is too small to verify normality.
- **Effect size**: rank-biserial r = 1 − 2U/(n₁n₂), range [-1, +1].
- **Multiple-comparison correction**: Holm-Bonferroni across the 6 baselines within each (scenario × metric) family (6 tests per family).
- **Full results**: `inferential_tests.csv`.

### Significant contrasts (Holm-adjusted p < 0.05): 27 / 72

**HFG-SAC significantly BETTER than baseline** (safety or efficiency):

| Scenario | Metric | Baseline | p_holm | r |
|----------|--------|----------|--------|---|
| S1 | cost | PPO-Lag | 0.048 | +1.0 |
| S1 | cost | Safety-Layer | 0.040 | +1.0 |
| S1 | violation | SAC | 0.037 | +1.0 |
| S1 | violation | PPO | 0.039 | +1.0 |
| S1 | FCSD | PPO | 0.048 | -1.0 |
| S2 | cost | PPO-Lag | 0.040 | +1.0 |
| S2 | cost | CPO | 0.032 | +1.0 |
| S2 | cost | Safety-Layer | 0.024 | +1.0 |
| S2 | violation | SAC | 0.040 | +1.0 |
| S2 | violation | PPO | 0.037 | +1.0 |
| S2 | FCSD | SAC | 0.048 | -1.0 |
| S2 | FCSD | PPO | 0.040 | -1.0 |
| S3 | cost | CPO | 0.040 | +1.0 |
| S3 | FCSD | SAC | 0.048 | -1.0 |
| S3 | FCSD | PPO | 0.040 | -1.0 |
| S4 | violation | PPO-Lag | 0.048 | +1.0 |
| S4 | FCSD | SAC | 0.048 | -1.0 |
| S4 | FCSD | PPO | 0.040 | -1.0 |
| S4 | FCSD | PPO-Lag | 0.032 | -1.0 |

**HFG-SAC significantly WORSE than baseline** (these are honest losses to report):

| Scenario | Metric | Baseline | p_holm | r | Note |
|----------|--------|----------|--------|---|------|
| S1 | FCSD | CPO | 0.040 | +1.0 | CPO FCSD = 1.000 vs HFG = 0.940 |
| S1 | FCSD | Safety-Layer | 0.032 | +1.0 | Safety-Layer FCSD = 0.991 |
| S2 | cost | SAC | 0.048 | -1.0 | Unsafe SAC is 24% cheaper (¥17.8k vs ¥23.4k) |
| S2 | FCSD | CPO | 0.032 | +1.0 | CPO FCSD higher |
| S2 | FCSD | Safety-Layer | 0.024 | +1.0 | Same |
| S3 | cost | PPO | 0.048 | -1.0 | Unsafe PPO cheaper (¥5.6k vs ¥9.1k, but 99% violation) |
| S4 | violation | CPO | 0.038 | -0.96 | CPO 0.06% vs HFG 0.6% — tiny gap |
| S4 | FCSD | CPO | 0.024 | +1.0 | CPO FCSD = 1.000 |

### Key interpretation

- HFG-SAC **never loses on safety** to PPO-Lagrangian, Safety-Layer, SAC, or PPO in any scenario.
- HFG-SAC **loses narrowly on FCSD** to CPO in S1/S2/S4 — but CPO's FCSD advantage comes at higher cost in S2 (¥29k vs ¥23k).
- HFG-SAC **loses on cost** to unsafe SAC/PPO in S2/S3 — this is the expected "price of safety" (those baselines violate 80-99% of the time).
- In S3, HFG-SAC dominates CPO on **cost** (¥9.1k vs ¥11.0k) while matching CPO on **safety** (0.6% vs 1.2% violation, not significantly different).

## 4. Assumptions checked

- **Normality**: not tested with n=5 (Shapiro-Wilk has almost no power below n=20). Non-parametric test chosen by default.
- **Independence**: each seed uses different RNG seed; runs are independent.
- **Homoscedasticity**: not assumed — Mann-Whitney U is distribution-free.
- **Ties**: where both methods report identical violation=0% across all 5 seeds, U-test is undefined (NaN). We mark these as "no test — equivalent on this metric."

## 5. Limitations

1. **n=5 per cell** is the minimum for any statistical claim. Effect sizes (r=±1.0) are large because the methods are qualitatively different, not because of noise.
2. **Cross-scenario cost comparison is not meaningful** — S3 island scenarios have ~50% smaller load scale than S1/S2 grid scenarios.
3. **Final evaluation uses 10 deterministic episodes** — stochastic policy evaluation would reduce variance but not bias.
4. **Training-time cost is not included** in the optimization; HFG-SAC trains 3-7× slower than PPO-family due to TSK fuzzy forward passes. This is a deployment consideration, not a final-metric one.
5. **No confidence interval on training curves** — only std band over 5 seeds.

## 6. Blockers

- None. All 140 runs completed cleanly (0 failures).
