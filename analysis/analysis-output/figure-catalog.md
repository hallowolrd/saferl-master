# Figure Catalog

All figures in `figures/`. PDF for submission, PNG for preview.

---

## Figure 1 — `figure-01-main-comparison.pdf/png`

- **Purpose**: Headline cost–safety comparison across all 7 algorithms and 4 scenarios.
- **Panels**: (a) mean cost per day with 95% CI; (b) mean violation rate with 95% CI. Red dotted line at 5% safety threshold.
- **Data source**: 84 JSON result files (`outputs/experiments/results/{algo}_{sc}_s{seed}.json`), fields `cost_per_day`, `violation_rate`.
- **Key observation**: HFG-SAC (red) is the only algorithm that is simultaneously low-cost on S1/S2/S3 and ≤5% violation on S1/S2/S3. On S4, red bar drops to ~33% — the only scenario where HFG-SAC does not meet the safety bar.
- **Interpretation checklist**:
  - Reader should notice HFG-SAC cost bars are among the shortest on S1/S2/S3 while violation bars are zero/near-zero.
  - Reader should notice S4 red bar at ~33% — the failure case.
  - Reader should note PPO-Lag/CPO/SL achieve 0% violation on S1/S2 but at 1.5–3× the cost.
- **Caption draft**: "Operational cost (a) and hard-constraint violation rate (b) for seven RL algorithms across four microgrid scenarios. Bars show mean over 3 seeds with 95% CI error bars. The red dotted line in (b) marks the 5% safety threshold."

---

## Figure 2 — `figure-02-training-curves.pdf/png`

- **Purpose**: Show training dynamics — episode cost convergence for PPO, SL, Fuzzy-SAC, HFG-SAC.
- **Data**: `episode_costs` field, 200 episodes, mean ± std across 3 seeds.
- **Key observation**: HFG-SAC converges fastest and most stably; Fuzzy-SAC diverges on S3/S4; SL oscillates.
- **Caption draft**: "Episode-level operational cost during training (mean ± std over 3 seeds). HFG-SAC (red) converges within ~50 episodes with the lowest variance; Fuzzy-SAC (teal) diverges on islanded scenarios."

---

## Figure 3 — `figure-03-violation-trajectory.pdf/png`

- **Purpose**: Episode-level violation rate trajectory on S3 (islanded normal) and S4 (islanded extreme) — the safety-critical scenarios.
- **Data**: `episode_violation_rates` field.
- **Key observation (left, S3)**: HFG-SAC drops from ~70% to ~5% within 10 episodes and stays below the 5% threshold. Every other method stays above 15% (PPO-Lag/PPO) or diverges (Fuzzy-SAC).
- **Key observation (right, S4)**: HFG-SAC drops from ~90% to ~30% and plateaus there — the structural ceiling. No algorithm crosses below 30%.
- **Interpretation checklist**:
  - Why does this figure exist? To show that HFG-SAC's safety advantage is not a final-episode artifact but a training-wide property, and that S4 is a hard ceiling not a slow convergence.
  - What should the reader notice? The red curve on the right flattens at ~30% — it has converged, not stalled.
  - What changes? Motivates load-shedding / tighter shield as a structural fix, not more training.
- **Caption draft**: "Episode-level hard-constraint violation rate during training on (left) S3 islanded normal and (right) S4 islanded extreme. Mean ± std over 3 seeds. The red curve shows HFG-SAC converging below 5% on S3 but plateauing at ~30% on S4."

---

## Figure 4 — `figure-04-fcsd-heatmap.pdf/png`

- **Purpose**: Fuzzy Constraint Satisfaction Degree (FCSD) heatmap — softer safety signal than binary violation.
- **Data**: `avg_fcsd` field, mean across 3 seeds. Color scale RdYlGn from 0.5 (red) to 1.0 (green).
- **Key observation**: PPO-Lag/CPO hit FCSD ≈ 1.0 on S1/S2/S3 but drop to 0.81/0.84 on S4. HFG-SAC maintains 0.92–0.98 on S1/S2/S3 and 0.89 on S4 — more balanced. SAC and Fuzzy-SAC are red/orange on most scenarios.
- **Caption draft**: "Mean Fuzzy Constraint Satisfaction Degree (FCSD) across algorithms and scenarios (n=3 seeds). FCSD ∈ [0, 1], higher indicates softer constraint satisfaction margin."

---

## Figure 5 — `figure-05-pareto.pdf/png`

- **Purpose**: Cost-safety Pareto frontier — each point is one (algorithm, scenario) mean.
- **Markers**: circles S1, squares S2, triangles S3, diamonds S4. Colors by algorithm.
- **Key observation**: HFG-SAC points (red) cluster at the lower-left (low cost, low violation) for S1/S2/S3. S4 red diamond sits at ~33% violation — outside the Pareto frontier desired region.
- **Caption draft**: "Cost–safety Pareto scatter. Each marker is the mean of 3 seeds. Marker shape denotes scenario; color denotes algorithm. The red dotted line marks the 5% safety threshold."

---

## Known caveats

- n=3 seeds: error bars / CIs are wide; treat as illustrative.
- S4 per-constraint breakdown not yet logged (freq vs voltage vs SOC) — future work.
- MILP/MPC oracle baselines not included in these figures (cost units not directly comparable).
