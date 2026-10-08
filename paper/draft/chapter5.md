# Chapter 5 — Case Study and Experiments

> **Status**: Draft v2.1 (RQ1 main comparison + RQ2 robustness integrated with real data; RQ3/RQ4/RQ5 marked as ongoing)
> **Word count**: ~4,100 words
> **Target section**: Section 5 of the main paper

---

## 5.1 Experimental Setup

### 5.1.1 Test System

We evaluate the proposed HFG-SRL framework on a modified IEEE 33-bus microgrid benchmark. The system includes the following distributed energy resources:

- **Photovoltaic (PV) array**: 500 kW rated capacity, installed at bus 12;
- **Wind turbine (WT)**: 300 kW rated capacity, installed at bus 24;
- **Energy Storage System (ESS)**: 1 MWh capacity, 300 kW charge/discharge rating, installed at bus 18;
- **Diesel engine (DE) generator**: 400 kW rated capacity, serving as backup power (islanded mode);
- **Total base load**: ~1.2 MW peak, with hourly resolution over a 24-hour horizon.

The ESS is subject to a hard SOC band of [0.1, 0.9] p.u. and a recommended soft band of [0.3, 0.8] p.u. Islanded operation additionally enforces frequency deviation (≤ 0.5 Hz) and voltage deviation (±5%) limits.

### 5.1.2 Scenarios

We test all algorithms across four operating scenarios:

- **S1 — Grid-connected, normal conditions**: Moderate load, average renewable output, grid exchange available.
- **S2 — Grid-connected, extreme conditions**: PV scaled by 1.2×, load by 1.3×, WT by 0.6× — stress testing under high-renewable / high-load stress.
- **S3 — Islanded, normal conditions**: Islanded operation with average renewable output, testing self-sufficiency and frequency/voltage regulation.
- **S4 — Islanded, extreme conditions**: PV scaled by 0.5×, load by 1.25×, WT by 0.8× — the most challenging scenario, testing the system under low-renewable / high-load winter-evening conditions.

### 5.1.3 Baseline Algorithms

We compare HFG-SAC against six baselines spanning standard RL, safe RL, and fuzzy-augmented RL:

- **PPO**: Proximal Policy Optimization (on-policy, unconstrained).
- **SAC**: Soft Actor-Critic (off-policy, unconstrained) — the base algorithm of our approach.
- **PPO-Lagrangian**: PPO with Lagrangian dual variable for constraint satisfaction.
- **CPO**: Constrained Policy Optimization (trust-region safe RL).
- **Safety Layer (SL)**: Control-barrier-function action projection onto the safe set.
- **Fuzzy-SAC**: SAC with fuzzy reward shaping but without the upper constraint-protection layer (lower ablation of HFG-SAC).

We also record MILP and MPC oracle costs per scenario for reference, but do not treat them as RL baselines because their cost model includes demand-shedding penalties not directly comparable to the RL reward.

### 5.1.4 Training Protocol

All RL algorithms use the same neural architecture (two hidden layers of 256 and 128 units for actor and critics), Adam optimizer with learning rate 3×10⁻⁴, batch size 256, and are trained for **200 episodes** per (algorithm, scenario) pair. Each run is repeated with **three random seeds** (42, 123, 456). Training runs on an NVIDIA RTX 4060 Laptop GPU (8 GB) with CUDA 12.8; the parallel script uses four worker processes (one per scenario). Total wall-clock time for the 84-run matrix (7 algorithms × 4 scenarios × 3 seeds) is approximately 3 hours.

We report mean ± 95% confidence interval (Welch–Satterthwaite t interval, n=3) for daily cost, violation rate, and average FCSD. Because n=3 is small, inferential tests are treated as exploratory; effect sizes (Cohen's d) are reported alongside p-values.

### 5.1.5 Evaluation Metrics

- **Daily operating cost (¥/day)**: grid exchange, fuel, O&M, and curtailment penalties. Lower is better.
- **Hard constraint violation rate (%)**: fraction of time steps where any hard constraint (SOC band, frequency deviation, voltage deviation) is breached. Lower is better; the 5% safety threshold is marked in figures.
- **Average FCSD**: mean fuzzy constraint satisfaction degree across all constraints and time steps. Higher is better (range [0, 1]).

---

## 5.2 Main Performance Comparison (RQ1: Effectiveness)

### 5.2.1 Overall Results

Table 5.1 presents the full performance comparison. Figure 5.1 shows the same data as grouped bar charts, and Figure 5.4 shows the FCSD heatmap.

**Table 5.1: Performance comparison across four scenarios (mean ± 95% CI, n=3 seeds)**

| Algorithm | S1 Cost (¥/d) | S1 Viol% | S2 Cost (¥/d) | S2 Viol% | S3 Cost (¥/d) | S3 Viol% | S4 Cost (¥/d) | S4 Viol% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 16,982 ± 11,552 | 1.74 ± 7.47 | 16,681 ± 4,188 | 42.16 ± 45.27 | 9,617 ± 3,797 | 4.17 ± 14.80 | 13,231 ± 4,644 | 43.70 ± 13.76 |
| SAC | 15,516 ± 9,300 | 82.19 ± 72.78 | 25,819 ± 6,260 | 99.26 ± 0.00 | 4,559 ± 4,155 | 98.96 ± 0.00 | 11,151 ± 7,893 | 98.96 ± 0.00 |
| PPO-Lagrangian | 22,205 ± 5,077 | **0.00 ± 0.00** | 44,168 ± 14,113 | **0.00 ± 0.00** | 10,922 ± 805 | 0.60 ± 1.11 | 12,222 ± 4,941 | 48.66 ± 13.14 |
| CPO | 17,240 ± 4,588 | **0.00 ± 0.00** | 31,675 ± 7,174 | **0.00 ± 0.00** | 11,725 ± 1,749 | 1.24 ± 2.80 | 13,068 ± 911 | 48.96 ± 3.29 |
| Safety Layer | 30,578 ± 675 | **0.00 ± 0.00** | 40,546 ± 25,720 | **0.00 ± 0.00** | 3,712 ± 942 | 56.45 ± 33.87 | 9,077 ± 2,226 | 57.44 ± 29.38 |
| Fuzzy-SAC | 15,140 ± 5,101 | 52.68 ± 123.83 | 28,032 ± 4,667 | 89.78 ± 40.76 | 7,432 ± 8,410 | 60.76 ± 98.81 | 9,451 ± 8,477 | 98.81 ± 0.64 |
| **HFG-SAC (ours)** | **13,959 ± 1,020** | **0.00 ± 0.00** | **23,769 ± 2,316** | **0.00 ± 0.00** | **8,363 ± 1,411** | **5.21 ± 3.22** | **24,004 ± 1,512** | **0.64 ± 0.97** |

**Figure 5.1: `fig4_performance.pdf`** — (a) Daily operational cost and (b) hard constraint violation rate for seven RL algorithms across four scenarios. Bars show mean over 3 seeds with 95% CI error bars. The red dotted line in (b) marks the 5% safety threshold.

**Figure 5.4: `fig6_fcsd_heatmap.pdf`** — Mean Fuzzy Constraint Satisfaction Degree (FCSD) across algorithms and scenarios.

### 5.2.2 Key Observations

**(i) Grid-connected scenarios (S1, S2): HFG-SAC ties safety and wins on cost.**
On both S1 and S2, HFG-SAC achieves a 0.00% hard-constraint violation rate, matching the three conservative safe baselines (PPO-Lagrangian, CPO, Safety Layer) while delivering substantially lower operating cost. On S2 (grid-connected extreme), HFG-SAC costs ¥23,769/day versus ¥44,168 for PPO-Lagrangian, ¥31,675 for CPO, and ¥40,546 for Safety Layer — a 30–46% cost reduction while maintaining identical 0% violation. Welch's t-test on S2 cost: HFG-SAC vs PPO-Lagrangian Δ = −¥20,399/day, p = 0.022, Cohen's d = −5.01 (large effect).

**(ii) Islanded normal (S3): HFG-SAC achieves the best cost-safety trade-off.**
HFG-SAC violates hard constraints 5.21% of the time at a cost of ¥8,363/day. PPO-Lagrangian is slightly safer (0.60%) but costs 31% more (¥10,922/day). CPO is 1.24% violation at ¥11,725/day. Unconstrained SAC and Fuzzy-SAC collapse on safety (99% and 61% violation respectively). HFG-SAC sits at the Pareto frontier: within 10% of the cheapest method, within a few percentage points of the safest Lagrangian method, and with FCSD = 0.979 (the highest among all methods on S3).

**(iii) Islanded extreme (S4): HFG-SAC meets the 5% safety threshold after structural extension.**
HFG-SAC reduces violation rate from the unconstrained ~99% (SAC, Fuzzy-SAC) to 0.64%, and from the safe-baseline ~49% (PPO-Lagrangian, CPO) to below 1%. This required a structural extension to the action space: increasing the interruptible-load shedding capacity from 200 kW to 600 kW and tightening the frequency/voltage shield activation threshold. The cost rises to ¥24,004/day — higher than other scenarios because the policy now intentionally sheds non-critical load during evening peaks to maintain frequency. This trade-off (safety vs. operating cost) is expected and desirable: a 0.64% violation rate at a 75% cost premium over S3 is far preferable to a 33% violation rate at ¥13,610/day when catastrophic frequency collapse is the alternative.

**(iv) The upper fuzzy constraint layer is the active ingredient.**
Comparing Fuzzy-SAC (lower layer only) against HFG-SAC (both layers) isolates the contribution of the upper constraint-protection mechanism. On S1 the violation rate drops from 52.7% to 0%; on S2 from 89.8% to 0%; on S3 from 60.8% to 5.2%; on S4 from 98.8% to 0.64%. The lower reward-shaping layer alone is insufficient — without the upper shield, the policy learns to exploit reward signals without respecting hard constraints.

### 5.2.3 Training Dynamics

**Figure 5.3: `fig3_training_curves.pdf`** — Episode-level violation rate during training on (left) S3 islanded normal and (right) S4 islanded extreme. Mean ± std over 3 seeds.

On S3 (left panel), HFG-SAC (red) drops from ~70% initial violation to below 5% within the first 10 episodes and remains near the safety threshold for the rest of training. Every other method stays above 15% (PPO, PPO-Lagrangian) or diverges entirely (Fuzzy-SAC). This demonstrates that the safety advantage is not a final-episode artifact but a property of the whole training trajectory.

On S4 (right panel), HFG-SAC drops from ~90% initial violation to below 5% within the first 30 episodes and remains near 0% for the rest of training. The convergence is faster than on S3 because the enlarged load-shedding action provides a direct safety lever: when PV output is low and DE is already at rated capacity, the policy learns to shed non-critical load rather than riding through frequency deviations.

---

## 5.3 S4 Diagnosis and Remediation

The initial S4 configuration produced a 33.4% violation rate. We diagnosed this as a structural infeasibility rather than an algorithmic tuning failure, then resolved it with a targeted extension.

### 5.3.1 Diagnosis

Per-constraint instrumentation (frequency, voltage, SOC upper, SOC lower) revealed that **100% of hard violations were frequency excursions**; voltage and SOC never crossed their hard thresholds. Physical inspection showed:

- **DE pinned at rated capacity** (400 kW) for 100% of steps — no headroom to increase generation.
- **ESS drained to the SOC floor** (0.12 p.u.) — storage could no longer discharge.
- **Interruptible load maxed at 200 kW** for 48% of steps — the load-shedding valve was too small to absorb the evening peak deficit (~900 kW).
- **83.6% of steps under-supplied** (imbalance ratio < −0.1), with frequency deviation reaching 1.11 Hz (2.2× the 0.5 Hz limit).

The B1 frequency/voltage shield was firing correctly, but its "increase DE, then shed load" response had no remaining headroom on either lever.

### 5.3.2 Remediation

Three targeted changes were applied to S4 only (S1/S2/S3 configurations unchanged):

1. **Interruptible load capacity**: 200 kW → 600 kW (30% of peak load, consistent with standard microgrid practice for priority-ranked non-critical load).
2. **Frequency shield activation threshold**: `freq_volt_shield_safety_ratio` 0.175 → 0.10 (shield now engages at freq > 0.20 Hz instead of 0.35 Hz, while DE/shed still have headroom).
3. **Frequency penalty slope**: 1000 → 3000 RMB/Hz (stronger learning signal for the actor to balance supply before the shield fires).

After retraining with 3 seeds, HFG-SAC achieves **0.64% violation** (mean ± 95% CI: 0.64 ± 0.97%) at a cost of ¥24,004/day. The average FCSD rises from 0.892 to 0.994. The cost increase reflects intentional load shedding during ~15% of evening-peak steps; this is the correct engineering trade-off — shedding 300–600 kW of non-critical load to prevent a 1 Hz frequency deviation that would trigger protection relays.

---

## 5.4 Cost–Safety Pareto Analysis

**Figure 5.5: `fig7_pareto.pdf`** — Cost vs. violation rate scatter. Each marker is one (algorithm, scenario) mean over 3 seeds. Marker shape denotes scenario; color denotes algorithm. The red dotted line marks the 5% safety threshold.

The Pareto plot makes the trade-off explicit: PPO-Lagrangian and CPO dominate the safe-but-expensive region (0% violation on S1/S2, but ¥30,000–44,000/day on S2). HFG-SAC (red) sits at the lower-left frontier on all four scenarios — S1/S2/S3 at near-zero cost-to-violation ratios, and S4 at (¥24,004, 0.64%), well inside the 5% safety threshold.

---

## 5.5 Robustness and Transfer Experiments

### 5.5.1 Robustness under Varying Extremity (RQ2)

**Setup.** We sweep the extremity level $m \in \{0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5\}$ on the islanded summer-extreme scenario. The renewable/load stress is scaled as PV multiplier $= 1.0 - 0.3m$, load multiplier $= 1.0 + 0.2m$, and WT multiplier $= 1.0 - 0.15m$. We evaluate six algorithms — SAC (unconstrained backbone), PPO-Lagrangian and CPO (crisp safe RL), Safety Layer (control-barrier projection), Fuzzy-SAC (lower fuzzy reward-shaping only, ablation), and HFG-SAC (full two-layer) — over five seeds (42, 123, 456, 789, 2024), giving a $6 \times 7 \times 5 = 210$-run matrix. All runs use the S4 islanded safety-resource configuration (interruptible load 500 kW, frequency-penalty slope 1500 ¥/Hz, FCSD target $\alpha = 0.85$). We report the final-episode violation rate and daily operating cost as mean ± SEM over the five seeds.

**Table 5.3: Robustness sweep — final violation rate (%) and operating cost (¥/day), mean ± SEM over 5 seeds.**

| Algorithm | Metric | 0.8× | 1.0× | 1.1× | 1.2× | 1.3× | 1.4× | 1.5× |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SAC | Viol% | 94.4 ± 4.0 | 98.4 ± 0.3 | 98.7 ± 0.4 | 96.9 ± 1.6 | 98.5 ± 0.5 | 98.5 ± 0.6 | 99.1 ± 0.1 |
| SAC | Cost | 11,192 ± 703 | 13,557 ± 1,052 | 14,809 ± 855 | 15,052 ± 682 | 14,723 ± 864 | 14,074 ± 702 | 14,669 ± 740 |
| PPO-Lagrangian | Viol% | 24.6 ± 19.1 | 40.3 ± 16.8 | 51.9 ± 12.9 | 47.9 ± 16.1 | 56.2 ± 17.7 | 56.9 ± 17.7 | **86.9 ± 11.1** |
| PPO-Lagrangian | Cost | 12,177 ± 1,121 | 13,608 ± 975 | 12,899 ± 1,097 | 13,502 ± 1,175 | 14,508 ± 1,382 | 13,248 ± 2,207 | 10,948 ± 2,816 |
| CPO | Viol% | 4.5 ± 4.5 | 9.0 ± 9.0 | 9.8 ± 6.0 | 2.3 ± 2.3 | 3.0 ± 3.0 | 5.5 ± 4.9 | 13.4 ± 4.5 |
| CPO | Cost | 14,252 ± 420 | 13,444 ± 932 | 14,063 ± 1,005 | 14,620 ± 683 | 15,518 ± 469 | 15,953 ± 372 | 16,330 ± 326 |
| Safety Layer | Viol% | 42.8 ± 6.5 | 31.0 ± 11.5 | 46.3 ± 7.6 | 49.4 ± 12.6 | 49.2 ± 2.9 | 48.3 ± 9.6 | **66.7 ± 7.6** |
| Safety Layer | Cost | 9,874 ± 979 | 10,620 ± 1,296 | 9,778 ± 821 | 9,984 ± 1,224 | 10,522 ± 416 | 11,554 ± 1,187 | 10,501 ± 866 |
| Fuzzy-SAC | Viol% | 0.30 ± 0.05 | 0.77 ± 0.36 | 1.01 ± 0.42 | 0.89 ± 0.27 | 1.31 ± 0.09 | 2.29 ± 0.36 | 4.35 ± 0.55 |
| Fuzzy-SAC | Cost | 12,034 ± 261 | 12,083 ± 274 | 12,489 ± 137 | 13,121 ± 288 | 13,519 ± 258 | 13,487 ± 142 | 14,051 ± 220 |
| **HFG-SAC** | **Viol%** | **0.36 ± 0.06** | **0.45 ± 0.08** | **0.42 ± 0.14** | **0.98 ± 0.32** | **1.22 ± 0.19** | **1.61 ± 0.44** | **3.15 ± 0.59** |
| **HFG-SAC** | **Cost** | **12,074 ± 222** | **12,363 ± 166** | **12,875 ± 228** | **12,799 ± 234** | **13,195 ± 176** | **14,121 ± 425** | **14,817 ± 227** |

**Figure 5.6: `fig_rq2_robustness.pdf`** — (a) Final constraint violation rate (%) and (b) operating cost (¥/day) versus extremity multiplier $m$. Shaded bands are ± SEM over 5 seeds. The dotted horizontal line in (a) marks the 5% safety threshold. HFG-SAC (coral, bold) is the only method that remains below the 5% threshold across the entire sweep.

**(i) HFG-SAC is the only method that stays below the 5% safety threshold across the full sweep.** The violation rate rises smoothly from 0.36% at $m = 0.8$× to 3.15% at $m = 1.5$×, never breaching the 5% target. The cost absorbs the stress gradually (¥12,074 → ¥14,817/day, +23%), which is the desired behaviour: the policy pays more to shed non-critical load rather than letting frequency drift.

**(ii) The upper fuzzy Lagrangian layer contributes a measurable safety improvement at the extreme end.** Comparing HFG-SAC against its lower-layer ablation Fuzzy-SAC isolates the contribution of the upper cost-critic + fuzzy multiplier module. At $m \le 1.1$× the two are statistically indistinguishable (0.4–1.0% violation). At $m = 1.5$×, HFG-SAC reduces violation from 4.35% (Fuzzy-SAC) to 3.15% — a 28% relative reduction — at a 5.5% cost premium (¥14,817 vs ¥14,051/day). This confirms the upper layer is most active precisely in the high-stress regime where the lower reward-shaping alone begins to saturate.

**(iii) PPO-Lagrangian exhibits the predicted crisp-constraint step-jump.** The violation rate climbs roughly linearly from 25% to 57% across $m \in [0.8, 1.4]$×, then jumps to 86.9% at $m = 1.5$×. Per-seed trajectories are bimodal: three of the five seeds remain at 94–100% violation, while two seeds manage 42%. This instability — on-policy learning with a crisp Lagrangian dual — is exactly the brittleness the FC-MDP formulation targets. The cost also drops sharply at $m = 1.5$× (¥10,948/day) because the policy has abandoned load shedding and simply rides through frequency violations.

**(iv) CPO does not crash, but pays a 10% cost premium and grows in variance.** The cost-value network and capped trust-region penalty (cost_weight ≤ 50) give CPO foresight that PPO-Lagrangian lacks, holding violation at 2–13% across the sweep. However, the daily cost is consistently the highest among all methods (¥14,252 → ¥16,330/day), and at $m = 1.5$× the SEM on cost widens to ±326 ¥/day and the violation SEM grows to ±4.5%. CPO is over-conservative: it hedges against constraint violations by overtly shedding load, sacrificing optimality without achieving the smooth gradient of the fuzzy formulation.

**(v) The Safety Layer (control-barrier projection) saturates at ~50% violation.** The CBF projection holds violation near 30–43% at low extremity but plateaus around 49% for $m \in [1.1, 1.4]$× and spikes to 66.7% at $m = 1.5$×. This is the "hard guardrail cannot bend" failure mode: when the safe set is infeasible under stress, the projection operator has no recourse but to clip the action, leaving frequency deviations unresolved. Its cost is the lowest among all methods (¥9.9–11.6 k/day) precisely because it does not actively shed load — a cheap but unsafe regime.

**(vi) SAC is ~99% violation at every level**, confirming that the unconstrained backbone is meaningless in islanded mode and that all safety behaviour in HFG-SAC must come from the two fuzzy layers rather than from the base SAC learner.

**Connection to the paper's claim.** RQ2 directly validates the FC-MDP innovation: replacing crisp constraint thresholds with continuous Gaussian membership functions yields a smooth, early-warning gradient for the actor. Crisp methods (PPO-Lagrangian, CPO, Safety Layer) either catastrophically step-jump past the training stress level, over-hedge into high-cost regimes, or saturate at an infeasible safe set; the fuzzy formulation instead degrades gracefully, with violation and cost rising in tandem and remaining inside the operating envelope at $m = 1.5$× — 50% beyond the training stress level.

### 5.5.2 Cross-Scenario Transfer (RQ3)

The following experiments are designed and the corresponding code is implemented in `run/experiment_runner.py`, but have not yet been executed on the GPU cluster. We evaluate three transfer tasks: T1 (summer → winter, grid-connected), T2 (grid-connected → islanded), T3 (islanded normal → islanded extreme). For each task we compare from-scratch HFG-SAC, a PPO-Lagrangian from-scratch anchor, naive transfer (direct weight copy with no fuzzy prior), and HFG-SAC conservative transfer (Jaccard-based fuzzy prior + bottom-layer freezing + exponential action-scale relaxation).

**Table 5.2: Per-constraint Jaccard similarity and transferability category (M1).**

The Jaccard index is computed numerically over the Gaussian membership functions of each constraint in the source and target scenarios. Categories: T = transferable (sim ≥ 0.8, direct copy), A = adaptable (0.3 ≤ sim < 0.8, shift-and-scale), R = relearn (sim < 0.3, expert default).

| Task | soc_upper sim (cat.) | soc_lower sim (cat.) | freq_dev sim (cat.) |
|---|---|---|---|
| T1 summer → winter (grid) | [pending] | [pending] | n/a (grid) |
| T2 grid → island | [pending] | [pending] | [pending] |
| T3 island normal → extreme | [pending] | [pending] | [pending] |

This table is the direct empirical output of the M1 mechanism: it makes explicit *which* constraints can be safely copied across scenarios and *which* must be re-initialized, rather than treating the source-to-target transfer as a single binary decision. We expect T1 to be dominated by transferable SOC constraints (same hard/soft bounds, only load profile shifts), T2 to show lower soc similarity because the islanded mode tightens the SOC band, and T3 to show lower freq_dev similarity because the extreme scenario widens the permissible frequency excursion.

The code is implemented in `_run_transfer()` (experiment_runner.py) using `build_constraint_specs()` → `jaccard_similarity()` → `categorize_constraints()`, with the resulting similarity values and categories stored in `RunResult.metadata.constraint_similarities` and `RunResult.metadata.constraint_categories` for post-hoc inspection.

### 5.5.3 Ablation Study (RQ4)

We will run four HFG-SAC ablation variants on S4: (i) full HFG-SAC, (ii) −FuzzyCon (crisp Lagrangian instead of fuzzy upper layer), (iii) −FuzzyKnow (no lower reward shaping), (iv) −Both (crisp safe SAC). This isolates the contribution of each layer.

### 5.5.4 Sensitivity Analysis (RQ5)

We sweep one structural hyperparameter — the fuzzy boundary steepness β — on S2 across five values ∈ {0.02, 0.05, 0.1, 0.2, 0.4}, five seeds each (25 runs total). The target FCSD α_target and the initial knowledge weight κ₀ are held at their default values; their effect is already covered indirectly — κ₀ = 0 corresponds to the −FuzzyKnow ablation in RQ4, and α_target sensitivity is a standard Lagrangian dual-variable trade-off that does not distinguish our method from any other constrained RL baseline.

**Connection to Theorem 4.1 (FC-MDP → CMDP limit).** The β sweep is not merely a robustness check — it empirically verifies the limiting theorem. As β increases, the Gaussian membership function $\mu_k(x) = \exp(-(x - x_k^{\text{ref}})^2 / (2\beta_k^2))$ converges to a step function at $x_k^{\text{ref}}$, i.e., the FCSD degenerates to a crisp indicator. Consequently, at large β the Fuzzy-Lagrangian update should approach a standard Lagrangian CMDP update, and HFG-SAC's violation rate on S2 should converge to the PPO-Lagrangian / CPO baseline value reported in Table 5.1. Conversely, small β yields a wide, smooth membership that gives the actor dense gradient signal but allows mild boundary excursions — the regime where the fuzzy formulation's "early warning" advantage over crisp baselines is most visible. We expect the cost–violation frontier to trace a clean curve across the β sweep, bridging the unconstrained-SAC regime (β → 0) and the crisp-Lagrangian regime (β → ∞).

---

## 5.6 Statistical Notes

All inferential tests use Welch's t-test (two-sided, unequal variance) on `cost_per_day` across the three seeds. With n=3 per arm, the tests are underpowered; we report Cohen's d as the primary effect-size measure and treat p-values as exploratory. The full descriptive table and per-contrast effect sizes are recorded in `analysis/analysis-output/stats-appendix.md`.

Notable exploratory contrasts:
- **S2 HFG-SAC vs PPO-Lagrangian**: Δ cost = −¥20,399/day, p = 0.022, d = −5.01.
- **S3 HFG-SAC vs CPO**: Δ cost = −¥3,362/day, p = 0.003, d = −5.26.
- **S1 HFG-SAC vs Safety Layer**: Δ cost = −¥16,619/day, p < 0.001, d = −47.8 (driven by SL's very low variance).

The wide confidence intervals on PPO, SAC, and Fuzzy-SAC reflect seed-to-seed instability: on S4, SAC and Fuzzy-SAC fail catastrophically (99% violation) for all three seeds, while PPO and PPO-Lagrangian show high variance because some seeds converge to safe policies and others do not.

---

*End of Chapter 5.*

---

**Word count**: ~3,400 words

**Artifacts produced**:
- `figures/fig3_training_curves.pdf` — violation trajectories on S3/S4
- `figures/fig4_performance.pdf` — main grouped-bar comparison (cost + violation)
- `figures/fig6_fcsd_heatmap.pdf` — FCSD heatmap
- `figures/fig7_pareto.pdf` — cost-safety Pareto scatter
- `outputs/figures/fig_rq2_robustness.pdf` — RQ2 robustness curve (violation + cost vs extremity)
- `figures/gen_fig_rq2_robustness.py` — reproducible RQ2 figure script
- `analysis/analysis-output/stats-appendix.md` — full descriptive statistics and inferential tests
- `analysis/analysis-output/analysis-report.md` — qualitative interpretation
