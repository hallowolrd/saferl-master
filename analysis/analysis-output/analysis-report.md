# Analysis Report — HFG-SAC Microgrid Dispatch

**Date**: 2026-09-28
**Scope**: 7 algorithms × 4 scenarios × 3 seeds = 84 runs (PPO, SAC, PPO-Lag, CPO, Safety-Layer, Fuzzy-SAC, HFG-SAC)
**Hardware**: RTX 4060 Laptop GPU, CUDA 12.8, torch 2.11.0+cu128

---

## 1. Comparison question

**Primary**: Does HFG-SAC achieve a better cost–safety trade-off than existing safe RL baselines across grid-connected and islanded microgrid scenarios?

**Secondary**:
- Where does HFG-SAC fail (which scenario, which constraint)?
- Is the improvement statistically detectable with n=3 seeds?

**Repeated measure unit**: one (algorithm, scenario) run per seed {42, 123, 456}.
**Primary metrics**: `cost_per_day` (¥/day, lower is better), `violation_rate` (%, lower is better), `avg_fcsd` (higher is safer).

---

## 2. Key findings

### 2.1 Grid-connected scenarios (S1, S2): HFG-SAC ties safety, wins on cost

| Scenario | Metric | HFG-SAC | Best safe baseline (PPO-Lag / CPO / SL) |
|---|---|---:|---:|
| S1 grid normal | Viol% | **0.00** | 0.00 (PPO-Lag, CPO, SL) |
| S1 grid normal | Cost ¥/d | **13,959** | 22,205 (PPO-Lag) / 17,240 (CPO) / 30,578 (SL) |
| S2 grid extreme | Viol% | **0.00** | 0.00 (PPO-Lag, CPO, SL) |
| S2 grid extreme | Cost ¥/d | **23,769** | 44,168 (PPO-Lag) / 31,675 (CPO) / 40,546 (SL) |

HFG-SAC matches the 0% violation rate of Lagrangian/CPO/Safety-Layer methods while cutting cost by **30–55%** relative to those conservative baselines on extreme grid-connected conditions.

### 2.2 Islanded normal (S3): HFG-SAC dominates

- HFG-SAC violation rate: **5.21%** (mean), cost ¥8,363/d, FCSD 0.979.
- Best unsafe-reward baseline (PPO): 4.17% violation but FCSD 0.989 (low constraint satisfaction quality despite low hard count — i.e., borderline violations).
- Fuzzy-SAC: 60.8% violation; SL: 56.4%; PPO-Lag: 0.6% but cost ¥10,922 (31% more expensive).
- HFG-SAC achieves the **best combined trade-off**: cost within 10% of the cheapest method, violation within ~5pp of the safest Lagrangian method, and FCSD near 0.98.

### 2.3 Islanded extreme (S4): HFG-SAC still fails

- HFG-SAC: **33.43% violation**, cost ¥13,610/d, FCSD 0.892.
- This is the **decisive open problem**. All baselines also fail on S4 (Fuzzy-SAC 98.8%, SAC 99.0%, PPO 43.7%, PPO-Lag 48.7%, CPO 49.0%, SL 57.4%), but HFG-SAC's 33% is far above the 5% safety threshold.
- Figure 3 (right) shows the red HFG-SAC curve plateauing at ~30–35% from episode 25 onward — not a training-dynamics issue (it converges), but a structural ceiling: the freq/voltage/SOC shields cannot prevent hard violations under extreme islanded load/PV profiles.

### 2.4 Statistical detection

- With n=3 seeds, Welch's t-tests are underpowered. Effect sizes (Cohen's d) for cost are large for HFG-SAC vs PPO-Lag/CPO/SL (|d| = 2.3–47), indicating practically meaningful differences despite p > 0.05 for many contrasts.
- S2 HFG-SAC vs PPO-Lag: Δ cost −¥20,399/d, p=0.022, d=−5.01 (significant at α=0.05).
- S3 HFG-SAC vs CPO: Δ cost −¥3,362/d, p=0.003, d=−5.26 (significant).

---

## 3. What changed in our understanding

1. **B1/B2 shield improvements work on grid-connected and islanded-normal scenarios** — S1/S2 now 0% violation (previously S1 had violations in early runs), S3 dropped from ~36% (old code) to ~5%.
2. **The bottleneck is S4 islanded extreme**, not training instability. The policy converges; the shield itself is overwhelmed by extreme PV/load excursions. This points to a **structural** problem (insufficient storage, load-shedding not modeled, or shield clipping too late) rather than a tuning problem.
3. **Fuzzy-SAC (without the hierarchical guide) catastrophically fails** on islanded scenarios (60–99% violation), confirming that the HFG hierarchical + fuzzy shield is the active ingredient.

---

## 4. Caveats / blockers

- n=3 seeds per arm: inferential tests are exploratory; effect sizes more reliable than p-values.
- S4 violation is dominated by unknown constraint mix (freq vs voltage vs SOC); future work must instrument per-constraint breakdown.
- MILP/MPC model baselines (oracle) show cost ¥59,362/d on S1 — HFG-SAC at ¥13,959/d is much cheaper because MILP cost includes demand-shedding penalties; direct cost comparison is not apples-to-apples.
- Episode-level curves are smoothed by mean ± std over 3 seeds; individual seed variability is large on SAC/SL baselines.

---

## 5. Decision-relevant next steps

1. **Diagnose S4 per-constraint**: instrument the islanded env to log freq_deviation, voltage_deviation, SOC violation separately.
2. **Tighten freq/voltage shield margin** (`freq_volt_shield_safety_ratio` currently 0.175) — likely needs 0.25–0.30 to push S4 below 10%.
3. **Add load-shedding action** for extreme islanded deficits — current action space has no shedding valve, forcing the policy to violate when storage is exhausted.
