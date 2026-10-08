# Pending Experiments TODO

> **Last updated**: 2026-10-07
> **Status legend**: ⏳ pending | 🔄 in progress | ✅ done

---

## Completed (2026-09-28)

- ✅ **RQ1 Main comparison**: 7 algorithms × 4 scenarios × 3 seeds = 84 runs
- ✅ **S4 structural fix**: interruptible_load 200→600 kW, shield ratio 0.175→0.10, freq penalty 1000→3000
- ✅ **S4 retrain**: 3 seeds parallel, violation 33.4% → 0.64%
- ✅ **Figures regenerated**: fig3/fig4/fig6/fig7 updated with new S4 data
- ✅ **Paper draft updated**: chapter5 §5.2/§5.3, chapter6 limitation, abstract

---

## ⏳ Plan D — S3/S4 Retune (IMMEDIATE, code ready)

**Status**: Code changes applied, smoke test passed, retraining paused.

**Why**: Current S4 has 0.6% violation (over-safe) at ¥24,004/d cost (too high).
S3 has 5.2% violation (over threshold) at ¥8,363/d. Ideal: S3 better than S4 on both metrics.

**Code changes applied**:
- `config.py`: new `load_shed_cost` field
- `islanded.py`: shed cost reads from config
- `experiment_runner.py` scenario overrides:
  - **S3**: shield_ratio=0.13, freq_slope=1500
  - **S4**: shield_ratio=0.15, freq_slope=1500, shed_cost=1.0, interruptible=600 kW

**Smoke test result (old actor + new shield, 5 episodes)**:
| Metric | S3 | S4 |
|--------|----|----|
| Hard freq viol | 4.3% | 4.0% |
| Mean cost/day | ¥7,348 | ¥11,451 |
| Mean shed | 68 kW | 237 kW |

**Action**: Run `run/rerun_s3_s4_plan_d.py` (6 workers parallel, ~1 hour).
- Old S3/S4 results backed up to `outputs/experiments/results_backup_pre_pland/`
- After retrain: re-run `analysis/run_analysis.py` and update chapter5 Table 5.1

**Expected post-retrain**: S3 ~1-2% viol at ~¥8k; S4 ~3-5% viol at ~¥12-13k.

---

## ⏳ RQ2 — Robustness under Varying Extremity

**Code**: `run/experiment_runner.py:920` (`run_extremity_experiments`)

**Design**: Vary extremity level from 0.8× to 1.5× on islanded scenario:
- PV multiplier = 1.0 − 0.3·level
- Load multiplier = 1.0 + 0.2·level
- WT multiplier = 1.0 − 0.15·level

**Matrix**: 4 algorithms (PPO, PPO-Lagrangian, CPO, HFG-SAC) × 7 levels × 3 seeds = **84 runs**

**Hypothesis**: HFG-SAC's fuzzy boundary provides early-warning gradient, yielding more graceful degradation than crisp-Lagrangian methods as extremity increases.

**Estimated time**: ~4 hours GPU (parallel)

**Deliverable**: `fig5_transfer_sensitivity.pdf` robustness curve

---

## ⏳ RQ3 — Cross-Scenario Transfer

### 🔄 NEXT — Plan A T1 real training pilot (M2 validation)

**Status**: Code ready (M1/M2/M3/M4 all executable end-to-end), pending run.

**Goal**: Verify on a REAL training run that M2 fuzzy-prior injection has a
measurable behavioral effect (cost / violation / convergence speed), not just
mechanically overwritten parameters.

**Task (Plan A, same mode, safety-spec micro-adjustment)**:
- Mode: grid_connected → grid_connected (obs dims stay 8, so M3 stays active)
- Source: default SoC optimal band [0.3, 0.8]
- Target: `soc_optimal_min=0.25`, `soc_optimal_max=0.75`
  (M1 expected: soc_upper ~0.65, soc_lower ~0.74 → both "adaptable")
- Seed: 42; episodes: 100–200; source pre-trained min(100, n_ep//2) episodes

**Arms (isolate M2 contribution)**:
1. A: from_scratch
2. B: naive transfer (M3 weights, no conservative mechanisms)
3. C: M3 + M4 (NO M2)
4. D: full M1 + M2 + M3 + M4

**Contrast of interest**: C vs D → any M2-specific delta on violation %,
cost/day, and convergence episode. A vs D → overall transfer benefit.

**Pass criteria**:
- D differs from C on at least one metric beyond run-to-run noise (single-seed
  pilot; treat as directional evidence, not inferential).
- No regression in hard violation vs from_scratch.

**After pilot**: if M2 shows signal → full RQ3 sweep (3 tasks × 4 arms ×
5 seeds, CPU); update paper §4.4 / §5.5.2. If no signal → debug M2 magnitude
or narrow the claim.

**Code entry**: `run/experiment_runner.py` (`_run_transfer`); pilot scripts
`run/pilot_rq3.py` / `run/pilot_rq3_t2.py` as templates.

---

**Code**: `run/experiment_runner.py` (`_run_transfer`)

**Tasks**:
- T1: summer → winter (grid-connected)
- T2: grid-connected → islanded (obs dims 8 vs 9, weight transfer SKIPPED)
- T3: islanded normal → islanded extreme

**Mechanisms claimed in paper §4.4**:
- M1: Jaccard similarity of fuzzy membership functions (analysis tool)
- M2: Fuzzy prior init (β^t=β^s·Δ, x_ref^t=x_ref^s+δ) — **NOT YET IMPLEMENTED in code**
- M3: Weight transfer + freeze bottom 50% of actor trunk — **implemented 2026-10-07**, only activates when dims_match
- M4: Exponential conservative action scaling α_cur(t)=1+(α_init-1)e^{-νt}, α_init=2.5

---

### 🔄 Pilot results (2026-10-07)

**T3 pilot** (`outputs/experiments/pilot_rq3/pilot_summary.json`): island normal → extreme, 60ep, seed=42
| Arm | cost/day | violation% | min_fcsd | converge_ep |
|-----|----------|------------|----------|-------------|
| from_scratch | ¥13,956 | 37.65% | 0.674 | 10 |
| naive_transfer | ¥14,571 | 23.36% | 0.422 | 39 |
| conservative (M1+M4) | ¥13,956 | 26.49% | 0.418 | 10 |

Note: T3 is the WORST task for M1 demonstration — constraint boundaries unchanged, Jaccard always 1.0.

**T2 diagnostic pilot** (`outputs/experiments/pilot_rq3_t2/pilot_summary.json`): grid summer → island summer, 200ep, seed=42
| Arm | cost/day | violation% | min_fcsd | converge_ep | time |
|-----|----------|------------|----------|-------------|------|
| A from_scratch | ¥10,589 | 9.38% | 0.677 | 10 | 631s |
| B naive | ¥8,917 | 11.16% | 0.677 | 17 | 2370s |
| C M4-only | ¥10,242 | **4.61%** | 0.677 | 25 | 5066s |
| D M1+M4 | ¥10,242 | **4.61%** | 0.677 | 25 | 2292s |

**Key findings**:
1. **M4 (conservative action scaling) is the ONLY mechanism that actually reduces violation** (9-11% → 4.61%)
2. **M1 has zero behavioral effect** — C and D arms produced identical results (M1 categorization does not modify target agent parameters)
3. **M2 (fuzzy prior init) is confirmed NOT implemented** in code
4. **M3 cannot activate on T2** due to obs dim mismatch (grid=8, island=9); `dims_match=false` for all transfer arms
5. Trade-off: M4 reduces violation but slows convergence (ep 10→25) and increases cost (~¥1.3k/day)

---

### ⏸️ PENDING DECISION — Route 1 vs Route 2

**Route 1 (narrow claim, fast)**:
- Reframe RQ3 narrative: "M4 conservative action scaling enables safe cross-mode transfer"
- M1 as analysis tool only (shows which constraints are similar), not a behavior mechanism
- Drop M2/M3 from claim (or mention as future work)
- Action: run 5 seeds × 3 tasks × 4 arms sweep (~6-8h CPU)
- Deliverable: RQ3 table + figure showing M4 violation reduction

**Route 2 (full implementation, larger effort)**:
- Implement M2 fuzzy prior initialization in code
  - After M1 categorization, modify target agent's fuzzy membership params (β, x_ref) for "adaptable" constraints
  - Need to modify `FuzzyConfig` or agent fuzzy layer to accept source params as init
  - Apply Δ (steepness scaling) and δ (reference shift) per paper §4.4.2 Step 2
- Re-run T2 diagnostic pilot to verify M2 has measurable effect
- Then run full 5-seed sweep
- Deliverable: full 4-mechanism story matching paper §4.4

**Hypothesis (original)**: Constraint-aware transfer reduces fine-tuning sample cost by ≥50% while maintaining safety.
**Estimated time**: Route 1 ~8h CPU; Route 2 ~8h code + 10h experiments

---

## ⏳ RQ4 — Ablation Study

**Code**: `run/experiment_runner.py:636` (`run_ablation_study`)

**Variants on S4**:
- (i) Full HFG-SAC
- (ii) −FuzzyCon (crisp Lagrangian, no fuzzy upper layer)
- (iii) −FuzzyKnow (no lower reward shaping)
- (iv) −Both (crisp safe SAC)

**Matrix**: 4 variants × 3 seeds = **12 runs**

**Hypothesis**: Both layers contribute; removing upper layer causes ~30% violation increase, removing lower layer causes ~2× slower convergence.

**Estimated time**: ~1 hour GPU

**Deliverable**: Ablation table in §5.5

---

## ⏳ RQ5 — Sensitivity Analysis

**Code**: `run/experiment_runner.py:680` (`run_sensitivity_analysis`)

**Sweeps on S2**:
- β (fuzzy boundary steepness): 1, 3, 5, 10
- α_target (FCSD target): 0.75, 0.85, 0.90, 0.95
- κ₀ (initial lambda): 0.5, 2.0, 5.0, 10.0

**Matrix**: 12 configs × 3 seeds = **36 runs**

**Hypothesis**: Performance is robust to β in [3, 10]; α_target=0.85 is the sweet spot.

**Estimated time**: ~2 hours GPU

**Deliverable**: Sensitivity heatmap / line plots

---

## ⏳ Paper Polish (after experiments)

- [ ] Re-run `analysis/run_analysis.py` with RQ2–RQ5 data
- [ ] Generate `fig5_transfer_sensitivity.pdf`
- [ ] Update Table 5.1 with ablation rows
- [ ] Add §5.5 results section (currently "ongoing")
- [ ] Update abstract with transfer/sensitivity numbers
- [ ] Run `/ars-citation-check` for all references
- [ ] Run `/ars-revision` for language polish
- [ ] Generate final PDF via docx skill

---

## Priority Order

0. **Plan D S3/S4 retrain** (code ready, smoke passed, ~1 hour GPU)
1. **RQ4 Ablation** (cheapest, directly supports paper claims about two-layer architecture)
2. **RQ2 Robustness** (main figure for §5.5, high impact)
3. **RQ5 Sensitivity** (ablation-adjacent, can run in same batch)
4. **RQ3 Transfer** — Plan A adopted (same-mode safety-spec transfer, M2
   implemented); NEXT: T1 real training pilot to validate M2 effect (see top
   of RQ3 section)
