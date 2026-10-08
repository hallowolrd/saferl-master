# Experimental Analysis: Comparison Methods, Metrics, and Innovation Mapping

**Date**: 2026-09-22
**Project**: saferl — Safe Reinforcement Learning for Microgrid Optimal Scheduling
**Focus**: Fuzzy Safe RL for Islanded Microgrid Economic Dispatch

---

## 1. Comparison Methods Across All Papers

### 1.1 Su et al. (Islanded MG TSEC, TSG 2025)

**Three levels of comparison:**

| Comparison Level | Methods Compared | Purpose |
|------------------|------------------|---------|
| **Estimation models** (Table I) | GP, SGP, DGP, DKL (probabilistic); SAE, ANN, DBN (deterministic) | Validate DSPP accuracy for transient stability estimation |
| **RL algorithms** (Table II) | RCPO, CPO, CPPO-PID, CUP, FOCOPS, PCPO, PPO-Lag, TRPO-Lag — total 8 safe RL methods | Compare RL training performance (reward + constraint satisfaction) |
| **TSEC methods** (Table III) | NSGA-II (evolutionary), MILP (sensitivity-based), PCPO, TRPO-Lag, RCPO | End-to-end comparison of full TSEC pipelines |

**Key findings from comparisons:**
- Only 4/8 safe RL methods converge to 0 constraint cost: PCPO, TRPO-Lag, CUP, RCPO
- Only NSGA-II, PCPO, TRPO-Lag, and RCPO satisfy transient stability constraint
- RCPO achieves highest reward (least load shedding) among constraint-satisfying methods
- RCPO + TRPO-Lag have identical convergence speed; CUP and PCPO converge slower

---

### 1.2 Xia et al. (NMG Hierarchical SDRL, TSTE 2024)

**Comparison methods:**

| Method | Description |
|--------|-------------|
| **Proposed method** | Multi-agent SAC (MASAC) + safety model (constraint estimation network + action correction) |
| **MASAC (baseline)** | Vanilla multi-agent SAC without safety model |
| **Penalty-based MASAC** | MASAC with constraint penalty added to reward function |

**Comparison dimensions:**
- Training stability (reward convergence curve)
- Number of unsafe cases during training (voltage violation count)
- P-management cost (local benefit + global benefit)
- Network power loss reduction
- Q-management (voltage violation reduction)
- Decentralized computational efficiency (~0.01s per step)

---

### 1.3 Bian et al. (Fuzzy RL MAS, TSMC 2026)

**Comparison structure:**

| Aspect | Details |
|--------|---------|
| **Problem compared** | Safe cooperative control of nonlinear MAS with composite constraints + multiactuator faults |
| **Fault types tested** | Drift fault (Agent 2), loss fault (Agent 3), bias fault (Agent 4), normal (Agent 1) |
| **Constraint types** | Asymmetric dynamic delayed state constraints + terminal time constraints |
| **Performance validated** | 1) Trajectory tracking within constraints; 2) Coordination error convergence; 3) NN weight convergence; 4) Controller response under different faults |
| **Theoretical guarantees** | Lyapunov stability + graph theory proof of boundedness and consensus |

---

### 1.4 Gu et al. (Safe RL Survey, TPAMI 2024)

**Methodological taxonomy (4 categories):**

| Category | Representative Methods |
|----------|-----------------------|
| **Policy Optimization** | CPO, PCPO, FOCOPS, CUP, RCPO, CPPO-PID, PPO-Lag, TRPO-Lag, CSAC, CVaR-constrained, chance-constrained |
| **Control Theory** | Lyapunov-based safe RL, MPC-based safe RL, CBF-based safe RL, robust CBF-SAC |
| **Formal Methods** | Neurosymbolic RL, shielding / reachability, formal verification |
| **Gaussian Processes** | PILCO-based safe search, GP reachability, GP uncertainty-aware exploration |

**Benchmarks reviewed:**
- Safety Gym, Safe MuJoCo, AISafety Gridworlds, Safe MAMuJoCo, Safe MAIG3, Safe MARobosuite

---

### 1.5 Yu et al. (Power System Safe RL Review, arXiv 2407.00681)

**Two-category taxonomy:**

| Category | Sub-types |
|----------|-----------|
| **Safe Layer (decoupled)** | Action projection: CBF-based, MPC-based, parameterized model analysis<br>Action replacement: shielding, human intervention |
| **Policy Optimization (coupled)** | Lagrange multipliers: PPO-Lag, SAC-Lag, CSAC, RCPO<br>Trust region: CPO, PCPO, FOCOPS<br>Lyapunov method |

**Three application areas:**
1. Frequency regulation
2. Voltage control
3. Energy management

---

### 1.6 Barbalho et al. (MG RL Survey, IEEE Access 2025)

**Algorithms compared in literature:**
- Tabular: Q-learning (QL), SARSA, Fitted Q-iteration (FQI)
- Deep RL: DQN, DDPG, TD3, PPO, A2C, A3C, SAC, TRPO
- Model-based: MPC, Dyna-Q
- Multi-agent: MARL, GT-based, MADDPG

**Novel concepts combined with DRL (Table 4):**
- GANs (adversarial training for robustness)
- Imitation learning
- Transformer architectures
- Fuzzy systems (fuzzy-PID tuning)
- Physics-informed approaches
- Hierarchical RL

---

## 2. Evaluation Metrics: Comprehensive Mapping

### 2.1 Metrics by Category

| Metric Category | Specific Metrics | Used In | Relevance to Our Work |
|-----------------|------------------|---------|---------------------|
| **Economic / Objective** | Total operating cost, load shedding amount, generation cost, network power loss | Su, Xia, Barbalho | ✅ Primary: minimize dispatch cost |
| **Safety / Constraint** | Constraint satisfaction rate, number of unsafe cases, voltage violation count, constraint cost, transient stability (TSI) | Su, Xia, Bian, Yu | ✅ Critical: hard safety guarantees |
| **Estimation Accuracy** | MAE, classification accuracy (AP), false alarm rate (FA), miss detection rate (MD) | Su | ✅ For fuzzy uncertainty quantification |
| **Learning Efficiency** | Convergence speed, training steps to converge, episodes to zero unsafe cases | Su, Xia | ✅ Sample efficiency claim |
| **Stability / Control** | Lyapunov stability, tracking error bound, consensus error, transient response | Bian, Barbalho, Yu | ✅ Theoretical safety guarantee |
| **Computational** | Online inference time, training time, memory usage | Xia, Barbalho | ⚠️ Secondary (real-time feasibility) |
| **Robustness** | Performance under unseen scenarios, fault tolerance, disturbance rejection | Bian, Su (dynamic loads) | ✅ Key: robust safe dispatch |

### 2.2 Metrics Gap Analysis

**What existing work measures:**
- ✅ Reward/cost (economic performance)
- ✅ Constraint violation count (safety)
- ✅ Convergence speed (learning efficiency)
- ✅ Estimation accuracy (for model-based approaches)

**What's under-measured / missing for islanded MG dispatch:**
- ❌ **Constraint satisfaction probability** with confidence intervals (probabilistic safety)
- ❌ **Worst-case performance guarantee** (robust safe RL metric)
- ❌ **Sample complexity bounds** for safe learning
- ❌ **Interpretability metrics** (fuzzy rule clarity, trustworthiness)
- ❌ **Multi-objective trade-off quantification** (cost vs. safety vs. robustness)
- ❌ **Real-world hardware validation** (HIL, power HIL) — noted as gap by Barbalho et al.

---

## 3. Experimental Dimensions: Systematic Comparison

### 3.1 Dimensions Used in Existing Work

| Dimension | Su et al. (TSEC) | Xia et al. (NMG) | Bian et al. (Fuzzy MAS) |
|-----------|-----------------|-----------------|----------------------|
| **Problem type** | Transient stability emergency control | P/Q hierarchical coordination | Consensus tracking control |
| **MG mode** | Islanded | Grid-connected (NMG) | N/A (general MAS) |
| **Constraint type** | Hard + probabilistic (chance) | Soft (voltage safety layer) | Hard (barrier function) |
| **Uncertainty source** | Renewable/load, model | PV/load prediction, historic data | Actuator faults, unknown dynamics |
| **RL algorithm** | RCPO (Lagrangian) | MASAC (safe layer) | Actor-critic-identifier (ADP) |
| **Safety mechanism** | Chance constraint (DSPP) | Constraint estimation network | Universal barrier function + FLS |
| **Time scale** | Emergency (ms-seconds) | Hourly + 3-min | Continuous-time |
| **Test system** | Real islanded MG (13 switches, DGs, BESS) | IEEE 33-bus NMG (4 MGs) | Numerical MAS (4 agents) |
| **Baselines** | 7 safe RL + 2 traditional | 2 MASAC variants | N/A (theoretical + sim) |

### 3.2 Underexplored Experimental Dimensions for Islanded MG Dispatch

| Dimension | Why It Matters | How Fuzzy Safe RL Addresses It |
|-----------|---------------|-------------------------------|
| **Multi-level uncertainty** | Islanded MGs face compound uncertainty: PV/load, model parameters, fault conditions | Fuzzy logic naturally handles multiple uncertainty types with membership functions |
| **Hard constraint satisfaction guarantee** | Penalty-based approaches can't guarantee zero violations | Fuzzy barrier functions + theoretical proof of safe set invariance |
| **Interpretability** | Power system operators need trustable AI decisions | Fuzzy rules are human-readable and verifiable |
| **Multi-objective trade-off** | Cost minimization vs. safety vs. robustness is a 3-way trade-off | Fuzzy multi-objective optimization with Pareto front |
| **Transfer to unseen scenarios** | Islanded MGs face diverse operating conditions | Fuzzy generalization + robust safe exploration |
| **Sparse reward / sample efficiency** | Real MG data is limited; unsafe exploration is costly | Fuzzy model-based rollouts in virtual buffer (like AR-SAC concept) |
| **Hardware / HIL validation** | Barbalho et al. flag this as a key gap | Fuzzy controllers have established HIL platforms |

---

## 4. Innovation Points: Mapping to Experimental Evidence

### 4.1 Innovation 1: Fuzzy Uncertainty-Aware Safe SAC (F-SAC)

**What's new:**
- Replace GP-based uncertainty (Su et al. DSPP) with **fuzzy logic system (FLS) uncertainty estimation**
- Fuzzy membership functions provide more flexible uncertainty representation than Gaussian assumptions
- FLS universal approximation theorem provides theoretical guarantee

**Experimental support from literature:**
- Su et al. show probabilistic estimation improves safety → Fuzzy estimation can do the same with more flexibility
- Bian et al. prove FLS can approximate unknown nonlinear dynamics → extend to MG dynamics
- Barbalho et al. identify fuzzy + DRL as emerging trend but under-explored for safe dispatch

**How to validate experimentally:**
- **Baselines**: Vanilla SAC, SAC-Lag, RCPO, GP-SAC (Su et al. approach), robust MPC
- **Metrics**: Operating cost, constraint violation rate, uncertainty estimation accuracy (fuzzy vs. GP), sample efficiency
- **Ablation**: Compare fuzzy uncertainty vs. GP uncertainty vs. no uncertainty model
- **Dimensions**: Vary uncertainty levels (low/medium/high renewable penetration), test on standard MG benchmarks

---

### 4.2 Innovation 2: Fuzzy Barrier Function Safety Layer

**What's new:**
- **Fuzzy control barrier function (F-CBF)** that handles non-crisp constraint boundaries
- Combines interpretability of fuzzy rules with rigor of barrier function theory
- Adaptive constraint tightening based on fuzzy uncertainty level

**Experimental support from literature:**
- Gu et al. (survey) identify CBF-based methods as rigorous but model-dependent → Fuzzy CBF reduces model dependency
- Xia et al. use NN-based safety layer → Fuzzy safety layer is more interpretable
- Bian et al. use universal barrier function → extend to fuzzy-valued barriers for MG

**How to validate experimentally:**
- **Baselines**: NN safety layer (Xia et al.), penalty-based reward, CBF-based projection
- **Metrics**: Number of constraint violations, constraint satisfaction probability, safety margin, interpretability score (rule count)
- **Ablation**: Fuzzy barrier vs. crisp barrier vs. no safety layer
- **Dimensions**: Test on various constraint types (SoC, voltage, frequency, ramp rate), extreme operating conditions

---

### 4.3 Innovation 3: Fuzzy-Augmented Virtual Replay Buffer

**What's new:**
- Fuzzy model generates **virtual safe trajectories** to augment training data
- Combines AR-SAC's residual learning concept with fuzzy system approximation
- Fuzzy rollouts provide conservative safety estimates

**Experimental support from literature:**
- Yu et al. (survey) note offline / model-based safe RL improves sample efficiency
- Barbalho et al. identify sample efficiency of model-free RL as key challenge
- Our own AR-SAC uses virtual replay buffer → Fuzzy version enhances it

**How to validate experimentally:**
- **Baselines**: AR-SAC (original), offline RL (BCQ, CQL), pure online safe RL
- **Metrics**: Samples needed to converge, performance with limited data, safety during learning
- **Ablation**: Buffer size, fuzzy model accuracy vs. sample complexity, virtual vs. real data ratio
- **Dimensions**: Data scarcity scenarios (cold start, rare events), transfer learning between MGs

---

### 4.4 Innovation 4: Multi-Objective Fuzzy Safe Dispatch Framework

**What's new:**
- Unified framework optimizing **economic cost, safety level, and robustness** simultaneously
- Fuzzy Pareto front for multi-objective decision making
- Operator-adjustable safety-robustness-cost trade-off via fuzzy weight tuning

**Experimental support from literature:**
- Gu et al. highlight "trade-off balances" as key future challenge
- Existing work focuses on single objective (cost with constraint) → not explicit multi-objective
- Barbalho et al. note reward function design is under-explored for MG RL

**How to validate experimentally:**
- **Baselines**: Single-objective safe RL (cost + constraint penalty), weighted sum method
- **Metrics**: Pareto front coverage, hypervolume indicator, decision space diversity
- **Ablation**: Different weight configurations, fuzzy vs. crisp multi-objective
- **Dimensions**: Different operator priorities (cost-sensitive vs. safety-sensitive scenarios)

---

### 4.5 Innovation 5: Hierarchical Fuzzy-RL for Multi-DER Coordination

**What's new:**
- **Upper level**: Fuzzy rule-based coordinator for DER setpoint allocation
- **Lower level**: Safe SAC for fast-timescale constraint satisfaction
- Hierarchical fuzzy-RL architecture for islanded MG with heterogeneous DERs

**Experimental support from literature:**
- Xia et al. use hierarchical RL (P/Q, two time-scales) → validate hierarchical approach
- Barbalho et al. identify hierarchical RL as trend for complex MG
- Bian et al. use backstepping (hierarchical control) → conceptually similar

**How to validate experimentally:**
- **Baselines**: Single-level safe RL, model-based hierarchical control, distributed optimization
- **Metrics**: Coordination efficiency, constraint satisfaction, scalability (number of DERs)
- **Ablation**: Fuzzy upper level vs. RL upper level, different time-scale splits
- **Dimensions**: MG size (small/medium/large), DER type diversity

---

## 5. Recommended Experimental Design for Fuzzy Safe RL (Islanded MG Dispatch)

### 5.1 Test Systems

| System | Description | Purpose |
|--------|-------------|---------|
| **Modified IEEE 33-bus** | Islanded mode with PV, WT, DG, BESS, controllable load | Primary benchmark |
| **CIGRE MV benchmark** | Standard European MV MG | Cross-validation |
| **Real MG dataset** | e.g., Australian CER, Pecan Street, or local data | Realism validation |
| **Small analytical MG** | 2-3 DER, simple model | Theoretical proof validation |

### 5.2 Baseline Methods (8-10 total)

| Category | Methods |
|----------|---------|
| **Traditional optimization** | Robust MPC, Stochastic MPC, MILP (if applicable) |
| **Standard RL** | DQN, DDPG, PPO, SAC |
| **Safe RL (constraint-based)** | PPO-Lag, RCPO, CPO, SAC-Lag |
| **Safe RL (model-based)** | GP-SAC, Lyapunov-based safe RL |
| **Our previous work** | AR-SAC |
| **Proposed** | Fuzzy-AR-SAC (full), F-SAC (no AR), Fuzzy-safe-SAC (no fuzzy model) |

### 5.3 Evaluation Metrics (5 categories, ~15 metrics)

| Category | Metrics |
|----------|---------|
| **Economic** | Total operating cost (¥), load shedding (kWh), renewable curtailment (kWh) |
| **Safety** | Constraint violation count, violation duration, worst-case violation magnitude, safety probability (%) |
| **Learning** | Convergence episodes, sample efficiency (cost per 1000 samples), training stability (reward variance) |
| **Robustness** | Performance degradation under uncertainty, unseen scenario performance, fault recovery time |
| **Computational** | Online inference time (ms), training time (hours), memory footprint |

### 5.4 Experimental Dimensions / Case Studies

1. **Base case**: Nominal operating conditions
2. **Uncertainty sweep**: Low / medium / high renewable penetration (σ scaling)
3. **Extreme scenarios**: Peak load, low renewable, equipment outage (N-1)
4. **Data scarcity**: 10% / 50% / 100% training data
5. **Multi-objective trade-off**: Cost-priority vs. safety-priority vs. balanced
6. **Ablation study**: Each component removed one at a time
7. **Transfer learning**: Train on MG A, test on MG B (zero-shot / few-shot)

### 5.5 Ablation Study Design

| Ablation | What's Removed | What It Tests |
|----------|---------------|---------------|
| Full model | — | Baseline for comparison |
| No fuzzy uncertainty | Replace FLS with deterministic model | Value of fuzzy uncertainty estimation |
| No barrier layer | Replace with penalty in reward | Value of rigorous safety guarantees |
| No virtual buffer | Pure online learning | Value of fuzzy model-based data augmentation |
| Crisp barriers | Fuzzy → crisp barrier function | Value of fuzzy-valued constraints |
| GP instead of fuzzy | Replace FLS with GP | Fuzzy vs. Gaussian uncertainty comparison |
| SAC only | No safety components at all | Upper bound of pure performance |

---

## 6. Gap Summary: What Differentiates Our Work

### 6.1 Positioning Relative to Key Papers

| Paper | What They Do | What We Add |
|-------|-------------|-------------|
| **Su et al. (2025)** | Safe RL for islanded MG TSEC (transient stability) | We do **economic dispatch** (different problem); use **fuzzy logic** instead of DSPP/GP |
| **Xia et al. (2024)** | Safe RL for NMG hierarchical control (grid-connected) | We focus on **islanded mode**; add **fuzzy interpretability**; **multi-objective** framework |
| **Bian et al. (2026)** | Fuzzy safe RL for general MAS | We apply to **islanded MG dispatch** (new domain); add **economic optimization**; **SAC-based** not ADP |
| **Gu et al. (2024)** | Safe RL survey (2H3W framework) | We fill the application gap: **fuzzy + safe RL + islanded MG dispatch** |
| **Yu et al. (2024)** | Power system safe RL review | We provide a **specific algorithmic contribution** (fuzzy safe SAC) vs. survey |
| **Barbalho et al. (2025)** | MG RL survey | We address two key gaps they identify: **safety guarantees** and **fuzzy-DRL integration** |
| **Our AR-SAC** | Safe robust SAC for islanded MG | We enhance with **fuzzy logic** for better uncertainty handling, interpretability, and theoretical grounding |

### 6.2 Core Differentiating Claims

1. **First fuzzy safe RL for islanded microgrid economic dispatch** (application novelty)
2. **Fuzzy barrier functions for power system constraints** (methodological novelty)
3. **Fuzzy-augmented virtual replay buffer for sample-efficient safe learning** (algorithmic novelty)
4. **Multi-objective fuzzy safe dispatch framework** (framework novelty)
5. **Comprehensive benchmark with 8+ baselines across 5 metric categories** (evaluation thoroughness)

---

## 7. Suggested Table of Results for Paper

### Table 1: Economic Performance Comparison
| Method | Avg. Cost (¥/day) | Cost Std. Dev. | Load Shed (kWh) | Renewable Curtail (kWh) |
|--------|------------------|----------------|-----------------|----------------------|
| Robust MPC | — | — | — | — |
| SAC | — | — | — | — |
| SAC-Lag | — | — | — | — |
| RCPO | — | — | — | — |
| AR-SAC | — | — | — | — |
| **Fuzzy-AR-SAC (Ours)** | **—** | **—** | **—** | **—** |

### Table 2: Safety Performance Comparison
| Method | Violation Count | Violation Rate (%) | Max Violation Mag. | Safety Probability |
|--------|-----------------|-------------------|--------------------|-------------------|
| Robust MPC | — | — | — | 100% (by design) |
| SAC | — | — | — | — |
| SAC-Lag | — | — | — | — |
| RCPO | — | — | — | — |
| AR-SAC | — | — | — | — |
| **Fuzzy-AR-SAC (Ours)** | **—** | **—** | **—** | **≥99% (theoretical)** |

### Table 3: Learning Efficiency Comparison
| Method | Convergence Episodes | Final Cost at 1k ep. | Sample Efficiency Score | Training Time (h) |
|--------|---------------------|---------------------|------------------------|-------------------|
| SAC | — | — | — | — |
| SAC-Lag | — | — | — | — |
| RCPO | — | — | — | — |
| AR-SAC | — | — | — | — |
| **Fuzzy-AR-SAC (Ours)** | **—** | **—** | **—** | **—** |

### Table 4: Ablation Study
| Variant | Cost (¥) | Violations | Convergence Ep. | Δ from Full |
|---------|---------|-----------|-----------------|-------------|
| Full Fuzzy-AR-SAC | — | — | — | — |
| - Fuzzy uncertainty | — | — | — | — |
| - Barrier layer | — | — | — | — |
| - Virtual buffer | — | — | — | — |
| - Crisp barriers | — | — | — | — |
| - GP instead of fuzzy | — | — | — | — |

---

*Generated by Claude Code — research-ideation + ml-paper-writing skills*
