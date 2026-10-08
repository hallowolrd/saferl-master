# Chapter 1 — Introduction

> **Status**: Draft v1.0
> **Word count**: ~1,250 words
> **Target section**: Section 1 of the main paper

---

## 1.1 Background and Motivation

The global transition toward low-carbon energy systems has driven rapid growth in renewable energy penetration. Microgrids—localized energy systems that integrate distributed energy resources (DERs) such as photovoltaic (PV) arrays, wind turbines (WTs), energy storage systems (ESSs), and controllable loads—have emerged as a key architecture for accommodating high shares of variable renewable generation while enhancing supply reliability and energy efficiency [1, 2]. Optimal dispatch of microgrids—determining the charge/discharge schedules of storage, output levels of dispatchable generators, and power exchange with the main grid—is fundamental to minimizing operating costs, reducing renewable curtailment, and maintaining power quality [3].

However, microgrid dispatch presents significant challenges due to the inherent uncertainty of renewable generation and load demand, the nonlinear dynamics of storage systems, and the need to satisfy multiple operational constraints simultaneously [4]. Traditional optimization methods such as mixed-integer linear programming (MILP) and model predictive control (MPC) rely on accurate system models and forecast data, and their performance degrades under high uncertainty or when the system model is imperfect [5, 6]. Deep reinforcement learning (DRL), with its ability to learn optimal control policies directly from interaction data and handle high-dimensional, uncertain environments, has emerged as a promising alternative for microgrid energy management [7, 8].

Despite remarkable progress, deploying DRL in real microgrids remains challenging due to a critical barrier: **safety**. In safety-critical energy systems, constraint violations—such as overcharging batteries, exceeding generator ramp limits, or violating voltage bounds—can lead to equipment damage, degraded power quality, or even blackouts. Standard DRL algorithms explore freely during training and offer no formal guarantees of constraint satisfaction, making them unsuitable for direct deployment in real systems [9].

Safe reinforcement learning (Safe RL) aims to address this gap by integrating constraint satisfaction into the RL framework [10, 11]. The dominant formulation is the Constrained Markov Decision Process (CMDP), in which safety is modeled as hard constraint thresholds that must not be exceeded [12]. Methods such as Lagrangian relaxation [13], constrained policy optimization (CPO) [14], and trust-region-based approaches [15] have been developed to solve CMDPs. However, these methods rest on a fundamental assumption: that constraints have precise, crisp boundaries.

## 1.2 Problem Statement and Research Gaps

In practice, the crisp constraint assumption poorly reflects the reality of microgrid operation. Engineering constraints are rarely binary—they exhibit inherent gradation and fuzziness for several reasons:

1. **Hierarchical safety levels**: Many constraints have a "recommended range" inside a "hard limit." For example, a battery's state-of-charge (SOC) may have a preferred band of 30–80% for cycle longevity and a hard safety limit of 10–90%. Operating between these bands is permissible but undesirable.
2. **Transient violation tolerance**: Under extreme conditions (sudden cloud cover, load spikes), short-duration, small-magnitude violations may be acceptable to avoid more severe consequences like load shedding.
3. **Priority-ranked constraints**: Not all constraints are equally important. Frequency stability is more critical than SOC bounds, yet standard CMDP treats all constraints as equally binding.

The crisp constraint idealization leads to three well-documented pathologies in Safe RL: (i) **over-conservatism**, where policies stay far inside the feasible region to avoid sharp boundaries, wasting safe interior space; (ii) **boundary oscillation**, where policies oscillate near constraint boundaries due to discontinuous gradient signals; and (iii) **sparse constraint gradients**, where the constraint cost provides no learning signal until the boundary is actually crossed, slowing convergence [16, 17].

Beyond the constraint formulation, two additional gaps limit the practical application of Safe RL in microgrids:

- **Neglect of domain expertise**. Purely data-driven Safe RL methods learn from scratch, requiring millions of environment interactions and extensive unsafe exploration during training. Microgrid operation, however, benefits from decades of accumulated expert knowledge—operational rules, heuristic guidelines, and proven dispatch strategies. Embedding this knowledge into Safe RL could substantially improve sample efficiency and initial safety, but existing methods lack a principled framework for doing so [18].
- **Unsafe cross-scenario transfer**. Microgrids operate under varying conditions—seasonal changes, grid-connected versus islanded modes, normal versus extreme weather. Training a new policy from scratch for each scenario is wasteful and requires unsafe exploration. Transfer learning has been applied to microgrid dispatch [19], but existing methods focus on accelerating reward optimization and do not guarantee constraint satisfaction during or after transfer.

## 1.3 Our Work

To address these gaps, we propose a **Hierarchical Fuzzy-Guided Safe Reinforcement Learning (HFG-SRL)** framework for microgrid optimal dispatch. Our approach integrates fuzzy logic into Safe RL at three levels: constraint formulation, knowledge embedding, and cross-scenario transfer.

First, we introduce the **Fuzzy Constrained Markov Decision Process (FC-MDP)**, which generalizes the standard CMDP by replacing crisp constraint thresholds with continuous Fuzzy Constraint Satisfaction Degrees (FCSD). In FC-MDP, each state-action pair has a satisfaction degree in [0, 1] for every constraint, rather than a binary feasible/infeasible label. We derive a **Fuzzy-Lagrangian** method for solving FC-MDP and prove its convergence to a saddle point under standard regularity conditions. We also show that FC-MDP recovers standard CMDP as a limiting case when the fuzzy boundary width approaches zero.

Second, we develop the **Hierarchical Fuzzy-Guided SAC (HFG-SAC)** algorithm, built on the Soft Actor-Critic (SAC) architecture with two complementary fuzzy layers. The **upper layer** implements fuzzy constraint protection: a TSK fuzzy system computes the joint multi-constraint satisfaction degree, and the Fuzzy-Lagrangian mechanism provides smooth constraint gradients to the actor network—eliminating boundary oscillation and enabling efficient learning. The **lower layer** performs fuzzy knowledge reward shaping: expert operational rules (encoded as TSK fuzzy rules) provide dense, potential-based reward guidance that accelerates learning while provably preserving the optimal policy. The knowledge influence decays adaptively over training to ensure asymptotic optimality.

Third, we design a **constraint-aware cross-scenario transfer mechanism** that safely adapts learned policies across microgrid operating conditions. A Jaccard similarity of fuzzy membership functions categorizes each constraint into transferable, adaptable, or relearn groups; the target fuzzy system is initialized from the source parameters under expert adjustment; and an exponentially relaxed conservative action scale keeps early fine-tuning inside a safe exploration tube around the source policy.

## 1.4 Contributions

The main contributions of this paper are threefold:

> **Contribution 1 — Theoretical**: We propose the Fuzzy Constrained Markov Decision Process (FC-MDP) framework, which generalizes standard CMDP to handle fuzzy safety constraints with continuous satisfaction degrees. A Fuzzy-Lagrangian method is derived with provable convergence properties, and the relationship to standard CMDP is established via a limiting theorem.
>
> **Contribution 2 — Methodological**: We develop the Hierarchical Fuzzy-Guided SAC (HFG-SAC) algorithm with two complementary fuzzy layers: (i) an upper-layer fuzzy constraint protection mechanism that ensures safe exploration with smooth constraint boundaries, and (ii) a lower-layer fuzzy knowledge reward shaping engine that embeds expert operational rules for accelerated learning with guaranteed policy invariance.
>
> **Contribution 3 — Application**: We design a constraint-aware cross-scenario transfer mechanism that safely adapts learned policies across microgrid operating modes (grid-connected / islanded) and seasonal conditions. The mechanism (i) quantifies per-constraint transferability via Jaccard similarity of fuzzy membership functions, (ii) initializes the target fuzzy system from the source parameters under expert adjustment, and (iii) controls early fine-tuning with an exponentially relaxed conservative action scale.

## 1.5 Organization

The remainder of this paper is organized as follows. Section 2 reviews related work on Safe RL, fuzzy logic in reinforcement learning, and DRL-based microgrid dispatch. Section 3 provides preliminaries on CMDP, SAC, and TSK fuzzy systems, formulates the microgrid dispatch problem, and introduces the fuzzy constraint formulation. Section 4 presents our proposed method: the FC-MDP framework, Fuzzy-Lagrangian solution, HFG-SAC algorithm, and constraint-aware transfer mechanism. Section 5 reports experimental results on a modified IEEE 33-bus microgrid system, including performance comparisons, ablation studies, sensitivity analysis, and transfer learning evaluations. Section 6 discusses theoretical implications, practical insights, and limitations. Section 7 concludes the paper and outlines future directions.

---

## References (placeholder)

[1] – Microgrid review paper
[2] – DER integration review
[3] – Microgrid optimal dispatch review
[4] – Uncertainty in microgrid dispatch
[5] – MILP for microgrid
[6] – MPC for microgrid
[7] – DRL for energy management
[8] – SAC for microgrid
[9] – Safe RL survey (García & Fernández)
[10] – Safe RL survey (recent)
[11] – Constrained MDP book (Altman)
[12] – CMDP definition
[13] – Lagrangian methods for CMDP
[14] – CPO (Achiam et al., 2017)
[15] – TRPO-Lagrangian
[16] – Constrained RL optimization issues
[17] – Reward constraints vs policy constraints
[18] – Knowledge embedding in RL
[19] – Transfer learning for microgrid dispatch

---

*End of Chapter 1.*

---

**Word count**: ~1,260 words
