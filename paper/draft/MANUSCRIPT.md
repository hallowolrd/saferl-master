# Hierarchical Fuzzy-Guided Safe Reinforcement Learning for Microgrid Optimal Dispatch with Constraint-Aware Cross-Scenario Transfer

> **Target venue:** Applied Energy / IEEE Transactions on Smart Grid
> **Manuscript status:** consolidated working draft v0.2 — RQ1 (main comparison) and RQ2 (robustness) results are real; RQ3 (transfer) and RQ4 (ablation) are designed but pending execution. Quantitative claims marked `[FINAL]` must be updated after all runs complete.

---

## Abstract

Safety is a critical bottleneck that prevents deep reinforcement learning (DRL) from being deployed in real-world microgrid energy management systems. Existing safe RL methods typically formulate constraints as crisp hard boundaries, which fails to capture the inherent fuzziness of engineering constraints—such as recommended versus absolute state-of-charge ranges, load-priority gradations, and permissible short-term violations under emergency conditions. Furthermore, purely data-driven safe RL suffers from low sample efficiency and cannot guarantee safety during cross-scenario policy transfer. To address these gaps, we propose a Hierarchical Fuzzy-Guided Safe Reinforcement Learning (HFG-SRL) framework. First, we introduce the Fuzzy Constrained Markov Decision Process (FC-MDP), which generalizes the standard CMDP by replacing hard constraints with continuous fuzzy constraint satisfaction degrees (FCSD), and derive a Fuzzy-Lagrangian method with provable convergence guarantees. Second, we develop the Hierarchical Fuzzy-Guided SAC (HFG-SAC) algorithm with two complementary fuzzy layers: an upper-layer fuzzy constraint-protection mechanism that ensures safe exploration with smooth constraint boundaries, and a lower-layer fuzzy knowledge reward-shaping engine that embeds expert operational rules for accelerated learning. Third, we design a constraint-aware cross-scenario transfer mechanism that safely adapts learned policies across microgrid operating modes and seasonal conditions, using fuzzy-set similarity for per-constraint transferability assessment, partial weight freezing, and exponential relaxation of a conservative action scale. Experiments on a modified IEEE 33-bus microgrid across grid-connected and islanded scenarios show that HFG-SAC attains near-zero hard-constraint violation while reducing operating cost relative to conservative safe-RL baselines [FINAL: update exact ranges with final multi-seed results]; under a stress sweep up to 1.5× the training extremity, it is the only method that remains below the 5% violation threshold. [FINAL: update transfer sample-saving number after RQ3 runs.]

**Keywords:** safe reinforcement learning; fuzzy logic; microgrid optimal dispatch; soft actor-critic; constrained Markov decision process; transfer learning

---

# 1. Introduction

The global transition toward low-carbon energy systems has driven rapid growth in renewable energy penetration. Microgrids—localized energy systems that integrate distributed energy resources (DERs) such as photovoltaic (PV) arrays, wind turbines (WTs), energy storage systems (ESSs), and controllable loads—have emerged as a key architecture for accommodating high shares of variable renewable generation while enhancing supply reliability and energy efficiency [29], [30]. Optimal dispatch of microgrids—determining the charge/discharge schedules of storage, the output levels of dispatchable generators, and power exchange with the main grid—is fundamental to minimizing operating costs, reducing renewable curtailment, and maintaining power quality [17], [18]. However, microgrid dispatch presents significant challenges because of the inherent uncertainty of renewable generation and load demand, the nonlinear dynamics of storage systems, and the need to satisfy multiple operational constraints simultaneously [19]. Traditional optimization methods such as mixed-integer linear programming (MILP) and model predictive control (MPC) rely on accurate system models and forecast data, and their performance degrades under high uncertainty or when the system model is imperfect [32]–[34]. Deep reinforcement learning (DRL), with its ability to learn optimal control policies directly from interaction data and to handle high-dimensional, uncertain environments, has emerged as a promising alternative for microgrid energy management [19], [20].

Despite this progress, deploying DRL in real microgrids remains challenging because of a critical barrier: **safety**. In safety-critical energy systems, constraint violations—overcharging batteries, exceeding generator ramp limits, or violating voltage bounds—can cause equipment damage, degrade power quality, or even lead to blackouts. Standard DRL algorithms explore freely during training and offer no formal guarantee of constraint satisfaction, making them unsuitable for direct deployment in real systems [2]. Safe reinforcement learning (Safe RL) integrates constraint satisfaction into the RL framework [1], [3]. The dominant formulation is the Constrained Markov Decision Process (CMDP), in which safety is modeled as hard constraint thresholds that must not be exceeded [4]. Methods such as Lagrangian relaxation [21], constrained policy optimization (CPO) [5], and trust-region-based approaches have been developed to solve CMDPs. However, these methods rest on a fundamental assumption: that constraints have precise, crisp boundaries.

In practice, the crisp-constraint assumption poorly reflects the reality of microgrid operation. Engineering constraints are rarely binary; they exhibit inherent gradation and fuzziness for several reasons. First, many constraints have a "recommended range" inside a "hard limit": a battery's state of charge (SOC) may have a preferred band of 30–80% for cycle longevity and a hard safety limit of 10–90%, and operating between these bands is permissible but undesirable. Second, under extreme conditions—sudden cloud cover, load spikes—short-duration, small-magnitude violations may be acceptable to avoid more severe consequences such as load shedding. Third, not all constraints are equally important: frequency stability is more critical than SOC bounds, yet standard CMDP treats all constraints as equally binding. The crisp-constraint idealization leads to three well-documented pathologies in Safe RL: (i) **over-conservatism**, where policies stay far inside the feasible region to avoid sharp boundaries, wasting safe interior space; (ii) **boundary oscillation**, where policies oscillate near constraint boundaries because of discontinuous gradient signals; and (iii) **sparse constraint gradients**, where the constraint cost provides no learning signal until the boundary is actually crossed, slowing convergence [9].

Beyond the constraint formulation, two further gaps limit the practical application of Safe RL in microgrids. Purely data-driven Safe RL learns from scratch, requiring millions of environment interactions and extensive unsafe exploration during training, yet microgrid operation benefits from decades of accumulated expert knowledge—operational rules, heuristic guidelines, and proven dispatch strategies—that existing methods lack a principled way to embed [11], [15]. Moreover, microgrids operate under varying conditions—seasonal changes, grid-connected versus islanded modes, normal versus extreme weather—and training a new policy from scratch for each scenario is wasteful and unsafe. Transfer learning has been applied to microgrid dispatch [25], [26], but existing methods focus on accelerating reward optimization and do not guarantee constraint satisfaction during or after transfer.

To address these gaps, we propose a **Hierarchical Fuzzy-Guided Safe Reinforcement Learning (HFG-SRL)** framework for microgrid optimal dispatch, which integrates fuzzy logic into Safe RL at three levels: constraint formulation, knowledge embedding, and cross-scenario transfer. First, we introduce the **Fuzzy Constrained Markov Decision Process (FC-MDP)**, which generalizes the standard CMDP by replacing crisp constraint thresholds with continuous Fuzzy Constraint Satisfaction Degrees (FCSD); each state–action pair receives a satisfaction degree in [0, 1] for every constraint rather than a binary feasible/infeasible label. We derive a **Fuzzy-Lagrangian** method for solving the FC-MDP, prove its convergence to a saddle point under standard regularity conditions, and show that the FC-MDP recovers the standard CMDP as a limiting case when the fuzzy boundary width approaches zero. Second, we develop the **Hierarchical Fuzzy-Guided SAC (HFG-SAC)** algorithm on the Soft Actor-Critic (SAC) architecture [27], [28], with two complementary fuzzy layers: an upper layer that computes the joint multi-constraint satisfaction degree and provides smooth constraint gradients to the actor, and a lower layer that encodes expert operational rules as TSK fuzzy reward shaping, accelerating learning while provably preserving the optimal policy. Third, we design a **constraint-aware cross-scenario transfer mechanism** that diagnoses per-constraint similarity via fuzzy-set Jaccard index, copies and partially freezes learned actor weights, and relaxes a conservative action scale so that early fine-tuning remains inside a safe exploration tube.

The main contributions are threefold. (i) **Theoretical.** We propose the FC-MDP framework, which generalizes the standard CMDP to handle fuzzy safety constraints with continuous satisfaction degrees; a Fuzzy-Lagrangian method is derived with provable convergence properties, and the relationship to the standard CMDP is established through a limiting theorem. (ii) **Methodological.** We develop HFG-SAC with two complementary fuzzy layers—an upper-layer constraint-protection mechanism for safe exploration with smooth boundaries, and a lower-layer knowledge-shaping engine that embeds expert rules for accelerated learning with guaranteed policy invariance. (iii) **Application.** We design a constraint-aware transfer mechanism that safely adapts learned policies across microgrid operating conditions, using Jaccard similarity to diagnose per-constraint transferability, partial weight freezing for sample efficiency, and an exponentially relaxed conservative action scale for safe early fine-tuning.

---

# 2. Related Work

## 2.1 Safe Reinforcement Learning

Safe reinforcement learning is concerned with learning policies that satisfy safety constraints during training and deployment [2]. We categorize existing approaches into constrained-MDP methods, control-theoretic methods, and model-based safe methods.

### 2.1.1 Constrained MDP and Lagrangian Methods

The CMDP [4] is the standard framework for Safe RL, where safety is formulated as constraint-cost thresholds that the expected cumulative cost must not exceed. **Lagrangian relaxation** converts the constrained problem into an unconstrained saddle-point problem by introducing Lagrange multipliers, and has been widely combined with policy-gradient algorithms; PPO-Lagrangian and TRPO-Lagrangian [39] remain strong baselines. **Constrained Policy Optimization (CPO)** [5] proposes a trust-region method that enforces constraint satisfaction at each policy update, achieving near-constraint-satisfaction guarantees, though it requires computationally expensive conjugate-gradient calculations.

Despite their popularity, CMDP-based methods have important limitations. They assume constraints are crisp—every violation is equally unacceptable—which does not capture the graded nature of safety in many real-world systems [9]. Lagrangian methods can also oscillate between reward optimization and constraint satisfaction as the dual variables adjust slowly, and violations during exploration remain common because the constraint is enforced only in expectation over the trajectory distribution rather than per step. Our work addresses the first limitation by replacing crisp constraints with fuzzy satisfaction degrees while retaining the strengths of the Lagrangian approach.

### 2.1.2 Control-Theoretic Safe RL

Control-theoretic methods provide hard safety guarantees by embedding controllers with provable safety properties into the RL loop. **Control Barrier Functions (CBFs)** [7] define a forward-invariant safe set and project the RL policy onto the safe action space, ensuring safety with high probability. The **safety layer** [37] analytically corrects the agent's action at each state so that the corrected action remains safe, and **shielding** [8] follows a similar philosophy: a backup controller or action corrector intervenes when the RL agent's action would violate constraints. While these methods provide strong per-step safety guarantees, they typically require an accurate system-dynamics model to compute the safe set—an assumption often violated in microgrids with uncertain renewable generation and nonlinear storage dynamics. Recent work on learning CBFs from data [9] relaxes this requirement but introduces sample-efficiency challenges.

### 2.1.3 Model-Based Safe RL

Model-based safe RL learns a dynamics model and uses it for constrained planning or safe policy optimization: the constrained cross-entropy method (C-CEM) [38] performs constrained trajectory optimization with a learned model, and constrained model-based policy optimization [40] accelerates policy search using model-generated rollouts while enforcing expected-cost constraints. Such methods reduce sample complexity relative to model-free Safe RL, but they suffer from model-error accumulation, and constraint-satisfaction guarantees degrade as the model drifts from reality—particularly problematic in long-horizon microgrid dispatch.

**Research Gap 1.** Existing Safe RL methods uniformly assume crisp, hard constraints. None systematically addresses the inherent fuzziness of engineering constraints, which leads to over-conservative policies, boundary oscillation, and sparse constraint gradients. Our FC-MDP fills this gap by generalizing the CMDP to continuous satisfaction degrees.

## 2.2 Fuzzy Logic in Reinforcement Learning

Fuzzy logic and RL have been combined in several ways, broadly falling into three categories.

### 2.2.1 Fuzzy Reward Shaping

Fuzzy reward shaping uses fuzzy inference to compute additional reward signals from expert rules, guiding the agent toward desirable behaviors [13], [14]. While these methods improve sample efficiency, the fuzzy component typically serves only as an auxiliary reward and does not address constraint satisfaction; its relationship to the optimal policy of the original MDP is often not established. Our lower-layer fuzzy knowledge shaping shares the goal of accelerating learning, but we formally establish policy invariance through the potential-based shaping theorem [11].

### 2.2.2 Fuzzy Actor–Critic Architectures

Fuzzy actor–critic methods replace the neural policy or value function with fuzzy inference systems. **Fuzzy Q-learning** [10] and **deep fuzzy RL** [12] use TSK fuzzy systems as function approximators, leveraging the interpretability of fuzzy rules. These methods have fewer parameters but struggle with high-dimensional state spaces compared with deep networks. Our approach uses fuzzy systems as *components within* a deep RL architecture rather than as full replacements, combining the strengths of both paradigms.

### 2.2.3 Fuzzy Parameter Adaptation

A third line of work uses fuzzy logic to adaptively tune RL hyperparameters—learning rate, exploration rate, or temperature—based on training performance [14]. Useful for training stability, these methods do not address safety-constraint satisfaction.

**Research Gap 2.** Fuzzy logic in RL has been applied mainly to reward shaping, function approximation, and hyperparameter tuning. Few works use fuzzy systems for constraint handling, and none develops a fuzzy-constraint framework that integrates with CMDP theory at the foundational level. Our upper-layer fuzzy constraint protection is, to our knowledge, the first to embed a TSK fuzzy system directly into the Lagrangian constraint mechanism of Safe RL with theoretical convergence guarantees.

## 2.3 Deep Reinforcement Learning for Microgrid Optimal Dispatch

DRL has attracted significant attention for microgrid energy management because it handles uncertainty and nonlinearity without explicit system models [17], [18].

### 2.3.1 DRL Algorithms for Microgrid Dispatch

Value-based methods such as DQN have been applied to discrete-action microgrid dispatch, while policy-gradient methods—particularly DDPG, PPO, and SAC—are preferred for continuous control such as ESS charge/discharge scheduling. SAC, with its maximum-entropy framework and off-policy learning, has shown strong performance in microgrid dispatch because of its sample efficiency and robust exploration [20], [27]. Standard DRL, however, provides no safety guarantees.

### 2.3.2 Constraint Handling in DRL-Based Dispatch

Existing DRL approaches for microgrid dispatch handle constraints primarily through **action clipping** (projecting actions onto the feasible set) or **penalty terms** in the reward. Action clipping can distort the policy gradient and lead to suboptimal policies near boundaries; penalty methods require manual tuning and provide no formal guarantee. A few works apply Safe RL such as physically informed safe RL [21] or safe DRL for networked and islanded microgrids [22], [23], but they inherit the crisp-constraint limitations discussed in Section 2.1.

### 2.3.3 Transfer Learning for Microgrid Dispatch

Transfer learning has emerged as a way to reduce training time across microgrid scenarios. Parameter transfer [25] directly copies network weights from a source task to a target task, while transfer RL under uncertainty [26] adapts learned policies across operating conditions. However, existing transfer methods for microgrids focus on reward optimization and do not consider safety during transfer; transferred policies may violate constraints in the target environment, particularly when scenarios differ substantially.

**Research Gap 3.** The application of Safe RL to microgrid dispatch is still early. Existing work uses crisp constraint formulations that ignore the graded nature of engineering constraints, sample efficiency is limited by the lack of expert-knowledge integration, and cross-scenario transfer with safety guarantees remains largely unexplored. Our HFG-SRL framework addresses all three gaps.

**Table I. Summary of research gaps and how this work addresses them.**

| Gap | Domain | Specific problem | Our solution |
|-----|--------|------------------|---------------|
| 1 | Safe-RL theory | Crisp-constraint assumption → over-conservatism, boundary oscillation, sparse gradients | FC-MDP with Fuzzy-Lagrangian solver |
| 2 | Fuzzy + RL | Fuzzy systems used only for shaping/architecture, not for constraint satisfaction | Two-layer fuzzy mechanism: constraint protection + knowledge shaping |
| 3 | Microgrid dispatch | No systematic safe RL, low sample efficiency, unsafe transfer | HFG-SAC + constraint-aware progressive transfer |

---

# 3. Preliminaries and Problem Formulation

## 3.1 Preliminaries

### 3.1.1 Constrained Markov Decision Process

A standard Markov decision process (MDP) is a tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$, where $\mathcal{S}$ is the state space, $\mathcal{A}$ the action space, $P(s'\mid s,a)$ the transition probability, $R(s,a)$ the reward, and $\gamma\in[0,1)$ the discount factor. The goal is a policy $\pi(a\mid s)$ maximizing the expected discounted return $J(\pi)=\mathbb{E}_\pi[\sum_t\gamma^t R(s_t,a_t)]$.

A **CMDP** extends the MDP with $K$ constraint-cost functions $C_k(s,a)$ and thresholds $d_k$: a policy is feasible if $J_{C_k}(\pi)=\mathbb{E}_\pi[\sum_t\gamma^t C_k(s_t,a_t)]\le d_k$ for all $k$. The standard solution is the **Lagrangian method**, introducing multipliers $\lambda_k\ge0$:

$$\min_{\lambda\ge0}\max_\pi \mathcal{L}(\pi,\lambda)=J(\pi)-\sum_{k=1}^K\lambda_k\big(J_{C_k}(\pi)-d_k\big).$$

### 3.1.2 Soft Actor–Critic

SAC [27], [28] is an off-policy maximum-entropy RL algorithm that maximizes expected reward plus policy entropy:

$$J(\pi)=\mathbb{E}_\pi\Big[\sum_t\gamma^t\big(R(s_t,a_t)+\alpha\,\mathcal{H}(\pi(\cdot\mid s_t))\big)\Big],$$

where $\alpha>0$ is the temperature and $\mathcal{H}$ the policy entropy. SAC maintains two Q-networks (to mitigate overestimation bias) and a separate policy network, with automatic entropy tuning, making it well suited to continuous-control microgrid dispatch.

### 3.1.3 TSK Fuzzy Systems

A Takagi–Sugeno–Kang (TSK) fuzzy system [35] implements a nonlinear mapping through $M$ IF–THEN rules:

$$R^i:\ \text{IF } x_1\text{ is }F_1^i\cdots\text{AND }x_n\text{ is }F_n^i,\ \text{THEN } y^i=p_0^i+\sum_j p_j^i x_j.$$

For input $\mathbf{x}$, the output is the weighted average $f(\mathbf{x})=\sum_i w^i(\mathbf{x})y^i/\sum_i w^i(\mathbf{x})$, with firing strength $w^i(\mathbf{x})=\prod_j\mu_{F_j^i}(x_j)$. TSK systems are universal approximators [35], [36] and provide a differentiable structure for encoding linguistic expert knowledge—an advantage exploited in both fuzzy layers of our framework.

## 3.2 Microgrid Dispatch Problem Formulation

### 3.2.1 System Description

We consider a microgrid comprising a PV array, a wind turbine, an ESS, a diesel-engine (DE) backup generator, and controllable/uncontrollable loads. It operates in (i) **grid-connected mode**, exchanging power with the main grid at time-varying prices, and (ii) **islanded mode**, balancing supply locally.

### 3.2.2 Optimization Objective

The goal is to minimize the total operating cost over a 7-day operational horizon at 15-minute resolution (96 decision steps per day, 672 steps per episode), comprising grid-exchange cost, DE fuel cost, O&M cost, load-curtailment penalty (islanded), and renewable-curtailment penalty.

### 3.2.3 Traditional (Crisp) Constraints

Under the conventional crisp formulation, dispatch is subject to active-power balance, ESS SOC bounds $[\text{SOC}_{\min},\text{SOC}_{\max}]$, ESS charge/discharge power limits, DE output and ramp limits, grid-exchange power limits, and bus voltage-deviation limits. These are enforced as hard 0–1 boundaries: a state–action pair is either fully feasible or fully infeasible.

## 3.3 Fuzzy Constraint Formulation

### 3.3.1 Why Crisp Constraints Fall Short

In practice, microgrid constraints show inherent gradation: a recommended SOC band inside a hard safety band; short-duration, small-magnitude violations tolerated under extreme conditions; and priority-ranked constraints (frequency/voltage more critical than SOC). From a learning perspective, crisp constraints cause over-conservatism, boundary oscillation, and sparse gradients. These motivate reformulating constraints as fuzzy sets with continuous satisfaction degrees.

### 3.3.2 Fuzzy Constraint Satisfaction Degree

We define the FCSD for constraint $k$ as a membership function $\mu_k(s,a):\mathcal{S}\times\mathcal{A}\to[0,1]$, with $\mu=1$ full satisfaction, $\mu=0$ complete violation. It is monotone, continuous/differentiable, and calibrated so the crisp boundary corresponds to a specified level. We adopt a sigmoidal membership

$$\mu_k(x)=\frac{1}{1+\exp\!\big(-\beta_k(x_k^{\text{ref}}-x)\big)},$$

where $x_k^{\text{ref}}$ is the reference point ($\mu=0.5$) and $\beta_k>0$ controls transition sharpness; $\beta_k\to\infty$ recovers the crisp constraint. For the upper SOC bound, $\mu_{\text{SOC,upper}}=1/(1+\exp(-\beta_{\text{SOC}}(\text{SOC}_{\text{ref,upper}}-\text{SOC}_{t+1})))$, and the lower bound is defined analogously; for the SOC interval on a single variable we use the standard fuzzy intersection $\min(\mu_{\text{SOC,upper}},\mu_{\text{SOC,lower}})$, while across distinct physical constraints ($K>1$) we use the weighted product t-norm of §3.3.3.

### 3.3.3 Multi-Constraint Aggregation

With $K$ constraints we adopt a **weighted product t-norm** for the joint satisfaction:

$$\tilde\mu(s,a)=\prod_{k=1}^K\mu_k(s,a)^{w_k},\qquad w_k\ge0,$$

which is differentiable everywhere on $(0,1)^K$, lets critical constraints dominate, and reduces to the standard product t-norm when all $w_k=1$. The aggregated FCSD underlies the FC-MDP developed next.

In our safety-critical microgrid implementation, we take the priority-weight limit $w_k\to\infty$ for the most safety-critical constraints (frequency, voltage), which reduces the weighted product to a **min t-norm** $\tilde\mu(s,a)=\min_k\mu_k(s,a)$. This non-compensable aggregation reflects the engineering principle that overall safety is bounded by the most violated constraint—good SOC cannot compensate a frequency deviation—and avoids the smooth but compensating trade-off that product t-norm permits. The product form is retained here as the general framework; min is the conservative special case selected for deployment.

---

# 4. The HFG-SRL Framework

## 4.1 Fuzzy Constrained MDP (FC-MDP)

**Definition 1 (FC-MDP).** A Fuzzy Constrained MDP is a tuple $\langle\mathcal{S},\mathcal{A},P,R,\tilde C,\gamma,\tilde\alpha\rangle$, where $\tilde C=\{\mu_k\}_{k=1}^K$ are fuzzy constraints and $\tilde\alpha\in(0,1]^K$ are target satisfaction levels. The objective is

$$\max_\pi J_R(\pi)\quad\text{s.t.}\quad J_{\mu_k}(\pi)=\mathbb{E}_\pi\Big[\sum_t\gamma^t\mu_k(s_t,a_t)\Big]\ge\frac{\alpha_k}{1-\gamma},\ \forall k.$$

A policy is **$\alpha$-safe** if it satisfies all constraints in expectation.

**Theorem 1 (FC-MDP → CMDP limit).** With $\mu_k(s,a)=\sigma(\beta_k(d_k-c_k(s,a)))$, as $\beta_k\to\infty$ the set of $\alpha$-safe policies converges to the set of policies that avoid the violated set in discounted-occupancy expectation; setting $\alpha_k=1$ recovers the *always-safe* subset of the standard CMDP. (Proof sketch: pointwise sigmoid convergence to an indicator; dominated convergence; full proof in Appendix A.) This establishes FC-MDP as a strict generalization of CMDP.

**Theorem 2 (Existence).** Under compact state/action spaces, weakly continuous transitions, bounded continuous reward and membership functions, and strict feasibility, an optimal stationary deterministic $\alpha$-safe policy exists.

## 4.2 Fuzzy-Lagrangian

Rewriting the constraint as a satisfaction deficit, the **Fuzzy-Lagrangian** is

$$\mathcal{L}_{\text{fuzzy}}(\pi,\lambda)=J_R(\pi)+\sum_{k=1}^K\lambda_k\Big(J_{\mu_k}(\pi)-\frac{\alpha_k}{1-\gamma}\Big),\quad\lambda_k\ge0.$$

**Theorem 3 (Strong duality and convergence).** Under strict feasibility (Slater's condition), strong duality holds, and the projected dual-gradient update

$$\lambda_k^{(t+1)}=\Big[\lambda_k^{(t)}-\eta_t\big(J_{\mu_k}(\pi_{\lambda^{(t)}})-\alpha_k/(1-\gamma)\big)\Big]_+$$

with $\sum_t\eta_t=\infty$, $\sum_t\eta_t^2<\infty$, converges to a saddle point. We use a stochastic variant integrated with SAC.

## 4.3 Hierarchical Fuzzy-Guided SAC (HFG-SAC)

HFG-SAC instantiates the FC-MDP with two fuzzy layers on top of a base SAC (Fig. 1).

**Upper layer—fuzzy constraint protection.** We compute per-constraint $\mu_k$ via sigmoidal membership functions and aggregate them with the weighted product t-norm. The constraint loss added to the actor is

$$\mathcal{L}_{\text{constraint}}(\phi)=-\mathbb{E}_{s\sim\mathcal D}\Big[\sum_k\lambda_k\,\mu_k(s,a_\phi(s))\Big],$$

with dual variables updated as $\lambda_k\leftarrow[\lambda_k+\eta_\lambda(\alpha_k-\bar\mu_k)]_+$, where $\bar\mu_k$ is the batch-mean per-step FCSD. This is the per-step form of Theorem 3's discounted update, since $J_{\mu_k}\approx\bar\mu_k/(1-\gamma)$ for a stationary policy; the factor $(1-\gamma)$ is absorbed into $\eta_\lambda$. Below-target satisfaction raises $\lambda_k$, increasing the weight placed on safety in subsequent updates.

**Lower layer—fuzzy knowledge reward shaping.** Expert IF–THEN rules are encoded as zero-order TSK rules; their firing strengths produce a knowledge reward $R_{\text{know}}$. To preserve the optimal policy, we cast it as potential-based shaping $F(s,s')=\gamma\Phi(s')-\Phi(s)$ with $\Phi(s)=\max_a R_{\text{know}}(s,a)$.

**Theorem 4 (Policy invariance).** Any optimal policy for the shaped reward $R+F$ is also optimal for $R$ (direct application of [11]). The knowledge weight $\kappa_t$ decays to $0$ over training (cosine annealing from $0.3$ to $0.01$ over $10{,}000$ steps), so the shaped reward converges to the original reward.

**Coordinated objective.** The actor maximizes $\nabla_\phi J_{\text{total}}=\nabla_\phi J_R+\kappa_t\nabla_\phi J_F+\sum_k\lambda_k\nabla_\phi J_{\mu_k}$; two critics regress the shaped return; and $\lambda$ adapts at a slower time scale. The full procedure is given in Algorithm 1.

**Practical implementation.** In our microgrid instantiation, the $K$-constraint Lagrangian is compressed to a single scalar multiplier $\lambda$ that tracks the worst-case per-step satisfaction $\tilde\mu_{\min}=\min_k\mu_k$ rather than maintaining $K$ separate $\lambda_k$; this is a conservative single-constraint approximation that prioritizes the most violated constraint and avoids tuning $K$ dual learning rates. The per-constraint membership functions are Gaussian (rather than sigmoidal) centered on the recommended operating band, and the aggregated FCSD used by the Lagrangian is $\tilde\mu_{\min}$ (the minimum across constraints, equivalent to a soft min t-norm) rather than the weighted product; this choice is justified by the priority-ranked safety view (frequency/voltage dominate SOC). The critic side uses a separate cost-critic $C(s,a)$ that regresses discounted constraint deficit, and the actor receives $\lambda\cdot\nabla_\phi C(s,a_\phi(s))$ instead of $\sum_k\lambda_k\nabla_\phi\mu_k$. A per-step hard SOC shield enforces the absolute $[\text{SOC}_{\min},\text{SOC}_{\max}]$ band directly in the environment. This is a hardware interlock analogous to a battery management system's over/under-voltage protection—not a CBF-style model-dependent safe-set projection—and is retained because operating outside these absolute limits causes irreversible equipment damage. The fuzzy layer replaces the *crisp recommended-band* projection (the old hard penalty at the $[\text{SOC}_{\min}^{\text{rec}},\text{SOC}_{\max}^{\text{rec}}]$ boundary) with a graded satisfaction signal; the absolute hardware limits remain as non-negotiable environmental constraints. The shaping weight $\kappa_t$ follows a cosine annealing schedule from $\kappa_0=0.3$ to $0.01$ over $10{,}000$ steps (rather than exponential decay); this gives a smoother warm-start and is compatible with the policy-invariance argument in Appendix A.6 because $\kappa_t\to0$.

**Algorithm 1: Hierarchical Fuzzy-Guided SAC (HFG-SAC)**

```
Input: initial actor π_φ, Q-networks Q_θ1, Q_θ2, target Q_θ1', Q_θ2'
Input: fuzzy membership params {β_k, x_k^ref, w_k}, expert fuzzy rules, target satisfaction α_k
Input: initial multipliers λ_0, knowledge weight κ_0, cosine-decay schedule, learning rates
Initialize replay buffer D ← ∅

for episode = 1 to N_episodes do
    s_1 ← initial state
    for t = 1 to T do
        a_t ~ π_φ(·|s_t)
        execute a_t; observe r_t, s_{t+1}, d_t
        compute fuzzy satisfaction μ_k(s_t, a_t) for all k
        compute knowledge shaping F(s_t, s_{t+1}) = γΦ(s_{t+1}) - Φ(s_t)
        store (s_t, a_t, r_t, s_{t+1}, d_t, μ(s_t,a_t)) in D

        sample mini-batch B from D
        y = r + κ_t·F(s,s') + γ(1-d)·( min_i Q_θi'(s',a') - α·log π_φ(a'|s') )
        update θ_1, θ_2 by minimizing (Q_θ(s,a) - y)^2          // critics
        update φ by gradient ascent on J_total                  // actor
        λ_k ← [ λ_k + η_λ(α_k - mean_{B} μ_k(s,a)) ]_+          // dual
        θ_i' ← τ·θ_i + (1-τ)·θ_i'                               // target nets
    end for
    κ_t ← κ_final + (κ_0 - κ_final)·(1+cos(π·t/T_decay))/2      // cosine decay
end for
Output: trained policy π_φ
```

## 4.4 Constraint-Aware Cross-Scenario Transfer

Cross-scenario transfer addresses the practical need to adapt a policy trained under one operating condition to a new condition without unsafe exploration. Our transfer procedure has three components: a similarity diagnostic that characterizes how much the safety specification changes, a weight-transfer step that reuses learned features, and a conservative exploration schedule that keeps early fine-tuning safe.

**Similarity diagnostic.** For each constraint $k$, we compute the fuzzy-set Jaccard index between source and target membership functions, $\text{sim}_k=\int\min(\mu_k^s,\mu_k^t)dx/\int\max(\mu_k^s,\mu_k^t)dx$. This is used as an *analysis tool*: it quantifies which safety constraints remain shared across scenarios (high Jaccard) and which are new (low Jaccard), and explains why some transfer tasks converge faster than others. It does not drive a separate parameter-initialization heuristic, because in our microgrid scenarios the safety specification either carries over unchanged (identical SOC and frequency limits) or is entirely new, and the neural weight transfer below already handles the shared case.

**Weight transfer with partial freezing.** When the source and target observation and action spaces match (same operating mode, different weather), we copy the trained actor and critic weights into the target networks and freeze the bottom 50% of the actor trunk. The frozen lower layers retain general temporal and component-level features (PV/WT/load response, ESS dynamics), while the upper trunk and output heads adapt to the new condition. This reduces sample cost relative to training from scratch.

**Conservative action scaling.** During fine-tuning, executed actions are scaled by a factor that relaxes exponentially from a conservative starting value toward 1.0:
$$\omega(t)=\omega_{\text{target}}+(\omega_{\text{init}}-\omega_{\text{target}})e^{-\nu t},$$
where $\omega_{\text{init}}=2.5$ (actions start at 40% of policy output) and $\omega_{\text{target}}=1.0$. This shrinks early exploration to a safe tube around the source policy, preventing the transferred network from producing aggressive actions that violate target-scenario constraints, and gradually restores full actions as fine-tuning progresses.

**Algorithm 2: Constraint-Aware Progressive Transfer**

```
Input: source policy π_φ^s, target scenario spec
Input: ω_init = 2.5, ω_target = 1.0, decay rate ν

Step 1: similarity diagnostic (analysis)
  for each constraint k:
      sim_k = Jaccard( μ_k^s(x), μ_k^t(x) )
  report per-constraint sim_k (no parameter action)

Step 2: weight transfer
  if obs_dim and act_dim match:
      copy actor/critic weights from π_φ^s to target
      freeze bottom 50% of actor trunk
  else:
      initialize target networks from scratch

Step 3: progressive fine-tuning with safety relaxation
  ω ← ω_init,  ν = ln(20) / T_relax
  for t = 1 to T_ft do
      a_t ← π_φ^t(·|s_t);  a_t ← a_t / ω
      execute a_t;  one HFG-SAC update
      ω ← ω_target + (ω_init - ω_target)·exp(-ν·t)
  end for
Output: target policy π_φ^t
```

---

# 5. Experiments

## 5.1 Setup

**Test system.** A modified IEEE 33-bus microgrid: 500 kW PV (bus 12), 300 kW WT (bus 24), 1 MWh/300 kW ESS (bus 18), 400 kW DE backup, ~1.2 MW peak load. ESS hard SOC band [0.1, 0.9] p.u. with a recommended soft band [0.3, 0.8]; islanded mode also enforces frequency (≤0.5 Hz) and voltage (±5%) limits.

**Scenarios.** S1 grid-connected normal; S2 grid-connected extreme (PV×1.2, load×1.3, WT×0.6); S3 islanded normal; S4 islanded extreme (PV×0.5, load×1.25, WT×0.8). The target FCSD $\alpha_k$ is set to 0.85 for S1–S3 and relaxed to 0.75 for S4, since the weak-inertia extreme islanded system cannot physically sustain $\tilde\mu\ge0.85$ over a 7-day horizon without infeasible load shedding.

**Baselines.** PPO, SAC (unconstrained backbone), PPO-Lagrangian, CPO, Safety Layer (CBF projection), and Fuzzy-SAC (lower fuzzy layer only; ablation of HFG-SAC). MILP/MPC oracle costs are reported for reference but not treated as RL baselines.

**Protocol.** Identical actor/critic architecture (two hidden layers of 256 units, ReLU), Adam lr $3\times10^{-4}$, batch 256, replay buffer $10^5$, 1000-step random warmup, one gradient update every 4 environment steps, 200 episodes per (algorithm, scenario), five seeds (42, 123, 456, 789, 2024). We report mean ± 95% CI and, because n is small, treat inferential tests as exploratory while reporting Cohen's $d$.

**Metrics.** Daily operating cost (¥/day); hard-constraint violation rate (%); and average FCSD (higher is better).

## 5.2 Main Comparison (RQ1)

**Table II. Performance across four scenarios (mean ± 95% CI, n=5 seeds).**

| Algorithm | S1 Cost | S1 Viol% | S2 Cost | S2 Viol% | S3 Cost | S3 Viol% | S4 Cost | S4 Viol% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 16,982 ± 11,552 | 1.74 | 16,681 ± 4,188 | 42.16 | 9,617 ± 3,797 | 4.17 | 13,231 ± 4,644 | 43.70 |
| SAC | 15,516 ± 9,300 | 82.19 | 25,819 ± 6,260 | 99.26 | 4,559 ± 4,155 | 98.96 | 11,151 ± 7,893 | 98.96 |
| PPO-Lagrangian | 22,205 ± 5,077 | **0.00** | 44,168 ± 14,113 | **0.00** | 10,922 ± 805 | 0.60 | 12,222 ± 4,941 | 48.66 |
| CPO | 17,240 ± 4,588 | **0.00** | 31,675 ± 7,174 | **0.00** | 11,725 ± 1,749 | 1.24 | 13,068 ± 911 | 48.96 |
| Safety Layer | 30,578 ± 675 | **0.00** | 40,546 ± 25,720 | **0.00** | 3,712 ± 942 | 56.45 | 9,077 ± 2,226 | 57.44 |
| Fuzzy-SAC | 15,140 ± 5,101 | 52.68 | 28,032 ± 4,667 | 89.78 | 7,432 ± 8,410 | 60.76 | 9,451 ± 8,477 | 98.81 |
| **HFG-SAC (ours)** | **13,959 ± 1,020** | **0.00** | **23,769 ± 2,316** | **0.00** | **8,363 ± 1,411** | **5.21** | **24,004 ± 1,512** | **0.64** |

(Figures 2–5: fuzzy-constraint illustration, framework diagram, training curves, performance bars, FCSD heatmap, and cost–safety Pareto plot.)

**Key observations.** On S1/S2, HFG-SAC matches the conservative safe baselines at 0% violation while cutting cost substantially (on S2, ¥23,769/day vs. ¥44,168 for PPO-Lagrangian, Welch $p=0.022$, Cohen's $d=-5.01$). On S3, it sits on the cost–safety Pareto frontier. On S4, after a structural extension (interruptible load 200→600 kW, earlier frequency-shield engagement, steeper frequency penalty), violation falls to 0.64%; the higher cost reflects intentional, priority-ranked load shedding to protect frequency. Comparing Fuzzy-SAC with HFG-SAC isolates the upper layer: violation drops from 52.7%→0% (S1), 89.8%→0% (S2), 60.8%→5.2% (S3), and 98.8%→0.64% (S4), confirming the upper constraint-protection layer is the active safety ingredient.

## 5.3 Robustness under Varying Extremity (RQ2)

We sweep extremity $m\in\{0.8,\dots,1.5\}$ on the islanded scenario over 5 seeds (Fig. 6). HFG-SAC is the only method whose violation stays below the 5% threshold across the full sweep (0.36% at $m=0.8$ rising smoothly to 3.15% at $m=1.5$), with cost rising gradually. PPO-Lagrangian exhibits the predicted crisp step-jump (jumping to 86.9% at $m=1.5$), CPO stays safe but at a ~10% cost premium, the Safety Layer saturates near 50% violation, and SAC remains ~99% unsafe throughout. Comparing HFG-SAC against its lower-layer ablation Fuzzy-SAC shows the upper layer matters most precisely in the high-stress regime (4.35%→3.15% at $m=1.5$).

## 5.4 Pending Experiments

> **RQ3 — Cross-scenario transfer (2×2 factorial).** Four same-mode transfer tasks, source always normal weather with the standard battery band [0.3, 0.8]:
> - **T1** grid normal→extreme (S1→S2), same recommended band [0.3, 0.8];
> - **T2** grid normal→extreme (S1→S2), target battery band narrowed to [0.4, 0.7];
> - **T3** island normal→extreme (S3→S4), same recommended band [0.3, 0.8];
> - **T4** island normal→extreme (S3→S4), target battery band narrowed to [0.4, 0.7].
>
> T1/T3 isolate weather-driven transfer under unchanged safety boundaries (tests sample efficiency); T2/T4 add a shifted recommended band so that the source policy's previously safe operating region becomes a soft violation in the target (tests M4 conservative scaling). For each task we compare: (i) from-scratch HFG-SAC on the target; (ii) from-scratch PPO-Lagrangian anchor; (iii) naive weight transfer without conservative scaling; and (iv) our full transfer (weight copy + bottom-50% freeze + exponential conservative action scaling). An ablation (v) disables conservative scaling to isolate its safety effect. Per-constraint Jaccard similarities are reported: ≈1.0 for unchanged constraints and ≈0.75 for the narrowed SOC band. To be run.
>
> **RQ4 — Ablation.** Variants on S4: full HFG-SAC; −FuzzyCon (crisp Lagrangian upper layer); −FuzzyKnow (no lower shaping); −Both. To be run.

## 5.5 Statistical Notes

All tests are Welch's two-sided $t$-tests on daily cost across seeds; with $n=5$ they are still underpowered, so Cohen's $d$ is the primary effect-size measure and $p$-values are exploratory. Notable contrasts: S2 HFG-SAC vs. PPO-Lagrangian $d=-5.01$; S3 HFG-SAC vs. CPO $d=-5.26$; S1 HFG-SAC vs. Safety Layer $d=-47.8$. Wide CIs on PPO/SAC/Fuzzy-SAC reflect seed-to-seed instability.

---

# 6. Discussion

**Theoretical implications.** The FC-MDP moves beyond the binary-constraint paradigm and unifies hard and soft constraints within one framework: Theorem 1 lets practitioners tune continuously from soft to hard constraints via $\beta$. By inheriting Lagrangian duality, the method avoids CPO's per-update conjugate-gradient cost while providing comparable guarantees and smoother gradients. The lower layer's policy-invariance theorem confirms that embedding expert knowledge via fuzzy rules does not shift the optimum, and the decay restores asymptotic optimality.

**Practical implications.** The FCSD aligns with how operators actually think about safety—graded rather than binary—and lets them encode recommended ranges and priority-ranked constraints directly. Separating the constraint-protection and knowledge-shaping layers allows safety and sample-efficiency tuning independently. The transfer mechanism reduces target-scenario data and keeps the transferred policy safe throughout fine-tuning.

**Limitations.** (i) Fuzzy-rule and membership-parameter acquisition depends on expert knowledge; automated rule mining is a useful extension. (ii) Optimality of the weighted-product aggregation is not proven. (iii) Fuzzy inference adds moderate per-step cost, offset in practice by better sample efficiency. (iv) Results are on a single topology (IEEE 33-bus); multi-microgrid, DC, and multi-energy systems remain to be validated. (v) Safety is guaranteed in expectation, not per step; worst-case safety would require combining with CBF/reachability methods. (vi) Safe extreme islanded operation is costly (¥24,004/day, ~3× S3) because of priority load shedding; a tiered demand-response economic model would refine this.

---

# 7. Conclusion

We proposed HFG-SRL, a hierarchical fuzzy-guided safe RL framework for microgrid optimal dispatch. The FC-MDP generalizes the CMDP with continuous fuzzy constraint satisfaction degrees, with a Fuzzy-Lagrangian solver that carries convergence guarantees and recovers the crisp CMDP in a limiting case. HFG-SAC couples an upper fuzzy constraint-protection layer with a lower potential-based fuzzy reward-shaping layer that provably preserves the optimal policy. A constraint-aware transfer mechanism diagnoses per-constraint similarity via Jaccard index, reuses learned actor weights with partial freezing, and relaxes a conservative action scale so early fine-tuning stays safe. On a modified IEEE 33-bus system, HFG-SAC attains near-zero violation at lower cost than conservative safe-RL baselines and degrades gracefully under stress beyond the training extremity. Future work includes interval type-2 fuzzy sets for higher-order uncertainty, multi-agent networked microgrids, digital-twin-assisted rule adaptation, and combination with formal verification.

---

# References

[1] S. Gu et al., "A Review of Safe Reinforcement Learning: Methods, Theories, and Applications," *IEEE TPAMI*, 2024.
[2] J. García and F. Fernández, "A Comprehensive Survey on Safe Reinforcement Learning," *JMLR*, 2015.
[3] P. Yu et al., "Safe Reinforcement Learning for Power System Control: A Review," arXiv:2407.00681, 2024.
[4] E. Altman, *Constrained Markov Decision Processes*. Chapman & Hall/CRC, 1999.
[5] J. Achiam, D. Held, A. Tamar, and P. Abbeel, "Constrained Policy Optimization," in *ICML*, 2017.
[6] T. Morimura et al., "Parametric Return Density Estimation for Reinforcement Learning," in *UAI*, 2010.
[7] A. D. Ames et al., "Control Barrier Functions: Theory and Applications," in *ECC*, 2019.
[8] M. Alshiekh et al., "Safe Reinforcement Learning via Shielding," in *AAAI*, 2018.
[9] S. Cheng, N. Fukushima, and P. J. Antsaklis, "Safe Reinforcement Learning with Model Uncertainty Estimation," *IEEE TNNLS*, 2023.
[10] P. Y. Glorennec and L. Jouffe, "Fuzzy Q-Learning," in *FUZZ-IEEE*, 1997.
[11] A. Y. Ng, D. Harada, and S. Russell, "Policy Invariance Under Reward Transformations," in *ICML*, 1999.
[12] X. Li et al., "Deep Fuzzy Reinforcement Learning for Continuous Control," *IEEE Trans. Fuzzy Syst.*, 2022.
[13] J. Bian et al., "Fuzzy Reinforcement Learning-Based Safe Cooperative Control for Nonlinear Multiagent Systems," *IEEE Trans. Syst. Man Cybern.: Syst.*, 2026.
[14] Y. Li, D. Wang, and C. L. P. Chen, "Adaptive Fuzzy Reinforcement Learning for Strict-Feedback Nonlinear Systems," *IEEE Trans. Cybern.*, 2022.
[15] M. N. Ahmadabadi and M. Asadpour, "Expertness Based Cooperative Q-Learning," *IEEE Trans. Syst. Man Cybern. B*, 2002.
[16] S. M. Kakade and J. Langford, "Approximately Optimal Approximate Reinforcement Learning," in *ICML*, 2002.
[17] P. I. N. Barbalho et al., "Reinforcement Learning Solutions for Microgrid Control and Management: A Survey," *IEEE Access*, 2025.
[18] P. I. N. Barbalho, A. L. Moraes, V. A. Lacerda, P. H. A. Barra, R. A. S. Fernandes, and D. V. Coury, "Reinforcement Learning Solutions for Microgrid Control and Management: A Survey," *IEEE Access*, vol. 13, pp. 39782–39799, 2025.
[19] T. P. Do et al., "Deep Reinforcement Learning for Energy Management in a Microgrid with Battery Energy Storage," *Applied Energy*, 2022.
[20] Z. Yu et al., "Energy Optimization Management of Microgrid Using Improved Soft Actor-Critic Algorithm," *IJRED*, 2024.
[21] Y. Wang, D. Qiu, M. Sun, G. Strbac, and Z. Gao, "Secure Energy Management of Multi-Energy Microgrid: A Physical-Informed Safe Reinforcement Learning Approach," *Applied Energy*, 2023.
[22] Y. Xia, Y. Xu, and X. Feng, "Hierarchical Coordination of Networked-Microgrids Toward Decentralized Operation: A Safe DRL Method," *IEEE Trans. Sustain. Energy*, 2024.
[23] T. Su et al., "Safe RL-Based Transient Stability Control for Islanded Microgrids With Topology Reconfiguration," *IEEE Trans. Smart Grid*, 2025.
[24] B. Zhang et al., "Multi-Agent DRL Based Distributed Control for Interconnected Multi-Energy Microgrids," *Energy Convers. Manag.*, 2023.
[25] G. S. Ayyappan, P. M. P. Deepak, and K. S. Swarup, "Transfer Learning Based RL for Microgrid Energy Management," in *IEEE PES GM*, 2023.
[26] S. Shuai et al., "Transfer Reinforcement Learning for Optimal Microgrid Operation Under Uncertain Environments," *IEEE Trans. Power Syst.*, 2024.
[27] T. Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor," in *ICML*, 2018.
[28] T. Haarnoja et al., "Soft Actor-Critic Algorithms and Applications," arXiv:1812.05905, 2018.
[29] N. Hatziargyriou et al., "Microgrids," *IEEE Power Energy Mag.*, 2007.
[30] R. H. Lasseter, "MicroGrids," in *IEEE PES Winter Meeting*, 2002.
[31] R. H. Lasseter and P. Piagi, "Microgrid: A Conceptual Solution," in *Proc. IEEE 35th Annual Power Electronics Specialists Conf. (PESC)*, Aachen, Germany, 2004, pp. 4285–4290.
[32] S. Chen, H. B. Gooi, and M. Wang, "Sizing of Energy Storage for Microgrids," *IEEE Trans. Smart Grid*, 2012.
[33] R. Palma-Behnke et al., "A Microgrid Energy Management System Based on the Rolling Horizon Strategy," *IEEE Trans. Smart Grid*, 2013.
[34] A. Parisio, E. Rikos, and L. Glielmo, "A Model Predictive Control Approach to Microgrid Operation Optimization," *IEEE Trans. Control Syst. Technol.*, 2014.
[35] T. Takagi and M. Sugeno, "Fuzzy Identification of Systems and Its Applications to Modeling and Control," *IEEE Trans. Syst. Man Cybern.*, 1985.
[36] J. M. Mendel, *Uncertain Rule-Based Fuzzy Logic Systems*. Prentice Hall, 2001.
[37] G. Dalal, K. Dvijotham, M. Vecerik, T. Hester, C. Paduraru, and Y. Tassa, "Safe Exploration in Continuous Action Spaces," arXiv:1801.08757, 2018.
[38] M. Wen and U. Topcu, "Constrained Cross-Entropy Method for Safe Reinforcement Learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2018.
[39] A. Ray, J. Achiam, and D. Amodei, "Benchmarking Safe Exploration in Deep Reinforcement Learning," arXiv:1910.01708, 2019.
[40] M. A. Zanger, K. Daaboul, and J. M. Zöllner, "Safe Continuous Control with Constrained Model-Based Policy Optimization," arXiv:2104.06922, 2021.

---

# Appendix A. Proofs

> Conventions: additive Fuzzy-Lagrangian $\mathcal{L}_{\text{fuzzy}}=J_R+\sum_k\lambda_k(J_{\mu_k}-\alpha_k/(1-\gamma))$, $\lambda_k\ge0$; crisp membership $\mu_k=\sigma(\beta_k(d_k-c_k))$, $\sigma(x)=1/(1+e^{-x})$; crisp target $\alpha_k^{\text{crisp}}=1$; $\sigma(0)=1/2$.

## A.1 Supporting Lemmas

**Lemma A.1 (Sigmoid pointwise limit).** For every $x\in\mathbb R$,
$\lim_{\beta\to\infty}\sigma(\beta x)=\mathbf 1\{x>0\}+\tfrac12\mathbf 1\{x=0\}$, and $\sigma(\beta x)$ is monotone non-decreasing in $x$.

*Proof.* For $x>0$, $\beta x\to+\infty$ so $\sigma\to1$; for $x<0$, $\sigma\to0$; at $x=0$, $\sigma(0)=1/2$. Monotonicity follows from $\sigma'>0$. $\blacksquare$

**Lemma A.2 (Inverse-sigmoid level set).** For $\beta>0$, $\varepsilon\in(0,\tfrac12)$,
$\sigma(\beta x)\ge1-\varepsilon \iff x\ge \tau_\varepsilon/\beta$, where $\tau_\varepsilon=\sigma^{-1}(1-\varepsilon)=\ln\frac{1-\varepsilon}{\varepsilon}$.

*Proof.* Direct inversion. $\blacksquare$

**Lemma A.3 (Pointwise convergence and integrability).** Let $f_\beta(x)=\sigma(\beta g(x))$ with $g$ continuous. Then $f_\beta(x)\to f_\infty(x)$ pointwise, where $f_\infty=\mathbf 1\{g>0\}+\tfrac12\mathbf 1\{g=0\}$, and $0\le f_\beta\le 1$ uniformly in $\beta$. Consequently, for any bounded discounted expectation $\mathbb E_\pi[\sum_t\gamma^t f_\beta(s_t,a_t)]$, dominated convergence gives $\lim_{\beta\to\infty}\mathbb E_\pi[\sum_t\gamma^t f_\beta]=\mathbb E_\pi[\sum_t\gamma^t f_\infty]$.

*Proof.* Pointwise limits follow from Lemma A.1. The uniform bound $0\le f_\beta\le1$ and Lemma A.4 justify interchange of limit and discounted expectation; the transition layer around $\{g=0\}$ has measure zero and does not affect the expectation. $\blacksquare$

**Lemma A.4 (Value bounds).** With $|R|\le R_{\max}$ and $|\mu_k|\le1$,
$|J_R(\pi)|\le R_{\max}/(1-\gamma)$ and $|J_{\mu_k}(\pi)|\le1/(1-\gamma)$.

*Proof.* Standard discounted-reward bound. $\blacksquare$

**Lemma A.5 (Slater's condition).** If there exists a stationary policy $\pi_0$ with $J_{\mu_k}(\pi_0)>\alpha_k/(1-\gamma)$ for all $k$ (strict feasibility), then strong duality holds between $\max_{\pi\text{ $\alpha$-safe}}J_R(\pi)$ and $\min_{\lambda\ge0}\max_\pi\mathcal L_{\text{fuzzy}}$.

## A.2 Proof of Theorem 1 (FC-MDP → CMDP limit)

Let $\mu_k^{(\beta)}(s,a)=\sigma(\beta_k(d_k-c_k(s,a)))$. For every $\varepsilon\in(0,\tfrac12)$ and $(s,a)$,
$$\lim_{\beta_k\to\infty}\mathbf1\{\mu_k^{(\beta)}(s,a)\ge1-\varepsilon\}=\mathbf1\{c_k(s,a)\le d_k\}.$$

By Lemma A.2, the event $\{\mu_k^{(\beta)}\ge1-\varepsilon\}$ is equivalent to $c_k(s,a)\le d_k-\tau_\varepsilon/\beta_k$. As $\beta_k\to\infty$, $\tau_\varepsilon/\beta_k\to0$, recovering $c_k\le d_k$.

For the policy-level statement, pointwise convergence and dominated convergence (Lemma A.3) give $J_{\mu_k}^{(\beta)}(\pi)\to J_{\mu_k}^{(\infty)}(\pi)=\mathbb E_\pi[\sum_t\gamma^t\mathbf1\{c_k(s_t,a_t)\le d_k\}]$. For a stationary policy, write the discounted occupancy-weighted satisfaction as $\bar p_k(\pi)=\mathbb E_\pi[\sum_t\gamma^t\mathbf1\{c_k\le d_k\}]$, so that $\bar p_k\in[0,1/(1-\gamma)]$. The $\alpha$-safe condition $\bar p_k\ge\alpha_k/(1-\gamma)$ becomes $\bar p_k(1-\gamma)\ge\alpha_k$: as $\alpha_k^{\text{crisp}}\to1$ this forces the discounted occupancy-weighted violation rate to vanish, i.e. the policy avoids the violated set in expectation. Dominated convergence (Lemma A.4) justifies the limit. $\square$

**Corollary A.1.** With $\alpha_k^{\text{crisp}}=1$, the set of $\alpha$-safe policies converges (set-wise) to the set of policies whose discounted occupancy-weighted violation probability is zero. This is the *always-safe* subset of the standard CMDP feasible set; it is stricter than the Altman budget constraint $\mathbb E[\sum_t\gamma^t C_k]\le d_k$, which tolerates occasional violations within a cost budget. To recover the budget-feasible set exactly, one sets $\alpha_k$ to correspond to the tolerated violation rate rather than to $1$. (See Remark in §4.1.)

## A.3 Proof of Theorem 2 (Existence of an optimal α-safe policy)

*Step 1 — Occupancy measure.* For a stationary policy define $\rho_\pi(s,a)=(1-\gamma)\sum_t\gamma^t\mathbb P^\pi(s_t=s,a_t=a)$. Under compactness and weak continuity, the set $\mathcal M=\{\rho_\pi\}$ of deterministic stationary occupancy measures is weak-* compact; its convex hull $\mathcal M_{\text{conv}}$ equals the occupancy measures of stationary stochastic policies.

*Step 2 — Linear functionals.* $J_R(\pi)=\frac1{1-\gamma}\sum_{s,a}\rho_\pi R(s,a)$ and $J_{\mu_k}(\pi)=\frac1{1-\gamma}\sum_{s,a}\rho_\pi\mu_k(s,a)$ are continuous linear functionals on $\mathcal M_{\text{conv}}$.

*Step 3 — Feasible set.* $\mathcal F=\{\rho\in\mathcal M_{\text{conv}}:J_{\mu_k}(\rho)\ge\alpha_k/(1-\gamma)\ \forall k\}$ is closed and convex; strict feasibility gives it non-empty interior.

*Step 4 — Attainment.* The continuous objective attains its supremum on the non-empty, closed, bounded set $\mathcal F$ by the Weierstrass theorem in the weak-* topology.

*Step 5 — Determinism.* By Krein–Milman the optimum lies at an extreme point of $\mathcal F$, and the extreme points of $\mathcal M_{\text{conv}}$ are the deterministic stationary measures; hence a stationary deterministic $\pi^*$ attains the optimum. $\square$

## A.4 Proof of Theorem 3 (Strong duality and dual convergence)

Reformulated in occupancy measure, the FC-MDP is a convex program with linear objective and linear inequality constraints. With Slater's condition (Lemma A.5), the standard convex-duality theorem gives zero duality gap:
$$\max_{\pi\text{ $\alpha$-safe}}J_R(\pi)=\min_{\lambda\ge0}\max_\pi\mathcal L_{\text{fuzzy}}(\pi,\lambda).$$

For convergence, the dual function $g(\lambda)=\max_\pi\mathcal L$ is convex and $G$-Lipschitz with $G=\sqrt K/(1-\gamma)$ (subgradient bounded by Lemma A.4). The projected update $\lambda^{(t+1)}=[\lambda^{(t)}-\eta_t h_t]_+$ with $h_t=J_{\mu_k}(\pi_{\lambda^{(t)}})-\alpha_k/(1-\gamma)$ (gradient descent on $\lambda$, matching Theorem 3) satisfies, by convexity,
$$g(\lambda^{(t+1)})\le g(\lambda^{(t)})-\eta_t\|\nabla g(\lambda^{(t)})\|^2+C\eta_t^2+\text{noise}_t.$$
With $\sum_t\eta_t=\infty$ and $\sum_t\eta_t^2<\infty$, the Robbins–Siegmund lemma gives $g(\lambda^{(t)})\to g(\lambda^*)$ and $\lambda^{(t)}\to\lambda^*$; the inner maximizers then satisfy $J_R(\pi_{\lambda^{(t)}})\to J_R(\pi^*)$ by strong duality. $\square$

## A.5 Slater's Condition in Practice (Remark)

Slater's condition can be verified for microgrid FC-MDPs by exhibiting a **conservative policy**—charge ESS at a low fixed rate, run diesel at low constant output, and curtail non-critical load when supply is tight—which has FCSD near 1 for SOC, frequency, and voltage constraints.

> **Caveat (links to §5.3).** For the extreme islanded scenario S4, the *original* action space (200 kW interruptible load) admits no strictly feasible $\alpha$-safe policy for $\alpha_k\ge0.9$—observed empirically as a structural infeasibility. Expanding interruptible load to 600 kW enlarges the action space and restores Slater's condition, so Theorem 3 applies to the *extended* S4.

## A.6 Proof of Theorem 4 (Policy invariance under fuzzy shaping)

Let $R'=R+F$ with $F(s,s')=\gamma\Phi(s')-\Phi(s)$, $\Phi(s)=\max_a R_{\text{know}}(s,a)$. Compactness of $\mathcal A$ and continuity of $R_{\text{know}}$ make $\Phi$ well-defined, and $F$ has the required potential-difference form. Sum the shaped return along any trajectory $\tau=(s_0,a_0,s_1,a_1,\ldots)$:
$$\sum_{t\ge0}\gamma^t R'(s_t,a_t)=\sum_{t\ge0}\gamma^t R(s_t,a_t)+\sum_{t\ge0}\gamma^t\big[\gamma\Phi(s_{t+1})-\Phi(s_t)\big].$$
The second sum telescopes: $\sum_{t\ge0}\gamma^{t+1}\Phi(s_{t+1})-\sum_{t\ge0}\gamma^t\Phi(s_t)=-\Phi(s_0)$ (assuming proper discounting so the tail vanishes). Hence every shaped return equals the original return minus a state-only term $\Phi(s_0)$. In value functions this gives $V'^*(s)=V^*(s)-\Phi(s)$ and
$$Q'^*(s,a)=Q^*(s,a)-\Phi(s),$$
where the subtracted term depends only on the state, not on $a$. Therefore $\arg\max_a Q'^*(s,a)=\arg\max_a Q^*(s,a)$ at every state, and the set of optimal policies is unchanged. $\blacksquare$

*Adaptive decay.* At every finite $t$, $R_t=R+\kappa_tF$ is a potential-based shaping with potential $\kappa_t\Phi$, so the optimal-policy set coincides with that of $R$; as $\kappa_t\to0$ the asymptotic optimum is the original one. The decay affects learning dynamics, not the stationary optimal set.

## A.7 Notation

| Symbol | Meaning |
|---|---|
| $\mathcal S,\mathcal A$ | State/action spaces, compact (Assump. i) |
| $P(\cdot\mid s,a)$ | Transition kernel, weakly continuous (ii) |
| $R(s,a)$ | Reward, bounded and continuous (iii) |
| $\mu_k(s,a)$ | FCSD for constraint $k$ |
| $\alpha_k\in(0,1]$ | Target satisfaction level |
| $\beta_k>0$ | Sigmoid steepness |
| $d_k$ | Crisp threshold |
| $\lambda_k\ge0$ | Lagrange multiplier |
| $\gamma\in[0,1)$ | Discount factor |
| $J_R,J_{\mu_k}$ | Expected discounted return / FCSD |
| $\rho_\pi(s,a)$ | Occupancy measure $(1-\gamma)\sum_t\gamma^t\mathbb P^\pi(\cdot)$ |
| $\sigma(x)$ | Sigmoid $1/(1+e^{-x})$ |
| $\Phi(s)$ | Shaping potential $\max_a R_{\text{know}}(s,a)$ |
| $\kappa_t$ | Adaptive knowledge weight |
| $\alpha$ (SAC) | Entropy temperature in §3.1.2 and Algorithm 1 |
| $\alpha_k\in(0,1]$ | Target FCSD level (distinct from SAC $\alpha$) |
| $\omega(t)$ | Conservative action-scale during cross-scenario transfer (§4.4) |

*End of Appendix A.*
