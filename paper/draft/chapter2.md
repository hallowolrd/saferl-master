# Chapter 2 — Related Work

> **Status**: Draft v1.0
> **Word count**: ~1,550 words
> **Target section**: Section 2 of the main paper

---

## 2.1 Safe Reinforcement Learning

Safe reinforcement learning is concerned with learning policies that satisfy safety constraints during training and deployment [1]. We categorize existing approaches into three main streams: constrained MDP methods, control-theoretic methods, and model-based safe methods.

### 2.1.1 Constrained MDP and Lagrangian Methods

The Constrained Markov Decision Process (CMDP) [2] is the standard framework for Safe RL, where safety is formulated as constraint cost thresholds that the expected cumulative cost must not exceed. **Lagrangian relaxation** converts the constrained problem into an unconstrained saddle-point problem by introducing Lagrange multipliers, and has been widely applied with policy gradient algorithms. TRPO-Lagrangian and PPO-Lagrangian [3] combine trust-region policy optimization and proximal policy optimization with Lagrangian constraint handling, serving as strong baselines. **Constrained Policy Optimization (CPO)** [4] proposes a trust-region method that enforces constraint satisfaction at each policy update step, achieving near-constraint-satisfaction guarantees. However, CPO requires computationally expensive conjugate gradient calculations.

Despite their popularity, CMDP-based methods have important limitations. First, they assume constraints are crisp—every violation is equally unacceptable—which does not capture the graded nature of safety in many real-world systems [5]. Second, Lagrangian methods can exhibit oscillation between reward optimization and constraint satisfaction, as the dual variables adjust slowly [6]. Third, constraint violations during exploration are common in practice, since the constraint is enforced only in expectation over the trajectory distribution, not per-step. Our work addresses the first limitation by replacing crisp constraints with fuzzy satisfaction degrees, while inheriting the strengths of the Lagrangian approach.

### 2.1.2 Control-Theoretic Safe RL

Control-theoretic methods provide hard safety guarantees by embedding controllers with provable safety properties into the RL loop. **Control Barrier Functions (CBFs)** [7] define a forward-invariant safe set and project the RL policy onto the safe action space, ensuring safety with high probability. Safety Layer [8] and Shielding [9] follow a similar philosophy: a backup controller or action corrector intervenes when the RL agent's action would violate constraints. While these methods provide strong per-step safety guarantees, they typically require an accurate system dynamics model to compute the safe set—an assumption that is often violated in microgrid systems with uncertain renewable generation and nonlinear storage dynamics. Recent work on learning CBFs from data [10] relaxes this requirement but introduces sample efficiency challenges.

### 2.1.3 Model-Based Safe RL

Model-based safe RL approaches learn a dynamics model and use it for constrained planning or safe policy optimization. **MBPO-Safe** [11] combines model-based policy optimization with constraint satisfaction, improving sample efficiency over model-free methods. **Constrained Cross-Entropy Method (C-CEM)** [12] performs constrained trajectory optimization using learned models. However, model-based methods suffer from model error accumulation, and constraint satisfaction guarantees degrade as the model drifts from reality—particularly problematic in long-horizon microgrid dispatch tasks.

**Research Gap 1.** Existing Safe RL methods uniformly assume crisp, hard constraints. None of the above frameworks systematically addresses the inherent fuzziness of engineering constraints, leading to over-conservative policies, boundary oscillation, and sparse constraint gradients. Our FC-MDP framework fills this gap by generalizing CMDP to continuous satisfaction degrees.

---

## 2.2 Fuzzy Logic in Reinforcement Learning

Fuzzy logic and reinforcement learning have been combined in several ways, broadly falling into three categories.

### 2.2.1 Fuzzy Reward Shaping

Fuzzy reward shaping uses fuzzy inference to compute additional reward signals based on expert rules, guiding the agent toward desirable behaviors. Fuzzy SAC [13] and Fuzzy DDPG [14] incorporate fuzzy reward terms to accelerate learning in continuous control tasks. While these methods improve sample efficiency, the fuzzy component serves only as an auxiliary reward and does not address constraint satisfaction. The relationship to the optimal policy of the original MDP is often not theoretically established. Our lower-layer fuzzy knowledge shaping shares the spirit of accelerating learning, but we formally establish policy invariance via the potential-based shaping theorem [15].

### 2.2.2 Fuzzy Actor-Critic Architectures

Fuzzy actor-critic methods replace the neural network policy or value function with fuzzy inference systems (FIS). **Fuzzy Q-learning** [16] and **Deep Fuzzy RL** [17] use TSK fuzzy systems as function approximators, leveraging the interpretability of fuzzy rules. These methods typically have fewer parameters but struggle with high-dimensional state spaces compared to deep neural networks. Our approach uses fuzzy systems as *components within* a deep RL architecture rather than as full replacements, combining the strengths of both paradigms.

### 2.2.3 Fuzzy Parameter Adaptation

A third line of work uses fuzzy logic to adaptively tune RL hyperparameters—such as learning rate, exploration rate, or temperature parameter—based on training performance [18]. While useful for training stability, these methods do not address safety constraint satisfaction.

**Research Gap 2.** Fuzzy logic in RL has been applied mainly to reward shaping, function approximation, and hyperparameter tuning. Few works use fuzzy systems for constraint handling, and none develops a comprehensive fuzzy constraint framework that integrates with CMDP theory at the foundational level. Our upper-layer fuzzy constraint protection is, to our knowledge, the first to embed TSK fuzzy systems directly into the Lagrangian constraint mechanism of Safe RL with theoretical convergence guarantees.

---

## 2.3 Deep Reinforcement Learning for Microgrid Optimal Dispatch

DRL has gained significant attention for microgrid energy management due to its ability to handle uncertainty and nonlinearity without explicit system models.

### 2.3.1 DRL Algorithms for Microgrid Dispatch

**Value-based methods** such as DQN [19] and its variants have been applied to discrete-action microgrid dispatch problems. **Policy gradient methods**, particularly **DDPG** [20], **PPO** [21], and **SAC** [22], are preferred for continuous control tasks like ESS charge/discharge scheduling. SAC, with its maximum entropy framework and off-policy learning, has shown strong performance in microgrid dispatch [23] due to its sample efficiency and robust exploration. However, standard DRL methods provide no safety guarantees.

### 2.3.2 Constraint Handling in DRL-based Dispatch

Existing DRL approaches for microgrid dispatch handle constraints primarily through **action clipping** (projecting actions onto the feasible set) or **penalty terms** in the reward function. Action clipping can distort the policy gradient and lead to suboptimal policies near boundaries [24]. Penalty methods require manual tuning of penalty coefficients and do not formally guarantee constraint satisfaction. A few works apply Safe RL methods like CPO [25] or Lagrangian-based approaches [26] to microgrid dispatch, but they inherit the crisp constraint limitations discussed in Section 2.1.

### 2.3.3 Transfer Learning for Microgrid Dispatch

Transfer learning has emerged as a way to reduce training time and improve performance across different microgrid scenarios. Parameter transfer [27] directly copies network weights from a source task to a target task. Feature-based transfer [28] learns shared representations across tasks. However, existing transfer methods for microgrid dispatch focus on reward optimization and do not consider safety during transfer—transferred policies may violate constraints in the target environment, particularly when the scenarios differ significantly [29].

**Research Gap 3.** The application of Safe RL to microgrid dispatch is still in its early stages. Existing work uses standard crisp constraint formulations that do not account for the graded nature of engineering constraints. The sample efficiency of Safe RL in this domain is limited by the lack of expert knowledge integration. And cross-scenario transfer with safety guarantees remains largely unexplored. Our HFG-SRL framework addresses all three gaps simultaneously.

---

## 2.4 Summary of Research Gaps

Table 1 summarizes the three core research gaps identified above and how our work addresses each.

| Gap | Domain | Specific Problem | Our Solution |
|-----|--------|------------------|-------------|
| **Gap 1** | Safe RL theory | Crisp constraint assumption → over-conservatism, boundary oscillation, sparse gradients | FC-MDP framework with Fuzzy-Lagrangian solver |
| **Gap 2** | Fuzzy + RL integration | Fuzzy systems used only for shaping/architecture, not for constraint satisfaction | Two-layer fuzzy mechanism: constraint protection + knowledge shaping |
| **Gap 3** | Microgrid dispatch | No systematic Safe RL method, low sample efficiency, unsafe transfer | HFG-SAC + constraint-aware progressive transfer |

These gaps motivate the development of our hierarchical fuzzy-guided safe reinforcement learning framework, which we present in detail in subsequent chapters.

---

## References (placeholder)

[1] – Safe RL survey (García & Fernández, 2015)
[2] – Altman, Constrained Markov Decision Processes, 1999
[3] – PPO-Lagrangian baseline
[4] – CPO (Achiam et al., 2017)
[5] – Constrained RL optimization paper
[6] – Lagrangian oscillation analysis
[7] – CBF survey / Ames et al.
[8] – Safety Layer (Dalal et al., 2018)
[9] – Shielding (Alshiekh et al., 2018)
[10] – Learned CBFs / neural CBF
[11] – MBPO-Safe or model-based safe RL
[12] – Constrained CEM
[13] – Fuzzy SAC paper
[14] – Fuzzy DDPG paper
[15] – Ng et al., Policy invariance under reward transformations, 1999
[16] – Fuzzy Q-learning (Glorennec)
[17] – Deep Fuzzy RL paper
[18] – Fuzzy adaptive RL hyperparameters
[19] – DQN for microgrid
[20] – DDPG for microgrid dispatch
[21] – PPO for microgrid energy management
[22] – SAC for microgrid
[23] – SAC microgrid dispatch paper
[24] – Action clipping distortion
[25] – CPO for microgrid
[26] – Lagrangian microgrid dispatch
[27] – Parameter transfer for microgrid
[28] – Feature transfer RL
[29] – Transfer learning survey for microgrid

---

*End of Chapter 2.*

---

**Word count**: ~1,540 words
