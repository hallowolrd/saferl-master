# Chapter 4 — Hierarchical Fuzzy-Guided Safe Reinforcement Learning Framework

> **Status**: Draft v1.0
> **Word count**: ~4,500 words
> **Target section**: Section 4 of the main paper

---

## 4.1 Fuzzy Constrained MDP (FC-MDP)

### 4.1.1 Definition

Building on the fuzzy constraint formulation of Section 3.3, we now introduce the **Fuzzy Constrained Markov Decision Process (FC-MDP)**, which generalizes the standard CMDP by replacing crisp constraint thresholds with continuous fuzzy constraint satisfaction degrees.

**Definition 4.1 (Fuzzy Constrained MDP).** A Fuzzy Constrained MDP is a tuple

$$
\langle \mathcal{S}, \mathcal{A}, P, R, \tilde{C}, \gamma, \tilde{\alpha} \rangle,
$$

where:
- $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$ is a standard MDP;
- $\tilde{C} = \{\tilde{C}_1, \dots, \tilde{C}_K\}$ is a set of $K$ fuzzy constraints, each characterized by a membership function $\mu_k(s, a): \mathcal{S} \times \mathcal{A} \to [0, 1]$ representing the satisfaction degree of constraint $k$ at state-action pair $(s, a)$;
- $\tilde{\alpha} = [\alpha_1, \dots, \alpha_K]^T \in (0, 1]^K$ is a vector of target satisfaction levels, one per constraint.

The goal in an FC-MDP is to find a policy $\pi$ that maximizes the expected return while maintaining the average fuzzy constraint satisfaction degree above the target level $\alpha_k$ for each constraint:

$$
\max_{\pi} J_R(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right],
$$

$$
\text{s.t.} \quad J_{\mu_k}(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t \mu_k(s_t, a_t) \right] \geq \alpha_k \cdot \frac{1}{1 - \gamma}, \quad \forall k = 1, \dots, K.
$$

The right-hand side $\alpha_k / (1 - \gamma)$ is the expected cumulative satisfaction degree for a policy that achieves exactly $\alpha_k$ at every step (normalized by the discount factor sum). We denote the constraint value function for the satisfaction degree as:

$$
V_{\mu_k}^{\pi}(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t \mu_k(s_t, a_t) \mid s_0 = s \right].
$$

**Definition 4.2 (α-safe Policy).** A policy $\pi$ is **$\alpha$-safe** with respect to target satisfaction vector $\alpha$ if it satisfies all $K$ fuzzy constraints in expectation, i.e., $J_{\mu_k}(\pi) \geq \alpha_k / (1 - \gamma)$ for all $k = 1, \dots, K$.

### 4.1.2 Relationship to Standard CMDP

FC-MDP is a strict generalization of the standard CMDP. The following theorem formalizes this relationship.

> **Notation convention.** To unify the limiting statement with the crisp CMDP literature, we introduce an auxiliary FCSD threshold $\alpha_k^{\text{crisp}} \in (0, 1]$ and state the correspondence with crisp constraints through the indicator function at level $\alpha_k^{\text{crisp}}$, not at the membership function's midpoint. Concretely, for the crisp CMDP with cost $c_k$ and threshold $d_k$, we set $\mu_k(s,a) = \sigma(\beta_k(d_k - c_k(s,a)))$ with $\alpha_k^{\text{crisp}} = 1$, so that the policy-level constraint $\mathbb{E}[\mu_k] \geq 1/(1-\gamma)$ recovers the crisp constraint in the limit $\beta_k \to \infty$. This convention is used throughout the appendix.

**Theorem 4.1 (FC-MDP → CMDP Limit, $\mu$-Level Form).** Let the membership function be $\mu_k(s,a) = \sigma(\beta_k(d_k - c_k(s,a)))$ with $\beta_k > 0$ and crisp threshold $d_k$. Define the level set
$$
L_k^{(\beta)} \;=\; \bigl\{(s,a)\in\mathcal{S}\times\mathcal{A} : \mu_k(s,a) \geq 1 - \varepsilon\bigr\},\qquad \varepsilon \in (0,\tfrac12).
$$
Then for every $\varepsilon > 0$ and every $(s,a)\in\mathcal{S}\times\mathcal{A}$,
$$
\lim_{\beta_k\to\infty} \mathbb{1}\bigl\{(s,a)\in L_k^{(\beta)}\bigr\} \;=\; \mathbb{1}\{c_k(s,a) \leq d_k\}
\quad \text{(pointwise)}.
$$
Consequently, for any stationary policy $\pi$ and the crisp threshold target $\alpha_k^{\text{crisp}} = 1$,
$$
\lim_{\beta_k\to\infty} \mathbb{1}\bigl\{J_{\mu_k}(\pi) \geq (1-\varepsilon)/(1-\gamma)\bigr\} \;=\; \mathbb{1}\bigl\{J_{C_k}(\pi) \leq d_k\bigr\},
$$
where $J_{C_k}(\pi) = \mathbb{E}_\pi[\sum_t \gamma^t c_k(s_t,a_t)]$.

**Proof sketch.** Sigmoid convergence. For $x 
eq 0$, $\sigma(\beta x) \to \mathbb{1}\{x > 0\}$ as $\beta \to \infty$ (pointwise, monotone). For the boundary level-set, since $\mu_k(s,a) \geq 1-\varepsilon$ iff $\beta_k(d_k - c_k(s,a)) \geq \sigma^{-1}(1-\varepsilon) = \ln(\tfrac{1-\varepsilon}{\varepsilon}) := \tau_\varepsilon$, we obtain $(s,a) \in L_k^{(\beta)}$ iff $c_k(s,a) \leq d_k - \tau_\varepsilon/\beta_k$. Letting $\beta_k \to \infty$ forces $\tau_\varepsilon/\beta_k \to 0$, recovering $c_k(s,a) \leq d_k$. The policy-level statement follows by dominated convergence (since $|J_{\mu_k}| \leq 1/(1-\gamma)$ and $|J_{C_k}| \leq \bar{c}_k/(1-\gamma)$ with $\bar{c}_k = \|c_k\|_\infty$). Full measure-theoretic justification and Dini-type uniform control on compact sets is given in Appendix A. $\blacksquare$

**Corollary 4.1.1 (Asymptotic Recovery of $\alpha$-safety).** With $\alpha_k^{\text{crisp}} = 1$ and the convention above, the set of $\alpha$-safe policies in FC-MDP converges (in the sense of set convergence) to the set of feasible policies in the standard CMDP as $\beta_k \to \infty$ for all $k$.

**Remark 4.1 (Why $\alpha = 1$ for the limit).** The midpoint convention $\mu_k(d_k) = 0.5$ would yield an inconsistent limiting object: $\alpha_k = 0.5$ implies the FCSD constraint $\mathbb{E}[\mu_k] \geq 0.5/(1-\gamma)$, which neither matches a hard-inequality CMDP nor a soft-inequality CMDP. The $\alpha_k^{\text{crisp}} = 1$ convention avoids this artifact and is the natural one for connecting to existing CMDP theory. In our HFG-SAC implementation we use $\alpha_k \in \{0.85, 0.90, 0.95\}$ depending on constraint criticality, which corresponds to operating in the genuinely *fuzzy* regime (away from the crisp limit), where the smooth gradient benefits of FC-MDP apply.

### 4.1.3 Existence of Optimal α-safe Policy

The following theorem establishes that under standard regularity conditions, an optimal $\alpha$-safe policy exists in FC-MDP.

**Theorem 4.2 (Existence of Optimal α-safe Policy).** Suppose that: (i) $\mathcal{S}$ and $\mathcal{A}$ are compact subsets of Euclidean spaces; (ii) the transition kernel $P(\cdot \mid s, a)$ is weakly continuous in $(s, a)$; (iii) $R(s, a)$ and all $\mu_k(s, a)$ are bounded and continuous in $(s, a)$; and (iv) there exists at least one policy $\pi_0$ satisfying all fuzzy constraints strictly (strict feasibility). Then there exists a stationary deterministic policy $\pi^*$ that is optimal among all $\alpha$-safe policies.

**Proof sketch.** The proof follows the standard convex-analytic approach to constrained MDPs (Altman, 1999). The set of occupancy measures induced by stationary policies is compact and convex in the weak* topology. The objective $J_R(\pi)$ is linear (hence continuous) on this set, and the constraint set defined by $J_{\mu_k}(\pi) \geq \alpha_k / (1 - \gamma)$ is closed and convex by continuity of $\mu_k$ and linearity of the expectation. By the Weierstrass extreme value theorem, the maximum is attained. Strict feasibility ensures the constraint set has non-empty interior, which is useful for Lagrangian duality. $\blacksquare$

---

## 4.2 Fuzzy-Lagrangian for FC-MDP

### 4.2.1 Formulation

To solve the FC-MDP, we extend the Lagrangian method from standard CMDP to the fuzzy constraint setting. The key insight is that fuzzy constraints are *upper-bound constraints on satisfaction deficit* rather than lower-bound constraints on cost. Rewriting the FC-MDP constraint:

$$
J_{\mu_k}(\pi) \geq \frac{\alpha_k}{1 - \gamma} \quad \Longleftrightarrow \quad \frac{\alpha_k}{1 - \gamma} - J_{\mu_k}(\pi) \leq 0.
$$

The left-hand side represents the **satisfaction deficit**—how much the policy falls short of the target satisfaction level, normalized in $[0, 1/(1-\gamma)]$.

**Definition 4.3 (Fuzzy-Lagrangian).** The Fuzzy-Lagrangian function for FC-MDP is:

$$
\mathcal{L}_{\text{fuzzy}}(\pi, \lambda) = J_R(\pi) + \sum_{k=1}^{K} \lambda_k \left( J_{\mu_k}(\pi) - \frac{\alpha_k}{1 - \gamma} \right),
$$

where $\lambda_k \geq 0$ are the Lagrange multipliers (dual variables) associated with the $k$-th fuzzy constraint.

Note the sign convention: since the constraint is $J_{\mu_k}(\pi) \geq \alpha_k/(1-\gamma)$, we add $\lambda_k \cdot (\text{constraint value} - \text{threshold})$ with $\lambda_k \geq 0$. This is equivalent to the standard CMDP Lagrangian after a sign flip: replacing cost $C_k$ with satisfaction $\mu_k$ and reversing the inequality direction.

### 4.2.2 Dual Problem and Saddle-Point Optimality

The **dual function** is:

$$
g(\lambda) = \max_{\pi} \mathcal{L}_{\text{fuzzy}}(\pi, \lambda).
$$

The **dual problem** is:

$$
\min_{\lambda \geq 0} g(\lambda).
$$

**Theorem 4.3 (Strong Duality and Convergence).** Under the conditions of Theorem 4.2 and assuming the FC-MDP is strictly feasible, strong duality holds:

$$
\max_{\pi \text{ is } \alpha\text{-safe}} J_R(\pi) = \min_{\lambda \geq 0} \max_{\pi} \mathcal{L}_{\text{fuzzy}}(\pi, \lambda).
$$

Furthermore, the Fuzzy-Lagrangian dual gradient ascent algorithm—where in each iteration we (i) solve the inner maximization for the current $\lambda$, producing a policy $\pi_{\lambda}$, and (ii) update $\lambda$ via projected gradient descent on the dual function:

$$
\lambda_k^{(t+1)} = \left[ \lambda_k^{(t)} - \eta_t \cdot \left( J_{\mu_k}(\pi_{\lambda^{(t)}}) - \frac{\alpha_k}{1 - \gamma} \right) \right]_{+}
$$

with step size $\eta_t$ satisfying $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$—converges to a saddle point $(\pi^*, \lambda^*)$ of $\mathcal{L}_{\text{fuzzy}}$.

**Proof sketch.** Strong duality follows from Slater's condition for convex programs: the feasible set of occupancy measures is convex, the objective is linear, and the constraint functions are linear (in the occupancy measure). Strict feasibility (Slater's condition) guarantees zero duality gap. Convergence of the projected subgradient method for the dual follows from standard results—since $g(\lambda)$ is convex and Lipschitz continuous with subgradient $\partial g(\lambda) = J_{\mu}(\pi_{\lambda}) - \alpha/(1-\gamma)$. The Robbins–Siegmund theorem gives convergence of the dual sequence to the optimal dual value. $\blacksquare$

In practice, we use a stochastic variant of this algorithm integrated with the SAC actor-critic architecture, as described in Section 4.3.

---

## 4.3 Hierarchical Fuzzy-Guided SAC (HFG-SAC)

We now present the **Hierarchical Fuzzy-Guided Soft Actor-Critic (HFG-SAC)** algorithm, which instantiates the FC-MDP and Fuzzy-Lagrangian framework with two complementary fuzzy layers built on top of a base SAC algorithm. The overall architecture is illustrated in Figure 1 (not yet rendered).

### 4.3.1 Upper Layer: Fuzzy Constraint Protection

The upper layer implements the Fuzzy-Lagrangian constraint mechanism via a TSK fuzzy system that computes the joint constraint satisfaction degree and provides gradient signals to the actor network.

**Multi-constraint FCSD computation.** Given the current state $s_t$ and action $a_t$, we first compute the individual satisfaction degrees $\mu_k(s_t, a_t)$ for each of the $K$ constraints using the sigmoidal membership functions defined in Section 3.3.2. We then aggregate them using the weighted product t-norm:

$$
\tilde{\mu}(s_t, a_t) = \prod_{k=1}^{K} \mu_k(s_t, a_t)^{w_k},
$$

where $w_k \geq 0$ are per-constraint importance weights.

**TSK fuzzy system implementation.** The individual membership functions and aggregation are implemented as a zero-order TSK fuzzy system with $M = 2^K$ rules (in the full form) or a parameterized product form (in the compact form we use). The compact form is differentiable end-to-end and has only $2K$ parameters per constraint (steepness $\beta_k$ and reference $x_k^{\text{ref}}$), plus $K$ importance weights $w_k$.

**Fuzzy-Lagrangian constraint loss.** The constraint loss added to the actor objective is:

$$
\mathcal{L}_{\text{constraint}}(\phi) = -\mathbb{E}_{s \sim \mathcal{D}} \left[ \sum_{k=1}^{K} \lambda_k \cdot \mu_k(s, a_{\phi}(s)) \right],
$$

where $\phi$ are the actor network parameters, $a_{\phi}(s)$ is the action sampled from the policy, and $\lambda_k$ are the Lagrange multipliers updated via dual gradient ascent:

$$
\lambda_k \leftarrow \left[ \lambda_k + \eta_{\lambda} \cdot \left( \alpha_k - \bar{\mu}_k \right) \right]_{+},
$$

where $\bar{\mu}_k$ is the average satisfaction degree of constraint $k$ over a recent batch, and $\eta_{\lambda}$ is the dual learning rate. When satisfaction is below target ($\bar{\mu}_k < \alpha_k$), the multiplier increases, putting more weight on constraint satisfaction in subsequent updates.

The total actor loss combines the standard SAC policy loss with the constraint loss:

$$
\mathcal{L}_{\text{actor}}(\phi) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \alpha \log \pi_{\phi}(a \mid s) - Q_{\theta}(s, a) \right] + \lambda^T \cdot \mathbb{E}_{s \sim \mathcal{D}} \left[ \mu(s, a_{\phi}(s)) \right]^{-}.
$$

Here the minus sign on the satisfaction term encodes the fact that maximizing satisfaction corresponds to minimizing the Lagrangian's penalty term.

### 4.3.2 Lower Layer: Fuzzy Knowledge Reward Shaping

The lower layer embeds expert operational knowledge into the learning process via potential-based fuzzy reward shaping. This accelerates learning while preserving the optimal policy.

**Expert rule encoding.** Domain experts provide operational rules in natural IF-THEN linguistic form. For example:
> *"If SOC is low and PV generation is high, then the ESS should charge."*

Each rule is converted into a TSK fuzzy rule with:
- **Antecedent**: fuzzy sets on relevant state variables (SOC level, PV generation, etc.);
- **Consequent**: a constant reward bonus (zero-order TSK) indicating the desirability of the corresponding state-action configuration.

Formally, expert rule $r$ has firing strength $w^r(s, a) = \prod_j \mu_{F_j^r}(s_j)$, and the fuzzy knowledge reward is:

$$
R_{\text{know}}(s, a) = \frac{\sum_r w^r(s, a) \cdot b^r}{\sum_r w^r(s, a) + \epsilon},
$$

where $b^r$ is the rule's consequent parameter (reward bonus), and $\epsilon$ is a small constant for numerical stability.

**Potential-based shaping and policy invariance.** To guarantee that the optimal policy of the shaped MDP equals the optimal policy of the original MDP (Ng et al., 1999), we cast the fuzzy knowledge reward as a potential-based shaping function:

$$
F(s, s') = \gamma \Phi(s') - \Phi(s),
$$

where the potential function $\Phi(s)$ is defined as the maximum expected expert reward from state $s$:

$$
\Phi(s) = \max_a R_{\text{know}}(s, a).
$$

**Theorem 4.4 (Policy Invariance under Fuzzy Shaping).** Let $R'(s, a, s') = R(s, a) + F(s, s')$ where $F(s, s') = \gamma \Phi(s') - \Phi(s)$ is the potential-based shaping reward derived from the fuzzy knowledge system. Then any optimal policy for the shaped reward $R'$ is also optimal for the original reward $R$.

**Proof.** This is a direct application of the potential-based shaping theorem (Ng et al., 1999). The fuzzy knowledge reward defines a potential function $\Phi(s)$, and the shaping term $F(s, s')$ satisfies the difference form required for policy invariance. $\blacksquare$

**Adaptive weight decay.** The influence of expert knowledge should gradually decrease as the agent learns, to avoid suboptimal convergence to the expert's (potentially imperfect) policy. We apply an exponential decay to the knowledge reward weight:

$$
\kappa_t = \kappa_0 \cdot \rho^{\lfloor t / T_{\text{decay}} \rfloor},
$$

where $\kappa_0$ is the initial weight, $\rho \in (0, 1)$ is the decay factor, and $T_{\text{decay}}$ is the decay period. The total shaped reward is:

$$
R_{\text{total}}(s, a, s') = R(s, a) + \kappa_t \cdot F(s, s').
$$

As $t \to \infty$, $\kappa_t \to 0$, and the shaped reward converges to the original reward—ensuring the policy converges to the true optimum.

### 4.3.3 Two-Layer Coordinated Optimization

The two fuzzy layers operate at different time scales and serve complementary purposes:

- **Upper layer (constraint protection)**: operates on the same time scale as policy updates; enforces safety by shaping the policy gradient through the Fuzzy-Lagrangian loss. Its multipliers $\lambda$ adapt slowly to maintain target satisfaction.
- **Lower layer (knowledge shaping)**: operates via reward signal; accelerates learning by providing dense, expert-informed reward guidance that decays over time.

The overall optimization objective is:

$$
\max_{\phi} J_{\text{total}}(\phi) = J_R(\phi) + \kappa_t \cdot J_F(\phi) + \sum_k \lambda_k \cdot \left( J_{\mu_k}(\phi) - \frac{\alpha_k}{1 - \gamma} \right),
$$

where $J_R(\phi)$ is the original return, $J_F(\phi)$ is the potential-based shaping return, and the last term is the Fuzzy-Lagrangian constraint penalty.

**Actor update.** The actor network parameters $\phi$ are updated by gradient ascent on the combined objective:

$$
\nabla_{\phi} J_{\text{total}} = \nabla_{\phi} J_R + \kappa_t \cdot \nabla_{\phi} J_F + \sum_k \lambda_k \cdot \nabla_{\phi} J_{\mu_k}.
$$

**Critic update.** Two Q-networks are trained to approximate the total shaped value (reward + shaping), with the constraint loss applied only to the actor:

$$
\mathcal{L}_{\text{critic}}(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( Q_{\theta}(s, a) - y \right)^2 \right],
$$

where the target $y = r + \kappa_t F(s, s') + \gamma (1 - d) \left( \min_{i=1,2} Q_{\theta_i'}(s', a') - \alpha \log \pi_{\phi}(a' \mid s') \right)$ uses the minimum of two target Q-networks.

**Dual variable update.** The Lagrange multipliers $\lambda$ are updated at a slower rate using running estimates of constraint satisfaction:

$$
\lambda_k \leftarrow \left[ \lambda_k + \eta_{\lambda} \cdot \left( \alpha_k - \frac{1}{|\mathcal{B}|} \sum_{(s,a) \in \mathcal{B}} \mu_k(s, a) \right) \right]_{+},
$$

where $\mathcal{B}$ is the most recent batch.

The complete HFG-SAC algorithm is summarized in Algorithm 1.

**Algorithm 1: Hierarchical Fuzzy-Guided SAC (HFG-SAC)**

```
Input: Initial actor π_φ, Q-networks Q_θ1, Q_θ2, target Q-networks Q_θ1', Q_θ2'
Input: Fuzzy membership parameters {β_k, x_k^ref, w_k}, expert fuzzy rules, α_target
Input: Initial multipliers λ_0, knowledge weight κ_0, decay rate ρ, learning rates
Initialize replay buffer D = ∅

for episode = 1 to N_episodes do
    Initialize state s_1
    for t = 1 to T do
        Sample action a_t ~ π_φ(·|s_t)
        Execute a_t, observe reward r_t, next state s_{t+1}, done d_t
        Compute fuzzy satisfaction degrees μ_k(s_t, a_t) for all k
        Compute knowledge shaping reward F(s_t, s_{t+1}) = γΦ(s_{t+1}) - Φ(s_t)
        Store transition (s_t, a_t, r_t + κ_t * F, s_{t+1}, d_t, μ(s_t, a_t)) in D
        
        // Update critic networks
        Sample mini-batch B from D
        Compute target y = r + κ_t*F + γ(1-d) * (min_i Q_θi'(s', a') - α log π_φ(a'|s'))
        Update θ_1, θ_2 by minimizing MSE loss
        
        // Update actor (policy + constraint)
        Update φ by gradient ascent: ∇_φ J_total = ∇_φ J_R + κ_t*∇_φ J_F + Σ λ_k * ∇_φ J_μk
        
        // Update dual variables
        Update λ_k ← [λ_k + η_λ * (α_k - mean(μ_k(B)))]_+
        
        // Soft update target networks
        θ_i' ← τθ_i + (1-τ)θ_i'
    end for
    
    // Decay knowledge weight
    κ_{t+1} = κ_t * ρ
end for

Output: Trained policy π_φ
```

---

## 4.4 Constraint-Aware Cross-Scenario Transfer

Real-world microgrids operate under varying conditions: seasonal changes in renewable generation, different load profiles, and switching between grid-connected and islanded modes. Training a policy from scratch for each scenario is sample-inefficient and may require unsafe exploration. We develop a constraint-aware transfer mechanism that safely adapts a policy trained on a source scenario to a target scenario.

### 4.4.1 Scenario Similarity Metric

Before transferring, we assess the similarity between the source scenario $\mathcal{S}_s$ and target scenario $\mathcal{S}_t$ to determine *what can be safely transferred* and *what must be re-learned*.

**Fuzzy rule similarity.** We measure scenario similarity by comparing the fuzzy constraint rules across scenarios. For each constraint $k$, let $R_k^s$ and $R_k^t$ be the fuzzy membership functions in the source and target, respectively. The per-constraint similarity is:

$$
\text{sim}_k = \frac{\int \min(\mu_k^s(x), \mu_k^t(x)) \, dx}{\int \max(\mu_k^s(x), \mu_k^t(x)) \, dx},
$$

which is the Jaccard index of the two fuzzy sets, ranging from 0 (disjoint) to 1 (identical).

**Constraint transferability categorization.** Based on the similarity score, constraints are categorized into three groups:

| Group | Similarity Range | Transfer Strategy |
|-------|-----------------|-------------------|
| **Transferable** | $\text{sim}_k \geq \tau_{\text{high}}$ (e.g., 0.8) | Direct transfer of fuzzy membership parameters |
| **Adaptable** | $\tau_{\text{low}} \leq \text{sim}_k < \tau_{\text{high}}$ | Transfer as initialization, then fine-tune with conservative prior |
| **Relearn** | $\text{sim}_k < \tau_{\text{low}}$ (e.g., 0.3) | Initialize from scratch (or from expert default rules) |

This categorization ensures that similar constraints benefit fully from transfer, while dissimilar constraints do not introduce unsafe prior biases.

### 4.4.2 Progressive Safety Transfer with Fuzzy Prior Initialization

We now describe the progressive transfer algorithm that safely adapts the source policy to the target scenario.

**Weight transfer with partial freezing.** We initialize the target actor and critic by copying all weights from the source policy. To preserve source knowledge while adapting to the target scenario, we then freeze the bottom 50% of the actor trunk (the layers that encode general temporal patterns and component dynamics) and fine-tune only the upper trunk layers plus the policy head. This partial freezing (i) prevents catastrophic forgetting of the source policy's low-level feature extraction and (ii) leaves enough trainable capacity for the target-specific constraint handling and reward optimization.

**Fuzzy prior initialization.** For adaptable constraints (those in the middle similarity group), we initialize the target fuzzy membership parameters by shifting and scaling the source parameters based on the target scenario's physical bounds. Specifically:

$$
\beta_k^t = \beta_k^s \cdot \Delta_k, \quad x_k^{\text{ref}, t} = x_k^{\text{ref}, s} + \delta_k,
$$

where $\Delta_k > 0$ and $\delta_k$ are scenario adjustment factors derived from the target configuration (e.g., the distance between the optimal SOC band and the hard SOC bound). For transferable constraints we copy $\beta_k^s$ and $x_k^{\text{ref}, s}$ directly; for relearn constraints we use expert default values. This initialization avoids bootstrapping the target fuzzy system from an unsafe random state.

**Progressive safety relaxation.** To balance safety and performance during fine-tuning, we start with a conservative action scale $\alpha_{\text{init}} > \alpha_{\text{target}} = 1$ and gradually relax it toward the unconservative level:

$$
\alpha_{\text{cur}}(t) = \alpha_{\text{target}} + (\alpha_{\text{init}} - \alpha_{\text{target}}) \cdot e^{-\nu \cdot t},
$$

where $\nu = -\ln(0.05) / T_{\text{relax}}$ is set so that $\alpha_{\text{cur}} \approx \alpha_{\text{target}} + 0.05 \cdot (\alpha_{\text{init}} - \alpha_{\text{target}})$ at step $T_{\text{relax}}$. The executed action is then scaled by $1/\alpha_{\text{cur}}$: a larger $\alpha_{\text{cur}}$ shrinks exploratory actions and keeps the policy inside a conservative tube around the source policy at the beginning of fine-tuning, while the exponential schedule lets exploration freedom recover smoothly as the target policy adapts.

**Algorithm 2: Constraint-Aware Progressive Transfer**

```
Input: Source policy π_φ_s, source fuzzy parameters {β_k^s, x_k^ref,s}
Input: Target scenario specification, similarity thresholds τ_high, τ_low
Input: Conservative factor α_init, target α_target = 1, relaxation horizon T_relax

Step 1: Scenario similarity assessment
  For each constraint k:
    Compute sim_k via Jaccard index of μ_k^s and μ_k^t
    Categorize as Transferable / Adaptable / Relearn

Step 2: Fuzzy prior initialization
  For Transferable: β_k^t = β_k^s, x_k^ref,t = x_k^ref,s
  For Adaptable:    β_k^t = β_k^s · Δ_k, x_k^ref,t = x_k^ref,s + δ_k
  For Relearn:      β_k^t, x_k^ref,t ← expert defaults

Step 3: Weight transfer and partial freezing
  Copy actor and critic weights from π_φ_s to target networks
  Freeze the bottom 50% of the actor trunk; upper trunk and head remain trainable

Step 4: Progressive fine-tuning with exponential safety relaxation
  Set α_cur = α_init, ν = -ln(0.05) / T_relax
  for fine-tuning step t = 1 to T_ft do
    Scale executed actions by 1 / α_cur
    Run HFG-SAC update on the target scenario
    α_cur ← α_target + (α_init - α_target) · exp(-ν · t)
  end for

Output: Target policy π_φ_t, target fuzzy parameters {β_k^t, x_k^ref,t}
```

---

*End of Chapter 4.*

---

**Word count**: ~4,280 words
