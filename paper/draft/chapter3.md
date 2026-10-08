# Chapter 3 — Preliminaries and Problem Formulation

> **Status**: Draft v1.0
> **Word count**: ~1,500 words
> **Target section**: Section 3 of the main paper

---

## 3.1 Preliminaries

### 3.1.1 Constrained Markov Decision Process

A standard Markov Decision Process (MDP) is defined as a tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $P(s' \mid s, a): \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$ is the transition probability, $R(s, a): \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the reward function, and $\gamma \in [0, 1)$ is the discount factor. The goal is to find a policy $\pi(a \mid s)$ that maximizes the expected cumulative discounted reward:

$$
J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right].
$$

A **Constrained MDP (CMDP)** extends the standard MDP by introducing $K$ constraint cost functions $C_k(s, a): \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ for $k = 1, \dots, K$, each associated with a constraint threshold $d_k \in \mathbb{R}$. A policy $\pi$ is considered *feasible* (safe) if it satisfies:

$$
J_{C_k}(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t C_k(s_t, a_t) \right] \leq d_k, \quad \forall k = 1, \dots, K.
$$

The CMDP objective is to maximize $J(\pi)$ subject to the above constraint. In microgrid dispatch, constraints typically include generator ramp limits, state-of-charge (SOC) bounds, and power balance. The standard approach to solving CMDPs is the **Lagrangian method**, which introduces Lagrange multipliers $\lambda_k \geq 0$ and converts the constrained problem into an unconstrained saddle-point problem:

$$
\min_{\lambda \geq 0} \max_{\pi} \mathcal{L}(\pi, \lambda) = J(\pi) - \sum_{k=1}^{K} \lambda_k \left( J_{C_k}(\pi) - d_k \right).
$$

### 3.1.2 Soft Actor-Critic

Soft Actor-Critic (SAC) is an off-policy maximum entropy reinforcement learning algorithm that optimizes a stochastic policy to maximize both expected reward and policy entropy. The modified objective is:

$$
J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t \left( R(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right) \right],
$$

where $\alpha > 0$ is the temperature parameter controlling the trade-off between reward maximization and entropy maximization, and $\mathcal{H}(\pi(\cdot \mid s)) = \mathbb{E}_{a \sim \pi} [-\log \pi(a \mid s)]$ is the policy entropy. SAC maintains two Q-networks (to mitigate overestimation bias) and a separate policy network, all updated via stochastic gradient descent. The dual value function formulation and automatic entropy tuning make SAC particularly suitable for continuous control tasks like microgrid dispatch.

### 3.1.3 TSK Fuzzy Systems

A Takagi–Sugeno–Kang (TSK) fuzzy system implements a nonlinear mapping $f: \mathbb{R}^n \to \mathbb{R}$ through a set of $M$ fuzzy IF-THEN rules of the form:

$$
R^i: \text{IF } x_1 \text{ is } F_1^i \text{ AND } \dots \text{ AND } x_n \text{ is } F_n^i, \text{ THEN } y^i = p_0^i + p_1^i x_1 + \dots + p_n^i x_n,
$$

where $F_j^i$ is the fuzzy set for the $j$-th input in the $i$-th rule, and $p_j^i$ are the consequent parameters. For a zero-order TSK system, the consequent is a constant $y^i = p_0^i$.

Given an input vector $\mathbf{x} = [x_1, \dots, x_n]^T$, the final output is computed via weighted averaging:

$$
f(\mathbf{x}) = \frac{\sum_{i=1}^{M} w^i(\mathbf{x}) \cdot y^i}{\sum_{i=1}^{M} w^i(\mathbf{x})},
$$

where $w^i(\mathbf{x}) = \prod_{j=1}^{n} \mu_{F_j^i}(x_j)$ is the firing strength of the $i$-th rule, and $\mu_{F_j^i}(x_j)$ denotes the membership function of the $j$-th input's $i$-th fuzzy set. Common membership functions include Gaussian, trapezoidal, and sigmoidal shapes. TSK fuzzy systems are universal approximators and offer a principled way to encode linguistic expert knowledge into differentiable parametric structures—an advantage we exploit in both the constraint protection layer and the knowledge reward shaping layer of our framework.

---

## 3.2 Microgrid Dispatch Problem Formulation

### 3.2.1 System Description

We consider a grid-connected/islanded microgrid system comprising the following components:

- **Photovoltaic (PV) array** with maximum power point tracking (MPPT);
- **Wind turbine (WT)** with variable power output;
- **Energy storage system (ESS)** with charge/discharge capability;
- **Diesel engine (DE) generator** serving as backup power (primarily in islanded mode);
- **Controllable and uncontrollable loads**.

The system operates in two modes: (i) **grid-connected mode**, where the microgrid can exchange power with the main grid at time-varying electricity prices; and (ii) **islanded mode**, where the microgrid operates autonomously and must maintain power balance using local resources only.

### 3.2.2 Optimization Objective

The goal of microgrid optimal dispatch is to minimize the total daily operating cost over a scheduling horizon $T = 24$ hours (with 1-hour time steps):

$$
\min \sum_{t=1}^{T} C_{\text{total},t},
$$

where the total cost at time step $t$ consists of:

- **Grid exchange cost** (grid-connected mode only): $C_{\text{grid},t} = c_{\text{buy},t} \cdot P_{\text{buy},t} - c_{\text{sell},t} \cdot P_{\text{sell},t}$;
- **Diesel generator fuel cost**: $C_{\text{DE},t} = a \cdot P_{\text{DE},t}^2 + b \cdot P_{\text{DE},t} + c$;
- **Operation and maintenance (O\&M) cost**: $C_{\text{OM},t} = \sum_{i \in \{\text{PV, WT, ESS, DE}\}} k_i \cdot |P_{i,t}|$;
- **Load curtailment penalty** (islanded mode): $C_{\text{curt},t} = c_{\text{curt}} \cdot P_{\text{curt},t}$;
- **Renewable curtailment penalty**: $C_{\text{curR},t} = c_{\text{curR}} \cdot P_{\text{curR},t}$.

### 3.2.3 Traditional (Crisp) Constraints

Under the conventional crisp constraint formulation, the dispatch problem is subject to the following hard constraints:

1. **Active power balance**:
   $$
   P_{\text{pv},t} + P_{\text{wt},t} + P_{\text{dis},t} + P_{\text{de},t} + P_{\text{buy},t} = P_{\text{load},t} + P_{\text{ch},t} + P_{\text{sell},t} + P_{\text{curR},t} + P_{\text{curt},t}.
   $$

2. **ESS state of charge (SOC) bounds**:
   $$
   \text{SOC}_{\text{min}} \leq \text{SOC}_t \leq \text{SOC}_{\text{max}}, \quad \forall t.
   $$

3. **ESS charge/discharge power limits**:
   $$
   0 \leq P_{\text{ch},t} \leq P_{\text{ch}}^{\text{max}}, \quad 0 \leq P_{\text{dis},t} \leq P_{\text{dis}}^{\text{max}}.
   $$

4. **Diesel generator output limits and ramp constraints**:
   $$
   P_{\text{de}}^{\text{min}} \leq P_{\text{de},t} \leq P_{\text{de}}^{\text{max}}, \quad |P_{\text{de},t} - P_{\text{de},t-1}| \leq R_{\text{de}}.
   $$

5. **Grid exchange power limit** (grid-connected mode):
   $$
   0 \leq P_{\text{buy},t} \leq P_{\text{buy}}^{\text{max}}, \quad 0 \leq P_{\text{sell},t} \leq P_{\text{sell}}^{\text{max}}.
   $$

6. **Bus voltage deviation limit** (simplified):
   $$
   |V_t - V_{\text{nom}}| \leq \Delta V_{\text{max}}.
   $$

These constraints are enforced as hard 0-1 boundaries: a state-action pair is either *fully feasible* or *fully infeasible*. As we discuss in the next section, this binary characterization does not align with real-world microgrid operation.

---

## 3.3 Fuzzy Constraint Formulation

### 3.3.1 Limitations of Crisp Constraints

The crisp constraint formulation in Section 3.2.3 assumes that each constraint has a well-defined, sharp boundary. In practice, however, microgrid operational constraints exhibit inherent gradation and fuzziness for several reasons:

- **Hierarchical safety levels**. Many constraints have a "recommended range" inside a "hard limit." For example, a battery's SOC may have a preferred band of $[0.3, 0.8]$ for longevity and a hard safety limit of $[0.1, 0.9]$. Operating within the recommended range incurs no penalty, operating between the recommended and hard boundaries is permissible but undesirable, and exceeding the hard limit is prohibited. A crisp constraint model cannot represent this hierarchy without introducing multiple artificial thresholds.

- **Transient violation tolerance**. Under extreme conditions (e.g., a sudden cloud covering a PV array or a load spike), short-duration, small-magnitude constraint violations may be acceptable to avoid more severe consequences (e.g., load shedding). Crisp constraints treat any violation, however brief or small, as equally unacceptable—which can lead to overly conservative policies that sacrifice significant economic performance for marginal safety gains.

- **Priority-ranked constraints**. Not all constraints are equally important. Voltage deviation and frequency stability are more critical than ESS SOC bounds, yet a standard CMDP treats all constraint violations as binary events with fixed threshold values.

From a learning perspective, crisp constraints cause three well-known pathologies:

1. **Over-conservatism**: Policies stay far inside the feasible region to avoid the sharp boundary, leaving the interior of the safe set under-explored.
2. **Boundary oscillation**: Near the constraint boundary, the policy oscillates between safe and unsafe states as the gradient signal switches abruptly.
3. **Sparse constraint gradient**: The constraint cost function is flat (zero) inside the feasible region, providing no gradient signal until the boundary is actually crossed—slowing learning significantly.

These limitations motivate a reformulation of constraints as fuzzy sets with continuous satisfaction degrees.

### 3.3.2 Fuzzy Constraint Satisfaction Degree (FCSD)

We formally define the **Fuzzy Constraint Satisfaction Degree (FCSD)** for a constraint $k$ as a membership function:

$$
\mu_k(s, a): \mathcal{S} \times \mathcal{A} \to [0, 1],
$$

where $\mu_k(s, a) = 1$ indicates full satisfaction of the constraint, $\mu_k(s, a) = 0$ indicates complete violation, and intermediate values represent partial satisfaction.

The FCSD function satisfies three key properties:

- **Monotonicity**: $\mu_k(s, a)$ increases monotonically as the state-action pair moves deeper into the safe region.
- **Continuity**: $\mu_k(s, a)$ is continuous (and typically differentiable) over $\mathcal{S} \times \mathcal{A}$, ensuring smooth gradient signals for learning.
- **Boundary calibration**: The crisp constraint boundary $c_k(s, a) = d_k$ corresponds to a specified satisfaction level (e.g., $\mu_k = 0.5$), establishing a direct correspondence between fuzzy and crisp formulations.

**Membership function selection.** Among the many possible membership function shapes (Gaussian, trapezoidal, triangular, sigmoidal), we adopt a **sigmoidal membership function** for inequality constraints due to its differentiability and natural S-shaped transition:

$$
\mu_k(x) = \frac{1}{1 + \exp\left(-\beta_k \cdot (x_k^{\text{ref}} - x)\right)},
$$

where $x$ is the constraint variable, $x_k^{\text{ref}}$ is the reference point (corresponding to $\mu_k = 0.5$), and $\beta_k > 0$ is the steepness parameter controlling the width of the fuzzy transition region. A larger $\beta_k$ yields a sharper transition, approaching a crisp constraint as $\beta_k \to \infty$.

**Example: ESS SOC constraint.** For the upper SOC bound $\text{SOC}_{\text{max}}$, the FCSD is:

$$
\mu_{\text{SOC, upper}}(s, a) = \frac{1}{1 + \exp\left(-\beta_{\text{SOC}} \cdot (\text{SOC}_{\text{ref, upper}} - \text{SOC}_{t+1})\right)},
$$

where $\text{SOC}_{t+1}$ is the predicted next-step SOC after taking action $a$ in state $s$. The lower bound is defined analogously. The combined SOC satisfaction degree is the minimum of the two: $\mu_{\text{SOC}} = \min(\mu_{\text{SOC, upper}}, \mu_{\text{SOC, lower}})$.

### 3.3.3 Multi-Constraint Aggregation

With $K$ individual fuzzy constraints, we need an aggregation operator that produces a single joint satisfaction degree $\tilde{\mu}(s, a) \in [0, 1]$. We consider three standard fuzzy aggregation operators:

1. **Minimum t-norm** (Gödel implication): $\tilde{\mu} = \min_{k=1,\dots,K} \mu_k$. This is the most conservative—any single constraint violation drags down the entire joint degree.
2. **Product t-norm**: $\tilde{\mu} = \prod_{k=1}^{K} \mu_k$. Moderately conservative; all constraints contribute multiplicatively.
3. **Weighted average**: $\tilde{\mu} = \sum_{k=1}^{K} w_k \mu_k$, where $\sum w_k = 1$ and $w_k \geq 0$. Allows explicit constraint prioritization but does not guarantee that a single critical constraint dominates.

**Our choice: weighted product aggregation.** We adopt a **weighted product t-norm** that combines the benefits of multiplicative aggregation and constraint prioritization:

$$
\tilde{\mu}(s, a) = \prod_{k=1}^{K} \mu_k(s, a)^{w_k},
$$

where $w_k \geq 0$ are constraint importance weights (normalized such that $\max w_k = 1$ for numerical stability). This formulation has two desirable properties: (i) critical constraints with higher weights have a stronger influence on the joint satisfaction degree; (ii) the function is differentiable everywhere on $(0, 1)^K$, facilitating gradient-based learning; and (iii) it reduces to the standard product t-norm when all $w_k = 1$.

The aggregated FCSD $\tilde{\mu}(s, a)$ serves as the foundation for our Fuzzy Constrained MDP framework, which we develop in Chapter 4.

---

*End of Chapter 3.*

---

**Word count**: ~1,480 words
