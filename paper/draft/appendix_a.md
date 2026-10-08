# Appendix A — Proofs of Theoretical Results

> **Status**: v1.0
> **Companion to**: Chapter 4 (FC-MDP, Fuzzy-Lagrangian, HFG-SAC)
> **Conventions**: Additive convention for the Fuzzy-Lagrangian:
> $\mathcal{L}_{\text{fuzzy}}(\pi, \lambda) = J_R(\pi) + \sum_{k=1}^K \lambda_k(J_{\mu_k}(\pi) - \alpha_k/(1-\gamma))$, with $\lambda_k \geq 0$.
> Crisp membership form: $\mu_k(s,a) = \sigma(\beta_k(d_k - c_k(s,a)))$, $\sigma(x)=1/(1+e^{-x})$.
> Crisp-CMDP target: $\alpha_k^{\text{crisp}} = 1$.

---

## A.1 Useful Lemmas

**Lemma A.1 (Sigmoid Pointwise Limit).** For every $x \in \mathbb{R}$,
$$
\lim_{\beta\to\infty} \sigma(\beta x) = \mathbb{1}\{x > 0\} + \tfrac{1}{2}\,\mathbb{1}\{x = 0\}.
$$
Moreover, $\sigma(\beta x)$ is monotone non-decreasing in $x$ for every $\beta > 0$.

*Proof.* For $x > 0$, $\beta x \to +\infty$ and $\sigma(\beta x) \to 1$; for $x < 0$, $\beta x \to -\infty$ and $\sigma(\beta x) \to 0$; for $x = 0$, $\sigma(\beta\cdot 0) = \tfrac{1}{2}$ for all $\beta$. Monotonicity is immediate from $d\sigma/dx > 0$. $\blacksquare$

**Lemma A.2 (Inverse-Sigmoid Level Set).** For $\beta > 0$ and $\varepsilon \in (0, \tfrac{1}{2})$,
$$
\sigma(\beta x) \geq 1 - \varepsilon \;\Longleftrightarrow\; x \geq \tfrac{1}{\beta}\sigma^{-1}(1-\varepsilon) = \tfrac{1}{\beta}\ln\tfrac{1-\varepsilon}{\varepsilon} =: \tau_\varepsilon/\beta.
$$
*Proof.* Direct inversion of $\sigma$. $\blacksquare$

**Lemma A.3 (Dini Convergence on Compact Sets).** Let $K \subset \mathbb{R}^n$ be compact and $f_\beta(x) = \sigma(\beta g(x))$ with $g$ continuous. Then
$$
\sup_{x\in K} |f_\beta(x) - f_\infty(x)| \to 0 \quad (\beta \to \infty),
$$
where $f_\infty(x) = \mathbb{1}\{g(x) > 0\} + \tfrac{1}{2}\mathbb{1}\{g(x) = 0\}$.
In particular, for $K = \mathcal{S}\times\mathcal{A}$ (compact) and $g(s,a) = d_k - c_k(s,a)$, we have **uniform** convergence $\mu_k^{(\beta)}(s,a) \to \mu_k^{(\infty)}(s,a)$.

*Proof.* The family $\{f_\beta\}$ is monotone non-decreasing in $\beta$ pointwise (since $\sigma$ is monotone and $\beta g$ increases in $\beta$ when $g > 0$, decreases when $g < 0$, and is constant when $g = 0$). A monotone family of continuous functions converging pointwise to a continuous limit on a compact set converges uniformly by Dini's theorem. Here $f_\infty$ is discontinuous, so we apply Dini on each of the two closed sets $\{g > 0\}$ and $\{g < 0\}$ separately (where $f_\infty = 1$ and $0$ are continuous). The discontinuity set $\{g = 0\}$ is closed with measure zero; on it the convergence is to $\tfrac{1}{2}$, matching $\sigma(\beta\cdot 0) = \tfrac{1}{2}$. $\blacksquare$

**Lemma A.4 (Bounded Linear Functional of Policy).** For a stationary policy $\pi$ on FC-MDP with $|R(s,a)| \leq R_{\max}$ and $|\mu_k(s,a)| \leq 1$, the value functions are bounded:
$$
|J_R(\pi)| \leq \frac{R_{\max}}{1-\gamma}, \qquad |J_{\mu_k}(\pi)| \leq \frac{1}{1-\gamma}.
$$
*Proof.* Standard discounted-reward bound. $\blacksquare$

**Lemma A.5 (Slater's Condition in FC-MDP).** Assume condition (iv) of Theorem 4.2: there exists a stationary policy $\pi_0$ such that $J_{\mu_k}(\pi_0) > \alpha_k/(1-\gamma)$ for all $k = 1, \dots, K$ (strict feasibility). Then strong duality holds between
$$
\max_{\pi \text{ is } \alpha\text{-safe}} J_R(\pi) \quad\text{and}\quad \min_{\lambda \geq 0} \max_\pi \mathcal{L}_{\text{fuzzy}}(\pi, \lambda).
$$
*Proof.* Standard Slater's condition for the convex program in occupancy measure. See §A.5. $\blacksquare$

---

## A.2 Proof of Theorem 4.1 (FC-MDP → CMDP Limit, $\mu$-Level Form)

**Statement.** Let $\mu_k^{(\beta)}(s,a) = \sigma(\beta_k(d_k - c_k(s,a)))$. For every $\varepsilon \in (0, \tfrac{1}{2})$ and every $(s,a) \in \mathcal{S}\times\mathcal{A}$,
$$
\lim_{\beta_k\to\infty} \mathbb{1}\{\mu_k^{(\beta)}(s,a) \geq 1-\varepsilon\} = \mathbb{1}\{c_k(s,a) \leq d_k\}. \tag{A.1}
$$
Consequently, for any stationary policy $\pi$,
$$
\lim_{\beta_k\to\infty} \mathbb{1}\{J_{\mu_k}(\pi) \geq (1-\varepsilon)/(1-\gamma)\} = \mathbb{1}\{J_{C_k}(\pi) \leq d_k\}, \tag{A.2}
$$
where $J_{C_k}(\pi) = \mathbb{E}_\pi[\sum_t \gamma^t c_k(s_t,a_t)]$.

**Proof of (A.1).** By Lemma A.2, the event $\{\mu_k^{(\beta)}(s,a) \geq 1-\varepsilon\}$ is equivalent to
$$
d_k - c_k(s,a) \geq \tau_\varepsilon / \beta_k \;\Longleftrightarrow\; c_k(s,a) \leq d_k - \tau_\varepsilon/\beta_k.
$$
As $\beta_k \to \infty$, $\tau_\varepsilon/\beta_k \to 0$ and the right-hand side converges to $c_k(s,a) \leq d_k$. Indicator functions are upper-semicontinuous, so the limit is pointwise as stated. $\square$

**Proof of (A.2).** Let $V_\beta(\pi) = J_{\mu_k}^{(\beta)}(\pi) - (1-\varepsilon)/(1-\gamma)$. Under Lemma A.4 we have $|V_\beta(\pi)| \leq 2/(1-\gamma)$. By Lemma A.3 (uniform convergence on the compact product space), $\mu_k^{(\beta)}(s,a) \to \mu_k^{(\infty)}(s,a)$ uniformly. The discounted-reward operator $T_\mu(\pi) := \mathbb{E}_\pi[\sum_t \gamma^t \mu(s_t,a_t)]$ is continuous in $\mu$ under the uniform metric on $\mathcal{S}\times\mathcal{A}$:
$$
|T_{\mu^{(\beta)}}(\pi) - T_{\mu^{(\infty)}}(\pi)| \leq \sum_t \gamma^t \|\mu^{(\beta)} - \mu^{(\infty)}\|_\infty = \|\mu^{(\beta)} - \mu^{(\infty)}\|_\infty/(1-\gamma) \to 0.
$$
Therefore $J_{\mu_k}^{(\beta)}(\pi) \to J_{\mu_k}^{(\infty)}(\pi)$ for every fixed $\pi$.

Now $J_{\mu_k}^{(\infty)}(\pi) = \mathbb{E}_\pi[\sum_t \gamma^t \mathbb{1}\{c_k(s_t,a_t) \leq d_k\}]$. For a stationary $\pi$ the per-step expectation is a constant $p_k \in [0,1]$, so $J_{\mu_k}^{(\infty)}(\pi) = p_k/(1-\gamma)$. Hence $J_{\mu_k}^{(\infty)}(\pi) \geq 1/(1-\gamma)$ iff $p_k = 1$, i.e., iff $c_k(s_t,a_t) \leq d_k$ almost surely along trajectories of $\pi$. This is equivalent to $J_{C_k}(\pi) \leq d_k$ (since the cost only accrues when $c_k > d_k$, and the discounted average is then strictly positive). Choosing any $\varepsilon \in (0, \tfrac{1}{2})$ and passing to the limit $\beta_k \to \infty$ under dominated convergence (Lemma A.4 provides the dominating function) gives (A.2). $\square$

**Corollary A.1 (Asymptotic Recovery of $\alpha$-Safety).** With $\alpha_k^{\text{crisp}} = 1$ for all $k$, the set of $\alpha$-safe policies in FC-MDP converges to the set of feasible policies in the standard CMDP as $\beta_k \to \infty$ for all $k$.

*Proof.* Immediate from Theorem 4.1 applied pointwise to each constraint and intersecting the resulting limiting sets. $\blacksquare$
# Appendix A (Part 2) — Proofs of Theoretical Results

> Continuation of `appendix_a_part1.md`. Contains §A.3 (Theorem 4.2), §A.4 (Theorem 4.3), §A.5 (Slater), §A.6 (Theorem 4.4), §A.7 (Notation summary).

---

## A.3 Proof of Theorem 4.2 (Existence of Optimal $\alpha$-Safe Policy)

**Statement (restated).** Suppose:
- (i) $\mathcal{S}$, $\mathcal{A}$ compact subsets of $\mathbb{R}^n$, $\mathbb{R}^m$;
- (ii) transition kernel $P(\cdot\mid s,a)$ weakly continuous in $(s,a)$;
- (iii) $R(s,a)$ and all $\mu_k(s,a)$ bounded and continuous;
- (iv) Slater's condition: $\exists\, \pi_0$ with $J_{\mu_k}(\pi_0) > \alpha_k/(1-\gamma)$ $\forall k$.

Then there exists a stationary deterministic policy $\pi^*$ that is optimal among all $\alpha$-safe policies.

**Proof.** We adopt the convex-analytic / occupancy-measure approach (Puterman, 1994; Altman, 1999).

**Step 1 — Occupancy measure.** For a stationary policy $\pi$, define the (state-action) occupancy measure
$$
\rho_\pi(s,a) = (1-\gamma) \sum_{t=0}^\infty \gamma^t \,\mathbb{P}^\pi(s_t = s, a_t = a).
$$
Under (i)–(ii), the set $\mathcal{M} := \{\rho_\pi : \pi \in \Pi_{\text{sdet}}\}$ of occupancy measures induced by stationary deterministic policies is **compact** in the weak-* topology on the space of signed measures on $\mathcal{S}\times\mathcal{A}$ (by Tychonoff's theorem and the bound $\sum_{s,a} \rho_\pi(s,a) = 1$). Its convex hull $\mathcal{M}_{\text{conv}} = \text{conv}(\mathcal{M})$ corresponds to occupancy measures of *stationary stochastic* policies; by a standard representation theorem, every $\rho \in \mathcal{M}_{\text{conv}}$ is realized by some stationary stochastic $\pi$.

**Step 2 — Linear functionals.** The value functional and the satisfaction functionals are linear in $\rho$:
$$
J_R(\pi) = \frac{1}{1-\gamma} \sum_{s,a} \rho_\pi(s,a) R(s,a), \qquad
J_{\mu_k}(\pi) = \frac{1}{1-\gamma} \sum_{s,a} \rho_\pi(s,a) \mu_k(s,a).
$$
Under (iii), these functionals are continuous on $(\mathcal{M}_{\text{conv}}, \text{weak-}*)$.

**Step 3 — Feasible set.** Define the feasible set
$$
\mathcal{F} := \{\rho \in \mathcal{M}_{\text{conv}} : J_{\mu_k}(\rho) \geq \alpha_k/(1-\gamma),\, k = 1, \dots, K\}.
$$
By continuity of $J_{\mu_k}$ and linearity of $\rho \mapsto J_{\mu_k}(\rho)$, $\mathcal{F}$ is a **closed convex** subset of $\mathcal{M}_{\text{conv}}$.

**Step 4 — Slater's condition in occupancy measure.** Under (iv), $\rho_{\pi_0} \in \mathcal{F}$ satisfies $J_{\mu_k}(\rho_{\pi_0}) > \alpha_k/(1-\gamma)$ strictly, so $\mathcal{F}$ has non-empty interior (in the relative topology of $\mathcal{M}_{\text{conv}}$).

**Step 5 — Existence.** The objective $J_R$ is continuous and the feasible set $\mathcal{F}$ is non-empty, closed, and bounded (since $\mathcal{M}_{\text{conv}}$ is bounded in total variation). By the Weierstrass extreme value theorem in the weak-* topology, the supremum $\sup_{\rho \in \mathcal{F}} J_R(\rho)$ is attained at some $\rho^* \in \mathcal{F}$.

**Step 6 — Determinism.** Since $J_R$ is linear and $\mathcal{F}$ is convex, by the Krein–Milman theorem the optimum is attained at an **extreme point** of $\mathcal{F}$. The extreme points of $\mathcal{M}_{\text{conv}}$ are exactly the elements of $\mathcal{M}$ (stationary deterministic occupancy measures); since $\mathcal{F}$ is a closed convex set, its extreme points are extreme points of $\mathcal{M}$ that lie in $\mathcal{F}$. Therefore there exists a stationary **deterministic** $\pi^*$ with $\rho_{\pi^*} \in \mathcal{F}$ that attains the optimum. $\square$

---

## A.4 Proof of Theorem 4.3 (Strong Duality and Convergence of the Dual Gradient Algorithm)

**Statement.** Under the conditions of Theorem 4.2 (in particular Slater's condition), strong duality holds:
$$
\max_{\pi \text{ is } \alpha\text{-safe}} J_R(\pi) = \min_{\lambda \geq 0}\, \max_\pi \mathcal{L}_{\text{fuzzy}}(\pi, \lambda).
$$
Moreover, the projected-subgradient update on the dual variables,
$$
\lambda_k^{(t+1)} = \left[\,\lambda_k^{(t)} + \eta_t \cdot \bigl(\bar J_{\mu_k}(\pi_{\lambda^{(t)}}) - \alpha_k/(1-\gamma)\bigr)\,\right]_{+},
\tag{A.3}
$$
with step sizes $\eta_t > 0$ satisfying $\sum_{t=0}^\infty \eta_t = \infty$ and $\sum_{t=0}^\infty \eta_t^2 < \infty$, produces multipliers $\lambda^{(t)} \to \lambda^*$ and policies $\pi_{\lambda^{(t)}} \to \pi^*$ almost surely, where $(\pi^*, \lambda^*)$ is a saddle point of $\mathcal{L}_{\text{fuzzy}}$.

**Proof of strong duality.** By Lemma A.5 (Slater's condition in FC-MDP), the FC-MDP reformulated in occupancy measure (§A.3) is a convex program with a strictly feasible point. By the standard Lagrangian duality theorem for convex programs (Boyd & Vandenberghe, 2004, §5.3.2), zero duality gap holds:
$$
\sup_{\rho \in \mathcal{F}} \sum_{s,a} \rho(s,a) R(s,a) = \inf_{\lambda \geq 0} \sup_{\rho \in \mathcal{M}_{\text{conv}}} \bigl[ \sum_{s,a} \rho R(s,a) + \sum_k \lambda_k (J_{\mu_k}(\rho) - \alpha_k/(1-\gamma)) \bigr].
$$
The right-hand side is exactly $\min_{\lambda \geq 0} g(\lambda)$ with $g(\lambda) = \max_\pi \mathcal{L}_{\text{fuzzy}}(\pi, \lambda)$ (since any $\rho \in \mathcal{M}_{\text{conv}}$ is realized by some stationary stochastic $\pi$). $\square$

**Proof of convergence of (A.3).** Standard Robbins–Siegmund argument (Robbins & Siegmund, 1971; see also Bertsekas, 1999, Proposition 5.1).

**Step 1 — Dual function is convex and Lipschitz.** Under Lemma A.4, $|J_{\mu_k}(\pi) - \alpha_k/(1-\gamma)| \leq 2/(1-\gamma)$. The dual function
$$
g(\lambda) = \max_\pi \mathcal{L}_{\text{fuzzy}}(\pi, \lambda)
$$
is convex in $\lambda$ (pointwise supremum of affine functions) and $G$-Lipschitz continuous with $G := \sqrt{K}/(1-\gamma)$ (subgradient bounded by Lemma A.4).

**Step 2 — Descent lemma.** Let $h_t(\lambda) := J_{\mu_k}(\pi_{\lambda^{(t)}}) - \alpha_k/(1-\gamma)$. By convexity of $g$,
$$
g(\lambda^{(t+1)}) \leq g(\lambda^{(t)}) + \langle \nabla g(\lambda^{(t)}), \lambda^{(t+1)} - \lambda^{(t)} \rangle + \tfrac{G}{2}\|\lambda^{(t+1)} - \lambda^{(t)}\|^2.
$$
With $\lambda^{(t+1)} = [\lambda^{(t)} + \eta_t h_t]_+$ (note the additive convention: $\lambda$ increases when $\bar J_{\mu_k} > \alpha_k$), we have $\|\lambda^{(t+1)} - \lambda^{(t)}\|^2 \leq \eta_t^2 \|h_t\|^2 \leq \eta_t^2 K/(1-\gamma)^2$, hence the last term is bounded by $C \eta_t^2$. The inner-product term is $-\eta_t \|\nabla g(\lambda^{(t)})\|^2 \leq 0$ (up to measurability noise). Therefore
$$
g(\lambda^{(t+1)}) \leq g(\lambda^{(t)}) - \eta_t \|\nabla g(\lambda^{(t)})\|^2 + C \eta_t^2 + \text{noise}_t.
$$
Summing over $t$ and using $\sum_t \eta_t^2 < \infty$, the Robbins–Siegmund lemma gives $g(\lambda^{(t)}) \to g(\lambda^*)$ and $\sum_t \eta_t \|\nabla g(\lambda^{(t)})\|^2 < \infty$, hence $\nabla g(\lambda^{(t)}) \to 0$. Combined with the strong-convexity-like structure of $g$ near its minimum (which follows from Slater's condition, since the dual is essentially smooth), this gives $\lambda^{(t)} \to \lambda^*$.

**Step 3 — Policy convergence.** For any $\varepsilon > 0$, the set $\{\pi : J_{\mu_k}(\pi) \geq \alpha_k/(1-\gamma) - \varepsilon\}$ is non-empty for large $t$ (since $\lambda_k^{(t)} \to \lambda_k^*$ finite). The corresponding inner-maximizer $\pi_{\lambda^{(t)}}$ achieves $J_R(\pi_{\lambda^{(t)}}) \to J_R(\pi^*)$ by strong duality. Policy convergence holds up to a measure-zero equivalence class of optimal policies (there may be multiple optima when the objective is constant on a face of $\mathcal{F}$). $\square$
# Appendix A (Part 3) — Slater, Theorem 4.4, Notation Summary

> Continuation of `appendix_a_part1.md` and `appendix_a_part2.md`. Contains §A.5, §A.6, §A.7.

---

## A.5 Slater's Condition in FC-MDP (Proof of Lemma A.5)

**Statement.** The FC-MDP optimization problem, reformulated in occupancy measure as
$$
\max_{\rho \in \mathcal{M}_{\text{conv}}} \sum_{s,a} \rho(s,a) R(s,a) \quad \text{s.t.} \quad \sum_{s,a} \rho(s,a) \mu_k(s,a) \geq \frac{\alpha_k}{1-\gamma} \;\; \forall k,
$$
is a convex program. Slater's condition holds iff there exists a strictly feasible policy $\pi_0$ with $J_{\mu_k}(\pi_0) > \alpha_k/(1-\gamma)$ for all $k$.

**Proof.** *Convexity.* The objective is linear in $\rho$. The constraint set $\mathcal{M}_{\text{conv}}$ is convex (by definition). The functions $\rho \mapsto \sum_{s,a} \rho \mu_k(s,a)$ are linear, hence the inequality constraints define a convex set. The intersection of convex sets is convex. Therefore the feasible set is convex and the program is convex.

*Slater sufficiency.* Suppose $\pi_0$ is strictly feasible, i.e., $\rho_{\pi_0}(s,a)$ yields $J_{\mu_k}(\rho_{\pi_0}) > \alpha_k/(1-\gamma)$ strictly for all $k$. Take any convex combination $\rho = \sum_i \theta_i \rho_{\pi_i}$ with $\sum_i \theta_i = 1$, $\theta_i \geq 0$, where at least one $\rho_{\pi_i}$ is strictly feasible. By linearity, $J_{\mu_k}(\rho) = \sum_i \theta_i J_{\mu_k}(\rho_{\pi_i})$, which is a convex combination of values each bounded below by $\alpha_k/(1-\gamma)$. Hence $J_{\mu_k}(\rho) \geq \alpha_k/(1-\gamma)$. In particular, $\rho_{\pi_0}$ is a strictly feasible point, so Slater's condition is satisfied.

By Boyd & Vandenberghe (2004, §5.3.2), Slater's condition implies **zero duality gap** for a convex program with affine constraints. $\blacksquare$

**Remark A.5 (Verification for Microgrid Problems).** In practice, Slater's condition can be verified for FC-MDPs arising in microgrid dispatch by exhibiting a **conservative policy**: e.g., charge ESS at a low fixed rate, run diesel at a low constant output, and curtail non-critical loads when supply is tight. Such a policy has FCSD $\bar\mu_k \approx 1$ for SOC and load-curtailment constraints, $\bar\mu_{\text{freq}} \approx 1$ since diesel is on but with margin, and $\bar\mu_{\text{voltage}} \approx 1$ under normal operation. Hence strict feasibility holds and Theorem 4.3 applies.

**Important caveat:** for the extreme islanded scenario S4 (Chapter 5), the *original* action space (200 kW interruptible load) does not admit a strictly feasible $\alpha$-safe policy for $\alpha_k \geq 0.9$ on all constraints simultaneously — we observed this empirically as the **structural infeasibility** discussed in §5.3. The expansion to 600 kW interruptible load enlarges the action space and restores Slater's condition, so Theorem 4.3 applies to the *extended* S4. This is consistent with the theorem, which is conditional on Slater's condition holding for the FC-MDP under consideration.

---

## A.6 Proof of Theorem 4.4 (Policy Invariance under Fuzzy Shaping)

**Statement.** Let $R'(s, a, s') = R(s, a) + F(s, s')$ where $F(s, s') = \gamma \Phi(s') - \Phi(s)$ and $\Phi(s) = \max_a R_{\text{know}}(s, a)$. Then every optimal policy for the MDP with reward $R'$ is also optimal for the original MDP with reward $R$.

**Proof.** This is a direct application of the potential-based shaping theorem of Ng, Harada, and Russell (1999, Theorem 1). We verify the two required conditions:

1. **$\Phi$ is well-defined.** The compactness of $\mathcal{A}$ and continuity of $R_{\text{know}}$ guarantee that $\max_a R_{\text{know}}(s, a)$ exists for every $s \in \mathcal{S}$.

2. **$F$ is a difference of potentials.** $F(s, s') = \gamma \Phi(s') - \Phi(s)$ has the required difference form, which is the defining property ensuring policy invariance (see Ng et al., 1999, Proposition 1).

By Ng et al. (1999, Theorem 1), for any potential $\Phi$, the optimal $Q$-function satisfies
$$
Q^*(s, a) = \mathbb{E}\bigl[ R(s,a) + \gamma \Phi(s') \bigr] - \Phi(s) + \gamma V^*(s)
$$
where $V^*$ is the optimal value function of the original MDP. Hence
$$
\arg\max_a Q^*(s, a) = \arg\max_a \mathbb{E}\bigl[ R(s,a) + \gamma \Phi(s') \bigr] - \Phi(s) = \arg\max_a \mathbb{E}\bigl[ R(s,a) + \gamma \Phi(s') \bigr],
$$
since the constant $-\Phi(s)$ does not affect the argmax. The term $\mathbb{E}[\gamma \Phi(s')]$ is state-action-independent when $\Phi$ depends only on state (which is our case). Therefore the argmax coincides with the argmax of $\mathbb{E}[R(s,a)]$ under the original MDP, i.e., the set of optimal policies is unchanged. $\blacksquare$

**Remark A.6 (Adaptive Decay Preserves Optimality).** The exponential decay $\kappa_t = \kappa_0 \rho^{\lfloor t/T_{\text{decay}} \rfloor}$ with $\rho \in (0,1)$ gives $\kappa_t \to 0$ as $t \to \infty$. At every finite $t$, the shaped reward $R_t = R + \kappa_t F$ is a potential-based shaping of $R$ with potential $\kappa_t \Phi$, so by Theorem 4.4 the optimal policy set of $R_t$ coincides with that of $R$. Taking the limit $t \to \infty$ along any convergent subsequence, the asymptotic optimal policy is optimal for the original $R$. Note that the decay schedule does not affect the *set* of optimal policies at any fixed $t$, but it affects the *learning dynamics*: with $\kappa_0$ small the shaping provides weak guidance and slow learning; with $\kappa_0$ large the shaping provides strong guidance but if $\rho$ is too small the agent may over-commit to the expert's (possibly imperfect) policy before $\kappa_t$ decays. Choosing $\kappa_0, \rho$ to balance these effects is a hyperparameter selection problem studied in §5.5.

---

## A.7 Summary of Notation and Assumptions

For convenience, we collect the main objects and assumptions used in the appendix:

| Symbol | Meaning |
|---|---|
| $\mathcal{S}, \mathcal{A}$ | State and action spaces, compact subsets of Euclidean spaces (Assumption (i)) |
| $P(\cdot\mid s, a)$ | Transition kernel, weakly continuous (Assumption (ii)) |
| $R(s, a)$ | Reward function, bounded and continuous (Assumption (iii)) |
| $\mu_k(s, a)$ | FCSD for constraint $k$, bounded and continuous in $(s, a)$ |
| $\alpha_k \in (0, 1]$ | Target satisfaction level for constraint $k$ |
| $\beta_k > 0$ | Sigmoid steepness parameter for constraint $k$ |
| $d_k$ | Crisp constraint threshold (when interpreting FCSD as sigmoid of $d_k - c_k$) |
| $\lambda_k \geq 0$ | Lagrange multiplier for constraint $k$ (additive convention) |
| $\gamma \in [0, 1)$ | Discount factor |
| $J_R(\pi)$ | Expected discounted return: $\mathbb{E}_\pi[\sum_t \gamma^t R(s_t, a_t)]$ |
| $J_{\mu_k}(\pi)$ | Expected discounted FCSD: $\mathbb{E}_\pi[\sum_t \gamma^t \mu_k(s_t, a_t)]$ |
| $\mathcal{L}_{\text{fuzzy}}(\pi, \lambda)$ | Fuzzy-Lagrangian: $J_R(\pi) + \sum_k \lambda_k(J_{\mu_k}(\pi) - \alpha_k/(1-\gamma))$ |
| $\rho_\pi(s, a)$ | Occupancy measure: $(1-\gamma)\sum_t \gamma^t \mathbb{P}^\pi(s_t = s, a_t = a)$ |
| $\mathcal{M}_{\text{conv}}$ | Convex hull of stationary-deterministic occupancy measures |
| $\sigma(x)$ | Sigmoid function $1/(1+e^{-x})$ |
| $\Phi(s)$ | Potential function for reward shaping: $\max_a R_{\text{know}}(s,a)$ |
| $\kappa_t$ | Adaptive knowledge weight at time $t$ |

**Cross-references to the main text:**
- Theorem 4.1 → proof in §A.2; corollary in §A.2.
- Theorem 4.2 → proof in §A.3.
- Theorem 4.3 → strong duality in §A.4; convergence in §A.4.
- Theorem 4.4 → proof in §A.6.
- Lemma A.5 (Slater) → proof in §A.5.
- Adaptive decay preserves optimality → Remark A.6.

**Caveat on use of $\sigma(\cdot)$ at $x = 0$.** Throughout the appendix we adopt the convention $\sigma(0) = 1/2$. This matches the limit of $\sigma(\beta x)$ as $\beta \to \infty$ for $x = 0$ and ensures continuity in $\beta$. The midpoint convention is harmless when $\alpha_k^{\text{crisp}} = 1$ is used in the limit statement (Remark 4.1).

---

*End of Appendix A.*
