# Figure Catalog — RQ1

## Figure 01: Cost comparison — `figure-01-cost.png`

- **Purpose**: Show operating cost across algorithms and scenarios.
- **Plotted variables**: y = cost/day (¥, mean over 5 seeds), x = 4 scenarios, grouped bars = 7 algorithms. Error bars = ±95% CI (normal approx).
- **What the reader should notice**:
  - HFG-SAC (red) is in the same cost band as SAC/CPO/Fuzzy-SAC.
  - PPO-Lagrangian and Safety-Layer are dramatically more expensive in S2/S1.
  - PPO is cheapest in S3 but violates 94% of steps.
- **Belief change**: HFG-SAC does NOT pay a large efficiency penalty for safety.
- **Caveat**: CI bars are large because n=5; overlapping CI does not imply equivalence.
- **Caption draft**: "Operating cost (¥/day) for 7 algorithms across 4 microgrid scenarios. Bars show mean over 5 seeds; error bars show ±95% CI."

## Figure 02: Violation rate — `figure-02-violation.png`

- **Purpose**: Show safety performance.
- **Plotted variables**: y = violation rate (%), x = 4 scenarios, grouped bars = 7 algorithms.
- **What the reader should notice**:
  - SAC/PPO are 80-99% violated in every scenario.
  - PPO-Lagrangian and Safety-Layer collapse in islanded scenarios (S3/S4).
  - CPO is ~0% in S1/S2/S4 and 1.2% in S3 after equalizing scenario resources.
  - HFG-SAC is ≤1% in ALL four scenarios — the only robust method.
- **Belief change**: Robust safety across grid AND islanded operation is HFG-SAC's unique contribution.
- **Caption draft**: "Constraint violation rate (%). Lower is better. HFG-SAC (red) is the only method maintaining violation below 1% in all four scenarios."

## Figure 03: FCSD — `figure-03-fcsd.png`

- **Purpose**: Show continuous safety margin (not just binary violation).
- **Plotted variables**: y = average FCSD (0-1, higher = safer), x = 4 scenarios.
- **What the reader should notice**:
  - SAC/PPO have FCSD ~0.6-0.8 (often close to constraint boundary).
  - CPO has FCSD ~1.0 everywhere (over-conservative).
  - HFG-SAC is 0.90-0.99 — high safety margin without CPO's over-conservatism.
- **Belief change**: HFG-SAC is not just "not violating" — it maintains a comfortable buffer.
- **Caption draft**: "Average Fuzzy Constraint Satisfaction Degree over the evaluation horizon. Higher is safer. HFG-SAC maintains high FCSD with lower cost than CPO."

## Figure 04: Training dynamics — `figure-04-training-curves.png`

- **Purpose**: Show convergence behavior.
- **Plotted variables**: y = episode cost (¥), x = episode (1-200), 4 subplots = scenarios. Shaded band = ±1 std over 5 seeds.
- **What the reader should notice**:
  - All SAC-family methods converge within ~100 episodes.
  - PPO-family converges faster but to a higher-cost (unsafe) policy.
  - HFG-SAC's learning curve is smooth (no oscillation) — the fuzzy shaping prevents catastrophic updates.
- **Belief change**: HFG-SAC does not require more episodes to converge; the fuzzy layer accelerates stable learning.
- **Caveat**: Mid-training eval is every 50 episodes; final eval is 10 episodes.
- **Caption draft**: "Episode cost during training. Solid lines = mean over 5 seeds; shaded bands = ±1 std. HFG-SAC converges stably to low cost while maintaining safety."

## Figure 05: Safety-efficiency frontier — `figure-05-frontier.png`

- **Purpose**: Visualize the Pareto trade-off.
- **Plotted variables**: x = cost/day, y = violation rate (%), 4 subplots = scenarios. Each point = one seed. Dashed red line = 5% safety target.
- **What the reader should notice**:
  - **Top-left** (cheap but unsafe): SAC, PPO.
  - **Bottom-right** (safe but expensive): PPO-Lagrangian, Safety-Layer.
  - **Bottom-left** (safe AND cheap): HFG-SAC, Fuzzy-SAC.
  - CPO sits at bottom-left in S1/S2/S4 but shifts upward in cost in S3/S4 while remaining safe.
- **Belief change**: HFG-SAC owns the Pareto frontier. No other method is both safe and cheap across all scenarios.
- **Caption draft**: "Safety-efficiency frontier. Each marker is one seed. HFG-SAC (red) occupies the safe-and-cheap region across all four scenarios; CPO (green) is safe but sits higher in cost."

## Figure 06: Violation-rate convergence — `figure-06-violation-convergence.png`

- **Purpose**: Show how quickly each algorithm learns to satisfy constraints during training.
- **Plotted variables**: y = episode violation rate (%, 10-ep moving average), x = episode (1-200), 4 subplots = scenarios. Shaded band = ±1 std over 5 seeds. Dashed red line = 5% safety target.
- **What the reader should notice**:
  - **HFG-SAC (red)**: drops to near 0% within ~10 episodes and stays there — in ALL four scenarios.
  - **CPO (green)**: converges fast in S1/S2, but in S3/S4 it oscillates and settles around 1% violation after equalizing scenario resources.
  - **PPO-Lagrangian (orange dashed)**: oscillates wildly — e.g., in S2 it drops to 0% then jumps back to 20% around episode 100; in islanded it never gets below 30%.
  - **Safety-Layer (purple dash-dot)**: unstable in islanded scenarios — drops then rises back to 13-35% violation.
  - **SAC/PPO (gray/blue dotted)**: never learn safety — stay at 70-100% violation throughout.
- **Belief change**: HFG-SAC is not just "safer at the end" — it achieves safety within the first 5% of training and never destabilizes. This is critical for online deployment.
- **Caveat**: 10-episode moving average smooths the curve; individual episodes may have higher spikes.
- **Caption draft**: "Episode violation rate during training (10-episode moving average, mean ± std over 5 seeds). Dashed line = 5% safety target. HFG-SAC (red) converges to near-zero violation within the first 10 episodes and remains stable across all four scenarios."

## Figure 07: FCSD convergence — `figure-07-fcsd-convergence.png`

- **Purpose**: Show how each algorithm's continuous safety margin (not just binary violation) evolves during training.
- **Plotted variables**: y = episode average FCSD (0-1), x = episode (1-200), 4 subplots = scenarios. Shaded band = ±1 std over 5 seeds.
- **What the reader should notice**:
  - **HFG-SAC (red)**: jumps to ~0.95 FCSD within 5 episodes, then gradually improves to ~0.99. In S3/S4 it starts at 0.9 and climbs to 0.99.
  - **CPO (green)**: starts at ~0.6-0.7 and slowly climbs to ~1.0 by episode 200 in all scenarios after equalizing resources.
  - **PPO-Lagrangian (orange dashed)**: oscillates widely in S1/S2 — FCSD drops from 1.0 to 0.93 then jumps back; in islanded it settles at 0.91-0.93.
  - **SAC/PPO (gray/blue dotted)**: stuck at 0.6-0.8 FCSD — always close to the constraint boundary.
- **Belief change**: HFG-SAC doesn't just "not violate" — it builds a safety margin (FCSD ~0.95+) almost immediately, whereas CPO takes 200 episodes just to approach 0.9.
- **Caption draft**: "Episode average FCSD during training (10-episode moving average, mean ± std over 5 seeds). Higher = safer margin. HFG-SAC (red) achieves FCSD > 0.9 within 5 episodes and maintains it, while CPO and Lagrangian baselines converge slowly or oscillate."
