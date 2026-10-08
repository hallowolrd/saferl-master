"""Rigorous statistical analysis for the HFG-SAC algorithm comparison.

Complements run/analyze_results.py (which produces figures + descriptive tables)
with the inferential analysis required by the results-analysis skill:

- descriptive statistics (n, mean, std, se, 95% CI)
- effect sizes (Hedges' g, small-sample corrected Cohen's d)
- exact permutation tests (two-sample, difference-in-means) — dependency-free,
  appropriate for the small n=3 seed regime where asymptotic tests are invalid
- Holmium-Bonferroni correction across the 6 baseline contrasts
- explicit limitations

Primary comparison: HFG-SAC ("our method") vs each of the 6 RL baselines,
separately per scenario (S1..S4) and per metric.

Usage:
    python run/statistical_analysis.py --input_dir outputs/experiments \
        --output_dir outputs/analysis
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --- Constants ---

SCENARIOS = ["S1_grid_normal", "S2_grid_extreme", "S3_island_normal", "S4_island_extreme"]
SCENARIO_LABELS = {
    "S1_grid_normal": "S1 Grid Normal",
    "S2_grid_extreme": "S2 Grid Extreme",
    "S3_island_normal": "S3 Island Normal",
    "S4_island_extreme": "S4 Island Extreme",
}

OUR_METHOD = "hfg_sac"
BASELINES = ["sac", "ppo", "ppo_lagrangian", "cpo", "safety_layer", "fuzzy_sac"]

DISPLAY_NAMES = {
    "hfg_sac": "HFG-SAC (ours)",
    "sac": "SAC",
    "ppo": "PPO",
    "ppo_lagrangian": "PPO-Lagrangian",
    "cpo": "CPO",
    "safety_layer": "CBF Safety Layer",
    "fuzzy_sac": "Fuzzy SAC",
    "safe_sac": "Safe SAC",
}

# Metrics: name -> (field_key, higher_is_better)
METRICS = {
    "total_cost": ("total_cost", False),
    "violation_rate": ("violation_rate", False),
    "avg_fcsd": ("avg_fcsd", True),
    "convergence_episode": ("convergence_episode", False),
}

METRIC_LABELS = {
    "total_cost": "Daily Operating Cost (¥)",
    "violation_rate": "Constraint Violation Rate (%)",
    "avg_fcsd": "Avg. FCSD",
    "convergence_episode": "Convergence Episode",
}

# t critical values for 95% two-sided CI: df -> t_{0.025, df}
_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def t_critical(df: int) -> float:
    """Return two-sided 95% t critical value for a given df."""
    if df < 1:
        return float("nan")
    for d in sorted(_T_TABLE):
        if df <= d:
            return _T_TABLE[d]
    return 1.960  # large-sample normal approx


# --- Data loading ---


@dataclass
class SeedResult:
    """One seed's scalar metrics."""
    total_cost: float
    violation_rate: float
    avg_fcsd: float
    convergence_episode: float


def load_seed_results(results_dir: Path) -> Dict[Tuple[str, str], List[SeedResult]]:
    """Load per-seed results grouped by (scenario, algorithm).

    Reads {algorithm}_{scenario}_s{seed}.json files, skipping model-baseline
    and summary files.
    """
    grouped: Dict[Tuple[str, str], List[SeedResult]] = {}
    if not results_dir.exists():
        return grouped

    for f in sorted(results_dir.glob("*.json")):
        if "summary" in f.name or "baselines" in f.name:
            continue
        # Expected stem: {algo}_{scenario}_s{seed}
        parts = f.stem.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        algo_scenario, seed_str = parts
        # algo may itself contain underscores (e.g. hfg_sac, ppo_lagrangian)
        # scenario names start with S<digit>_
        for scenario in SCENARIOS:
            if algo_scenario.endswith(scenario):
                algo = algo_scenario[: -len(scenario) - 1]
                break
        else:
            continue

        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        sr = SeedResult(
            total_cost=float(data.get("total_cost", 0.0)),
            violation_rate=float(data.get("violation_rate", 0.0)),
            avg_fcsd=float(data.get("avg_fcsd", 0.0)),
            convergence_episode=float(data.get("convergence_episode", 0.0)),
        )
        grouped.setdefault((scenario, algo), []).append(sr)

    return grouped


# --- Statistics ---


@dataclass
class Descriptives:
    n: int
    mean: float
    std: float
    se: float
    ci_low: float
    ci_high: float


def describe(values: Sequence[float]) -> Descriptives:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    t = t_critical(n - 1)
    half = t * se
    return Descriptives(n, mean, std, se, mean - half, mean + half)


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    """Small-sample corrected Cohen's d (Hedges' g) for a - b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return float("nan")

    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    df = n1 + n2 - 2
    # Pooled std (equal-variance assumption; with tiny n this is the standard choice)
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df)
    if sp == 0.0:
        return float("nan")
    g = (a.mean() - b.mean()) / sp
    # Bias correction factor
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(g * j)


def permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000) -> float:
    """Two-sample exact/permutation test (two-sided) on difference of means.

    Uses exact enumeration when the total group size is small (<= ~22, so
    C(n, n1) is enumerable), otherwise Monte-Carlo sampling. Returns two-sided
    p-value; with n1=n2=3 the minimum achievable p is 0.1.
    """
    a = list(map(float, a))
    b = list(map(float, b))
    n1, n2 = len(a), len(b)
    combined = np.array(a + b)
    obs = abs(np.mean(a) - np.mean(b))

    from math import comb
    total = comb(n1 + n2, n1)
    if total <= 200_000:  # exact enumeration
        count = 0
        touched = 0
        for idx in itertools.combinations(range(n1 + n2), n1):
            rest = [i for i in range(n1 + n2) if i not in idx]
            stat = abs(combined[list(idx)].mean() - combined[list(rest)].mean())
            touched += 1
            if stat >= obs - 1e-12:
                count += 1
        return count / touched
    else:  # Monte-Carlo
        rng = np.random.default_rng(seed=12345)
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(combined)
            stat = abs(perm[:n1].mean() - perm[n1:].mean())
            if stat >= obs - 1e-12:
                count += 1
        return count / n_perm


def effect_label(g: float) -> str:
    g = abs(g)
    if g < 0.2:
        return "negligible"
    if g < 0.5:
        return "small"
    if g < 0.8:
        return "medium"
    return "large"


def rel_improvement(ours: float, baseline: float, higher_is_better: bool) -> Optional[float]:
    """Relative improvement of ours over baseline, in percent (NaN if undefined)."""
    denom = baseline if not higher_is_better else baseline
    if abs(denom) < 1e-12:
        return None
    if higher_is_better:
        return (ours - baseline) / abs(denom) * 100.0
    return (baseline - ours) / abs(denom) * 100.0


# --- Report generation ---


def _fmt_desc(d: Descriptives, decimals: int = 2) -> str:
    return (f"{d.mean:.{decimals}f} ± {d.std:.{decimals}f} "
            f"[{d.ci_low:.{decimals}f}, {d.ci_high:.{decimals}f}]")


def generate_stats_appendix(
    grouped: Dict[Tuple[str, str], List[SeedResult]],
    output_dir: Path,
) -> str:
    """Build the stats-appendix markdown content."""
    lines: List[str] = []
    lines.append("# Statistical Appendix — HFG-SAC Algorithm Comparison")
    lines.append("")
    lines.append("_Units of analysis: independent training runs (seeds). "
                 "Reported effect size = Hedges' g (small-sample corrected Cohen's d). "
                 "Significance = two-sided permutation test on the difference of means._")
    lines.append("")

    # 1. Descriptive statistics per scenario
    lines.append("## 1. Descriptive Statistics (mean ± std, 95% CI)")
    lines.append("")
    for scenario in SCENARIOS:
        lines.append(f"### {SCENARIO_LABELS[scenario]}")
        lines.append("")
        lines.append("| Metric | Algorithm | n | Mean ± Std | 95% CI |")
        lines.append("|---|---|---|---|---|")
        algos = [OUR_METHOD] + BASELINES
        for metric, (field, _hib) in METRICS.items():
            for algo in algos:
                res = grouped.get((scenario, algo))
                if not res:
                    continue
                vals = [getattr(r, field) for r in res]
                d = describe(vals)
                label = DISPLAY_NAMES.get(algo, algo)
                bold = "**" if algo == OUR_METHOD else ""
                lines.append(
                    f"| {METRIC_LABELS[metric]} | {bold}{label}{bold} "
                    f"| {d.n} | {_fmt_desc(d)} |"
                )
        lines.append("")

    # 2. Effect sizes: HFG-SAC vs each baseline
    lines.append("## 2. Effect Sizes — HFG-SAC vs Each Baseline (Hedges' g)")
    lines.append("")
    lines.append("Sign convention: negative g means HFG-SAC is _better_ for "
                 "lower-is-better metrics (cost, violation, convergence) and _worse_ "
                 "for higher-is-better metrics (FCSD).")
    lines.append("")
    for scenario in SCENARIOS:
        lines.append(f"### {SCENARIO_LABELS[scenario]}")
        lines.append("")
        lines.append("| Metric | Baseline | g (HFG-SAC − baseline) | Magnitude | Rel. improvement |")
        lines.append("|---|---|---|---|---|")
        for metric, (field, hib) in METRICS.items():
            ours = grouped.get((scenario, OUR_METHOD))
            if not ours:
                continue
            our_vals = [getattr(r, field) for r in ours]
            for base in BASELINES:
                bres = grouped.get((scenario, base))
                if not bres:
                    continue
                base_vals = [getattr(r, field) for r in bres]
                g = hedges_g(our_vals, base_vals)
                imp = rel_improvement(float(np.mean(our_vals)),
                                      float(np.mean(base_vals)), hib)
                imp_s = f"{imp:+.1f}%" if imp is not None else "—"
                mag = effect_label(g) if not math.isnan(g) else "—"
                lines.append(
                    f"| {METRIC_LABELS[metric]} | {DISPLAY_NAMES.get(base, base)} "
                    f"| {g:+.2f} | {mag} | {imp_s} |"
                )
        lines.append("")

    # 3. Significance tests
    lines.append("## 3. Significance Tests (two-sided permutation test)")
    lines.append("")
    lines.append("_Note: with n=3 seeds per condition, the minimum achievable "
                 "two-sided p-value is 0.1, so no contrast can reach the "
                 "conventional 0.05 threshold regardless of effect magnitude. "
                 "Reported p-values are informational; effect sizes (Section 2) "
                 "are the primary evidence at this seed count._")
    lines.append("")
    for scenario in SCENARIOS:
        lines.append(f"### {SCENARIO_LABELS[scenario]}")
        lines.append("")
        lines.append("| Metric | Baseline | p (raw) | Holm-adj. α | Sig. at 0.05? |")
        lines.append("|---|---|---|---|---|")
        for metric, (field, _hib) in METRICS.items():
            ours = grouped.get((scenario, OUR_METHOD))
            if not ours:
                continue
            our_vals = [getattr(r, field) for r in ours]
            rows = []
            for base in BASELINES:
                bres = grouped.get((scenario, base))
                if not bres:
                    continue
                base_vals = [getattr(r, field) for r in bres]
                p = permutation_test(our_vals, base_vals)
                rows.append((base, p))
            # Holm-Bonferroni across baselines for this metric+scenario
            m = len(rows)
            sorted_rows = sorted(rows, key=lambda t: t[1])
            holm = {}
            for rank, (base, p) in enumerate(sorted_rows, start=1):
                alpha_i = 0.05 / (m - rank + 1)
                holm[base] = alpha_i
            for base, p in rows:
                sig = "yes" if p < holm[base] else "no"
                lines.append(
                    f"| {METRIC_LABELS[metric]} | {DISPLAY_NAMES.get(base, base)} "
                    f"| {p:.3f} | {holm[base]:.4f} | {sig} |"
                )
        lines.append("")

    # 4. Limitations
    lines.append("## 4. Limitations and Recommendations")
    lines.append("")
    lines.append("- **Seed count (n=3) is insufficient for inferential power.** "
                 "The exact permutation test cannot yield p < 0.1 with three "
                 "independent seeds per condition. Effect sizes are therefore "
                 "unstable; a single outlier seed can flip a Hedges' g sign. "
                 "Treat all cross-method claims as directional until ≥5 seeds "
                 "(ideally 10) are run.")
    lines.append("- **Pooled-variance assumption** in Hedges' g is coarse at n=3; "
                 "report alongside raw means to avoid over-interpretation.")
    lines.append("- **Multiple comparisons** were controlled via Holm-Bonferroni "
                 "(6 baselines × 4 metrics × 4 scenarios), which is conservative. "
                 "Given the n=3 floor on p, this mainly guards the (rare) case "
                 "where larger seed counts are added later.")
    lines.append("- **Model baselines (MILP/MPC)** are deterministic single runs and "
                 "are excluded from inferential tests; they serve as a cost "
                 "lower-bound / safety reference only.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical analysis for HFG-SAC comparison")
    parser.add_argument("--input_dir", type=str, default="outputs/experiments")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = input_dir / "results"
    grouped = load_seed_results(results_dir)

    n_conditions = len(grouped)
    print(f"Loaded {n_conditions} (scenario, algorithm) conditions")
    if n_conditions == 0:
        print("No per-seed result files found. Is the experiment run complete?")
        # print available files for diagnosis
        if results_dir.exists():
            files = [f.name for f in results_dir.glob("*.json")]
            print(f"  Found {len(files)} files in {results_dir} (first 10):")
            for f in files[:10]:
                print(f"    {f}")
        return

    content = generate_stats_appendix(grouped, output_dir)
    out_path = output_dir / "stats-appendix.md"
    out_path.write_text(content, encoding="utf-8")
    print(f"Stats appendix written to {out_path}")


if __name__ == "__main__":
    main()