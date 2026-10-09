"""S3 boosted with S4's safety resources — 1-seed smoke test.

Tests: if S3 gets 600 kW interruptible load + 1.0 ¥/kWh shed cost + 0.15 shield,
do violation, cost, and FCSD improve?
"""
import sys, os
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from run.experiment_runner import train_agent

S3_BASE = {
    "env_mode": "islanded",
    "scenario_name": "typical_week_summer",
    "extreme_multiplier": None,
    "freq_penalty_slope": 1500.0,
    "freq_volt_shield_safety_ratio": 0.13,
    "interruptible_load_kw": 200.0,
    "load_shed_cost": 2.0,
}

S3_BOOSTED = {
    "env_mode": "islanded",
    "scenario_name": "typical_week_summer",
    "extreme_multiplier": None,
    "freq_penalty_slope": 1500.0,
    "freq_volt_shield_safety_ratio": 0.15,
    "interruptible_load_kw": 600.0,
    "load_shed_cost": 1.0,
}

SEED = 42

print("=" * 60)
print(f"Running S3 BASE (interruptible=200kW, shed_cost=2.0) seed={SEED}")
print("=" * 60)
r_base = train_agent(
    "hfg_sac", "S3_base_smoke", S3_BASE, seed=SEED,
    n_episodes=200, device="cpu", output_dir=None,
)

print("\n" + "=" * 60)
print(f"Running S3 BOOSTED (interruptible=600kW, shed_cost=1.0) seed={SEED}")
print("=" * 60)
r_boost = train_agent(
    "hfg_sac", "S3_boosted_smoke", S3_BOOSTED, seed=SEED,
    n_episodes=200, device="cpu", output_dir=None,
)

print("\n" + "=" * 60)
print("COMPARISON (same seed=42, 200 episodes)")
print("=" * 60)
print(f"{'Metric':<25} {'S3 base':>15} {'S3 boosted':>15} {'Delta':>15}")
print("-" * 70)
for label, key in [
    ("Cost (¥/day)", "cost_per_day"),
    ("Violation (%)", "violation_rate"),
    ("Avg FCSD", "avg_fcsd"),
    ("Min FCSD", "min_fcsd"),
    ("Training time (s)", "training_time_s"),
]:
    b = getattr(r_base, key)
    p = getattr(r_boost, key)
    d = p - b
    print(f"{label:<25} {b:>15.2f} {p:>15.2f} {d:>+15.2f}")
