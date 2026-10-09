"""Rerun S3_island_normal for all 7 algorithms x 5 seeds with corrected config."""
import sys, os, json, time
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from run.experiment_runner import train_agent, SCENARIOS

SEEDS = [42, 123, 456, 789, 2024]
ALGOS = ["sac", "ppo", "ppo_lagrangian", "cpo", "safety_layer", "fuzzy_sac", "hfg_sac"]
SCEN = "S3_island_normal"
SCEN_CFG = SCENARIOS[SCEN]
OUTPUT_DIR = Path("outputs/experiments")
N_EPISODES = 200

def run_one(job):
    algo, seed = job
    os.environ["HFG_NUM_THREADS"] = "3"
    import torch
    torch.set_num_threads(3)
    t0 = time.time()
    r = train_agent(algo, SCEN, SCEN_CFG, seed,
                     n_episodes=N_EPISODES, device="cpu", output_dir=OUTPUT_DIR)
    elapsed = time.time() - t0
    return algo, seed, r.cost_per_day, r.violation_rate, r.avg_fcsd, elapsed

if __name__ == "__main__":
    jobs = [(a, s) for a in ALGOS for s in SEEDS]
    print(f"Rerunning S3: {len(jobs)} jobs (7 algos x 5 seeds)")
    print(f"Config: interruptible=600kW, shed_cost=1.0, shield=0.15")
    print()

    done = 0
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futures):
            algo, seed, cost, viol, fcsd, elapsed = fut.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {algo:18s} s{seed:5d}  "
                  f"cost={cost:>10,.0f}  viol={viol:>5.1f}%  fcsd={fcsd:.4f}  "
                  f"({elapsed:.0f}s)", flush=True)
    print(f"\nDone. {done} jobs completed.")
