"""Diagnose S3: why is violation 5.2% while S4 is 0.6%?"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"src"))

from run.experiment_runner import SCENARIOS, create_env_config
from hfg_srl.env.base_env import make_env
from hfg_srl.algorithms import make_agent
from hfg_srl.utils.config import AlgorithmConfig, FuzzyConfig, HFGConfig
from hfg_srl.utils.seed import set_seed


def build(scen, seed=123):
    set_seed(seed)
    cfg = SCENARIOS[scen]
    env_cfg, dl = create_env_config(cfg["env_mode"], cfg, seed=seed, num_days=7)
    algo = AlgorithmConfig(name="hfg_sac", hidden_dims=[128,128], batch_size=256,
                          buffer_size=int(1e5), warmup_steps=100, eval_interval=1000,
                          eval_episodes=1, log_interval=1000)
    shield = cfg.get("freq_volt_shield_safety_ratio", 0.175)
    fuzz = FuzzyConfig(num_rules=16, num_inputs=9,
                       fcsd_target=0.75 if scen=="S4_island_extreme" else 0.85,
                       use_expert_rules=True, reward_shaping_weight_init=0.0,
                       reward_shaping_weight_final=0.0,
                       lambda_init=2.0, lambda_lr=0.05, lambda_max=50.0,
                       freq_volt_shield_safety_ratio=shield)
    hfg = HFGConfig(env=env_cfg, algorithm=algo, fuzzy=fuzz, seed=seed, device="cpu")
    env = make_env(cfg["env_mode"], env_cfg, data_loader=dl); env.seed(seed)
    ag = make_agent("hfg_sac", hfg, env.observation_space_shape[0],
                    env.action_space_shape[0], env.action_high, "cpu")
    ag.eval()
    return ag, env, env_cfg


def rollout(ag, env, episodes=5):
    stats = {"steps":0,"freq":[],"volt":[],"soc":[],"de":[],"shed":[],"ess":[],"imb":[],
             "freq_hard":0,"volt_hard":0,"soc_hi":0,"soc_lo":0}
    for ep in range(episodes):
        obs,_ = env.reset(seed=200+ep); done=False
        while not done:
            a = ag.select_action(obs, deterministic=True)
            obs,r,term,trunc,info = env.step(a); done=term or trunc
            stats["steps"] += 1
            stats["freq"].append(info["freq_deviation"])
            stats["volt"].append(info["voltage_deviation"])
            stats["soc"].append(info["soc"])
            stats["de"].append(getattr(env,"de_power",0))
            stats["shed"].append(info["load_shed"])
            stats["ess"].append(getattr(env,"ess_power",0))
            stats["imb"].append(info.get("imbalance_ratio",0))
            if info["freq_deviation"]>0.5: stats["freq_hard"]+=1
            if info["voltage_deviation"]>0.05: stats["volt_hard"]+=1
            if info["soc"]>0.9: stats["soc_hi"]+=1
            if info["soc"]<0.1: stats["soc_lo"]+=1
    return stats


def main():
    for scen in ["S3_island_normal","S4_island_extreme"]:
        ag, env, ecfg = build(scen)
        ckpt = ROOT/f"outputs/experiments/checkpoints/hfg_sac_{scen}_s123/hfg_sac.pt"
        if ckpt.exists():
            c = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            ag.actor.load_state_dict(c["actor"])
            if "log_alpha" in c: ag.log_alpha.data = c["log_alpha"].data
        s = rollout(ag, env)
        n = s["steps"]
        print(f"\n=== {scen} (interruptible={ecfg.interruptible_load_kw:.0f}kW, "
              f"slope={ecfg.freq_penalty_slope:.0f}) ===")
        print(f"  hard freq viol: {s['freq_hard']}/{n} = {100*s['freq_hard']/n:.1f}%")
        print(f"  hard volt viol: {s['volt_hard']}/{n} = {100*s['volt_hard']/n:.1f}%")
        print(f"  SOC hi/lo:      {s['soc_hi']}/{s['soc_lo']}")
        f = np.array(s["freq"]); d = np.array(s["de"]); sh = np.array(s["shed"])
        print(f"  freq_dev mean/max: {f.mean():.3f}/{f.max():.3f} Hz")
        print(f"  DE mean/max:  {d.mean():.0f}/{d.max():.0f} kW (rated 400)")
        print(f"  Shed mean/max: {sh.mean():.0f}/{sh.max():.0f} kW")


if __name__=="__main__":
    main()
