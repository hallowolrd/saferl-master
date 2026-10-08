"""
Main training script for HFG-SRL.

Usage:
    python run/train.py --env grid_connected --algorithm hfg_sac --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hfg_srl.env import make_env
from hfg_srl.algorithms import make_agent
from hfg_srl.buffers import ReplayBuffer
from hfg_srl.utils.config import HFGConfig
from hfg_srl.utils.seed import set_seed, get_device
from hfg_srl.utils.logger import get_logger
from hfg_srl.utils.metrics import EpisodeMetrics, EvalResults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HFG-SRL Training")
    parser.add_argument("--env", type=str, default="grid_connected",
                        help="Environment name")
    parser.add_argument("--algorithm", type=str, default="hfg_sac",
                        help="Algorithm name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--experiment", type=str, default="default",
                        help="Experiment name")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps")
    return parser.parse_args()


def evaluate(
    agent,
    env,
    num_episodes: int = 10,
    max_steps_per_episode: int = 96 * 7,
) -> EvalResults:
    """Evaluate agent over multiple episodes."""
    results = EvalResults()
    agent.eval()

    for _ in range(num_episodes):
        obs, info = env.reset()
        metrics = EpisodeMetrics()
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            action = agent.select_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)

            metrics.total_reward += reward
            metrics.num_steps += 1
            metrics.total_cost += info.get("cost", 0.0)
            metrics.total_constraint_violation += info.get("constraint_violation", 0.0)
            if info.get("violated", False):
                metrics.constraint_violation_count += 1
            metrics.mean_fcsd += info.get("fcsd", 1.0)
            metrics.min_fcsd = min(metrics.min_fcsd, info.get("fcsd", 1.0))

            obs = next_obs
            done = terminated or truncated
            step += 1

        if metrics.num_steps > 0:
            metrics.mean_fcsd /= metrics.num_steps
        results.add_episode(metrics)

    agent.train()
    return results


def main() -> None:
    args = parse_args()

    # Config
    cfg = HFGConfig(seed=args.seed, device=get_device(args.device))

    # Output dir
    exp_dir = Path(args.output_dir) / args.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir / "logs"
    ckpt_dir = exp_dir / "checkpoints"

    # Logger
    logger = get_logger("train", str(log_dir))
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {cfg.device}")

    # Seed
    set_seed(cfg.seed)

    # Environment
    env = make_env(args.env, cfg.env)
    env.seed(cfg.seed)

    obs_dim = env.observation_space_shape[0]
    act_dim = env.action_space_shape[0]
    action_high = env.action_high

    logger.info(f"Observation dim: {obs_dim}, Action dim: {act_dim}")

    # Agent
    agent = make_agent(
        args.algorithm, cfg, obs_dim, act_dim, action_high, cfg.device
    )
    agent.train()

    # Replay buffer
    buffer = ReplayBuffer(
        obs_dim, act_dim, cfg.algorithm.buffer_size, cfg.device
    )

    # Training loop
    max_steps = args.max_steps or cfg.algorithm.max_steps
    warmup_steps = cfg.algorithm.warmup_steps

    total_steps = 0
    episode = 0
    best_eval_reward = -float("inf")

    logger.info(f"Starting training for {max_steps} steps...")
    logger.info(f"Warmup steps: {warmup_steps}")
    start_time = time.time()

    while total_steps < max_steps:
        episode += 1
        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done and total_steps < max_steps:
            # Action selection
            if total_steps < warmup_steps:
                # Random action during warmup
                action = np.random.uniform(
                    low=env.action_low, high=env.action_high, size=act_dim
                ).astype(np.float32)
            else:
                action = agent.select_action(obs, deterministic=False)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                fcsd=info.get("fcsd", 1.0),
                constraint_cost=info.get("constraint_violation", 0.0),
            )

            # Update
            if total_steps >= warmup_steps:
                for _ in range(cfg.algorithm.updates_per_step):
                    batch = buffer.sample(cfg.algorithm.batch_size)
                    update_info = agent.update(batch, total_steps)

            # Logging
            total_steps += 1
            episode_steps += 1
            episode_reward += reward
            obs = next_obs

            if total_steps % cfg.algorithm.log_interval == 0 and total_steps > warmup_steps:
                elapsed = time.time() - start_time
                fps = total_steps / elapsed if elapsed > 0 else 0
                log_msg = (
                    f"Step {total_steps}/{max_steps} | "
                    f"Episode {episode} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"FPS: {fps:.1f}"
                )
                if "fcsd_batch_mean" in update_info:
                    log_msg += f" | FCSD: {update_info['fcsd_batch_mean']:.3f}"
                if "lambda_val" in update_info:
                    log_msg += f" | λ: {update_info['lambda_val']:.3f}"
                logger.info(log_msg)

            # Evaluation
            if total_steps % cfg.algorithm.eval_interval == 0 and total_steps > warmup_steps:
                logger.info(f"--- Evaluation at step {total_steps} ---")
                eval_results = evaluate(
                    agent, env,
                    num_episodes=cfg.algorithm.eval_episodes,
                )
                summary = eval_results.summary()
                logger.info(
                    f"Eval Reward: {summary['reward_mean']:.2f} ± {summary['reward_std']:.2f}"
                )
                logger.info(
                    f"Eval Cost: {summary['cost_mean']:.2f} ± {summary['cost_std']:.2f}"
                )
                logger.info(
                    f"Eval FCSD: {summary['fcsd_mean']:.4f} | "
                    f"Violation Rate: {summary['violation_rate']:.4f}"
                )

                # Save best model
                if summary["reward_mean"] > best_eval_reward:
                    best_eval_reward = summary["reward_mean"]
                    best_dir = ckpt_dir / "best"
                    agent.save(str(best_dir))
                    logger.info(f"New best model saved (reward: {best_eval_reward:.2f})")

                # Save latest
                latest_dir = ckpt_dir / "latest"
                agent.save(str(latest_dir))

        # Episode end
        logger.info(
            f"Episode {episode} finished: {episode_steps} steps, "
            f"reward={episode_reward:.2f}"
        )

    # Final save
    final_dir = ckpt_dir / "final"
    agent.save(str(final_dir))
    logger.info(f"Training complete. Final model saved to {final_dir}")

    # Save config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time / 3600:.2f} hours")


if __name__ == "__main__":
    main()
