# HFG-SRL: Hierarchical Fuzzy-Guided Safe Reinforcement Learning

Safe reinforcement learning for microgrid optimal dispatch, with hierarchical
fuzzy guidance combining fuzzy constraint protection and fuzzy knowledge-based
reward shaping.

## Overview

This project implements a hierarchical fuzzy-guided safe reinforcement learning
(HFG-SRL) framework for microgrid optimal dispatch, addressing:

1. **Fuzzy constraint satisfaction** — Many real-world safety constraints are
   not crisp 0/1 boundaries but have gradual transition zones. We propose a
   Fuzzy Constrained MDP (FC-MDP) formulation with Fuzzy Constraint Satisfaction
   Degree (FCSD) metrics.

2. **Expert knowledge embedding** — Operational heuristics from domain experts
   are encoded as fuzzy rules to shape the reward signal, accelerating learning
   and improving policy interpretability.

3. **Cross-scenario safe transfer** — Constraint-aware transfer learning enables
   safe adaptation across microgrid operating scenarios (grid-connected/islanded,
   seasonal changes, topology variations).

## Architecture

```
HFG-SRL
├── env/               # Microgrid environments (grid-connected, islanded)
├── fuzzy_system/      # TSK fuzzy systems, rule bases, reward shaping
├── algorithms/        # RL algorithms (HFG-SAC, Safe SAC baseline)
├── transfer/          # Cross-scenario transfer learning
├── models/            # Neural network building blocks
├── buffers/           # Experience replay
└── utils/             # Config, logging, metrics
```

## Installation

```bash
# With uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

## Quick Start

```bash
# Train HFG-SAC on grid-connected microgrid
python run/train.py --env grid_connected --algorithm hfg_sac --seed 42

# Train baseline Safe SAC
python run/train.py --env islanded --algorithm safe_sac --seed 42

# Run tests
pytest tests/ -v
```

## Key Algorithms

| Algorithm | Description |
|-----------|-------------|
| `hfg_sac` | Hierarchical Fuzzy-Guided SAC (proposed) |
| `safe_sac` | Safe SAC with standard Lagrangian (baseline) |

## Environments

| Environment | Mode | Obs Dim | Act Dim |
|-------------|------|---------|---------|
| `grid_connected` | Grid-connected | 8 | 3 |
| `islanded` | Islanded | 9 | 4 |

## Project Structure

```
saferl/
├── src/hfg_srl/        # Core package
├── run/                # Entry scripts and configs
│   ├── train.py        # Training entry point
│   └── conf/           # Hydra configuration files
├── tests/              # Unit tests
├── data/               # Dataset directory
├── outputs/            # Training outputs (logs, checkpoints, results)
└── plan/               # Research plans and documentation
```

## Research Context

This is a research project on safe reinforcement learning applied to microgrid
optimal dispatch. For detailed research plans, see `plan/research-proposal.md`.

## License

MIT
