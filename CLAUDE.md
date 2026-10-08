# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**saferl** — Safe Reinforcement Learning for Microgrid Optimal Scheduling

**Primary target: a high-level SCI journal paper** (e.g., Applied Energy, IEEE Transactions on Smart Grid / Sustainable Energy / Power Systems), NOT a master's thesis. All experimental design, baseline fidelity, statistics, and writing must meet journal peer-review standards: faithful (non-strawman) baselines with equal tuning, multi-seed runs with inferential statistics and effect sizes, complete coverage of every claimed contribution, and reproducible artifacts.

Background: the work originates from a master's thesis at Xiamen University of Technology (厦门理工学院), School of Computer and Information Engineering. Advisor: Li Lin (李林, 高级工程师). Co-advisor: Du Shasha (杜莎莎, 高级工程师). The thesis is the seed material; the SCI journal paper is the current deliverable.

### Research Topic
Microgrid optimal dispatch using offline pre-training and safe online fine-tuning with deep reinforcement learning. Addresses two operating modes:
- **Grid-connected (并网)**: DEA-ITSAC — Data-Enhanced Imitation-Guided Transformer Soft Actor-Critic
- **Islanded (孤岛)**: AR-SAC — Absolute-safe & Robust Soft Actor-Critic

### Key Artifacts
- `终稿（杨爱惠）（5.16) .docx` — Original master's thesis manuscript (Chinese, ~27MB). Seed/background material, not the current target.
- `paper/draft/` — English journal manuscript in progress (current primary writing artifact).

### Current State
The repo is an active research workspace for the SCI paper: HFG-SRL framework with implemented algorithms, environments, experiments, and an English manuscript. The thesis docx remains for reference. Code and experiments must satisfy journal-level reproducibility and completeness.

---

## Research Domain Context

**Domain keywords**: microgrid (微电网), optimal dispatch (优化调度), deep reinforcement learning (DRL), offline pre-training (离线预训练), safe reinforcement learning (安全强化学习), soft actor-critic (SAC), imitation learning, diffusion models, transformer.

**Core challenges addressed in the thesis**:
1. Poor generalization of RL policies under extreme/long-tail operating conditions
2. Policy oscillation during offline-to-online transition
3. Hard safety constraint satisfaction during exploration in islanded mode

**Two proposed algorithms**:
1. **DEA-ITSAC**: Conditional diffusion model for expert trajectory augmentation → GAIL-based offline pre-training → Transformer SAC with online fine-tuning for grid-connected microgrids
2. **AR-SAC**: Model-based residual learning with virtual replay buffer → uncertainty-constrained decay function → safe exploration mechanism for islanded microgrids

---

## Working with This Repository

### Language
- **Conversation language**: Respond to the user in Chinese (简体中文). This overrides the global English-by-default setting for this repo.
- **Paper/writing language**: All academic output (papers, journal/conference submissions, abstracts, rebuttal letters) must be written in English. The master's thesis manuscript itself remains in Chinese.
- Technical terms remain in English (DRL, SAC, microgrid, etc.).

### Document Workflows
- The primary artifact is a `.docx` file. When editing or generating academic content, use the `document-skills:docx` skill or the `academic-research-skills` plugin.
- For paper/rebuttal/abstract tasks, use `/ars-*` slash commands from the academic-research-skills plugin.

### Future Code Work
If source code is added to this repo, it is expected to be a Python-based RL project using:
- PyTorch (model training)
- Gym/Gymnasium (environment interfaces)
- Likely custom microgrid environment
- Hydra + OmegaConf for config management (per user preferences)

When code is added, follow the global coding style rules:
- Factory & Registry patterns for modules
- Dataclass-based immutable configs
- Type hints and docstrings
- 200-400 line file sizes

### Academic Writing
This is an academic research project. When assisting with writing tasks:
- Target venue quality: high-level SCI journal (e.g., Applied Energy, IEEE Transactions on Smart Grid / Sustainable Energy / Power Systems). Every claimed contribution must have complete, statistically rigorous experimental support; partial or placeholder results are not acceptable.
- Use `ml-paper-writing` skill for paper drafting
- Use `/ars-lit-review` for literature review support
- Use `/ars-revision` for revision tasks
- Use `writing-anti-ai` for AI-style removal (bilingual Chinese/English)

### Obsidian Knowledge Base
Per global config, this research project should be integrated with Obsidian project memory. Use `/obsidian-init` to bootstrap the vault and `/obsidian-ingest` for note ingestion.

---

## Commands

No build/test/lint commands exist yet. This section should be updated when source code is added.

Common research workflow commands (from global config):
- `/research-init` — Start Zotero-integrated research ideation
- `/zotero-review` — Literature review from Zotero collection
- `/analyze-results` — Experiment result analysis
- `/rebuttal` — Generate rebuttal document
- `/ars-*` — Academic research skills plugin commands
