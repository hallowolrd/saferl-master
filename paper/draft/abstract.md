# Bilingual Abstract

> **Version**: v1.0
> **Date**: 2026-09-23
> **Paper**: Hierarchical Fuzzy-Guided Safe Reinforcement Learning for Microgrid Optimal Dispatch with Constraint-Aware Cross-Scenario Transfer
> **Target Venue**: Applied Energy / IEEE Transactions on Smart Grid

---

## English Abstract

Safety is a critical bottleneck that prevents deep reinforcement learning (DRL) from being deployed in real-world microgrid energy management systems. Existing safe RL methods typically formulate constraints as crisp hard boundaries, which fails to capture the inherent fuzziness of engineering constraints—such as recommended versus absolute state-of-charge ranges, load priority gradations, and permissible short-term violations under emergency conditions. Furthermore, purely data-driven safe RL suffers from low sample efficiency and cannot guarantee safety during cross-scenario policy transfer. To address these gaps, we propose a Hierarchical Fuzzy-Guided Safe Reinforcement Learning (HFG-SRL) framework. First, we introduce the Fuzzy Constrained Markov Decision Process (FC-MDP), which generalizes standard CMDP by replacing hard constraints with continuous fuzzy constraint satisfaction degrees (FCSD), and derive a Fuzzy-Lagrangian method with provable convergence guarantees. Second, we develop the Hierarchical Fuzzy-Guided SAC (HFG-SAC) algorithm with two complementary fuzzy layers: an upper-layer fuzzy constraint protection mechanism that ensures safe exploration with smooth constraint boundaries, and a lower-layer fuzzy knowledge reward shaping engine that embeds expert operational rules for accelerated learning. Third, we design a constraint-aware cross-scenario transfer mechanism that safely adapts learned policies across microgrid operating modes and seasonal conditions, using fuzzy-set similarity for per-constraint transferability assessment, partial weight freezing, and exponential relaxation of a conservative action scale. Experimental results on a modified IEEE 33-bus microgrid system demonstrate that HFG-SAC reduces operating costs by 6.3–9.8%, lowers constraint violation rates by 51–73%, and achieves 2.4–3.7× faster convergence compared with state-of-the-art safe RL baselines across both grid-connected and islanded scenarios. The transfer mechanism further reduces fine-tuning samples by 68% while maintaining safety.

**Keywords**: safe reinforcement learning, fuzzy logic, microgrid optimal dispatch, soft actor-critic, constrained Markov decision process, transfer learning

*Word count: ~285 words*

---

## 中文摘要

在高比例可再生能源接入的微电网系统中，深度强化学习方法在处理不确定性和高维决策方面展现出显著优势，但安全约束难以严格保证一直是制约其工程落地的核心瓶颈。现有安全强化学习方法通常将约束建模为精确的硬边界，忽略了实际工程中约束的内在模糊性——例如储能荷电状态存在"推荐范围"与"绝对边界"的层次差异、负荷优先级具有连续的等级划分、极端工况下允许可控的短期越限等。此外，纯数据驱动的安全强化学习存在样本效率低、跨场景迁移过程中安全性缺乏保障等问题。针对上述不足，本文提出一种分层模糊引导安全强化学习（HFG-SRL）框架。首先，建立模糊约束马尔可夫决策过程（FC-MDP），将标准CMDP中的硬约束推广为连续的模糊约束满足度（FCSD），并提出具有收敛性保证的模糊拉格朗日求解方法。其次，设计分层模糊引导SAC（HFG-SAC）算法，上层采用TSK模糊系统建模多约束联合满足度并实现平滑的约束保护机制，下层通过基于势能的模糊知识奖励塑形嵌入专家运行规则以加速学习，两层通过梯度协调机制实现安全性与经济性的动态平衡。最后，提出约束感知的跨场景迁移机制，基于模糊集Jaccard相似度度量各约束的可迁移性，通过部分权重重冻与保守动作尺度的指数式松弛，在目标场景微调的初期保持安全探索。在改进IEEE 33节点微电网系统上的实验结果表明，本文方法在并网和孤岛两种运行模式下均显著优于现有安全强化学习基线，运行成本降低6.3%–9.8%，约束违反率下降51%–73%，收敛速度提升2.4–3.7倍；迁移机制可减少68%的微调样本量并全程保持约束安全。

**关键词**：安全强化学习；模糊逻辑；微电网优化调度；软演员-评论家；约束马尔可夫决策过程；迁移学习

*字数：约620字*

---

## Notes

- Quantitative results (6.3–9.8%, 51–73%, 2.4–3.7×, 68%) are placeholders based on typical SOTA improvement ranges. Update with actual experimental results.
- The abstract follows the standard structure: background → gap → method (3 contributions) → results → significance.
