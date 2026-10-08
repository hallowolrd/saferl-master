# Bilingual Abstract

> **Version**: v1.0
> **Date**: 2026-09-23
> **Paper**: Hierarchical Fuzzy-Guided Safe Reinforcement Learning for Microgrid Optimal Dispatch with Constraint-Aware Cross-Scenario Transfer
> **Target Venue**: Applied Energy / IEEE Transactions on Smart Grid

---

## English Abstract

Safety is a critical bottleneck that prevents deep reinforcement learning (DRL) from being deployed in real-world microgrid energy management systems. Existing safe RL methods typically formulate constraints as crisp hard boundaries, which fails to capture the inherent fuzziness of engineering constraints—such as recommended versus absolute state-of-charge ranges, load priority gradations, and permissible short-term violations under emergency conditions. Furthermore, purely data-driven safe RL suffers from low sample efficiency and cannot guarantee safety during cross-scenario policy transfer. To address these gaps, we propose a Hierarchical Fuzzy-Guided Safe Reinforcement Learning (HFG-SRL) framework. First, we introduce the Fuzzy Constrained Markov Decision Process (FC-MDP), which generalizes standard CMDP by replacing hard constraints with continuous fuzzy constraint satisfaction degrees (FCSD), and derive a Fuzzy-Lagrangian method with provable convergence guarantees. Second, we develop the Hierarchical Fuzzy-Guided SAC (HFG-SAC) algorithm with two complementary fuzzy layers: an upper-layer fuzzy constraint protection mechanism that ensures safe exploration with smooth constraint boundaries, and a lower-layer fuzzy knowledge reward shaping engine that embeds expert operational rules for accelerated learning. Third, we design a constraint-aware cross-scenario transfer mechanism that safely adapts learned policies across microgrid operating modes and seasonal conditions with theoretical safety guarantees during transfer. Experimental results on a modified IEEE 33-bus microgrid system across four scenarios (grid-connected normal/extreme, islanded normal/extreme) with three random seeds show that HFG-SAC achieves zero hard-constraint violations on both grid-connected scenarios while reducing operating cost by 19–46% relative to conservative safe-RL baselines (PPO-Lagrangian, CPO, Safety Layer). On islanded normal operation, HFG-SAC reaches the cost–safety Pareto frontier (5.2% violation, ¥8,363/day). On the most challenging islanded extreme scenario, after adding a priority-ranked load-shedding valve and tightening the frequency shield, HFG-SAC reduces violations from ~99% (unconstrained SAC) or ~49% (Lagrangian/CPO) to 0.64%, meeting the 5% safety threshold at a cost of ¥24,004/day — a trade-off between safety and operating cost that reflects intentional load shedding during evening peaks.

**Keywords**: safe reinforcement learning, fuzzy logic, microgrid optimal dispatch, soft actor-critic, constrained Markov decision process, transfer learning

*Word count: ~285 words*

---

## 中文摘要

在高比例可再生能源接入的微电网系统中，深度强化学习方法在处理不确定性和高维决策方面展现出显著优势，但安全约束难以严格保证一直是制约其工程落地的核心瓶颈。现有安全强化学习方法通常将约束建模为精确的硬边界，忽略了实际工程中约束的内在模糊性——例如储能荷电状态存在"推荐范围"与"绝对边界"的层次差异、负荷优先级具有连续的等级划分、极端工况下允许可控的短期越限等。此外，纯数据驱动的安全强化学习存在样本效率低、跨场景迁移过程中安全性缺乏保障等问题。针对上述不足，本文提出一种分层模糊引导安全强化学习（HFG-SRL）框架。首先，建立模糊约束马尔可夫决策过程（FC-MDP），将标准CMDP中的硬约束推广为连续的模糊约束满足度（FCSD），并提出具有收敛性保证的模糊拉格朗日求解方法。其次，设计分层模糊引导SAC（HFG-SAC）算法，上层采用TSK模糊系统建模多约束联合满足度并实现平滑的约束保护机制，下层通过基于势能的模糊知识奖励塑形嵌入专家运行规则以加速学习，两层通过梯度协调机制实现安全性与经济性的动态平衡。最后，提出约束感知的跨场景迁移机制，基于模糊规则相似度度量场景可迁移性，通过渐进式网络适配和初始保守策略确保迁移过程中的安全。在改进IEEE 33节点微电网系统上，针对并网正常/极端、孤岛正常/极端四个场景、三种随机种子的实验表明：本文方法在两个并网场景下均实现0%硬约束违反，同时相比PPO-Lagrangian、CPO、安全层等保守安全RL基线降低运行成本19%–46%；在孤岛正常场景下达到成本-安全Pareto前沿（违反率5.21%，日成本8,363元）；在最具挑战性的孤岛极端场景下，通过增加优先级负荷切除阀并收紧频率保护阈值，将违反率从无约束SAC的约99%或Lagrangian/CPO的约49%降至0.64%，达到5%安全阈值（日成本24,004元），体现了极端工况下安全性与经济性的权衡。

**关键词**：安全强化学习；模糊逻辑；微电网优化调度；软演员-评论家；约束马尔可夫决策过程；迁移学习

*字数：约620字*

---

## Notes

- Quantitative results (6.3–9.8%, 51–73%, 2.4–3.7×, 68%) are placeholders based on typical SOTA improvement ranges. Update with actual experimental results.
- The abstract follows the standard structure: background → gap → method (3 contributions) → results → significance.
