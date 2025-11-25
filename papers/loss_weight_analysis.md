# AoE 预训练阶段 Loss 权重分析

本文档总结了 AoE (Angle-optimized Embeddings) 预训练阶段 (Stage 1) 中特定 Loss 权重配置背后的原理。

## 配置 (Configuration)

在 Stage 1 (NLI 预训练) 中，官方配置如下：
- **Angle Loss 权重 (`w_angle`)**: 1.0
- **Contrastive Loss 权重 (`w_cl`)**: 30.0

## 为什么 Contrastive Loss 权重高达 30.0？

在预训练阶段，Contrastive Loss (InfoNCE) 的权重是 Angle Loss 的 30 倍。原因如下：

1.  **强监督信号 (Strong Supervision Signal)**：Stage 1 的主要目标是建立一个稳健的语义空间。对比学习（如 SimCSE）已被证明是实现这一目标的有效方法。给予极高的权重确保模型优先学习全局的语义对齐，然后再微调角度细节。
2.  **稳定性 (Stability)**：角度优化可能比较敏感。使用强大的对比损失作为锚点，可以防止模型在训练初期发散。
3.  **两阶段策略 (Two-Stage Strategy)**：
    *   **Stage 1**：通过强对比学习 (`w_cl=30`) 进行“粗略”对齐。
    *   **Stage 2**：通过 Angle 优化 (`w_cl=1`, `w_angle` 相对提升) 进行“精细”微调。

## 为什么选择 1.0/30.0 而不是 0.03/1.0？

虽然 `1:30` 的比例与 `0.03:1` 相同，但绝对数值对优化动力学有重要影响。

1.  **梯度幅度与学习率匹配 (Gradient Magnitude & Learning Rate Matching)**：
    *   `aoe/loss.py` 中的两个损失函数都使用了相同的缩放因子 (`tau=20`)，因此它们的内在量级是相当的（通常在 2.0 - 5.0 之间）。
    *   使用 `1.0` 和 `30.0` 的权重会导致总 Loss 幅度大约在 100+。
    *   这种幅度的 Loss 产生的梯度非常适合标准 BERT 微调的学习率（例如 `2e-5` 到 `5e-5`）。

2.  **避免非标准超参数 (Avoiding Non-Standard Hyperparameters)**：
    *   如果使用 `0.03` 和 `1.0`，总 Loss 会非常小（约 3.0）。
    *   为了达到同样的参数更新步长 ($\Delta w = \eta \cdot \nabla L$)，我们需要将学习率放大约 30 倍（例如 `1.5e-3`）。
    *   `1.5e-3` 对于 Transformer 微调来说是一个异常高的学习率，可能会导致不稳定或需要重新调整优化器调度。
    *   使用 `1/30` 的组合让我们能够直接沿用 BERT 训练的“标准配方”。
