# AnglE 训练 Loss 预期值分析 (Loss Expectation Analysis)

本文档基于代码实现 (`aoe/loss.py`) 和超参数配置，对 AnglE 模型在预训练 (NLI) 和微调 (STS) 阶段的 Loss 预期值进行理论分析。由于官方并未直接提供 Loss 曲线数值，以下分析主要基于 InfoNCE Loss 的数学特性。

## 1. 理论基础

AnglE 的总 Loss 由两部分组成：
$$ L_{total} = w_{angle} \times L_{angle} + w_{cl} \times L_{cl} $$

其中 $L_{cl}$ (Contrastive Loss) 是标准的 InfoNCE Loss，但在实现上有一个关键缩放：
- 代码中 `in_batch_negative_loss` 使用了 `tau=20.0` 进行缩放。
- 这相当于传统 InfoNCE 中的温度系数 $T = 1/20 = 0.05$。
- 对于 Batch Size 为 $N$ 的随机初始化模型，InfoNCE Loss 的初始值约为 $\ln(N)$。

## 2. 预训练阶段 (Pre-training on NLI)

**配置：**
- `w_angle = 1.0`
- `w_cl = 30.0`
- `Batch Size (Per GPU) = 64` (假设使用 2x3090 配置)

**Loss 估算：**
1.  **Contrastive Loss ($L_{cl}$):**
    - 初始值 $\approx \ln(64) \approx 4.16$。
    - 加权后：$30.0 \times 4.16 \approx 124.8$。
2.  **Angle Loss ($L_{angle}$):**
    - 作为一个 Ranking Loss，其初始值通常较小，但在加权后相比 $L_{cl}$ 占比很低。
3.  **总 Loss ($L_{total}$):**
    - **初始预期值：** **120 ~ 130** 左右。
    - **收敛预期值：** 随着训练进行，模型对正负样本的区分能力增强，$L_{cl}$ 会显著下降。通常 InfoNCE Loss 难以降到 0（因为要保持 Uniformity），最终可能收敛在 $1.0 \sim 2.0$ (未加权) 左右。
    - **加权后收敛值：** $30 \times (1.0 \sim 2.0) + L_{angle} \approx$ **30 ~ 60**。


## 3. 微调阶段 (Fine-tuning on STS)

**配置：**
- `w_angle = 0.02`
- `w_cl = 1.0`
- `Batch Size (Per GPU) = 64`

**Loss 估算：**
1.  **Contrastive Loss ($L_{cl}$):**
    - 初始值 $\approx \ln(64) \approx 4.16$。
    - 加权后：$1.0 \times 4.16 \approx 4.16$。
2.  **Angle Loss ($L_{angle}$):**
    - 权重仅为 0.02，对总 Loss 贡献微乎其微。
3.  **总 Loss ($L_{total}$):**
    - **初始预期值：** **4.0 ~ 5.0** 左右。
    - **收敛预期值：** 由于微调数据量较少且使用了预训练模型，Loss 下降可能不如预训练剧烈，但最终应低于初始值。预期收敛在 **1.0 ~ 2.0** 之间。

## 4. 总结

| 阶段 | 关键权重 | 初始 Loss 预期 | 收敛 Loss 预期 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **NLI Pre-training** | `w_cl=30.0` | **~120+** | **~30 - 60** | Loss 数值较大是由于 `w_cl` 的高权重导致的，属正常现象。 |
| **STS Fine-tuning** | `w_cl=1.0` | **~4 - 5** | **~1 - 2** | Loss 数值回归正常范围。 |

**注意：**
- Loss 的绝对数值高度依赖于 Batch Size。如果你增加了 Batch Size（例如通过梯度累积或更多 GPU），初始 Loss 会变大（$\ln(N)$ 增大），这是正常的。
- 只要 Loss 曲线呈现下降趋势，且评测指标（Spearman Correlation）在上升，即说明训练正常。
