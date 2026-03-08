# Lecture 14：Transformer

---

## 📌 概览

本讲介绍彻底改变深度学习格局的架构——Transformer：

1. **Transformer 架构** — 完全基于注意力的序列模型
2. **位置编码** — 注入序列位置信息
3. **Vision Transformer (ViT)** — 将 Transformer 应用于计算机视觉

---

## 第一部分：Transformer 架构（⭐⭐ 核心）

### 1.1 "Attention is All You Need"

Transformer 完全抛弃了 RNN 和 CNN，仅用注意力机制构建整个模型。

### 1.2 Transformer Block

```
输入 x
  → Multi-Head Self-Attention → + x (残差连接) → LayerNorm
  → Feed-Forward Network (MLP) → + (残差连接) → LayerNorm
  → 输出
```

每个 Transformer Block 包含：
1. **多头自注意力层**：全局信息交互
2. **前馈网络 (FFN)**：逐位置的非线性变换
3. **残差连接 + Layer Normalization**：稳定训练

### 1.3 前馈网络 (FFN)

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

- 先升维（通常 4 倍），再降回原维度
- 对每个位置独立应用（不混合位置信息）
- 提供非线性变换能力

### 1.4 Layer Normalization vs Batch Normalization

| 特性 | BatchNorm | LayerNorm |
| ---- | --------- | --------- |
| 归一化维度 | 跨 batch，每个特征独立 | 跨特征，每个样本独立 |
| 依赖 batch size | ✅ 是 | ❌ 否 |
| 适用场景 | CNN（固定输入大小） | Transformer/RNN（变长序列） |

**为什么 Transformer 用 LayerNorm？**
- 序列长度可变，不同样本的 batch 统计量不稳定
- LayerNorm 对每个样本独立归一化，不受 batch 影响

---

## 第二部分：位置编码 (Positional Encoding)（⭐ 重要）

### 2.1 为什么需要位置编码？

自注意力是**排列不变的** (permutation invariant)——打乱输入顺序，输出也只是相应打乱。但序列的顺序很重要（"猫吃鱼" ≠ "鱼吃猫"）。

### 2.2 正弦位置编码

$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

- $pos$：位置索引
- $i$：维度索引
- 不同频率的正弦/余弦函数编码不同粒度的位置信息

### 2.3 可学习位置编码

$$x_i' = x_i + \text{PE}[i]$$

其中 PE 是一个可训练的参数矩阵。ViT 和 BERT 常用此方式。

---

## 第三部分：Vision Transformer (ViT)（⭐⭐ 重要）

### 3.1 核心思想

> "An image is worth 16×16 words."

将图像拆分为若干个不重叠的 patch，每个 patch 类比 NLP 中的一个 token。

### 3.2 ViT 流程

```
输入图像 (3, 224, 224)
  → 拆分为 14×14 = 196 个 patch，每个 16×16×3
  → 线性投影每个 patch 为 D 维向量 → (196, D)
  → 加上 [CLS] token → (197, D)
  → 加上位置编码 → (197, D)
  → Transformer Encoder × L 层
  → 取 [CLS] token 的输出
  → MLP Head → 分类结果
```

### 3.3 Patch Embedding

```python
# 等价于 stride=patch_size 的卷积
patch_embed = nn.Conv2d(3, D, kernel_size=16, stride=16)
# 输入 (B, 3, 224, 224) → 输出 (B, D, 14, 14) → reshape → (B, 196, D)
```

### 3.4 [CLS] Token

一个可学习的特殊 token，拼接在 patch 序列前面。经过 Transformer 后，它聚合了全图信息，用于最终分类。

### 3.5 ViT vs CNN

| 特性 | CNN | ViT |
| ---- | --- | --- |
| 归纳偏置 | 局部性 + 平移不变性 | 几乎无（仅 patch 结构） |
| 全局感受野 | 需要堆叠多层 | 第一层就有 |
| 数据需求 | 中等即可 | 需要大量数据（或预训练） |
| 小数据性能 | 较好（归纳偏置帮助） | 较差（需要数据补偿） |
| 大数据性能 | 有上限 | 可持续提升 |

### 3.6 关于归纳偏置

CNN 的归纳偏置（局部性、平移不变性）是一种**先验知识**：
- 优点：小数据下也能学得不错
- 缺点：限制了模型的灵活性

ViT 几乎没有归纳偏置：
- 优点：大数据下更灵活、性能上限更高
- 缺点：小数据下泛化差，需要更多数据来"重新发现"这些规律

---

## 💡 重点总结

1. **Transformer = Self-Attention + FFN + Residual + LayerNorm**
2. **位置编码是必须的**：自注意力本身是位置无关的
3. **ViT 将图像视为 patch 序列**，用纯 Transformer 处理
4. **LayerNorm 替代 BatchNorm**：不依赖 batch size，适合变长输入
5. **ViT 在大数据集上超越 CNN**，但在小数据集上不如 CNN（缺少归纳偏置）
6. **[CLS] token** 充当全局信息聚合器，最终用于分类

