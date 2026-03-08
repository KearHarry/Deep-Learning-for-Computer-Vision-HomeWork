# Lecture 11：CNN 架构（二）

---

## 📌 概览

本讲延续 Lecture 8 的 CNN 架构演进，介绍更现代的网络设计：

1. **ResNet 深入分析** — 残差连接的本质
2. **ResNeXt** — 分组卷积 + 残差
3. **Squeeze-and-Excitation (SE)** — 通道注意力
4. **MobileNet / ShuffleNet** — 轻量化网络
5. **Neural Architecture Search (NAS)** — 自动搜索架构

---

## 第一部分：残差网络深入分析

### 1.1 ResNet 的残差块回顾

$$y = F(x) + x$$

网络只需学习残差 $F(x) = y - x$，而不是完整映射 $H(x) = y$。

### 1.2 恒等映射的深层含义（⭐ 重要）

如果某一层不需要做任何变换（恒等映射），学习 $F(x) = 0$ 比学习 $H(x) = x$ 容易得多。

**梯度传播视角**：
- 没有残差连接时：梯度需要经过每一层的变换，容易消失
- 有残差连接时：梯度可以通过 skip connection **直接流到前面的层**

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$

其中 "1" 就是跳跃连接的贡献——即使 $\frac{\partial F}{\partial x}$ 很小，梯度也不会消失。

### 1.3 Bottleneck 残差块

```
输入 (256-d)
  → 1×1 Conv (降维到 64)
  → 3×3 Conv (64-d)
  → 1×1 Conv (升维回 256)
  + 输入 (skip connection)
  → 输出 (256-d)
```

- 1×1 conv 先降维再升维，大幅减少 3×3 conv 的计算量
- ResNet-50+ 使用 Bottleneck，ResNet-18/34 使用 Basic Block

### 1.4 Pre-activation ResNet

将 BN 和 ReLU 放在卷积**之前**而非之后：

```
标准：  x → Conv → BN → ReLU → Conv → BN → + → ReLU
预激活：x → BN → ReLU → Conv → BN → ReLU → Conv → +
```

预激活版本在极深网络中表现更好，因为跳跃连接传递的是"干净"的信号。

---

## 第二部分：其他现代架构

### 2.1 ResNeXt：分组卷积

将一个宽的卷积拆分为多条并行的窄路径（cardinality），最后求和：

$$y = \sum_{i=1}^{C} T_i(x) + x$$

- 增加 cardinality（路径数）比增加宽度或深度更有效
- 与 Inception 的多分支类似，但每条路径结构相同

### 2.2 Squeeze-and-Excitation (SE) 模块

```
输入 (C, H, W)
  → Global Avg Pool → (C, 1, 1)           # Squeeze
  → FC → ReLU → FC → Sigmoid → (C, 1, 1)  # Excitation
  → 逐通道乘回输入                           # Scale
```

**核心思想**：不是所有通道都同等重要，让网络自动学习通道之间的权重关系。

### 2.3 轻量化网络

#### Depthwise Separable Convolution（⭐ MobileNet 核心）

标准卷积：$C_{in} \times K \times K \times C_{out}$ 参数

拆分为两步：
1. **Depthwise Conv**：每个通道独立卷积，$C_{in} \times K \times K$ 参数
2. **Pointwise Conv**：1×1 conv 混合通道，$C_{in} \times C_{out}$ 参数

计算量减少到约 $\frac{1}{C_{out}} + \frac{1}{K^2}$，通常为原来的 $\frac{1}{8}$ ~ $\frac{1}{9}$。

### 2.4 Neural Architecture Search (NAS)

用算法（强化学习、进化算法等）自动搜索最优的网络架构：
- **搜索空间**：可选的层类型、通道数、连接方式
- **搜索策略**：RL、进化、可微分搜索 (DARTS)
- **评估方法**：训练候选架构并评估准确率

代表性成果：EfficientNet — 同时优化深度、宽度和分辨率的缩放系数。

---

## 💡 重点总结

1. **残差连接**本质上提供了梯度的"高速公路"，让梯度能跳过中间层直接传播
2. **Bottleneck 设计**用 1×1 conv 降维来减少计算量，是深层 ResNet 的标准配置
3. **分组卷积** (ResNeXt) 和 **通道注意力** (SE) 是两种正交的改进方向
4. **Depthwise Separable Convolution** 将计算量减少约 8-9 倍，是移动端网络的核心
5. **NAS** 代表了从人工设计到自动搜索架构的范式转变

