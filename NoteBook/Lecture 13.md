# Lecture 13：注意力机制 (Attention)

---

## 📌 概览

本讲引入深度学习最重要的机制之一——注意力：

1. **Seq2Seq 与注意力** — 从翻译任务理解注意力的动机
2. **注意力的通用形式** — Query、Key、Value
3. **自注意力 (Self-Attention)** — 序列内部的交互

---

## 第一部分：Sequence-to-Sequence 与注意力

### 1.1 Seq2Seq 模型（无注意力）

```
编码器 (Encoder): x1, x2, ..., xT → 最终隐藏状态 c
解码器 (Decoder): c → y1, y2, ..., yT'
```

**瓶颈问题**：所有输入信息被压缩到一个固定长度的向量 $c$ 中。对于长序列，$c$ 无法保留所有信息。

### 1.2 注意力的动机（⭐ 关键直觉）

> 解码时，不同的输出词应该"关注"输入序列的不同部分。

比如翻译"I love cats"为"我喜欢猫"时：
- 生成"猫"时，应该重点关注输入中的"cats"
- 生成"喜欢"时，应该重点关注"love"

### 1.3 注意力机制（Bahdanau Attention）

```python
# 在解码器每个时间步 t：
# 1. 计算注意力分数：当前解码状态与每个编码状态的相关性
e_t = [score(s_t, h_i) for i in range(T)]  # T 个分数

# 2. 归一化为注意力权重
a_t = softmax(e_t)  # 和为 1 的权重分布

# 3. 加权求和得到上下文向量
context = sum(a_t[i] * h_i for i in range(T))

# 4. 用上下文向量辅助生成当前输出
output = decode(s_t, context)
```

---

## 第二部分：注意力的通用形式（⭐⭐ 核心）

### 2.1 Query-Key-Value 框架

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- **Query (Q)**：我在找什么？（查询向量）
- **Key (K)**：我有什么可以被查找的？（键向量）
- **Value (V)**：实际要取出的信息（值向量）

**流程**：
1. Query 与所有 Key 计算相似度（点积）
2. 除以 $\sqrt{d_k}$ 防止点积过大（数值稳定性）
3. Softmax 归一化为概率分布
4. 用概率加权 Value，得到注意力输出

### 2.2 不同的注意力评分函数

| 方法 | 公式 | 说明 |
| ---- | ---- | ---- |
| 点积 (Dot) | $e = q^T k$ | 最简单，要求维度相同 |
| 缩放点积 (Scaled Dot) | $e = q^T k / \sqrt{d}$ | 防止 softmax 饱和 |
| 加法 (Additive) | $e = w^T \tanh(W_1 q + W_2 k)$ | 可处理不同维度 |

### 2.3 为什么除以 $\sqrt{d_k}$？

当 $d_k$ 很大时，$q^T k$ 的方差约为 $d_k$，值会很大。Softmax 对大值非常敏感——输出几乎是 one-hot，梯度接近零。除以 $\sqrt{d_k}$ 让方差回到 1，保持 Softmax 的有效梯度。

---

## 第三部分：自注意力 (Self-Attention)（⭐⭐ Transformer 的基石）

### 3.1 核心思想

在一个序列内部，每个元素都与其他所有元素计算注意力。Q、K、V 都来自**同一个输入序列**。

```python
# 输入 X: (N, D) — N 个 token，每个 D 维
Q = X @ W_Q  # (N, d_k)
K = X @ W_K  # (N, d_k)
V = X @ W_V  # (N, d_v)

# 注意力
A = softmax(Q @ K.T / sqrt(d_k))  # (N, N) 注意力矩阵
output = A @ V                      # (N, d_v)
```

### 3.2 自注意力 vs 卷积 vs 全连接

| 特性 | 自注意力 | 卷积 | 全连接 |
| ---- | -------- | ---- | ------ |
| 感受野 | 全局（所有位置） | 局部（核大小） | 全局 |
| 参数量 | 与序列长度无关 | 与序列长度无关 | 与序列长度相关 |
| 位置关系 | 需额外编码 | 内置局部性 | 无 |
| 计算复杂度 | $O(N^2 D)$ | $O(NKD)$ | $O(N^2 D)$ |

### 3.3 多头注意力 (Multi-Head Attention)

```python
# 将 Q, K, V 分成 h 个头
Q_i = Q @ W_Q_i  # 每个头有独立的投影矩阵
K_i = K @ W_K_i
V_i = V @ W_V_i

head_i = Attention(Q_i, K_i, V_i)  # 每个头独立计算注意力

# 合并所有头
output = Concat(head_1, ..., head_h) @ W_O
```

**为什么多头？**
- 单头注意力只能学习一种"关注模式"
- 多头让模型同时关注不同位置的不同类型信息
- 类似于 CNN 中多个滤波器捕捉不同的特征

---

## 💡 重点总结

1. **注意力的本质**：动态地、有选择性地聚焦于输入的不同部分
2. **QKV 框架**：Query 查找与 Key 最匹配的项，用匹配度加权 Value
3. **缩放因子 $\sqrt{d_k}$**：防止点积过大导致 Softmax 饱和
4. **自注意力**：序列内部的全局交互，是 Transformer 的核心组件
5. **多头注意力**：多种注意力模式并行，捕捉多样化的关系

