# Lecture 12：循环神经网络 (RNN)

---

## 📌 概览

本讲引入处理序列数据的核心架构——循环神经网络：

1. **序列建模的动机** — 为什么需要处理变长序列
2. **Vanilla RNN** — 基本结构与计算
3. **反向传播 (BPTT)** — 梯度消失/爆炸问题
4. **LSTM** — 门控机制解决长期依赖

---

## 第一部分：为什么需要 RNN？

### 1.1 序列问题类型

| 类型 | 输入 → 输出 | 例子 |
| ---- | ----------- | ---- |
| one-to-one | 固定 → 固定 | 图像分类 |
| one-to-many | 固定 → 序列 | 图像描述 (Image Captioning) |
| many-to-one | 序列 → 固定 | 情感分析 |
| many-to-many | 序列 → 序列 | 机器翻译 |
| many-to-many (同步) | 序列 → 序列 | 视频分类（逐帧） |

### 1.2 CNN 的局限

CNN 处理固定大小的输入和输出。对于变长的序列数据（文本、语音、视频），我们需要一种能处理**任意长度**输入的架构。

---

## 第二部分：Vanilla RNN

### 2.1 基本结构（⭐ 核心公式）

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

- $h_t$：时间步 $t$ 的隐藏状态（"记忆"）
- $x_t$：当前输入
- $y_t$：当前输出
- 权重 $W_{hh}, W_{xh}, W_{hy}$ **跨时间步共享**

### 2.2 展开视图

```
x1     x2     x3     x4
↓      ↓      ↓      ↓
[RNN] → [RNN] → [RNN] → [RNN]
  ↓      ↓      ↓      ↓
  y1     y2     y3     y4
```

每个时间步使用**相同的权重**，但隐藏状态 $h_t$ 不同——它携带了之前所有输入的信息。

### 2.3 计算图与参数共享

```python
h = torch.zeros(hidden_size)
for t in range(seq_length):
    h = torch.tanh(W_hh @ h + W_xh @ x[t] + b)
    y[t] = W_hy @ h
```

**参数数量与序列长度无关**——这是 RNN 能处理任意长度序列的关键。

---

## 第三部分：反向传播 BPTT（⭐⭐ 核心难点）

### 3.1 时间反向传播 (Backpropagation Through Time)

将 RNN 展开后，就是一个非常深的普通网络。反向传播需要从最后一个时间步一路传回到第一个。

### 3.2 梯度消失/爆炸（⭐ 关键问题）

$$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=2}^{T} W_{hh}^T \cdot \text{diag}(\tanh'(\cdot))$$

- 如果 $W_{hh}$ 的最大特征值 $> 1$：梯度**指数级增长**（爆炸）
- 如果 $W_{hh}$ 的最大特征值 $< 1$：梯度**指数级衰减**（消失）

**梯度爆炸的缓解**：梯度裁剪 (Gradient Clipping)

```python
if grad.norm() > max_norm:
    grad = grad * (max_norm / grad.norm())
```

**梯度消失的缓解**：→ LSTM

---

## 第四部分：LSTM（⭐⭐ 必须掌握）

### 4.1 核心思想

引入**细胞状态 (Cell State)** $c_t$ 作为长期记忆的"高速公路"，通过三个**门 (Gates)** 控制信息的流动。

### 4.2 LSTM 公式

$$\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) & \text{（遗忘门）} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) & \text{（输入门）} \\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) & \text{（候选记忆）} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t & \text{（更新细胞状态）} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) & \text{（输出门）} \\
h_t &= o_t \odot \tanh(c_t) & \text{（隐藏状态）}
\end{aligned}$$

### 4.3 三个门的直觉

| 门 | 作用 | 直觉 |
| -- | ---- | ---- |
| 遗忘门 $f_t$ | 决定丢弃多少旧记忆 | "我该忘掉什么？" |
| 输入门 $i_t$ | 决定写入多少新信息 | "我该记住什么？" |
| 输出门 $o_t$ | 决定暴露多少内部状态 | "我该输出什么？" |

### 4.4 为什么 LSTM 解决了梯度消失？

细胞状态 $c_t$ 的更新是**加法操作**：

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

反向传播时：

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

- 遗忘门 $f_t$ 的值在 (0, 1) 之间，由 sigmoid 控制
- 只要 $f_t$ 接近 1，梯度就能**无衰减地**通过任意长的时间步
- 这就像 ResNet 的跳跃连接——为梯度提供了"高速公路"

---

## 💡 重点总结

1. **RNN 通过隐藏状态传递信息**，权重跨时间步共享
2. **Vanilla RNN 的致命缺陷**：长序列中梯度消失/爆炸，无法学习长期依赖
3. **LSTM 的核心创新**：细胞状态的加法更新 + 门控机制，让梯度可以无损传播
4. **梯度裁剪**解决梯度爆炸，但无法解决梯度消失（需要 LSTM/GRU）
5. **LSTM 之于 RNN，如同 ResNet 之于普通深层网络**——都是通过加法跳跃连接解决梯度传播问题

