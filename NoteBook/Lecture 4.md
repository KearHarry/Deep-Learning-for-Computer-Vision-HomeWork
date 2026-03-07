# Lecture 4：正则化与优化

---

## 📌 概览

本讲衔接 Lecture 3 的损失函数，回答一个核心问题：**如何找到使损失最小的权重 $W$？**

1. **正则化** — 防止过拟合，偏好"简单"模型
2. **优化算法** — SGD、Momentum、AdaGrad、Adam 等

---

## 第一部分：优化基础

### 1.1 优化的目标

$$W^* = \arg\min_W L(W)$$

找到一组权重 $W$，使得总损失（数据损失 + 正则化）最小。

### 1.2 策略一：随机搜索（❌ 不靠谱）

随便试很多组 $W$，挑最好的。在高维空间中这几乎不可能碰到好的解。

### 1.3 策略二：跟着梯度走（✅ 正确方法）

**梯度 (Gradient)**：损失函数对每个参数的偏导数组成的向量，指向损失**增长最快**的方向。

$$\nabla_W L = \begin{bmatrix} \frac{\partial L}{\partial W_1} \\ \frac{\partial L}{\partial W_2} \\ \vdots \end{bmatrix}$$

> **下山的比喻**：你在一个雾天站在山坡上，看不到全貌。梯度告诉你脚下哪个方向最陡，你就朝**反方向**迈一步——这就是梯度下降。

### 1.4 数值梯度 vs 解析梯度（⭐ 重要区分）

**数值梯度**（慢但简单，用于验证）：

$$\frac{\partial L}{\partial W_i} \approx \frac{L(W + h \cdot e_i) - L(W - h \cdot e_i)}{2h}$$

**解析梯度**（快且精确，用于实际训练）：

通过微积分推导出梯度的解析公式，直接计算。

> **Gradient Check**：实践中常用数值梯度来验证解析梯度的正确性。如果两者差异很大，说明代码有 bug。

---

## 第二部分：随机梯度下降 (SGD)

### 2.1 Vanilla Gradient Descent

```python
while True:
    grad = compute_gradient(loss_fn, data, W)
    W -= learning_rate * grad
```

**问题**：每次计算梯度要遍历**整个**训练集，太慢了。

### 2.2 随机梯度下降 (SGD)（⭐ 核心算法）

每次只从训练集中随机取一小批数据（**mini-batch**）来估计梯度：

```python
while True:
    mini_batch = sample(data, batch_size=256)
    grad = compute_gradient(loss_fn, mini_batch, W)
    W -= learning_rate * grad
```

- **Mini-batch 大小**：常用 32、64、128、256
- **关键假设**：小批量梯度是全量梯度的**无偏估计**
- **优点**：每步计算快很多，且噪声有助于跳出局部极值

### 2.3 SGD 的问题（⭐ 面试常考）

| 问题 | 说明 |
| ---- | ---- |
| 损失在不同方向上敏感度不同 | 梯度在陡峭方向震荡，在平缓方向前进缓慢（"锯齿形"轨迹） |
| 局部极小值 (Local Minima) | 梯度为 0，卡住不动 |
| 鞍点 (Saddle Point) | 高维空间中更常见，某些方向上升某些方向下降 |
| 噪声 | mini-batch 的梯度估计有噪声 |

---

## 第三部分：改进的优化算法

### 3.1 SGD + Momentum（⭐ 重要）

核心思想：**加入"惯性"**。不仅看当前梯度，还考虑之前梯度的累积方向。

```python
v = 0
while True:
    grad = compute_gradient(loss_fn, mini_batch, W)
    v = rho * v + grad          # rho 通常取 0.9 或 0.99
    W -= learning_rate * v
```

- $v$：**速度 (velocity)**，是历史梯度的指数衰减平均
- **效果**：在一致方向上加速，在震荡方向上减速
- **直觉**：就像球从山上滚下来，会积累速度，能冲过小坑（局部极小值）

### 3.2 Nesterov Momentum

先按照当前速度"预看一步"，再在预看位置计算梯度：

```python
v = 0
while True:
    W_ahead = W + rho * v            # 先看一步
    grad = compute_gradient(loss_fn, mini_batch, W_ahead)
    v = rho * v - learning_rate * grad
    W += v
```

比标准 Momentum 有更好的理论收敛保证。

### 3.3 AdaGrad

对每个参数**自适应调整学习率**：频繁更新的参数学习率变小，不常更新的参数学习率变大。

```python
grad_squared = 0
while True:
    grad = compute_gradient(loss_fn, mini_batch, W)
    grad_squared += grad * grad                        # 累积梯度平方
    W -= learning_rate * grad / (sqrt(grad_squared) + 1e-7)
```

**问题**：`grad_squared` 只增不减，学习率会单调递减，最终趋近于 0，训练停滞。

### 3.4 RMSProp

解决 AdaGrad 学习率持续衰减的问题——用指数衰减平均代替简单累加：

```python
grad_squared = 0
while True:
    grad = compute_gradient(loss_fn, mini_batch, W)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * grad * grad
    W -= learning_rate * grad / (sqrt(grad_squared) + 1e-7)
```

### 3.5 Adam（⭐⭐ 最常用的优化器）

**Adam = Momentum + RMSProp**，同时维护一阶矩（均值）和二阶矩（方差）的指数衰减估计：

```python
m, v = 0, 0     # 一阶矩, 二阶矩
t = 0
while True:
    t += 1
    grad = compute_gradient(loss_fn, mini_batch, W)

    m = beta1 * m + (1 - beta1) * grad          # 一阶矩估计 (类似 Momentum)
    v = beta2 * v + (1 - beta2) * grad * grad    # 二阶矩估计 (类似 RMSProp)

    # 偏差校正（⭐ 初始阶段 m 和 v 偏小，需要修正）
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    W -= learning_rate * m_hat / (sqrt(v_hat) + 1e-8)
```

**超参数经验值**：

| 参数 | 典型值 | 含义 |
| ---- | ------ | ---- |
| `learning_rate` | `1e-3` | 初始学习率 |
| `beta1` | `0.9` | 一阶矩衰减率 |
| `beta2` | `0.999` | 二阶矩衰减率 |
| `epsilon` | `1e-8` | 防止除以零 |

> **实践建议**：**不知道用什么就用 Adam**。它在绝大多数情况下都表现良好，是默认选择。

---

## 第四部分：学习率

### 4.1 学习率的影响（⭐ 最重要的超参数）

| 学习率 | 效果 |
| ------ | ---- |
| 太大 | 损失爆炸或剧烈震荡 |
| 太小 | 收敛极慢，浪费计算资源 |
| 刚好 | 稳定下降，最终收敛 |

### 4.2 学习率衰减策略

训练初期用较大学习率快速下降，后期用小学习率精细调整：

| 策略 | 公式 / 说明 |
| ---- | ----------- |
| Step Decay | 每隔若干 epoch 乘以 0.1 |
| Cosine Decay | $\text{lr}_t = \frac{1}{2}\text{lr}_0(1 + \cos(\frac{t\pi}{T}))$ |
| Linear Warmup | 前几个 epoch 从很小的学习率线性增大到目标值 |

---

## 💡 重点总结

1. **梯度下降**：沿着损失函数的负梯度方向更新参数
2. **SGD**：用 mini-batch 估计梯度，每步计算量小
3. **Momentum**：积累历史梯度方向，克服震荡和局部极小
4. **Adam**：融合 Momentum 和 RMSProp，自适应学习率，是最常用的优化器
5. **偏差校正**：Adam 初始阶段的矩估计偏小，必须校正
6. **学习率**：最重要的超参数，太大爆炸太小龟速，需要配合衰减策略
7. **Gradient Check**：用数值梯度验证解析梯度，是调试的重要手段
