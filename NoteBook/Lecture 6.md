# Lecture 6：反向传播

---

## 📌 概览

本讲解决神经网络训练中的核心问题：**如何高效计算损失对所有参数的梯度？**

1. **计算图** — 把前向计算表示为有向无环图 (DAG)
2. **反向传播** — 利用链式法则在计算图上反向传播梯度
3. **矩阵运算的反向传播** — 全连接层的梯度推导

---

## 第一部分：计算图 (Computational Graph)

### 1.1 什么是计算图？

将复杂的数学表达式拆解为一系列**简单的基本操作**，每个操作是图中的一个节点。

**例子**：SVM Loss 的计算图

$$L = \frac{1}{N}\sum_i \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1) + \lambda \sum W^2$$

可以拆解为：

```
x, W → [矩阵乘法] → scores → [hinge loss] → data_loss
                                                  ↘
W → [平方] → [求和] → [× λ] → reg_loss → [+] → total_loss
```

### 1.2 为什么用计算图？

- 前向传播：从左到右，计算最终的 loss
- 反向传播：从右到左，利用链式法则计算每个节点的梯度
- **模块化**：每个节点只需要知道自己的局部梯度，就能参与全局梯度计算

---

## 第二部分：反向传播算法（⭐⭐ 核心）

### 2.1 链式法则 (Chain Rule)

如果 $y = f(x)$，$z = g(y)$，那么：

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}$$

**上游梯度** × **局部梯度** = **下游梯度**

### 2.2 反向传播的流程

对计算图中的每个节点：

1. **前向传播时**：计算输出值，并保存必要的中间结果
2. **反向传播时**：
   - 接收来自下游的**上游梯度** $\frac{\partial L}{\partial \text{output}}$
   - 计算自己的**局部梯度** $\frac{\partial \text{output}}{\partial \text{input}}$
   - 两者相乘，传给上游

### 2.3 基本操作的局部梯度（⭐ 必须记住）

| 操作 | 前向 | 局部梯度 |
| ---- | ---- | -------- |
| 加法 $z = x + y$ | $z = x + y$ | $\frac{\partial z}{\partial x} = 1, \; \frac{\partial z}{\partial y} = 1$ |
| 乘法 $z = x \cdot y$ | $z = xy$ | $\frac{\partial z}{\partial x} = y, \; \frac{\partial z}{\partial y} = x$ |
| max $z = \max(x, y)$ | $z = \max(x,y)$ | 梯度只传给较大的那个输入 |
| ReLU $z = \max(0, x)$ | $z = \max(0, x)$ | $x > 0$ 时传 1，$x \leq 0$ 时传 0 |
| 指数 $z = e^x$ | $z = e^x$ | $\frac{\partial z}{\partial x} = e^x = z$ |

### 2.4 具体例子

计算 $f(x, y, z) = (x + y) \cdot z$，其中 $x = -2, y = 5, z = -4$

**前向传播**：

```
q = x + y = -2 + 5 = 3
f = q * z = 3 * (-4) = -12
```

**反向传播**：

```
df/df = 1                              (起点)

df/dz = q = 3                          (乘法节点：局部梯度是另一个输入)
df/dq = z = -4                         (乘法节点：局部梯度是另一个输入)

df/dx = df/dq · dq/dx = -4 · 1 = -4   (加法节点：局部梯度是 1)
df/dy = df/dq · dq/dy = -4 · 1 = -4   (加法节点：局部梯度是 1)
```

### 2.5 反向传播中的模式（⭐ 直觉理解）

| 门类型 | 反向传播行为 | 直觉 |
| ------ | ------------ | ---- |
| **加法门** | 梯度"分发"给两个输入 | 梯度分配器 |
| **乘法门** | 梯度"交换"并缩放 | 梯度交换器 |
| **max 门** | 梯度只传给最大值 | 梯度路由器 |
| **复制/分支** | 梯度"相加" | 梯度累加器 |

---

## 第三部分：向量与矩阵的反向传播

### 3.1 向量化的链式法则

当变量是向量时，局部梯度变成**雅可比矩阵 (Jacobian)**：

如果 $\mathbf{y} = f(\mathbf{x})$，其中 $\mathbf{x} \in \mathbb{R}^N$，$\mathbf{y} \in \mathbb{R}^M$，则：

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_N} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_M}{\partial x_1} & \cdots & \frac{\partial y_M}{\partial x_N} \end{bmatrix} \in \mathbb{R}^{M \times N}$$

### 3.2 矩阵乘法的反向传播（⭐⭐ 最重要的推导）

前向：$Y = XW$，其中 $X \in \mathbb{R}^{N \times D}$，$W \in \mathbb{R}^{D \times C}$，$Y \in \mathbb{R}^{N \times C}$

已知上游梯度 $\frac{\partial L}{\partial Y} \in \mathbb{R}^{N \times C}$，求：

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T \quad \in \mathbb{R}^{N \times D}$$

$$\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y} \quad \in \mathbb{R}^{D \times C}$$

> **记忆技巧**：维度匹配法
>
> - $\frac{\partial L}{\partial X}$ 形状 = $(N, D)$ = $(N, C) \times (C, D)$ = $\frac{\partial L}{\partial Y} \cdot W^T$
> - $\frac{\partial L}{\partial W}$ 形状 = $(D, C)$ = $(D, N) \times (N, C)$ = $X^T \cdot \frac{\partial L}{\partial Y}$

### 3.3 全连接层的完整反向传播

前向：$Y = XW + b$

```python
# 前向
Y = X.mm(W) + b

# 反向（已知 dY = dL/dY）
dX = dY.mm(W.t())        # (N, D)
dW = X.t().mm(dY)        # (D, C)
db = dY.sum(dim=0)       # (C,)  — 偏置梯度是上游梯度按 batch 求和
```

---

## 第四部分：实现层面的组织

### 4.1 模块化设计

每一层实现为一个模块，包含 `forward()` 和 `backward()` 方法：

```python
class Layer:
    def forward(self, x):
        # 计算输出，保存中间值
        self.cache = ...
        return output

    def backward(self, d_output):
        # 利用 cache 和上游梯度计算局部梯度
        d_input = ...
        d_params = ...
        return d_input
```

### 4.2 前向传播 = 从左到右

```python
# Layer 1
h1 = relu(X @ W1 + b1)
# Layer 2
scores = h1 @ W2 + b2
# Loss
loss = softmax_loss(scores, y)
```

### 4.3 反向传播 = 从右到左

```python
# Loss 的梯度
dscores = softmax_grad(scores, y)
# Layer 2 反向
dh1 = dscores @ W2.T
dW2 = h1.T @ dscores
db2 = dscores.sum(0)
# ReLU 反向
dh1[h1 <= 0] = 0
# Layer 1 反向
dX = dh1 @ W1.T
dW1 = X.T @ dh1
db1 = dh1.sum(0)
```

---

## 💡 重点总结

1. **计算图**：把复杂计算拆成简单操作的有向无环图
2. **链式法则**：反向传播的数学基础——上游梯度 × 局部梯度
3. **四种门的行为**：加法（分发）、乘法（交换）、max（路由）、分支（累加）
4. **矩阵乘法反向传播**：$dX = dY \cdot W^T$，$dW = X^T \cdot dY$，通过维度匹配来记忆
5. **ReLU 反向传播**：$x > 0$ 时梯度直通，$x \leq 0$ 时梯度为零（门控效应）
6. **模块化实现**：每层封装 forward/backward，像搭积木一样组合
7. **实践意义**：现代框架（PyTorch）自动构建计算图并执行反向传播，但理解原理对调试至关重要
