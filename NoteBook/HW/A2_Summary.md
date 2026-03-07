# A2 作业总结：线性分类器 & 两层神经网络

---

## 📌 概览

A2 包含三个核心部分：

1. **SVM 线性分类器** — 多类 SVM (Hinge Loss) 的实现
2. **Softmax 线性分类器** — Softmax + 交叉熵损失的实现
3. **两层神经网络** — FC → ReLU → FC 结构的从零实现

数据集：CIFAR-10（10类，32×32彩色图像）

---

## 第一部分：SVM 线性分类器

### 1.1 知识点总结

| 知识点 | 说明 |
| ------ | ---- |
| 线性分类器 | 分数函数 $f(x) = Wx$，W 为权重矩阵 |
| Hinge Loss | $L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$ |
| L2 正则化 | $R(W) = \sum_{k} \sum_{l} W_{k,l}^2$ |
| 随机梯度下降 (SGD) | $W \leftarrow W - \alpha \nabla_W L$ |
| Mini-batch 训练 | 每次用一小批数据计算梯度 |

### 1.2 SVM Loss 朴素实现（带循环）

```python
for i in range(num_train):
    scores = W.t().mv(X[i])                   # 计算第 i 个样本对所有类的分数
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
        if j == y[i]:
            continue
        margin = scores[j] - correct_class_score + 1  # delta = 1
        if margin > 0:
            loss += margin
            dW[:, j] += X[i]           # 错误类梯度 +X[i]
            dW[:, y[i]] -= X[i]        # 正确类梯度 -X[i]

# 平均化 + 正则化
loss /= num_train
dW /= num_train
loss += reg * torch.sum(W * W)
dW += 2 * reg * W
```

**梯度推导**（⭐ 重点）：

对于 Hinge Loss $L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)$：
- 对错误类 j 的分数 $s_j$：$\frac{\partial L_i}{\partial s_j} = \mathbb{1}(s_j - s_{y_i} + 1 > 0)$
- 对正确类 $y_i$ 的分数：$\frac{\partial L_i}{\partial s_{y_i}} = -\sum_{j \neq y_i} \mathbb{1}(s_j - s_{y_i} + 1 > 0)$
- 因为 $s = Wx$，所以 $\frac{\partial L}{\partial W_j} = \frac{\partial L}{\partial s_j} \cdot x$

### 1.3 SVM Loss 向量化实现（⭐⭐ 核心）

```python
num_train = X.shape[0]

# 1. 计算所有分数 (N, C)
scores = X.mm(W)

# 2. 提取正确类分数 → (N, 1) 用于广播
correct_class_scores = scores[torch.arange(num_train), y].view(-1, 1)

# 3. 计算 margins (N, C)
margins = scores - correct_class_scores + 1

# 4. 正确类位置置零
margins[torch.arange(num_train), y] = 0

# 5. Hinge: max(0, margin)
margins = margins.clamp(min=0)

# 6. 损失 = 平均 + 正则化
loss = margins.sum() / num_train + reg * torch.sum(W * W)
```

**向量化梯度**：

```python
# 构造梯度矩阵
dscores = (margins > 0).to(W.dtype)                       # (N, C)
row_sum = dscores.sum(dim=1)                               # 每行违规类数量
dscores[torch.arange(num_train), y] = -row_sum             # 正确类梯度

# 反向传播到 W
dW = X.t().mm(dscores) / num_train + 2 * reg * W           # (D, C)
```

**关键技巧**：
1. `view(-1, 1)` 将正确类分数变为列向量，利用广播与所有类分数相减
2. `clamp(min=0)` 等价于 `max(0, x)`
3. `(margins > 0)` 生成布尔掩码，转为浮点型作为梯度指示

---

## 第二部分：Softmax 分类器

### 2.1 知识点总结

| 知识点 | 说明 |
| ------ | ---- |
| Softmax 函数 | $P(y=k \mid x) = \frac{e^{s_k}}{\sum_j e^{s_j}}$ |
| 交叉熵损失 | $L_i = -\log P(y_i \mid x_i)$ |
| 数值稳定性 | 减去最大值：$s_k \leftarrow s_k - \max_j s_j$ |
| 概率解释 | Softmax 输出可解释为类别概率分布 |

### 2.2 Softmax Loss 朴素实现

```python
for i in range(num_train):
    f_i = W.t().mv(X[i])
    f_i -= torch.max(f_i)              # ⭐ 数值稳定性处理

    exp_scores = torch.exp(f_i)
    sum_scores = torch.sum(exp_scores)
    p = exp_scores / sum_scores         # softmax 概率

    loss += -torch.log(p[y[i]])         # 交叉熵

    for j in range(num_classes):
        dscores = p[j]
        if j == y[i]:
            dscores -= 1               # 正确类梯度: p_k - 1
        dW[:, j] += dscores * X[i]
```

### 2.3 Softmax Loss 向量化实现（⭐⭐ 核心）

```python
num_train = X.shape[0]

# 1. 计算分数
scores = X.mm(W)                                          # (N, C)

# 2. 数值稳定性（⭐ 防止 exp 溢出）
scores -= torch.max(scores, dim=1, keepdim=True).values

# 3. Softmax 概率
exp_scores = torch.exp(scores)
sum_scores = torch.sum(exp_scores, dim=1, keepdim=True)
probs = exp_scores / sum_scores                            # (N, C)

# 4. 交叉熵损失
correct_class_probs = probs[torch.arange(num_train), y]
loss = -torch.sum(torch.log(correct_class_probs)) / num_train
loss += reg * torch.sum(W * W)

# 5. 梯度
dscores = probs.clone()
dscores[torch.arange(num_train), y] -= 1
dscores /= num_train
dW = X.t().mm(dscores) + 2 * reg * W
```

**数值稳定性**（⭐⭐ 必须掌握）：

$$\frac{e^{s_k}}{\sum_j e^{s_j}} = \frac{e^{s_k - \max(s)}}{\sum_j e^{s_j - \max(s)}}$$

如果不减去最大值，$e^{s_k}$ 可能溢出为 `inf`。

**Softmax 梯度推导**（⭐ 重点）：

$$\frac{\partial L_i}{\partial s_k} = P(y=k|x_i) - \mathbb{1}(k = y_i)$$

即：概率值减去独热编码。对于正确类 $y_i$，梯度为 $p_{y_i} - 1$；对于错误类，梯度为 $p_k$。

---

## 第三部分：训练框架

### 3.1 Mini-Batch 采样

```python
def sample_batch(X, y, num_train, batch_size):
    indices = torch.randint(num_train, (batch_size,), device=X.device)
    X_batch = X[indices]
    y_batch = y[indices]
    return X_batch, y_batch
```

### 3.2 SGD 训练循环

```python
for it in range(num_iters):
    X_batch, y_batch = sample_batch(X, y, num_train, batch_size)
    loss, grad = loss_func(W, X_batch, y_batch, reg)

    # 参数更新
    W -= learning_rate * grad

    # 每个 epoch 记录一次 loss
    loss_history.append(loss.item())
```

### 3.3 超参数搜索

```python
# SVM 超参数范围
learning_rates = [1e-9, 2e-9, 5e-8, 1e-8, 2e-8]
regularization_strengths = [2.5e3, 5e3, 1e4, 2.5e4, 5e4]

# Softmax 超参数范围
learning_rates = [1e-7, 5e-7, 1e-6, 5e-6]
regularization_strengths = [2.5e3, 5e3, 1e4, 2.5e4, 5e4]
```

**注意**：SVM 和 Softmax 的学习率范围差异很大，SVM 的梯度量级更大，所以需要更小的学习率。

### 3.4 预测函数

```python
def predict_linear_classifier(W, X):
    scores = X.mm(W)                    # (N, C)
    y_pred = torch.argmax(scores, dim=1) # 取分数最高的类
    return y_pred
```

---

## 第四部分：两层神经网络（two_layer_net.py）

### 4.1 知识点总结

| 知识点 | 说明 |
| ------ | ---- |
| 网络结构 | FC → ReLU → FC (输入层 → 隐藏层 → 输出层) |
| ReLU 激活 | $\text{ReLU}(x) = \max(0, x)$ |
| Softmax 损失 | 同 Section 2 |
| 反向传播 | 链式法则逐层计算梯度 |
| 学习率衰减 | $\text{lr} \leftarrow \text{lr} \times \text{decay}$ |

### 4.2 网络架构

```
输入 X (N, D)
    ↓
全连接层 1: X @ W1 + b1 → (N, H)
    ↓
ReLU 激活: max(0, ·) → (N, H)
    ↓
全连接层 2: hidden @ W2 + b2 → (N, C)
    ↓
Softmax + Cross-Entropy Loss
```

### 4.3 前向传播（⭐ 重点）

```python
def nn_forward_pass(params, X):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    hidden = X.mm(W1).add(b1)     # 全连接层1: (N, D) × (D, H) + (H,) = (N, H)
    hidden = hidden.clamp(min=0)   # ReLU 激活（不允许用 torch.relu）
    scores = hidden.mm(W2).add(b2) # 全连接层2: (N, H) × (H, C) + (C,) = (N, C)

    return scores, hidden
```

**注意**：这里使用 `.clamp(min=0)` 代替 `torch.relu()`，因为作业要求不使用 `torch.relu` 和 `torch.nn` 模块。

### 4.4 反向传播（⭐⭐ 核心难点）

```python
# ============ 前向：计算损失 ============
# Softmax + 数值稳定
shifted_logits = scores - scores.max(dim=1, keepdim=True)[0]
Z = shifted_logits.exp().sum(dim=1, keepdim=True)
log_probs = shifted_logits - Z.log()
probs = log_probs.exp()

# 交叉熵损失
loss = -log_probs[torch.arange(N), y].sum() / N
loss += reg * (W1.pow(2).sum() + W2.pow(2).sum())  # L2 正则化

# ============ 反向：计算梯度 ============
# Step 1: dL/d(scores) — Softmax 梯度
dscores = probs.clone()
dscores[torch.arange(N), y] -= 1
dscores /= N                                        # (N, C)

# Step 2: W2 和 b2 的梯度
grads['W2'] = h1.t().mm(dscores) + 2 * reg * W2     # (H, C)
grads['b2'] = dscores.sum(dim=0)                     # (C,)

# Step 3: 反向传播到隐藏层
dhidden = dscores.mm(W2.t())                         # (N, H)
dhidden[h1 <= 0] = 0                                 # ⭐ ReLU 反向传播

# Step 4: W1 和 b1 的梯度
grads['W1'] = X.t().mm(dhidden) + 2 * reg * W1       # (D, H)
grads['b1'] = dhidden.sum(dim=0)                      # (H,)
```

**反向传播流程图**（⭐⭐ 必须理解）：

```
Loss
  ↓ dscores = probs - one_hot(y)           [Softmax 反向]
scores (N, C)
  ↓ dW2 = hidden.T @ dscores               [FC2 权重梯度]
  ↓ db2 = sum(dscores, dim=0)              [FC2 偏置梯度]
  ↓ dhidden = dscores @ W2.T               [传播到隐藏层]
hidden (N, H)
  ↓ dhidden[h1 <= 0] = 0                   [ReLU 反向：门控]
ReLU input
  ↓ dW1 = X.T @ dhidden                    [FC1 权重梯度]
  ↓ db1 = sum(dhidden, dim=0)              [FC1 偏置梯度]
```

**ReLU 反向传播的关键**（⭐⭐ 难点）：

$$\frac{\partial \text{ReLU}(x)}{\partial x} = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

代码中 `dhidden[h1 <= 0] = 0` 实现了这个门控效应 — 前向传播中被 ReLU "关闭"的神经元，反向传播时梯度也为零。

### 4.5 训练过程

```python
for it in range(num_iters):
    X_batch, y_batch = sample_batch(X, y, num_train, batch_size)
    loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)

    # SGD 参数更新
    params["W1"] -= learning_rate * grads["W1"]
    params["b1"] -= learning_rate * grads["b1"]
    params["W2"] -= learning_rate * grads["W2"]
    params["b2"] -= learning_rate * grads["b2"]

    # 每个 epoch 学习率衰减
    if it % iterations_per_epoch == 0:
        learning_rate *= learning_rate_decay
```

### 4.6 预测

```python
def nn_predict(params, loss_func, X):
    scores, _ = nn_forward_pass(params, X)
    y_pred = scores.argmax(dim=1)
    return y_pred
```

### 4.7 超参数搜索

```python
learning_rates = [1e-3, 1e-2, 5e-2, 1e-1]
hidden_sizes = [32, 64, 128]
regularization_strengths = [1e-3, 1e-4]
learning_rate_decays = [0.95, 1.0]
```

网格搜索流程：

```python
for lr in learning_rates:
    for hidden_size in hidden_sizes:
        for reg in regularization_strengths:
            for decay in learning_rate_decays:
                model = TwoLayerNet(input_size, hidden_size, output_size, ...)
                stats = model.train(X_train, y_train, X_val, y_val,
                                    num_iters=2000, batch_size=200,
                                    learning_rate=lr,
                                    learning_rate_decay=decay, reg=reg)
                val_acc = stats['val_acc_history'][-1]
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_net = model
```

---

## 💡 重点与难点总结

### 重点

1. **损失函数理解**：SVM Hinge Loss vs Softmax Cross-Entropy Loss 的区别
   - SVM：只关心正确类分数是否比错误类高出 margin
   - Softmax：将分数转为概率分布，优化正确类的对数似然
2. **向量化实现**：从循环版本到无循环版本的转化技巧
3. **反向传播**：链式法则在每一层的应用
4. **ReLU 的反向传播**：门控机制 `dhidden[h1 <= 0] = 0`
5. **数值稳定性**：Softmax 计算前减去最大值

### 难点

1. **SVM 梯度的向量化**：
   - 理解 `(margins > 0)` 掩码的含义
   - 正确类的梯度 = 负的违规类数量

2. **Softmax 梯度推导**：
   - $\frac{\partial L}{\partial s_k} = p_k - \mathbb{1}(k=y_i)$ 的证明

3. **两层网络反向传播**：
   - 链式法则的正确应用顺序
   - 矩阵维度匹配（转置的时机）
   - ReLU 反向时使用前向的激活值 `h1` 而非梯度

4. **超参数搜索**：
   - SVM 和 Softmax 的学习率范围差几个数量级
   - 正则化强度的选择对模型性能影响巨大

### SVM vs Softmax 对比

| 特性 | SVM | Softmax |
| ---- | --- | ------- |
| 损失函数 | Hinge Loss: $\max(0, s_j - s_{y_i} + 1)$ | Cross-Entropy: $-\log \frac{e^{s_{y_i}}}{\sum_j e^{s_j}}$ |
| 输出解释 | 分数（无概率含义） | 概率分布 |
| 梯度特性 | 稀疏（仅违规类有梯度） | 密集（所有类都有梯度） |
| 学习率 | 更小 (~1e-9) | 更大 (~1e-7) |
| 数值问题 | 无特殊问题 | 需要数值稳定性处理 |

### 线性分类器 vs 两层神经网络

| 特性 | 线性分类器 | 两层神经网络 |
| ---- | ---------- | ------------ |
| 决策边界 | 线性 | 非线性 |
| 参数 | W (D×C) | W1(D×H), b1(H), W2(H×C), b2(C) |
| 表达能力 | 有限 | 更强（通用近似定理） |
| 训练技巧 | 固定学习率 | 学习率衰减 |
| CIFAR-10 准确率 | ~37-39% | ~50%+ |

> 从线性分类器到神经网络，关键突破在于引入了**非线性激活函数 (ReLU)**，使模型能够学习非线性决策边界。
