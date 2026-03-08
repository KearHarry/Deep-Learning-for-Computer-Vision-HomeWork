# A3 作业总结：全连接网络 & 卷积神经网络

---

## 📌 概览

A3 包含两大部分：

1. **全连接网络 (Fully Connected Networks)** — 模块化层设计、优化器、Dropout
2. **卷积神经网络 (Convolutional Networks)** — 卷积层、池化层、BatchNorm、深度网络

数据集：CIFAR-10（10类，32×32彩色图像）

---

## 第一部分：全连接网络（fully_connected_networks.py）

### 1.1 知识点总结

| 知识点 | 说明 |
| ------ | ---- |
| 模块化设计 | 每一层拆分为 `forward` 和 `backward` 两个函数 |
| Linear 层 | $y = xW + b$，前向/反向传播 |
| ReLU 层 | $\max(0, x)$ 的前向/反向传播 |
| Sandwich 层 | Linear_ReLU 组合层，简化网络搭建 |
| Dropout | Inverted Dropout 正则化 |
| 优化器 | SGD、SGD+Momentum、RMSProp、Adam |
| 任意深度网络 | FullyConnectedNet 支持 L 层隐藏层 |

### 1.2 Linear 层实现

#### 前向传播

```python
# 输入 x 形状: (N, d_1, ..., d_k)，需先展平为 (N, D)
out = x.reshape(x.shape[0], -1).mm(w) + b
```

#### 反向传播（⭐ 重点）

```python
dx = dout.mm(w.t()).reshape(x.shape)   # 恢复原始形状
dw = x.reshape(x.shape[0], -1).t().mm(dout)
db = dout.sum(dim=0)
```

**维度分析**：
- `dout`: (N, M) — 上游梯度
- `dx`: (N, D) → reshape 回 (N, d_1, ..., d_k)
- `dw`: (D, M) — 与 w 同形
- `db`: (M,) — 对 batch 维度求和

### 1.3 ReLU 层实现

```python
# 前向：不能用 in-place 操作
out = x.clamp(min=0)

# 反向：门控机制
dx = dout.clone()
dx[x <= 0] = 0
```

**反向传播关键**：ReLU 的导数是 0/1 的门控——前向被"关闭"（x ≤ 0）的位置，反向梯度也为零。

### 1.4 TwoLayerNet 模块化实现

```
Input → Linear_ReLU (W1, b1) → Linear (W2, b2) → Softmax Loss
```

```python
# 前向传播
a1, cache1 = Linear_ReLU.forward(X, W1, b1)
scores, cache2 = Linear.forward(a1, W2, b2)

# 反向传播
data_loss, dscores = softmax_loss(scores, y)
reg_loss = reg * (W1.pow(2).sum() + W2.pow(2).sum())

da1, dW2, db2 = Linear.backward(dscores, cache2)
dW2 += 2 * reg * W2

dx, dW1, db1 = Linear_ReLU.backward(da1, cache1)
dW1 += 2 * reg * W1
```

**关键理解**：模块化设计的好处——每层的 `forward` 保存 `cache`，`backward` 利用 `cache` 计算局部梯度，层层传递。

### 1.5 FullyConnectedNet 任意深度网络

```
{Linear - ReLU - [Dropout]} × (L-1) → Linear → Softmax
```

```python
# 前向传播：循环处理每一层
out = X
caches = []
for i in range(1, self.num_layers):
    out, cache = Linear_ReLU.forward(out, W_i, b_i)
    caches.append(cache)
    if self.use_dropout:
        out, dropout_cache = Dropout.forward(out, self.dropout_param)
        caches.append(dropout_cache)
# 最后一层
scores, cache = Linear.forward(out, W_L, b_L)

# 反向传播：逆序处理
dout = dscores
for i in reversed(range(1, num_layers + 1)):
    if i == num_layers:
        dout, dW, db = Linear.backward(dout, caches.pop())
    else:
        if use_dropout:
            dout = Dropout.backward(dout, caches.pop())
        dout, dW, db = Linear_ReLU.backward(dout, caches.pop())
    dW += reg * W_i  # 正则化梯度
```

**设计模式**：使用列表 `caches` 作为栈，前向 append、反向 pop，完美配对每一层的 cache。

### 1.6 优化器实现

#### SGD with Momentum（⭐ 重点）

```python
v = momentum * v - learning_rate * dw   # 更新速度
next_w = w + v                           # 更新参数
```

**物理直觉**：小球在损失曲面上滚动，`v` 是速度，`momentum` 控制惯性。即使当前梯度为零，小球仍会因惯性继续运动，有助于逃离局部最优和鞍点。

#### RMSProp

```python
cache = decay_rate * cache + (1 - decay_rate) * (dw ** 2)  # 梯度平方的滑动平均
next_w = w - learning_rate * dw / (sqrt(cache) + epsilon)   # 自适应学习率
```

**核心思想**：对梯度大的参数用小学习率（防止震荡），对梯度小的参数用大学习率（加速收敛）。

#### Adam（⭐⭐ 最常用）

```python
t = t + 1
m = beta1 * m + (1 - beta1) * dw          # 一阶矩估计（动量）
v = beta2 * v + (1 - beta2) * (dw ** 2)   # 二阶矩估计（RMSProp）
m_hat = m / (1 - beta1 ** t)              # 偏差校正
v_hat = v / (1 - beta2 ** t)              # 偏差校正
next_w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

**Adam = Momentum + RMSProp + 偏差校正**

偏差校正的必要性：初始时 `m` 和 `v` 都是零，前几步估计值会偏小，除以 `(1 - beta^t)` 来补偿。

### 1.7 Dropout 实现

#### Inverted Dropout（训练时）

```python
mask = (torch.rand_like(x) > p).float() / (1 - p)  # 生成并缩放 mask
out = x * mask
```

#### 测试时

```python
out = x  # 直接传递，不做任何操作
```

**关键理解**：
- `p` 是**丢弃**概率（不是保留概率）
- "Inverted" 的含义：训练时就除以 `(1-p)` 缩放，这样测试时无需任何修改
- 作用：防止过拟合，强迫网络学习冗余表示

---

## 第二部分：卷积神经网络（convolutional_networks.py）

### 2.1 知识点总结

| 知识点 | 说明 |
| ------ | ---- |
| 卷积层 (Conv) | 滑动窗口计算，保留空间结构 |
| 最大池化 (MaxPool) | 降采样，增强平移不变性 |
| BatchNorm | 归一化中间层输出，稳定训练 |
| Spatial BatchNorm | BatchNorm 在卷积特征图上的扩展 |
| Kaiming 初始化 | 考虑 ReLU 的权重初始化方案 |
| 三层卷积网络 | Conv-ReLU-Pool → FC-ReLU → FC |
| DeepConvNet | 任意深度的 VGG 风格网络 |

### 2.2 卷积层实现

#### 前向传播（Naive 四重循环）

```python
N, C, H, W = x.shape
F, _, HH, WW = w.shape
stride, pad = conv_param['stride'], conv_param['pad']

# 1. 零填充输入
x_pad = F.pad(x, (pad, pad, pad, pad))

# 2. 计算输出尺寸
H_out = 1 + (H + 2 * pad - HH) // stride
W_out = 1 + (W + 2 * pad - WW) // stride

# 3. 卷积（四重循环）
for n in range(N):          # 遍历样本
    for f in range(F):      # 遍历滤波器
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                x_slice = x_pad[n, :, h_start:h_start+HH, w_start:w_start+WW]
                out[n, f, i, j] = torch.sum(x_slice * w[f]) + b[f]
```

**输出尺寸公式**（⭐ 必须记住）：

$$H' = \frac{H + 2 \times \text{pad} - K}{\text{stride}} + 1$$

#### 反向传播

```python
for n in range(N):
    for f in range(F):
        for i in range(H_out):
            for j in range(W_out):
                dout_val = dout[n, f, i, j]
                h_start, w_start = i * stride, j * stride
                
                db[f] += dout_val                                              # 偏置梯度
                dw[f] += x_pad[n, :, h_start:h_end, w_start:w_end] * dout_val # 权重梯度
                dx_pad[n, :, h_start:h_end, w_start:w_end] += w[f] * dout_val # 输入梯度

# 去掉 padding
dx = dx_pad[:, :, pad:-pad, pad:-pad] if pad > 0 else dx_pad
```

**梯度直觉**：
- `dw`：输入片段与上游梯度的"相关"（类似于卷积）
- `dx`：滤波器权重与上游梯度的"反卷积"
- `db`：上游梯度在所有空间位置和样本上的求和

### 2.3 最大池化层实现

#### 前向传播

```python
for each (n, c, i, j):
    window = x[n, c, h_start:h_end, w_start:w_end]
    out[n, c, i, j] = torch.max(window)
```

#### 反向传播（⭐ 难点）

```python
for each (n, c, i, j):
    window = x[n, c, h_start:h_end, w_start:w_end]
    max_val = torch.max(window)
    mask = (window == max_val)  # 找到最大值位置
    dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, i, j]
```

**关键思想**：梯度只传回最大值所在的位置，其他位置梯度为零。"赢者通吃"的路由机制。

### 2.4 Batch Normalization 实现（⭐⭐ 核心）

#### 训练时前向传播

```python
# 1. 计算 mini-batch 的均值和方差
sample_mean = torch.mean(x, dim=0)
sample_var = torch.var(x, dim=0, unbiased=False)

# 2. 归一化
x_norm = (x - sample_mean) / torch.sqrt(sample_var + eps)

# 3. 缩放和平移（可学习参数）
out = gamma * x_norm + beta

# 4. 更新运行统计量（用于测试时）
running_mean = momentum * running_mean + (1 - momentum) * sample_mean
running_var = momentum * running_var + (1 - momentum) * sample_var
```

#### 测试时前向传播

```python
x_norm = (x - running_mean) / torch.sqrt(running_var + eps)
out = gamma * x_norm + beta
```

#### 反向传播

```python
dbeta = torch.sum(dout, dim=0)
dgamma = torch.sum(dout * x_norm, dim=0)

# 简化版反向传播公式
dx = (1/N) * (1/sqrt(var + eps)) * (N * dx_norm - sum(dx_norm) 
     - x_norm * sum(dx_norm * x_norm))
```

**为什么需要 BatchNorm？**
1. 减少内部协变量偏移 (Internal Covariate Shift)
2. 允许使用更大的学习率
3. 有轻微的正则化效果
4. 加速训练收敛

### 2.5 Spatial BatchNorm

将卷积特征图 `(N, C, H, W)` 视为 `(N*H*W, C)` 来做标准 BatchNorm：

```python
# 前向
x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
out_flat, cache = BatchNorm.forward(x_flat, gamma, beta, bn_param)
out = out_flat.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
```

**关键**：对每个通道独立归一化，统计量跨 (N, H, W) 三个维度计算。

### 2.6 Kaiming 初始化

```python
# 线性层
std = sqrt(gain / Din)
weight = torch.randn(Din, Dout) * std

# 卷积层
fan_in = Din * K * K
std = sqrt(gain / fan_in)
weight = torch.randn(Dout, Din, K, K) * std
```

- `gain = 2`（ReLU 后）或 `gain = 1`（无 ReLU / Xavier）
- **原理**：保持每一层输出的方差与输入方差相同，防止信号在深层网络中消失或爆炸。

### 2.7 ThreeLayerConvNet

```
Conv(W1) → ReLU → 2×2 MaxPool → Linear(W2) → ReLU → Linear(W3) → Softmax
```

```python
# 前向
conv_out, conv_cache = Conv.forward(X, W1, b1, conv_param)
relu_out, relu_cache = ReLU.forward(conv_out)
pool_out, pool_cache = MaxPool.forward(relu_out, pool_param)
hidden_out, hidden_cache = Linear_ReLU.forward(pool_out.view(N, -1), W2, b2)
scores, scores_cache = Linear.forward(hidden_out, W3, b3)

# 反向（逆序）
dscores → Linear.backward → Linear_ReLU.backward → 
  reshape → MaxPool.backward → ReLU.backward → Conv.backward
```

### 2.8 DeepConvNet（VGG 风格）

```
{Conv(3×3) - [BatchNorm?] - ReLU - [MaxPool?]} × (L-1) → Linear → Softmax
```

- 所有卷积使用 3×3 核，padding=1 保持尺寸
- 池化层使用 2×2，stride=2 减半尺寸
- 通过 `max_pools` 列表指定哪些层后面接池化
- 支持可选的 BatchNorm

---

## 💡 重点与难点总结

### 重点

1. **模块化设计**：`forward` 保存 cache，`backward` 利用 cache 计算梯度——这是现代深度学习框架的核心设计模式
2. **Adam 优化器**：融合 Momentum + RMSProp + 偏差校正，是实践中最常用的优化器
3. **BatchNorm**：训练与测试行为不同（mini-batch 统计量 vs running 统计量）
4. **卷积输出尺寸公式**：$H' = (H + 2P - K) / S + 1$
5. **Kaiming 初始化**：深层网络的标配，保持信号方差一致

### 难点

1. **卷积反向传播**：理解 `dw`、`dx`、`db` 各自如何通过局部梯度和上游梯度计算
2. **BatchNorm 反向传播**：计算图复杂，需要跟踪 mean、var 的梯度贡献
3. **MaxPool 反向传播**：梯度只传回最大值位置的 "赢者通吃" 机制
4. **Dropout 的 Inverted 版本**：训练时缩放而非测试时缩放
5. **DeepConvNet 的灵活架构**：用循环和字典管理任意深度的前向/反向传播

### 优化器对比

| 优化器 | 公式核心 | 特点 |
| ------ | -------- | ---- |
| SGD | $w \leftarrow w - \alpha \nabla L$ | 最简单，容易震荡 |
| SGD + Momentum | 引入速度 $v$，惯性运动 | 加速收敛，减少震荡 |
| RMSProp | 自适应学习率（梯度平方的滑动平均） | 处理非平稳目标 |
| Adam | Momentum + RMSProp + 偏差校正 | 最常用，通常效果最好 |

### 正则化方法对比

| 方法 | 作用位置 | 机制 |
| ---- | -------- | ---- |
| L2 正则化 | 损失函数 | 惩罚大的权重值 |
| Dropout | 隐藏层输出 | 随机丢弃神经元，迫使冗余学习 |
| BatchNorm | 每层输入 | 归一化分布，轻微正则化效果 |

> 从 A2 的手动梯度计算，到 A3 的模块化层设计，核心转变在于：**不再关注整个网络的梯度推导，而是把注意力放在每个模块的局部梯度上**，然后通过链式法则自动组合。这正是 PyTorch、TensorFlow 等框架的设计哲学。

