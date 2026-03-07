# A1 作业总结：PyTorch 基础 & K-近邻分类器

---

## 📌 概览

A1 包含两个部分：
1. **PyTorch 101** — PyTorch 张量操作基础
2. **KNN 分类器** — 基于 K-近邻算法在 CIFAR-10 数据集上的图像分类

---

## 第一部分：PyTorch 101（pytorch101.py）

### 1.1 知识点总结

| 知识点 | 说明 |
|--------|------|
| 张量创建 | `torch.zeros()`, `torch.randn()`, `torch.tensor()` |
| 张量索引 | 切片索引、整数数组索引、布尔索引 |
| 张量变形 | `view()`, `reshape()`, `permute()`, `contiguous()` |
| 广播机制 | Broadcasting 规则，自动维度扩展 |
| 归约操作 | `sum()`, `mean()`, `min()`, `max()`, `argmax()` |
| 矩阵运算 | `mm()`, `bmm()`, 批量矩阵乘法 |
| GPU 计算 | `.cuda()`, `.cpu()` 设备间数据迁移 |
| One-hot 编码 | 利用高级索引实现向量化的 one-hot |

### 1.2 具体实现与关键代码

#### 1.2.1 张量创建与修改

```python
# 创建指定形状的零张量并赋值
x = torch.zeros(3, 2)
x[0][1] = 10
x[1][0] = 100

# 批量修改张量元素
for i in range(len(values)):
    x[indices[i][0], indices[i][1]] = values[i]

# 计算张量元素总数（不使用 numel）
num_elements = 1
for i in x.shape:
    num_elements *= i
```

#### 1.2.2 切片索引（⭐ 重点）

```python
# 获取最后一行（降维为1D）
last_row = x[-1, :]

# 获取第三列（保持2D，注意 2:3 而非 2）
third_col = x[:, 2:3]

# 偶数行、奇数列（步长索引）
even_rows_odd_cols = x[::2, 1::2]
```

**难点**：理解切片 `x[:, 2:3]` 与 `x[:, 2]` 的区别 — 前者保留维度(M,1)，后者降维为(M,)。

#### 1.2.3 切片赋值

```python
# 用不超过6次切片操作构造目标矩阵
x[0:2, 0:1] = 0
x[0:2, 1:2] = 1
x[0:2, 2:6] = 2
x[2:4, 0:3] = 3
x[2:4, 1:4:2] = 4  # 步长为2的切片赋值
x[2:4, 4:6] = 5
```

**难点**：用步长切片 `1:4:2` 精确选中索引1和3的列。

#### 1.2.4 整数数组索引（⭐ 重点）

```python
# 列重排：使用整数数组索引一步完成
y = x[:, [0, 0, 2, 1]]

# 行反转：构造倒序索引
y = x[torch.arange(x.shape[0]-1, -1, -1), :]

# 从每列取特定元素：行索引列表 + 列索引列表
y = x[[1, 0, 3], [0, 1, 2]]
```

#### 1.2.5 One-Hot 编码（⭐ 难点）

```python
# 不使用循环实现 one-hot 编码
y = torch.zeros(len(x), max(x)+1, dtype=torch.float32)
y[torch.arange(len(x)), x] = 1
```

**关键技巧**：利用 `torch.arange` 生成行索引，配合整数列表 `x` 作为列索引，实现向量化赋值。这是 PyTorch 高级索引的经典应用。

#### 1.2.6 布尔索引与条件求和

```python
# 求正数之和（不使用循环）
pos_sum = x[x > 0].sum().item()
```

#### 1.2.7 Reshape 实践（⭐ 难点）

```python
# 将 (24,) 变为 (3,8)，且元素排列有特殊要求
medium = x[:12].reshape(3, 4)
y = torch.cat((medium, x[12:].reshape(3, 4)), dim=1)
```

**难点**：需要理解目标排列规律 — 前12个元素和后12个元素各 reshape 为 (3,4) 后拼接。

#### 1.2.8 行最小值置零

```python
y = x.clone()
_, row_min_idxs = x.min(dim=1)
y[torch.arange(x.shape[0]), row_min_idxs] = 0
```

**技巧**：`min(dim=1)` 返回值和索引，配合 arange 高级索引精确定位。

#### 1.2.9 批量矩阵乘法

```python
# 使用循环版本
for i in range(B):
    z[i] = x[i].mm(y[i])

# 使用 torch.bmm（无循环版本）
z = torch.bmm(x, y)
```

#### 1.2.10 列标准化（⭐ 难点）

```python
# 不使用 torch.mean/torch.std 实现列标准化
M, N = x.shape
col_sum = x.sum(dim=0)
col_mean = col_sum / M
col_var = ((x - col_mean) ** 2).sum(dim=0)
col_std = torch.sqrt(col_var / (M - 1))  # 注意：使用无偏估计 (M-1)
y = (x - col_mean) / col_std
```

**难点**：
- 广播机制：`x - col_mean` 中 col_mean 形状 (N,) 会自动广播到 (M,N)
- 标准差使用贝塞尔校正 (Bessel's correction)，除以 `M-1` 而非 `M`

#### 1.2.11 GPU 计算

```python
x_gpu = x.cuda()
w_gpu = w.cuda()
y_gpu = x_gpu.mm(w_gpu)
y = y_gpu.cpu()
```

#### 1.2.12 挑战题：无循环求均值

```python
# 利用 cumsum 技巧，不使用循环对变长张量列表求均值
flat_x = torch.cat(xs)
x_cumsum = torch.cumsum(flat_x.float(), dim=0)
end_indices = torch.cumsum(ls, dim=0) - 1
pass_sums = x_cumsum[end_indices]
chunk_sums = pass_sums.clone()
chunk_sums[1:] = pass_sums[1:] - pass_sums[:-1]
y = chunk_sums / ls
```

**核心思想**：先拼接所有张量，通过前缀和 (cumulative sum) + 差分，在 O(N) 时间内计算每个子张量的均值。

#### 1.2.13 挑战题：无循环获取唯一值

```python
sorted_x, sort_idx = x.sort(stable=True)
unique_mask = torch.cat([
    torch.tensor([True], device=x.device, dtype=torch.bool),
    sorted_x[1:] != sorted_x[:-1]
])
uniques = sorted_x[unique_mask]
indices = sort_idx[unique_mask]
```

**核心思想**：稳定排序后，相邻不等的元素即为唯一值，对应的原始索引即为首次出现位置。

---

## 第二部分：KNN 分类器（knn.py）

### 2.1 知识点总结

| 知识点 | 说明 |
|--------|------|
| K-近邻算法 | 基于距离的非参数分类方法 |
| 欧氏距离计算 | 三种实现方式（双循环/单循环/无循环） |
| 向量化编程 | 消除 Python 循环，利用矩阵运算加速 |
| 交叉验证 | K-fold cross validation 选择超参数 |
| CIFAR-10 数据集 | 10 类彩色图像分类基准 |

### 2.2 KNN 算法核心流程

```
训练阶段：仅存储训练数据（懒学习）
预测阶段：
  1. 计算测试样本与所有训练样本的距离
  2. 找到 K 个最近邻
  3. 多数投票确定类别
```

### 2.3 距离计算（⭐⭐ 核心难点）

#### 双循环版本（朴素实现）

```python
x_flatten = x_train.view(num_train, -1)
x_test_flatten = x_test.view(num_test, -1)
for i in range(num_train):
    for j in range(num_test):
        dists[i, j] = torch.sum((x_flatten[i] - x_test_flatten[j]) ** 2)
```

#### 单循环版本（部分向量化）

```python
for i in range(num_train):
    # 广播：(D,) - (num_test, D) -> (num_test, D)
    dists[i] = torch.sum((x_flatten[i] - x_test_flatten) ** 2, dim=1)
```

#### 无循环版本（⭐⭐ 完全向量化，最重要）

```python
x_train_square = torch.sum(x_flatten ** 2, dim=1, keepdim=True)    # (N_train, 1)
x_test_square = torch.sum(x_test_flatten ** 2, dim=1, keepdim=True) # (N_test, 1)
cross_term = torch.mm(x_flatten, x_test_flatten.t())                # (N_train, N_test)
dists = x_train_square - 2 * cross_term + x_test_square.t()
```

**数学推导**（⭐⭐ 必须理解）：

$$\|a - b\|^2 = \|a\|^2 - 2a^Tb + \|b\|^2$$

展开后：
- $\|a\|^2$：每个训练样本的平方和 → `(N_train, 1)` 广播
- $\|b\|^2$：每个测试样本的平方和 → `(1, N_test)` 广播
- $a^Tb$：矩阵乘法 → `(N_train, N_test)`

**难点**：
1. `keepdim=True` 的作用：保持维度以便广播
2. 不能创建 O(N_train × N_test × D) 的中间张量
3. 利用矩阵乘法代替逐元素运算

### 2.4 标签预测

```python
for j in range(num_test):
    _, topk_indices = torch.topk(dists[:, j], k, largest=False)
    topk_labels = y_train[topk_indices]
    y_pred[j] = torch.bincount(topk_labels).argmax()
```

**关键 API**：
- `torch.topk()`：找到最大/最小的 k 个元素
- `torch.bincount()`：统计每个整数出现次数
- `.argmax()`：返回最大值的索引（平局时自动选最小标签）

### 2.5 KNN 分类器封装

```python
class KnnClassifier:
    def __init__(self, x_train, y_train):
        self.train_data = x_train      # 记忆训练数据
        self.train_labels = y_train

    def predict(self, x_test, k=1):
        dists = compute_distances_no_loops(self.train_data, x_test)
        return predict_labels(dists, self.train_labels, k=k)
```

### 2.6 K-折交叉验证（⭐ 重点）

```python
# 数据划分
x_train_folds = torch.chunk(x_train, num_folds)
y_train_folds = torch.chunk(y_train, num_folds)

# 交叉验证
for k in k_choices:
    accuracies = []
    for i in range(num_folds):
        x_val_fold = x_train_folds[i]
        y_val_fold = y_train_folds[i]
        x_train_fold = torch.cat(x_train_folds[:i] + x_train_folds[i+1:])
        y_train_fold = torch.cat(y_train_folds[:i] + y_train_folds[i+1:])
        classifier = KnnClassifier(x_train_fold, y_train_fold)
        accuracy = classifier.check_accuracy(x_val_fold, y_val_fold, k=k)
        accuracies.append(accuracy)
    k_to_accuracies[k] = accuracies
```

**流程**：
1. 将训练数据均分为 `num_folds` 份
2. 对每个 k 值，轮流选一份作验证集，其余作训练集
3. 计算每折的准确率，取平均作为该 k 值的性能指标

### 2.7 选择最佳 K 值

```python
best_k = max(k_to_accuracies,
             key=lambda k: (torch.tensor(k_to_accuracies[k]).mean(), -k))
```

**要点**：平均准确率最高的 k 值；若平局则选最小的 k。

---

## 💡 重点与难点总结

### 重点
1. **PyTorch 张量操作**：索引、切片、reshape 是所有后续作业的基础
2. **向量化编程思想**：用矩阵运算替代 Python 循环，大幅提升速度
3. **距离计算的平方展开公式**：$\|a-b\|^2 = \|a\|^2 - 2a^Tb + \|b\|^2$
4. **交叉验证**：标准的超参数选择流程

### 难点
1. **高级索引**：整数数组索引 `x[[rows], [cols]]` 的正确使用
2. **广播机制**：理解不同形状张量间的自动扩展规则
3. **无循环距离计算**：平方展开技巧 + 避免大中间张量
4. **前缀和技巧**：用 cumsum 实现变长子张量的无循环求均值
5. **数值计算**：标准差使用贝塞尔校正（除以 N-1）

### 性能对比
| 实现方式 | 时间复杂度 | 实际速度 |
|----------|-----------|---------|
| 双循环 | O(N×M×D) | 最慢 |
| 单循环 | O(N×M×D) | 中等 |
| 无循环 | O(N×M×D) | 最快（利用底层优化） |

> 三者时间复杂度相同，但无循环版本利用了 PyTorch/BLAS 的底层矩阵运算优化，速度可快数十倍。
