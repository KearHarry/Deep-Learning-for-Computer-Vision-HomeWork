# Lecture 2

##    K-Nearest Neighbor

KNN 的核心思想一句话就能总结：**“物以类聚，人以群分。”**

当你给它一张新图片（测试集）让它分类时，它的操作流程如下：

1. **计算距离**：拿这张新图片，去和库里**所有**已知的图片（训练集）逐一计算“距离”。
2. **寻找邻居**：找出距离最近的 **K** 张图片。
3. **投票决策**：这 K 个邻居中，哪个类别最多，新图片就属于哪个类别

在图像分类中，我们把图片看作是一个巨大的高维向量（每一个像素点都是一个维度）。课程中重点讲了两种计算距离的方式：

### L1 距离 (Manhattan Distance)

直接把两个向量对应位置的数值相减，取绝对值，然后求和。

$$d_1(I_1, I_2) = \sum_{p} |I_1^p - I_2^p|$$

- **直观理解**：就像在曼哈顿街区开车，你只能沿水平或垂直方向走。
- **特点**：它与坐标轴的选择有关。如果你旋转了坐标轴，L1 距离会改变。

### L2 距离 (Euclidean Distance)

对应位置相减后取平方，求和，再开根号。

$$d_2(I_1, I_2) = \sqrt{\sum_{p} (I_1^p - I_2^p)^2}$$

- **直观理解**：这就是两点之间的“直线距离”。
- **特点**：它是**旋转不变**的。无论你如何旋转坐标轴，两点间的 L2 距离都不会变。

## 交叉验证（Cross-validation）

### 什么是 K-折交叉验证 (K-Fold CV)？

在数据量比较小的时候（比如传统的机器学习任务），如果只分出一个固定的验证集，可能会因为运气不好，分出的那部分数据太简单或太难，导致评估不准。

**K-折交叉验证**的操作如下：

1. 把训练数据平均分成 $K$ 份（注意：这里的 $K$ 和 KNN 的 $K$ 不是一个意思，通常取 $5$ 或 $10$）。
2. **轮流坐庄**：进行 $K$ 次循环。每次循环中，取其中 1 份作为“验证集”，剩下的 $K-1$ 份合并作为“训练集”。
3. **计算平均分**：把这 $K$ 次模拟考的准确率加起来取平均值。

> **举个例子**：
>
> 如果你做 5-折交叉验证（5-Fold）：
>
> - 第一次：用第 1,2,3,4 份练，第 5 份考。
> - 第二次：用第 1,2,3,5 份练，第 4 份考。
> - ...以此类推。
> - 最后你看 $K=3$ 在这五次考试中的平均分是多少。



总结：交叉验证的核心目的就是**为了选出最优的超参数**（比如 KNN 里的 $K$ 值和距离度量公式），并确保这个选择在未知数据上也是有效的，而不是“碰巧”在某堆数据上表现好。



```python
import numpy as np
from collections import Counter

class NearestNeighbor:
    def __init__(self, k=3, dist_metric='l2'):
        self.k = k
        self.dist_metric = dist_metric
def train(self, X, y):
    """
    KNN 的训练其实就是“死记硬背”
    X: 训练数据 (num_train, D)
    y: 训练标签 (num_train,)
    """
    self.X_train = X
    self.y_train = y

def predict(self, X_test):
    """
    预测过程：计算距离 -> 找最近的 K 个 -> 投票
    """
    num_test = X_test.shape[0]
    y_pred = np.zeros(num_test, dtype=self.y_train.dtype)

    for i in range(num_test):
        # 1. 计算当前测试图片与所有训练图片的距离
        if self.dist_metric == 'l1':
            # L1 距离: sum(|x1 - x2|)
            distances = np.sum(np.abs(self.X_train - X_test[i, :]), axis=1)
        else:
            # L2 距离: sqrt(sum((x1 - x2)^2))
            distances = np.sqrt(np.sum(np.square(self.X_train - X_test[i, :]), axis=1))

        # 2. 找到距离最近的 K 个邻居的索引
        # np.argsort 会返回排序后的索引
        closest_indices = np.argsort(distances)[:self.k]
        
        # 3. 获取这些邻居的标签
        closest_y = self.y_train[closest_indices]

        # 4. 投票：看哪个标签出现次数最多
        vote = Counter(closest_y).most_common(1)
        y_pred[i] = vote[0][0]

    return y_pred
```

