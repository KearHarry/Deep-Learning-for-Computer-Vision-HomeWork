# Lecture 15：目标检测 (Object Detection)

---

## 📌 概览

本讲是计算机视觉最核心的应用之一——目标检测：

1. **问题定义** — 分类 + 定位
2. **两阶段检测器** — R-CNN 家族
3. **单阶段检测器** — YOLO、SSD、FCOS
4. **Feature Pyramid Network (FPN)** — 多尺度特征融合

---

## 第一部分：目标检测问题

### 1.1 从分类到检测

| 任务 | 输出 | 说明 |
| ---- | ---- | ---- |
| 图像分类 | 类别标签 | 图里有什么？ |
| 目标定位 | 类别 + 1个框 | 它在哪里？ |
| 目标检测 | 多个(类别 + 框) | 所有目标在哪里？ |
| 实例分割 | 多个(类别 + 像素级掩码) | 每个目标的精确轮廓 |

### 1.2 评估指标

- **IoU (Intersection over Union)**：$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$
- **AP (Average Precision)**：Precision-Recall 曲线下的面积
- **mAP (mean AP)**：所有类别 AP 的平均值

---

## 第二部分：两阶段检测器（⭐⭐ 核心）

### 2.1 R-CNN (2014)

```
输入图像
  → Selective Search 提取 ~2000 个候选区域 (Region Proposals)
  → 对每个区域：Resize → CNN 提取特征 → SVM 分类 + 框回归
```

**问题**：每个 proposal 独立过 CNN，极其缓慢。

### 2.2 Fast R-CNN (2015)

```
输入图像
  → CNN 提取整张图的特征图（只过一次 CNN！）
  → Selective Search 提取 proposals
  → ROI Pooling：从特征图上裁剪每个 proposal 的特征
  → FC 层 → 分类 + 框回归
```

**改进**：共享 CNN 计算，速度大幅提升。

### 2.3 Faster R-CNN (2015)（⭐ 里程碑）

```
输入图像
  → Backbone CNN → 特征图
  → RPN (Region Proposal Network) 生成 proposals
  → ROI Align：从特征图裁剪 proposal 特征
  → 分类头 + 框回归头
```

**核心创新**：用**可学习的 RPN** 替代手工的 Selective Search，整个检测器端到端训练。

#### RPN 的工作方式

- 在特征图的每个位置放置多个**锚框 (Anchors)**
- 对每个锚框预测：是否包含目标（objectness）+ 框偏移量
- 通过 NMS 筛选出高质量 proposals

#### ROI Align（⭐ 重要）

ROI Pooling 的改进版，使用双线性插值避免量化误差：

```python
# 将任意大小的 ROI 映射到固定大小 (如 7×7) 的特征
roi_feats = torchvision.ops.roi_align(
    feature_map, proposals, output_size=(7, 7), aligned=True
)
```

### 2.4 Box Parameterization

Faster R-CNN 不直接预测框坐标，而是预测相对于锚框的**偏移量**：

$$t_x = (x - x_a) / w_a, \quad t_y = (y - y_a) / h_a$$
$$t_w = \log(w / w_a), \quad t_h = \log(h / h_a)$$

- 平移量归一化到锚框尺寸
- 缩放量取对数（保证正值 + 数值稳定）

---

## 第三部分：单阶段检测器

### 3.1 YOLO (You Only Look Once)

```
输入图像
  → CNN → S×S 网格
  → 每个格子预测 B 个框 + C 个类别概率
  → NMS → 最终检测结果
```

**核心思想**：不需要 proposal 阶段，一次前向传播直接预测所有框。

### 3.2 SSD (Single Shot MultiBox Detector)

在**多个特征图尺度**上做预测（类似 FPN 的思想），小特征图检测大目标，大特征图检测小目标。

### 3.3 FCOS (Fully Convolutional One-Stage)（⭐ A4 作业）

**Anchor-free** 方法：
- 不使用预定义锚框
- 每个特征图位置直接预测到 GT 框四条边的距离 (LTRB)
- 额外预测 centerness 来抑制低质量预测
- 使用 Focal Loss 处理正负样本不平衡

---

## 第四部分：Feature Pyramid Network (FPN)（⭐⭐ 核心）

### 4.1 多尺度检测的挑战

- 小目标在高层特征图上可能只有 1-2 个像素
- 大目标在低层特征图上没有足够的语义信息

### 4.2 FPN 的设计

```
Backbone 金字塔（自底向上）:
  C2(大) → C3 → C4 → C5(小)

FPN 金字塔（自顶向下 + 横向连接）:
  P5(小) → 上采样 + C4横向 → P4 → 上采样 + C3横向 → P3(大)
```

每个级别都同时拥有：
- **强语义**（来自高层的自顶向下路径）
- **高分辨率**（来自低层的横向连接）

### 4.3 FPN 的目标分配

不同大小的目标分配到不同的 FPN 级别：
- P3 (stride=8)：小目标
- P4 (stride=16)：中等目标
- P5 (stride=32)：大目标

---

## 💡 重点总结

1. **两阶段 vs 单阶段**：精度 vs 速度的权衡
2. **Faster R-CNN = Backbone + RPN + ROI Head**：端到端可训练的两阶段检测器
3. **FPN 是现代检测器的标配**：多尺度特征融合解决大小目标检测问题
4. **Anchor-based vs Anchor-free**：Faster R-CNN 用锚框，FCOS 不用
5. **NMS 是后处理必需步骤**：去除冗余的重叠检测框
6. **Box 参数化**：预测偏移量而非绝对坐标，更容易学习

