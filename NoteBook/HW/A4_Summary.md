# A4 作业总结：目标检测 — FCOS & Faster R-CNN

---

## 📌 概览

A4 包含两大部分：

1. **单阶段检测器 (FCOS)** — Fully Convolutional One-Stage Detector，无锚框的密集预测
2. **双阶段检测器 (Faster R-CNN)** — Region Proposal Network + ROI 分类，经典的两阶段框架

核心公共模块：
- **FPN (Feature Pyramid Network)** — 多尺度特征金字塔
- **NMS (Non-Maximum Suppression)** — 去除重叠检测框

数据集：Pascal VOC（20类目标检测）

---

## 第零部分：公共模块（common.py）

### 0.1 知识点总结

| 知识点 | 说明 |
| ------ | ---- |
| DetectorBackboneWithFPN | RegNet backbone + FPN 多尺度特征提取 |
| FPN 横向连接 | 1×1 conv 对齐通道数 + 上采样融合 |
| FPN 位置坐标 | 特征图位置到图像绝对坐标的映射 |
| NMS | 非极大值抑制，去除冗余检测框 |

### 0.2 Feature Pyramid Network (FPN) 实现（⭐⭐ 核心）

#### 网络结构

```
Backbone (RegNet)
├── c3: stride=8  → lateral_c3 (1×1 conv) → m3 → output_p3 (3×3 conv) → p3
├── c4: stride=16 → lateral_c4 (1×1 conv) → m4 → output_p4 (3×3 conv) → p4
└── c5: stride=32 → lateral_c5 (1×1 conv) → m5 → output_p5 (3×3 conv) → p5
                                                ↗ upsample
                                           m5 ──────→ m4 = lateral_c4 + upsample(m5)
                                                       ↗ upsample
                                                  m4 ──────→ m3 = lateral_c3 + upsample(m4)
```

#### 自顶向下路径

```python
# P5：最高层，直接 1×1 conv
m5 = lateral_c5(c5)
p5 = output_p5(m5)

# P4：1×1 conv + 上采样融合
m4 = lateral_c4(c4) + F.interpolate(m5, size=c4.shape[-2:], mode="nearest")
p4 = output_p4(m4)

# P3：同理
m3 = lateral_c3(c3) + F.interpolate(m4, size=c3.shape[-2:], mode="nearest")
p3 = output_p3(m3)
```

**FPN 的意义**：
- 底层特征（p3）空间分辨率高，适合检测小目标
- 高层特征（p5）语义信息丰富，适合检测大目标
- 横向连接 + 自顶向下融合，让每一层都同时拥有强语义和高分辨率

### 0.3 FPN 位置坐标计算

```python
# 每个特征图位置映射到图像坐标（感受野中心）
xc = (shift_x + 0.5) * stride
yc = (shift_y + 0.5) * stride
locations = torch.stack((xc, yc), dim=-1).reshape(-1, 2)
```

**+0.5 的原因**：特征图的第 (0,0) 个位置对应的感受野中心在图像的 `(0.5*stride, 0.5*stride)` 处，而非原点。

### 0.4 NMS 实现（⭐ 重点）

```python
sorted_indices = torch.argsort(scores, descending=True)
keep_indices = []

while sorted_indices.numel() > 0:
    current = sorted_indices[0]
    keep_indices.append(current.item())
    
    if sorted_indices.numel() == 1:
        break
    
    remaining = sorted_indices[1:]
    remaining_boxes = boxes[remaining]
    current_box = boxes[current]
    
    # 计算 IoU
    x1 = torch.max(current_box[0], remaining_boxes[:, 0])
    y1 = torch.max(current_box[1], remaining_boxes[:, 1])
    x2 = torch.min(current_box[2], remaining_boxes[:, 2])
    y2 = torch.min(current_box[3], remaining_boxes[:, 3])
    
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    union = area_current + area_remaining - inter
    iou = inter / union
    
    # 保留 IoU ≤ 阈值的框
    sorted_indices = remaining[iou <= iou_threshold]
```

**NMS 算法流程**：
1. 按分数降序排列所有检测框
2. 取最高分的框加入结果集
3. 计算该框与所有剩余框的 IoU
4. 移除 IoU 超过阈值的框（被抑制）
5. 重复直到没有剩余框

---

## 第一部分：FCOS 单阶段检测器（one_stage_detector.py）

### 1.1 知识点总结

| 知识点 | 说明 |
| ------ | ---- |
| Anchor-free 检测 | 不使用预定义锚框，每个位置直接预测 |
| LTRB 回归 | 预测到 GT 框四条边的距离 (Left, Top, Right, Bottom) |
| Centerness | 衡量位置到目标中心的程度，抑制低质量预测 |
| Focal Loss | 解决正负样本不平衡问题 |
| 多级别匹配 | 不同大小的目标分配到不同 FPN 级别 |

### 1.2 FCOS 预测网络（FCOSPredictionNetwork）

```
FPN 特征 (p3/p4/p5)
    ├── stem_cls (Conv3×3 + ReLU) × N → pred_cls (Conv3×3) → 分类 logits (num_classes)
    └── stem_box (Conv3×3 + ReLU) × N → pred_box (Conv3×3) → 框回归 deltas (4)
                                       → pred_ctr (Conv3×3) → 中心度 logits (1)
```

- 分类和框回归使用**独立的 stem**（不同于 RPN 共享 stem）
- 中心度与框回归共享 stem（近期论文的做法）
- 权重初始化：`N(0, 0.01)`，分类偏置初始化为 `-log(99)` 以稳定训练

### 1.3 GT 匹配与目标计算

#### 位置到 GT 框的匹配

```python
# 每个 FPN 位置 (xc, yc) 必须满足：
# 1. 在 GT 框内部（LTRB 都 > 0）
# 2. 最大边距在该级别的尺度范围内：
#    lower_bound = stride × 4 (p3 为 0)
#    upper_bound = stride × 8 (p5 为 ∞)
# 3. 匹配面积最小的 GT 框（解决重叠）
```

#### LTRB Deltas 计算

```python
# 从位置到 GT 框边缘的距离，归一化到 stride
deltas = [xc - x0, yc - y0, x1 - xc, y1 - yc] / stride
# 背景位置设为 (-1, -1, -1, -1)
```

#### 反向：从 Deltas 恢复框坐标

```python
deltas = deltas * stride
deltas = deltas.clamp(min=0)  # 模型可能预测负值，需要裁剪
x1 = xc - l;  y1 = yc - t
x2 = xc + r;  y2 = yc + b
```

#### Centerness 计算（⭐ 重要）

$$\text{centerness} = \sqrt{\frac{\min(l, r)}{\max(l, r)} \times \frac{\min(t, b)}{\max(t, b)}}$$

- 中心位置：centerness ≈ 1
- 边缘位置：centerness ≈ 0
- 背景位置：centerness = -1

### 1.4 FCOS 损失函数

```python
# 1. 分类损失：Sigmoid Focal Loss
loss_cls = sigmoid_focal_loss(pred_cls_logits, gt_one_hot, alpha=0.25, gamma=2.0)

# 2. 框回归损失：L1 Loss（仅前景）
loss_box = F.l1_loss(pred_deltas, gt_deltas, reduction="none")
loss_box[background] = 0

# 3. 中心度损失：Binary Cross-Entropy（仅前景）
loss_ctr = F.binary_cross_entropy_with_logits(pred_ctr, gt_centerness)
loss_ctr[background] = 0

# 归一化：除以前景位置数量的 EMA
total_loss = (loss_cls + loss_box + loss_ctr) / normalizer
```

**Focal Loss 的作用**：

$$FL(p) = -\alpha (1 - p)^\gamma \log(p)$$

- 大量背景位置的分类损失很小（容易样本），被 $(1-p)^\gamma$ 下调
- 少量前景位置的分类损失较大（难样本），保持正常权重
- 有效解决了密集检测器中正负样本极度不平衡的问题

### 1.5 FCOS 推理流程

```python
for each FPN level:
    # 1. 计算置信度 = sqrt(class_prob × centerness)
    scores = sqrt(sigmoid(cls_logits) * sigmoid(ctr_logits))
    
    # 2. 取每个位置得分最高的类别
    scores, classes = scores.max(dim=1)
    
    # 3. 过滤低分预测（score_thresh）
    keep = scores > threshold
    
    # 4. 将 deltas 转换为框坐标
    boxes = apply_deltas_to_locations(deltas, locations, stride)
    
    # 5. 裁剪到图像边界

# 所有级别合并后执行 Class-Specific NMS
```

---

## 第二部分：Faster R-CNN 双阶段检测器（two_stage_detector.py）

### 2.1 知识点总结

| 知识点 | 说明 |
| ------ | ---- |
| RPN | Region Proposal Network，生成候选区域 |
| Anchor Boxes | 预定义的参考框，多尺度多比例 |
| IoU 匹配 | 基于 IoU 阈值分配前景/背景 |
| Box Deltas (dx, dy, dw, dh) | 相对于锚框的偏移和缩放 |
| ROI Align | 从特征图裁剪并对齐候选区域特征 |
| 两阶段设计 | 第一阶段提 proposals，第二阶段分类+精修 |

### 2.2 RPN 预测网络（RPNPredictionNetwork）

```
FPN 特征
  → stem_rpn (Conv3×3 + ReLU，共享) × N
    → pred_obj (1×1 Conv) → objectness logits (num_anchors)
    → pred_box (1×1 Conv) → box deltas (num_anchors × 4)
```

与 FCOS 的区别：
- RPN 的 stem 是**共享**的（objectness 和 box regression 共用）
- 使用 **1×1 conv** 而非 3×3 conv 做最终预测
- 每个位置有多个锚框（默认 A=3 种宽高比）

### 2.3 Anchor Box 生成

```python
# 对于每个 FPN 级别和每种宽高比
area = (stride_scale * stride) ** 2
new_width = sqrt(area / aspect_ratio)
new_height = area / new_width

# 以位置为中心生成 XYXY 锚框
x1 = xc - new_width / 2
y1 = yc - new_height / 2
x2 = xc + new_width / 2
y2 = yc + new_height / 2
```

**尺度规则**：
- P3 (stride=8)：锚框大小 ≈ 64×64
- P4 (stride=16)：锚框大小 ≈ 128×128
- P5 (stride=32)：锚框大小 ≈ 256×256
- 每个级别有 3 种宽高比：[0.5, 1.0, 2.0]

### 2.4 IoU 计算

```python
# 计算 M×N 对框的 IoU
# boxes1: (M, 4), boxes2: (N, 4)
x1 = max(boxes1[:, 0], boxes2[:, 0])  # 交集左上角
y1 = max(boxes1[:, 1], boxes2[:, 1])
x2 = min(boxes1[:, 2], boxes2[:, 2])  # 交集右下角
y2 = min(boxes1[:, 3], boxes2[:, 3])

intersection = clamp(x2 - x1, min=0) * clamp(y2 - y1, min=0)
union = area1 + area2 - intersection
iou = intersection / union
```

### 2.5 锚框匹配与 Delta 计算

#### 匹配规则

```python
# 计算每个锚框与所有 GT 框的 IoU
match_quality, matched_idxs = iou_matrix.max(dim=1)

# IoU > 0.6 → 前景（正样本）
# IoU ≤ 0.3 → 背景（负样本）
# 0.3 < IoU < 0.6 → 中性（忽略）
```

#### Box Delta 计算（⭐ 重点）

$$\begin{aligned}
d_x &= (x_{gt} - x_a) / w_a \\
d_y &= (y_{gt} - y_a) / h_a \\
d_w &= \log(w_{gt} / w_a) \\
d_h &= \log(h_{gt} / h_a)
\end{aligned}$$

- $(x_a, y_a, w_a, h_a)$：锚框的中心和尺寸
- $(x_{gt}, y_{gt}, w_{gt}, h_{gt})$：GT 框的中心和尺寸
- 平移量归一化到锚框尺寸，缩放量取对数

#### 反向：Delta → 框坐标

$$\begin{aligned}
x &= d_x \cdot w_a + x_a \\
y &= d_y \cdot h_a + y_a \\
w &= w_a \cdot e^{d_w} \\
h &= h_a \cdot e^{d_h}
\end{aligned}$$

### 2.6 RPN 训练

```python
# 1. 采样锚框（50% 前景 + 50% 背景）
fg_idx, bg_idx = sample_rpn_training(matched_gt_boxes, num_samples=256, fg_fraction=0.5)

# 2. Objectness Loss: Binary Cross-Entropy
loss_obj = F.binary_cross_entropy_with_logits(pred_obj[sampled], gt_labels[sampled])

# 3. Box Regression Loss: 仅前景锚框
loss_box = F.l1_loss(pred_deltas[fg_idx], gt_deltas[fg_idx])
# 背景锚框的 box loss = 0
```

### 2.7 RPN 推理：Proposal 生成

```python
for each FPN level:
    # 1. 用预测的 delta 变换锚框 → proposal boxes
    boxes = apply_deltas_to_anchors(pred_deltas, anchors)
    boxes = boxes.clamp(min=0, max=image_size)  # 裁剪
    
    # 2. 按 objectness 排序，取 top-K
    topk_scores, topk_indices = torch.topk(obj_logits, pre_nms_topk)
    
    # 3. 对 top-K proposals 执行 NMS
    keep = nms(topk_boxes, topk_scores, nms_thresh=0.7)
    proposals = topk_boxes[keep[:post_nms_topk]]
```

### 2.8 第二阶段：ROI 分类

```python
# 1. ROI Align：从 FPN 特征中裁剪 proposal 对应的特征
roi_feats = torchvision.ops.roi_align(fpn_feats, proposals, roi_size, aligned=True)

# 2. 分类头：Conv stem → Flatten → Linear → C+1 类 logits
pred_cls = cls_pred(roi_feats)  # 包含背景类

# 3. 训练：Cross-Entropy Loss
# 采样 25% 前景 + 75% 背景
# 类标签 +1（背景从 -1 变为 0），交叉熵损失

# 4. 推理：
# Softmax 获取概率 → 去掉背景 → 类标签 -1 还原 → NMS
```

### 2.9 训练稳定性技巧

```python
# 将 GT 框混入 RPN proposals，提高早期训练质量
proposals = mix_gt_with_proposals(proposals, gt_boxes)
# GT 框按面积分配到不同 FPN 级别（FPN 论文 Equation 1）
level = floor(5 + log2(sqrt(area) / 224))
```

---

## 💡 重点与难点总结

### 重点

1. **FPN 的自顶向下路径**：高层语义 + 低层细节的融合方式
2. **NMS 算法**：贪心策略逐步抑制重叠框
3. **FCOS vs Faster R-CNN**：无锚框 vs 基于锚框的检测范式
4. **Box Delta 编码/解码**：FCOS 用 LTRB，Faster R-CNN 用 (dx, dy, dw, dh)
5. **Focal Loss**：解决密集检测中的类别不平衡问题

### 难点

1. **多级别匹配**：不同大小的目标如何分配到不同的 FPN 级别
2. **正负样本采样**：RPN 的 50/50 采样 vs ROI 的 25/75 采样，为什么不同
3. **Centerness 的作用**：为什么需要它来抑制边缘位置的低质量预测
4. **训练稳定性**：分类偏置初始化 `-log(99)`、EMA normalizer、GT 混入 proposals
5. **Box Delta 的数值稳定性**：`dw/dh` 使用 log/exp，需要 clamp 防止溢出

### FCOS vs Faster R-CNN 对比

| 特性 | FCOS (单阶段) | Faster R-CNN (双阶段) |
| ---- | ------------- | -------------------- |
| 锚框 | ❌ 无锚框（anchor-free） | ✅ 有锚框（anchor-based） |
| 阶段数 | 1（直接从特征预测） | 2（RPN + ROI Head） |
| 框参数化 | LTRB deltas（到四边距离） | (dx, dy, dw, dh)（中心偏移+尺度缩放） |
| 分类损失 | Sigmoid Focal Loss | Cross-Entropy |
| 正负样本处理 | Focal Loss 自动调权 | 手动采样固定比例 |
| 额外预测 | Centerness | Objectness |
| 推理速度 | 较快（无 ROI 操作） | 较慢（需要 ROI Align） |
| 精度 | 接近 | 通常略高 |

### 检测流程对比

```
FCOS:
  Image → Backbone+FPN → Dense Predictions → NMS → Final Boxes

Faster R-CNN:
  Image → Backbone+FPN → RPN (Proposals) → ROI Align → Classification → NMS → Final Boxes
```

> A4 是从图像分类到目标检测的关键跨越。分类只需回答"图里有什么"，检测还需回答"在哪里"和"有多大"。FPN 解决了多尺度问题，NMS 解决了冗余检测问题，而 FCOS 和 Faster R-CNN 分别代表了无锚框和有锚框两种主流检测范式。

