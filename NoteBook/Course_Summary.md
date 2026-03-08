# Deep Learning for Computer Vision — 课程总结

---

## 📌 课程概览

本课程（EECS 498-007/598-005）系统地介绍了深度学习在计算机视觉中的应用，从基础理论到前沿方法，覆盖了以下主要模块：

```
基础模块 (Lecture 1-9)
├── 课程导论与计算机视觉概述
├── KNN 与图像分类
├── 线性分类器（SVM / Softmax）
├── 正则化与优化（SGD, Adam）
├── 神经网络基础
├── 反向传播与计算图
├── 卷积神经网络 (CNN)
├── CNN 经典架构（AlexNet → ResNet）
└── 训练技巧（激活函数、初始化、BatchNorm）

进阶模块 (Lecture 10-15)
├── 训练技巧（学习率调度、数据增强、迁移学习）
├── CNN 现代架构（ResNeXt, MobileNet, NAS）
├── 循环神经网络 (RNN / LSTM)
├── 注意力机制 (Attention)
├── Transformer & Vision Transformer
└── 目标检测（Faster R-CNN, FCOS, FPN）

应用模块 (Lecture 16-22)
├── 图像分割（FCN, U-Net, Mask R-CNN）
├── 视频理解（3D CNN, SlowFast）
├── 生成模型 — VAE
├── 生成模型 — GAN
├── 网络可视化与可解释性
├── 自监督学习（SimCLR, MoCo）
└── 3D 视觉（点云, 深度估计）
```

---

## 🔗 核心知识链条

### 从分类到生成的完整路径

```
图像分类 (Lecture 2-4)
  → 线性分类器: f(x) = Wx
  → 损失函数: SVM Hinge / Softmax Cross-Entropy
  → 优化: SGD → Momentum → Adam
  ↓
神经网络 (Lecture 5-6)
  → 非线性激活突破线性限制
  → 反向传播高效计算梯度
  → 计算图 → 自动微分框架
  ↓
卷积神经网络 (Lecture 7-11)
  → 卷积: 局部连接 + 权重共享 + 平移等变
  → 架构演进: AlexNet → VGG → GoogLeNet → ResNet
  → 残差连接解决深度网络的梯度传播
  ↓
序列模型 (Lecture 12-14)
  → RNN: 隐藏状态传递时序信息
  → LSTM: 门控机制解决长期依赖
  → Attention: 动态聚焦重要信息
  → Transformer: 纯注意力架构，取代 RNN/CNN
  ↓
视觉应用 (Lecture 15-17)
  → 目标检测: 分类 + 定位（Faster R-CNN, FCOS）
  → 图像分割: 像素级分类（U-Net, Mask R-CNN）
  → 视频理解: 时空建模（3D CNN, SlowFast）
  ↓
生成与理解 (Lecture 18-22)
  → VAE: 学习数据的概率分布
  → GAN: 对抗训练生成逼真图像
  → 可视化: 理解网络学到了什么
  → 自监督: 无标注数据的表示学习
  → 3D 视觉: 从 2D 走向 3D 世界
```

---

## ⭐ 每讲核心知识点速查

| Lecture | 主题 | 一句话核心 |
| ------- | ---- | ---------- |
| 1 | 课程导论 | 深度学习 = 数据驱动的特征学习，取代手工特征工程 |
| 2 | KNN | 基于距离的非参数分类；$\|a-b\|^2 = \|a\|^2 - 2a^Tb + \|b\|^2$ |
| 3 | 线性分类器 | $f(x) = Wx + b$；SVM 关注边界，Softmax 输出概率 |
| 4 | 优化 | 梯度下降沿负梯度方向更新；SGD → Momentum → Adam |
| 5 | 神经网络 | 非线性激活 (ReLU) 打破线性限制；通用近似定理 |
| 6 | 反向传播 | 计算图 + 链式法则；上游梯度 × 局部梯度 = 下游梯度 |
| 7 | CNN | 卷积 = 局部连接 + 权重共享；$H' = (H+2P-K)/S + 1$ |
| 8 | CNN 架构 (一) | AlexNet → VGG → GoogLeNet → ResNet；残差连接是关键 |
| 9 | 训练技巧 (一) | ReLU 是默认激活；Kaiming 初始化适配 ReLU |
| 10 | 训练技巧 (二) | Cosine LR + Warmup；迁移学习是默认选择 |
| 11 | CNN 架构 (二) | Depthwise Separable Conv (MobileNet)；NAS 自动搜索 |
| 12 | RNN / LSTM | LSTM 的门控 + 加法更新解决梯度消失 |
| 13 | Attention | QKV 框架；Multi-Head 并行捕捉多种关系 |
| 14 | Transformer | Self-Attention + FFN + LayerNorm；ViT 将图像切为 patch |
| 15 | 目标检测 | FPN 多尺度特征；Faster R-CNN (两阶段) vs FCOS (单阶段) |
| 16 | 图像分割 | FCN 全卷积；U-Net 编码器-解码器 + Skip Connection |
| 17 | 视频理解 | 3D 卷积建模时空；SlowFast 双路径设计 |
| 18 | VAE | 重建损失 + KL 散度；重参数化技巧使采样可微 |
| 19 | GAN | 生成器 vs 判别器博弈；模式坍塌是主要挑战 |
| 20 | 可视化 | Grad-CAM 类别热力图；对抗样本揭示网络脆弱性 |
| 21 | 自监督学习 | 对比学习拉近正样本推开负样本；数据增强是关键 |
| 22 | 3D 视觉 | PointNet 处理无序点云；隐式函数突破分辨率限制 |

---

## 🔑 贯穿全课程的核心思想

### 1. 数据驱动 (Data-Driven)

从 KNN 到 Transformer，所有方法都依赖数据而非手工规则。数据量和质量决定了模型的上限。

### 2. 端到端学习 (End-to-End Learning)

从原始输入到最终输出，中间所有步骤都可微分、可优化。避免了手工特征工程的瓶颈。

### 3. 梯度是一切的基础

- 反向传播计算梯度
- 梯度下降优化参数
- 梯度消失/爆炸限制了网络深度（ResNet 和 LSTM 用加法跳跃连接解决）
- 梯度可视化帮助理解模型

### 4. 归纳偏置 (Inductive Bias) 的权衡

| 模型 | 归纳偏置 | 数据效率 | 灵活性 |
| ---- | -------- | -------- | ------ |
| CNN | 局部性 + 平移不变性 | 高 | 受限 |
| RNN | 时序依赖 | 中 | 中 |
| Transformer | 几乎无 | 低（需大数据） | 极高 |

### 5. 从特定到通用

```
特定任务的模型 (AlexNet, VGG)
  → 通用骨架 + 任务头 (ResNet + FPN + 检测头/分割头)
  → 预训练 + 微调 (ImageNet → 下游任务)
  → 自监督预训练 (SimCLR, MoCo)
  → Foundation Model (ViT 大规模预训练)
```

---

## 📊 作业与课程对应关系

| 作业 | 对应 Lecture | 核心内容 |
| ---- | ----------- | -------- |
| A1 | Lecture 1-2 | PyTorch 基础、KNN 分类器 |
| A2 | Lecture 3-5 | SVM/Softmax 线性分类器、两层神经网络 |
| A3 | Lecture 6-9 | 模块化反向传播、CNN、BatchNorm、优化器 |
| A4 | Lecture 15 | 目标检测（FCOS、Faster R-CNN、FPN、NMS） |
| A5 | Lecture 12-14 | RNN/LSTM 图像描述、Transformer |
| A6 | Lecture 18-20 | VAE、GAN、网络可视化、风格迁移 |

