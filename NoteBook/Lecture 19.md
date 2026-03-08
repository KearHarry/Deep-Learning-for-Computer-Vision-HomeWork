# Lecture 19：生成模型（二）— 生成对抗网络 (GAN)

---

## 📌 概览

本讲介绍最具影响力的生成模型之一——GAN：

1. **GAN 的核心思想** — 博弈论视角
2. **训练过程** — 判别器与生成器的对抗
3. **训练挑战** — 模式坍塌、训练不稳定
4. **经典 GAN 变体** — DCGAN、WGAN、StyleGAN

---

## 第一部分：GAN 的核心思想（⭐⭐ 核心）

### 1.1 博弈框架

GAN 由两个网络组成：

- **生成器 G**：从随机噪声 $z$ 生成假图像，试图欺骗判别器
- **判别器 D**：区分真假图像，试图不被欺骗

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

### 1.2 直觉理解

> G 是造假币的人，D 是验钞机。G 不断提高造假技术，D 不断提高鉴别能力。最终 G 造的假币连 D 都分辨不出来。

### 1.3 训练流程

```python
for each iteration:
    # 1. 训练判别器 D（最大化）
    real_images = sample_from_dataset()
    fake_images = G(random_noise())
    loss_D = -[log(D(real_images)) + log(1 - D(fake_images))]
    update D to minimize loss_D
    
    # 2. 训练生成器 G（最小化）
    fake_images = G(random_noise())
    loss_G = -log(D(fake_images))  # 非饱和版本
    update G to minimize loss_G
```

### 1.4 生成器目标的两种形式

**原始版本**：$\min_G \mathbb{E}[\log(1 - D(G(z)))]$

- 当 G 很差时，$D(G(z)) \approx 0$，$\log(1 - 0) \approx 0$，梯度很小
- 训练初期梯度消失

**非饱和版本**（⭐ 实际使用）：$\max_G \mathbb{E}[\log D(G(z))]$

- 当 G 很差时，$\log(D(G(z)))$ 的梯度更大
- 提供更强的学习信号

---

## 第二部分：训练挑战

### 2.1 模式坍塌 (Mode Collapse)

**现象**：G 只生成少数几种样本，而非覆盖数据分布的全部模式。

例如：训练集有 10 种数字，但 G 只学会生成 "1" 和 "7"。

**原因**：G 找到了少数能骗过 D 的"安全样本"，不愿意探索其他模式。

### 2.2 训练不稳定

- D 太强 → G 得不到有效梯度（D 总是输出 0）
- D 太弱 → G 收不到有意义的反馈
- 需要精心平衡 D 和 G 的训练节奏

### 2.3 评估困难

GAN 没有显式的似然函数，无法直接量化生成质量。常用替代指标：

| 指标 | 说明 |
| ---- | ---- |
| IS (Inception Score) | 生成图像的多样性和清晰度 |
| FID (Fréchet Inception Distance) | 生成分布与真实分布的距离（越小越好） |

---

## 第三部分：经典 GAN 变体

### 3.1 DCGAN (Deep Convolutional GAN)

将 GAN 的 G 和 D 替换为卷积网络的最佳实践：

**生成器规则**：
- 用转置卷积进行上采样（不用全连接层）
- 使用 BatchNorm（除了最后一层）
- 使用 ReLU 激活（最后一层用 Tanh）

**判别器规则**：
- 用步长卷积进行下采样（不用池化）
- 使用 BatchNorm（除了第一层）
- 使用 LeakyReLU 激活

### 3.2 WGAN (Wasserstein GAN)

用 Wasserstein 距离替代原始 GAN 的 JS 散度：

$$\min_G \max_{D \in \text{1-Lip}} \; \mathbb{E}_{x}[D(x)] - \mathbb{E}_{z}[D(G(z))]$$

- D 的输出不再经过 sigmoid（不是概率，而是"分数"）
- D 必须满足 Lipschitz 约束（通过权重裁剪或梯度惩罚实现）
- 训练更稳定，loss 曲线与生成质量相关

### 3.3 StyleGAN

- 使用**映射网络**将 $z$ 转换为"风格"向量 $w$
- 在生成器的每一层注入不同的风格（通过 AdaIN）
- 不同层控制不同粒度的特征（粗糙结构 vs 细节纹理）
- 生成极其逼真的人脸图像

### 3.4 条件 GAN (Conditional GAN)

给 G 和 D 额外的条件信息（如类别标签）：

$$G(z, c) \rightarrow \text{生成类别 } c \text{ 的图像}$$

应用：文本到图像 (Text-to-Image)、图像到图像翻译 (Pix2Pix)

---

## 第四部分：GAN vs VAE

| 特性 | VAE | GAN |
| ---- | --- | --- |
| 训练方式 | 最大化似然下界 | 对抗博弈 |
| 生成质量 | 偏模糊 | 清晰锐利 |
| 训练稳定性 | ✅ 稳定 | ⚠️ 不稳定 |
| 模式覆盖 | ✅ 较好 | ❌ 模式坍塌风险 |
| 似然估计 | ✅ 可以 | ❌ 不行 |
| 隐空间 | 连续、规整 | 不一定规整 |

---

## 💡 重点总结

1. **GAN = 生成器 vs 判别器的博弈**，纳什均衡时 G 生成的分布 = 真实分布
2. **非饱和损失**是实际训练中使用的目标，避免早期梯度消失
3. **模式坍塌**是 GAN 最大的问题：G 只学会少数模式
4. **WGAN** 通过 Wasserstein 距离大幅提升训练稳定性
5. **FID** 是评估 GAN 生成质量的标准指标（越低越好）
6. **StyleGAN** 代表了 GAN 在人脸生成领域的巅峰

