# SenseNova-U1 论文解析：原生统一多模态理解与生成范式

## ✦ 核心洞察与挑战

### 核心问题
- **问题一**：传统 VLM 在视觉编码器（VE）和语言模型之间存在表征割裂——VE 负责理解，VAE 负责生成，两套目标函数、两套训练流程导致理解与生成的特征空间严重不对齐。
- **问题二**：现有统一多模态模型（UMM）虽然在架构上追求统一，但仍然依赖离散 tokenizer、扩散解码器或辅助桥接模块，而非真正的端到端原生学习。
> **关键判断**：SenseNova-U1 的核心主张是——理解与生成不应是"翻译"关系，而应是同一个原生多模态过程的两个协同视角。

### 传统方案局限
- **局限一**：VE + LLM 架构继承预训练语义偏差，且理解与生成共享同一套表示但优化目标冲突。
- **局限二**：离散 token 化方法（Chameleon、Emu3 等）压缩视觉信号严重，视觉保真度受限；连续表征方法（扩散 VAE）则面临语义抽象与像素粒度之间的权衡。
- **局限三**：拼接式两阶段训练（先预训练 VE，再微调生成）无法让两个能力协同进化。
> **关键判断**：作者认为这些局限不是工程问题，而是结构性问题——只有从原始像素和文字出发端到端学习，才能释放原生多模态智能。

## 研究动机

当前多模态基础模型的进展让人振奋：VLMs 在视觉语言理解上不断突破，扩散模型在图像生成上取得了惊人成就。然而，这两条技术路线长期平行演化——理解依赖预训练视觉编码器，生成依赖 latent diffusion VAE，两者的特征空间存在系统性割裂。

作者指出了一个根本性问题：碎片化的架构（VE + LLM + VAE）不只是工程权衡，而是阻碍原生多模态智能涌现的结构性瓶颈。真正的统一应该让理解与生成作为同一过程的两个协同视图共同进化，而不是通过 tokenizer、latent space 或辅助模块松散连接。

SenseNova-U1 的目标是从第一性原理出发，构建一个不依赖预训练 VE 和深度解码头的原生统一架构，直接在像素和文字上端到端学习。

## 方法论（主要模块简介）

- **NEO-unify 框架**：Near-Lossless Visual Interface（近无损视觉接口）+ Native Mixture-of-Transformers（原生 MoT 主干），用 32× 压缩比同时保留语义结构和细粒度像素细节。
- **理解分支（Und Stream）**：轻量 2 层卷积 patch encoding + 线性投影头，预测文本 token；共享 LLM 的全部参数。
- **生成分支（Gen Stream）**：同构轻量 encoding + MLP 头，直接在像素空间做 flow matching（pixel-space JiT），绕过 VAE/扩散解码器。
- **Native MoT 架构**：理解流处理干净图像和文本，生成流处理噪声条件输入；两流全参数解耦，路由由 token 类型决定。
- **联合训练目标**：$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{Und} + \lambda_2 \mathcal{L}_{Gen}$，AR 交叉熵 + pixel-space flow matching MSE。

---

## 🌟 引言

2026 年 5 月，SenseTime 发布了 SenseNova-U1 系列，提出了一个极具野心的目标：**用一个原生统一的架构，同时搞定多模态理解和生成，让理解与生成作为同一个过程的两个协同视图共同进化，而非通过模块桥接**。

论文发布在 arXiv（arXiv:2605.12500），两位主要作者是 Haiwen Diao 和 Penghao Wu，团队共 58 人。提出了两个规模的模型：

- **SenseNova-U1-8B-MoT**：Dense 8B × 8B 双流对称配置
- **SenseNova-U1-A3B-MoT**：MoE 版本，理解流 128 experts / 30B 总参，生成流 32 experts / 8B 总参，推理时激活约 3B 参数

两个模型的核心架构一致，都基于 **NEO-unify** 框架——一种原生统一多模态范式，直接在像素和文字上端到端学习，不依赖预训练视觉编码器（VE）和深度 VAE 解码器。

## 🧩 背景与动机

### 1.1 视觉-语言模型的二分法困境

近年 VLMs（Qwen3-VL、InternVL3 等）将视觉编码器与 LLM 耦合，在理解任务上取得了出色表现。然而，这种"VE + LLM"设计天然存在能力天花板：视觉编码器在预训练时以语义对比学习为目标，而 LLM 以语言建模为目标，两者的表征空间和优化方向存在系统性偏移。

与此同时，图像生成领域长期由 latent diffusion 主导——VAE/VQ-VAE 将图像压缩到低维 latent 空间，扩散模型在该空间进行生成。虽然重建驱动目标能保证视觉保真度，但低维压缩严重限制了语义表达能力。

结果是，理解任务和生成任务各自发展出了不同的技术栈：理解依赖 CLIP-based VE，生成依赖 diffusion + VAE。要在同一模型中同时做好两件事，就需要额外的模块来弥合表征鸿沟——这正是现有 UMM 的基本思路。

### 1.2 现有统一方案的局限

现有的统一多模态模型（Show-o、Janus、OmniGen、BAGEL 等）已经证明了感知和合成可以在单个模型中共存，但它们仍然在 tokenization、扩散头或解码路径上存在根本性分歧——理解流和生成流并非真正的联合学习。

另一条路线是连续表征路线（Transfusion、PRISM、Tuna 等），追求在统一连续视觉接口中调和概念结构与高保真重建，但仍然受制于中间表征——语义抽象和像素粒度之间的权衡没有消失，只是被推迟到了表示层。

### 1.3 作者的核心主张

**原生多模态建模的真正统一应该从第一性原理出发：直接在原始像素和文字上学习，摆脱预训练编码器的语义先验和固定表征的扩展限制。**

这意味着：
1. 不使用预训练 VE，用轻量 patch encoding layer 从像素直接学习
2. 不使用深度 VAE/扩散解码头，用轻量 MLP 在像素空间直接回归
3. 用 Native MoT 架构替代两套独立模块，通过 token 类型动态路由

## 🛠️ 方法详解

### 2.1 Near-Lossless Visual Interface

**Patch Encoding Layer**：完全沿用 NEO 的设计，使用两层卷积（stride 16 + 2）将图像映射为 visual token 序列，每个 token 对应 32×32 的图像块。使用 `<img>` 和 `</img>` 特殊 token 包裹视觉内容。文本使用原始 LLM tokenizer 无修改。视觉 token 和文本 token 投影到统一 embedding 空间后，在统一主干内联合处理。

**Patch Decoding Layer**：
- 理解分支：线性投影头，映射到词表做文本预测
- 生成分支：MLP head，直接预测像素块——**绕过 VAE 解码器和深度扩散头**

这种极轻量的 encoding/decoding 接口（只有两层卷积 + MLP）使得模型能在 32× 压缩比下同时保留语义结构和细粒度像素信息。

**Dynamic Noise Scale（动态噪声尺度）**：生成流在高分辨率下，若噪声来自单位方差高斯，会出现 SNR 不一致的问题。作者引入分辨率自适应的噪声尺度：

$$\sigma_R(H,W) = \sigma_0 \sqrt{N(H,W)/N_0}$$

其中 $N(H,W) = (H \cdot W) / 32^2$ 是该分辨率下的 token 数，$\sigma_0$ 是基础噪声尺度。这个平方根缩放保持了单位 token 噪声能量的大致恒定。

**Noise-Scale Conditioning**：将归一化噪声尺度 $\bar{\sigma} = \sigma_R / \sigma_{\max} \in [0,1]$ 用正弦 MLP 编码后，与时间步 embedding 拼接：

$$\mathbf{s}_t = \tau_t + \text{NSEmb}(\bar{\sigma}(H,W))$$

### 2.2 Native Mixture-of-Transformers（原生 MoT）

Native RoPE 的核心设计：文本 token 在时间轴 $T$ 上演化（$H=W=0$），图像 token 额外携带空间索引（$H, W$ 轴）。三种轴向共享预训练 LLM 的 head dimensions，各自关联独立频率基，且不引入额外参数。

**双流注意力模式**：
- 文本 token：causal attention（只看前面 token）
- 同一图像块内的 clean token：双向 attention
- 同一图像块内的 noise token：双向 attention，且可看到 clean 输入
- **关键约束**：clean token 永远不能 attending 到 noise token

**全参数解耦**：理解流和生成流使用独立的投影、正则化和前向块，每层通过 token 类型动态路由。

**两个模型变体的配置**（表 1）：

| 配置 | SenseNova-U1-8B-MoT | SenseNova-U1-A3B-MoT |
|------|---------------------|---------------------|
| Patch Size | 32×32 | 32×32 |
| Pre-Buffer | ✓ | ✗ |
| # Layers | 42 | 48 |
| # Heads (Q/KV) | 32 / 8 | 32 / 4 |
| Hidden Size | 4,096 | 2,048 |
| # Und / Gen Experts | 1 / 1 | 128 / 32 (A8) |
| # Und / Gen 参数 | 8.2B / 8.2B | 30.0B / 8.2B (A3B) |

8B-MoT 采用对称并行双流 Dense 配置；A3B-MoT 使用流-wise MoE，理解流 128 experts（top-8 激活），生成流 32 experts。

### 2.3 联合训练目标

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{Und} + \lambda_2 \mathcal{L}_{Gen}$$

**理解目标**（标准 AR LM）：
$$\mathcal{L}_{Und} = -\frac{1}{N}\sum_{n=1}^{N}\log p_\theta(x_n \mid x_{<n}, \mathbf{c})$$

**像素空间 Flow Matching**：直接回归干净图像 $\hat{\mathbf{x}}$，通过 velocity loss：

$$\mathbf{v}_\theta(\mathbf{z}_t, t) = \frac{\hat{\mathbf{x}}(\mathbf{z}_t, t, \mathbf{s}_t) - \mathbf{z}_t}{1-t}$$

$$\mathcal{L}_{Gen} = \mathbb{E}_{t,\mathbf{x},\boldsymbol{\epsilon},(H,W)}[\|\mathbf{v}_\theta - \mathbf{v}^*\|_2^2]$$

这里 $\mathbf{z}_t = t\mathbf{x} + (1-t)\sigma_R\boldsymbol{\epsilon}$，$t=0$ 时为纯噪声，$t=1$ 时为干净图像。

**CFG（Classifier-Free Guidance）**：训练时随机丢弃文本条件（10%）和文本+图像条件（额外 10%）。推理时：

$$\nabla_\mathbf{x}\log p(\mathbf{x}\mid c_{img}, c_{txt}) = \gamma(\nabla_\mathbf{x}\log p(\mathbf{x}\mid c_{img}, c_{txt}) - \nabla_\mathbf{x}\log p(\mathbf{x}\mid c_{img})) + \gamma_{img}(\nabla_\mathbf{x}\log p(\mathbf{x}\mid c_{img}) - \nabla_\mathbf{x}\log p(\mathbf{x})) + \nabla_\mathbf{x}\log p(\mathbf{x})$$

默认 $\gamma=4$（文本引导），$\gamma_{img}=1$（图像上下文引导），时间步 shift 为 3.0。

### 2.4 训练流程

SenseNova-U1 的训练分为 4 个阶段（表 2）：

**Stage 1: 理解 Warmup**（从预训练 NEO 初始化）
- (i) Attention-Fusion Phase：统一 NEO 的 QK 投影和归一化（时空轴共享），QK 参数量减半。冻结其余网络，只微调 attention 层直到恢复精度。
- (ii) Full-Model Continuation Phase：解冻整个理解分支，在更新后的 mid-training corpus 上继续训练，lr=2×10⁻⁵。

**Stage 2: 生成 Pre-Training**（冻结理解分支）
- Phase I（120K steps）：text-to-image 数据，分辨率 256²→512²，lr=2×10⁻⁴
- Phase II（60K steps）：分辨率 ≥512²，lr=1×10⁻⁴
- Phase III（120K steps）：加入编辑、推理、交错数据，cosine lr 衰减至 2×10⁻⁵

**Stage 3: 联合 Mid-Training**（两端联合训练）
- 数据配比：理解 33% / T2I 生成 37% / 编辑 24% / 交错 6%
- 全模型 84K steps，lr=2×10⁻⁵，$\lambda_1=0.1, \lambda_2=1.0$

**Stage 4: 联合 SFT**
- 全模型 9K steps，cosine lr 衰减至 0
- 数据配比同 Stage 3

**Stage 5（可选）: T2I Post-Training**：Flow-GRPO + DMD 提升生成质量和效率。

## 📈 实验结果

### 3.1 多模态理解

在 MMMU、MMMU-Pro、MathVision、MMBench、AI2D、OCRBench 等基准上，SenseNova-U1-A3B 达到了 80.55% MMMU、91.59% MMBench-EN、91.90% OCRBench，**媲美甚至超越顶级理解专用模型**。

| Benchmark | SenseNova-U1-8B | SenseNova-U1-A3B | Qwen2.5-VL-72B | InternVL3-8B |
|-----------|-----------------|-----------------|-----------------|---------------|
| MMMU | 72.83 | **80.55** | 73.3 | 69.1 |
| MMBench-EN | 88.41 | **91.59** | 86.9 | 86.4 |

### 3.2 文本理解

在 IFEval、MMLU-Pro、SuperGPQA 等纯文本基准上，8B 版本达到 91.13% IFEval，A3B 版本达到 84.04% MMLU-Pro、59.71% SuperGPQA，**保持了强大的语言能力**。

### 3.3 图像生成

在 GenEval（0.91 overall）、DPG-Bench（94.19 Global / 88.14 overall）、DPG-Bench 等多个基准上，SenseNova-U1 实现了**开源最佳任意图像生成**。

| Model | GenEval Overall |
|-------|----------------|
| **SenseNova-U1-8B** | **0.91** |
| Qwen-Image | 0.87 |
| Lumina-DiMOO | 0.88 |
| BAGEL | 0.82 |
| FLUX.1-dev | 0.90 |

在知识密集型图像生成（DPG-Bench）和文图表生成（IGenBench）、长文本渲染（LongText-Bench）、商业图像（BizGenEval）等场景中同样表现出色。

### 3.4 图像编辑

在 GEdit-Bench、RISEBench 上，SenseNova-U1-A3B（with CoT）达到 30.0 RISEBench（相比无 CoT 的 25.3 显著提升），展示了思维模式在复杂编辑推理中的价值。

### 3.5 交错生成

在 OpenING 基准上，SenseNova-U1-A3B with CoT 达到 **9.16 overall**，超越 Nano Banana（8.85）、Wan-Weaver（8.67）、GPT-4o+DALL-E3（8.20），在开源模型中排名第一。

### 3.6 消融实验

关键消融结论：
1. **原生无 encoder 设计**：保留语义和像素双重表征能力（2B NEO-unify 在 MS-COCO 2017 重建上达到 31.56 PSNR / 0.85 SSIM，接近 FLUX.1-dev VAE）
2. **MoT 主干**：理解与生成在联合训练中协同增益，而非相互干扰
3. **数据扩展效率**：原生统一架构在小样本场景下展现出更高学习效率

### 3.7 VLA & World Model 初探

论文给出了 VLA（机器人操作）和 World Model（机械臂视角预测）的可视化结果，展示了 SenseNova-U1 在感知和生成之外跨模态行动的能力。官方 Demo：https://unify.light-ai.top/

## 💡 亮点总结

1. **原生统一架构**：不依赖预训练 VE 和 VAE，直接在像素和文字上端到端学习，用极轻量 encoding/decoding 接口（2 层卷积 + MLP）实现近无损视觉接口
2. **Native MoT 主干**：双流全参数解耦的 Mixture-of-Transformers，理解流和生成流通过 token 类型动态路由，在共享自注意力中每层原生交互
3. **极简压缩比（32×）下同时保证语义抽象和像素保真**：为理解保留了语义结构，为生成保留了细粒度视觉细节
4. **联合训练目标**：AR 交叉熵 + pixel-space flow matching，通过动态噪声尺度和 CFG 实现稳定的联合优化
5. **两阶段 RL + DMD 后训练**：Flow-GRPO 提升生成质量，Distribution Matching Distillation 提升推理效率
6. **VLA & WM 初步验证**：为真正原生统一的感知-推理-行动多模态智能指出了一条更广阔的路线图

## ⚖️ 局限性与思考

1. **架构复杂度和训练成本**：Native MoT 双流设计虽然推理效率通过解耦得以优化，但预训练阶段需要多阶段渐进式训练，流程比单流模型更复杂
2. **生成质量与顶级闭源模型的差距**：虽然开源最佳，但 FLUX.1-dev 等顶级闭源系统仍是天花板
3. **MoE 路由策略对双流协同的影响**：两流使用独立 MoE 路由，是否会在某些跨模态任务上出现路由不一致，还有待进一步分析
4. **VLA/WM 仍是初步探索**：论文仅给出了可视化示例，尚无系统性基准评估
5. **32× 压缩比是否是语义-保真度最优平衡点**：更低的压缩比（如 16×）是否会带来进一步提升，值得探索

## ✅ 结语

SenseNova-U1 提出的 NEO-unify 范式代表了一次从"模块连接"到"原生统一"的范式转变——理解与生成不再是通过 adapter 或 latent space 桥接的两个独立系统，而是在同一个端到端架构中作为协同视图共同进化。

从技术层面，其核心贡献在于证明了：极轻量的视觉接口（无预训练 encoder）配合全参数解耦的 MoT 主干，可以在 32× 压缩比下同时实现顶级理解和生成水平，且两者可以相互增益而非相互干扰。

从更宏观的角度，SenseNova-U1 为多模态 AI 的发展指出了一条更激进的路线：不是在不同模态之间做翻译，而是让模型在原生统一的架构中思考和行动，**相信必要的能力会从内部涌现**。

---

**论文信息**：Diao et al., "SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-unify Architecture", arXiv:2605.12500, 2026.
**开源地址**：https://github.com/OpenSenseNova/SenseNova-U1
**模型权重**：https://huggingface.co/collections/sensenova/sensenova-u1
**官方 Demo**：https://unify.light-ai.top/