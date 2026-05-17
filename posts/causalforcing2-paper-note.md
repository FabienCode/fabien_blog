# Causal Forcing++ 论文解析：面向实时交互视频生成的 Scalable Few-Step AR Diffusion 蒸馏

## 📖 论文信息

- **标题**：[Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation](https://arxiv.org/abs/2605.15141)
- **GitHub**：[thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing)、[shengshu-ai/minWM](https://github.com/shengshu-ai/minWM)
- **作者**：Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, Jun Zhu
- **单位**：Tsinghua University / ShengShu / Renmin University of China
- **arXiv**：arXiv:2605.15141 (cs.CV)

## ✦ 核心洞察与挑战

### 核心问题
- **问题一**：实时交互视频生成要求低延迟、流式和可控 rollout，现有 AR diffusion 蒸馏方法在 chunk-wise 4-step  regime 下已取得不错效果，但仍受限于粗糙的响应粒度（chunk 而非 frame）和不可忽视的采样延迟。
- **问题二**：在更激进的 frame-wise 1-2 步 setting 下，此前所有初始化策略均失效——bidirectional teacher 的 ODE 初始化存在架构错位，多步 AR 直接初始化在激进设置下严重退化，而 Causal Forcing 的 causal ODE 初始化虽然有效但代价高昂（需预计算并存储完整 PF-ODE 轨迹）。
> **关键判断**：Causal Forcing++ 的核心洞察是——causal ODE 蒸馏和 causal CD（一致性蒸馏）共享同一个学习目标（AR-conditional flow map），但 causal CD 通过单步在线 teacher ODE  supervision 替代昂贵的离线轨迹预计算，使初始化既正确、又高效、更易扩展。

### 传统方案局限
- **局限一**：Self Forcing 的 ODE 初始化（bidirectional teacher）违反 frame-level injectivity，回归目标变成条件期望而非 AR flow map，在激进设置下最终崩溃（Dynamic Degree → 0）。
- **局限二**：直接使用多步 AR 模型初始化（LiveAvatar、WorldPlay）在激进设置下误差被自回归放大，近乎崩溃。
- **局限三**：Causal Forcing 的 causal ODE 初始化虽然正确，但每个训练样本需 teacher 生成完整多步 PF-ODE 轨迹（如 48 步），需预计算存储，在 80K 视频规模下约需 11,600 A800 GPU 小时 + 1,900 GiB 额外存储，成本过高难以扩展。
> **关键判断**：需要一个同时满足 **AR**、**few-step**、**scalable** 三个条件的初始化策略——causal CD 正是这样的策略。

## 研究动机

实时交互视频生成是视频生成模型从被动内容生产者向交互式世界模型演进的关键能力。低延迟、流式 rollout 和用户可控交互是核心需求，AR diffusion 模型天然适合这一目标——它在帧级别进行因果 rollout，同时每帧内部保留扩散生成的高质量。

近期 AR diffusion 蒸馏方法（如 CausVid、Self Forcing、Causal Forcing）通过将双向视频扩散模型蒸馏为 few-step AR 学生，已经取得了不错的效果。但这些方法都在 **chunk-wise 4-step**  regime 下操作——这对于真正的实时交互来说仍然不够：响应粒度太粗（chunk 而非 frame），采样延迟不可忽视。

Causal Forcing++ 研究了一个更激进的设置：**frame-wise 自回归，仅 1-2 采样步**。在这个设置下，asymmetric DMD 的 few-step AR 学生初始化成为关键瓶颈——所有现有策略都因互补的原因失效。Causal Forcing++ 提出的 **causal consistency distillation（causal CD）** 作为 causal ODE 初始化的原则性替代方案，同时实现了更高的质量和 ~4× 的训练效率提升。

## 方法论（主要模块简介）

- **Causal Consistency Distillation（causal CD）**：学习与 causal ODE 相同的 AR-conditional flow map，但通过单步在线 teacher ODE supervision（相邻时间步之间）替代昂贵的离线多步 PF-ODE 轨迹预计算与存储。
- **两阶段 AR diffusion 蒸馏管线**：Stage 1（teacher forcing AR 扩散训练）+ Stage 2（causal CD 初始化，替代 causal ODE）+ Stage 3（asymmetric DMD + self-rollout）。
- **核心理论保证**：causal CD 的最优模型与目标 flow map 的误差受 ODE 求解器数值误差约束，误差为 $O((\Delta t)^p)$，可忽略。
- **Action-conditioned 世界模型扩展**：将 pipeline 扩展到相机姿态条件的生成器蒸馏，应用于 Genie3 风格的世界模型。

---

## 🌟 引言

2026 年 5 月，清华大学联合其他机构发布了 **Causal Forcing++**（arXiv:2605.15141），提出了一个面向实时交互视频生成的 Scalable Few-Step AR Diffusion 蒸馏框架。

Causal Forcing++ 的核心贡献是发现了 **few-step AR 学生初始化是激进低延迟 regime 的关键瓶颈**，并提出用 **causal consistency distillation（causal CD）** 替代此前 Causal Forcing 中的 causal ODE 初始化——两者共享同一个学习目标（AR-conditional flow map），但 causal CD 通过相邻时间步之间的单步在线 supervision 避免了昂贵的离线轨迹预计算。

在 frame-wise 2-step 设置下，Causal Forcing++ 相比此前 SOTA 4-step chunk-wise Causal Forcing 在 VBench Total 上提升 0.1、Quality 提升 0.3、VisionReward 提升 0.335，同时 **首帧延迟降低 50%**、Stage 2 训练成本降低约 **4×**。

## 🧩 背景与动机

### 1.1 AR Diffusion 的交互优势

扩散模型在视频生成中取得了巨大成功（如 Wan、HunyuanVideo），但通常是双向的——一次性生成整个视频，导致高延迟（首帧到完成的漫长等待）且无法交互。AR 生成在更小的单元上操作，天然具有更低延迟；它还支持交互——用户可以根据已生成的内容提供反馈并调整后续条件信号。

AR Diffusion 结合了两者优势：在帧级别做因果 rollout，同时每帧内部用扩散模型生成质量。

### 1.2 AR Diffusion 蒸馏的三代演进

**CausVid**：两阶段框架——(1) 用 bidirectional teacher 的 ODE 初始化，(2) asymmetric DMD 训练。Bidirectional teacher 的ODE轨迹到AR学生的回归存在架构错位。

**Self Forcing**：修正了 asymmetric DMD 阶段——用 self-rollout 替代 diffusion forcing，使训练与推理一致，但 ODE 初始化仍基于 bidirectional teacher，frame-level injectivity 问题未解决。

**Causal Forcing**：修正了 ODE 初始化——先 fine-tune bidirectional model 为 AR model，再用 AR teacher 生成 ODE 轨迹初始化 AR 学生。架构正确但代价极高（每个样本需预计算48步PF-ODE轨迹）。

### 1.3 激进设置下所有初始化策略的失效

在 frame-wise 1-2 步的激进设置下，三种现有策略因互补原因全部失效：

| 初始化策略 | 问题 | 在 1-step 下的表现 |
|-----------|------|-----------------|
| Bidirectional teacher ODE | 违反 frame-level injectivity，回归目标变为条件期望而非 AR flow map | 灾难性崩溃 |
| 直接多步 AR 模型 | 缺乏 few-step 能力，self-rollout 误差放大 | 近乎崩溃 |
| Causal ODE | 正确但高昂——需预计算并存储完整PF-ODE轨迹 | 可行但成本无法扩展 |

## 🛠️ 方法详解

### 2.1 核心洞察：causal ODE 和 causal CD 共享同一目标

Causal Forcing++ 的关键发现是：**causal ODE 蒸馏和 causal CD 试图学习同一个对象——AR teacher 的 AR-conditional flow map（即 consistency function）**。它们的不同仅在于如何获得 supervision：

- **Causal ODE 蒸馏**：需要 AR teacher 为每个训练样本生成完整多步 PF-ODE 轨迹，必须预计算并离线存储
- **Causal CD**：只需一个在线 teacher ODE 步骤（在相邻时间步之间），无需预计算轨迹

### 2.2 Causal CD 的理论保证

在 flow matching 参数化下，causal CD 的最优模型 $G_{\theta^*}$ 与目标 flow map $f_\phi$ 的误差受 ODE 求解器精度约束：

$$\sup \| f_\phi(\mathbf{x}_t^i, \mathbf{x}_{gt}^{<i}, t) - G_{\theta^*}(\mathbf{x}_t^i, \mathbf{x}_{gt}^{<i}, t) \|_2 = O((\Delta t)^p)$$

这意味着 causal CD 和 causal ODE 共享相同的原则性保证，但 causal CD 的相邻时间步监督提供了更小的 per-step 优化间隙。

### 2.3 Causal Forcing++ 管线

Causal Forcing++ 继承 Causal Forcing 的 Stage 1（teacher forcing AR 扩散训练）和 Stage 3（asymmetric DMD + self-rollout），但将 **Stage 2 的 causal ODE 初始化替换为 causal CD**：

![图 1：Causal Forcing++ 框架对比](assets/causalforcing2-paper-note/x1.png)

**causal CD 的两大优势**：

1. **效率**：无需预计算和存储完整轨迹，Stage 2 训练时间从 ~11,600 降至 ~2,900 A800 GPU 小时（约 4× 加速），额外存储从 ~1,900 GiB 降至 0
2. **质量**：相邻时间步的一致性监督比大跳 ODE 回归提供更小的 per-step 优化间隙，更易优化

### 2.4 为什么 Causal DMD 不适合做初始化

论文还讨论了 causal DMD（score distillation 形式）作为初始化的可行性。尽管 DMD 在双向设置中通常产生更锐利的图像，但在 AR 设置中 causal DMD 的表现不如 causal CD——

原因在于 DMD 优化 reverse KL，倾向于 mode-seeking，在自回归 rollout 中对累积误差更敏感，产生更强的 exposure bias。虽然最初几帧可能更锐利，但随着生成进行，误差快速累积，质量迅速崩溃。

### 2.5 Action-conditioned 世界模型扩展

Causal Forcing++ 还展示了在 Genie3 风格 action-conditioned 世界模型中的应用：将相机姿态条件的双向 generator 蒸馏为交互式 AR 世界模型。

## 📈 实验结果

### 3.1 主要性能对比

基于 Wan2.1-1.3B，在 frame-wise 2-step 设置下：

| Model | Throughput (↑) | Latency (↓) | VBench Total (↑) | Quality (↑) | VisionReward (↑) |
|-------|----------------|-------------|-------------------|-------------|-----------------|
| CausVid | 10.4 FPS | 0.60s | 81.33 | 83.98 | 5.741 |
| Self Forcing | 10.4 FPS | 0.60s | 83.74 | 84.48 | 5.820 |
| Causal Forcing | 10.4 FPS | 0.60s | 84.04 | 84.59 | 6.326 |
| **Causal Forcing++ (2-step)** | **14.1 FPS** | **0.27s** | **84.14** | **84.89** | **6.661** |

**关键提升**：
- 相比 Causal Forcing（4-step），VBench Total +0.1，Quality +0.3，VisionReward +0.335
- 首帧延迟降低 **50%**（0.27s vs 0.60s）
- 吞吐量提升 **~36%**（14.1 vs 10.4 FPS）

### 3.2 消融实验：各类初始化策略对比

在 1-step / 2-step / 4-step 各类设置下，causal CD 均与或超越 causal ODE：

| 步数设置 | 初始化策略 | VBench Total | Stage 2 成本 | 额外存储 |
|---------|-----------|-------------|-------------|---------|
| 1-step | Causal ODE | 83.06 | 11,600 GPUh | 1,900 GiB |
| 1-step | **Causal CD** | **83.35** | **2,900 GPUh** | **0** |
| 2-step | Causal ODE | 83.77 | 11,600 GPUh | 1,900 GiB |
| 2-step | **Causal CD** | **84.14** | **2,900 GPUh** | **0** |
| 4-step | Causal ODE | 83.78 | 11,600 GPUh | 1,900 GiB |
| 4-step | **Causal CD** | **84.10** | **2,900 GPUh** | **0** |

Self Forcing ODE 和直接 AR 初始化在 1-step 下均灾难性崩溃（Dynamic Degree → 0）。

### 3.3 训练效率

![图 3：causal CD vs causal ODE 效率和性能对比](assets/causalforcing2-paper-note/x3.png)

Causal CD 在所有指标上匹配或超越 causal ODE，同时：
- Stage 2 训练时间：**~4× 加速**（11,600 → 2,900 A800 GPU hours）
- 额外存储：**降至 0**（原来需要 ~1,900 GiB）

## 💡 亮点总结

1. **原则性理论框架**：causal CD 和 causal ODE 共享同一学习目标（AR-conditional flow map），causal CD 的误差由 ODE 求解器精度约束，有严格理论保证
2. **4× 训练效率提升**：用在线单步 supervision 替代离线多步轨迹预计算，Stage 2 训练成本从 11,600 降至 2,900 GPU 小时，存储从 1,900 GiB 降至 0
3. **50% 延迟降低**：frame-wise 2-step 生成相比 chunk-wise 4-step 将首帧延迟从 0.60s 降至 0.27s，同时 VBench 分数更高
4. **更小的优化间隙**：causal CD 的相邻时间步监督比 causal ODE 的大跳回归提供更小 per-step 优化间隙，更易优化
5. **在所有步数设置下有效**：causal CD 在 1/2/4-step 所有设置下均与或超越 causal ODE，且无需任何额外存储
6. **Causal DMD 不适合的原因被理论分析**：DMD 的 mode-seeking reverse KL 在 AR rollout 中对累积误差更敏感，产生严重 exposure bias

## ⚖️ 局限性与思考

1. **帧级别实时交互仍有挑战**：即使 2-step 设置，0.27s 的首帧延迟在极端实时场景（如 VR/AR 交互）仍可能不够低
2. **对 teacher 模型质量的依赖**：整个蒸馏管线的上限仍然受限于 teacher（ Wan2.1-1.3B）的生成质量
3. **action-conditioned 变体的 frame-wise 扩展未探索**：论文仅展示了 chunk-wise 4-step 的 action-conditioned 扩展，frame-wise 实时交互版本留待未来工作
4. **与更激进的 few-step 设置（1-step）的权衡**：2-step 在质量和延迟间取得了更好平衡，1-step 虽然延迟最低但 Dynamic Degree 有所下降
5. **曝光偏差仍存在**：虽然 causal CD 比 causal DMD 更鲁棒，自回归 rollout 中的质量衰减仍是开放问题

## ✅ 结语

Causal Forcing++ 为实时交互视频生成的 few-step AR diffusion 蒸馏提供了一个原则性且可扩展的解决方案。它解决了此前方法中初始化策略"正确但昂贵"的核心矛盾——causal CD 和 causal ODE 共享同一原则性目标，但 causal CD 以在线单步 supervision 替代了昂贵的离线轨迹预计算，实现了 4× 效率提升和更强的初始化质量。

从更宏观的角度，这项工作指向了一个更高效的 AR 世界模型蒸馏范式：**action-conditioned 的 Causal Forcing++ 为 Genie3 风格的交互式世界模型提供了一个可扩展的基础**——更低的延迟和更低的训练成本意味着更广泛的探索和迭代成为可能。

---

**论文信息**：Zhao et al., "Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation", arXiv:2605.15141, 2026.
**arXiv**：https://arxiv.org/abs/2605.15141
**GitHub**：https://github.com/thu-ml/Causal-Forcing