# 离散型 Self-Paced Deep Reinforcement Learning：理论推导与求解算法（学位论文草稿）

## 4.3 离散型自步课程学习机制

为避免深度强化学习策略直接在完整复杂任务空间中训练而产生探索效率低、收敛不稳定甚至频繁违规截断等问题，本文将课程学习建模为有限上下文任务池上的概率分布优化问题。该机制在每轮策略迭代后动态更新任务采样分布：一方面优先选择当前策略已具备一定胜任能力、能产生有效学习信号的任务；另一方面逐步驱动采样分布向目标任务分布迁移，并通过 KL 信赖域严格限制相邻课程之间的分布突变。

本节提出的离散型自步深度强化学习（Discrete Self-Paced Deep Reinforcement Learning, Discrete SPDL）是一种通用的外层分布优化机制，其内层可与任意 Actor–Critic 类深度强化学习算法（如 PPO、SAC、TD3 等）协同工作。

---

### 4.3.1 任务池建模与基于任务胜任力的课程优化模型

#### （1）有限上下文任务池与基本记号

设有限上下文任务池为

$$
\mathcal C = \{c_1, c_2, \ldots, c_N\}.
$$

每个上下文 $c_i$ 对应一个具体的训练工况或起始子任务。设由任务 $c_i$ 映射生成的环境初始完整状态为 $s_0 = g(c_i)$。对于高速磁浮列车速度曲线优化问题，$c_i$ 可由不同的初始位置、初始速度以及目标停车条件等参数构成；$g(c_i)$ 包含了策略与评估模块所需的所有状态特征。

定义相关分布与符号如下：

- $p = (p_1, \ldots, p_N)^\top \in \Delta_N$：下一轮训练中抽取各上下文任务的概率分布（待求解量）；
- $q = p^{(k)} \in \Delta_N$：当前轮训练中抽取各上下文任务的概率分布（已知量）；
- $\mu = (\mu_1, \ldots, \mu_N)^\top \in \Delta_N$：最终希望达到的目标任务分布（先验设定）；
- $\Delta_N = \left\{p \in \mathbb R_+^N : \sum_{i=1}^N p_i = 1\right\}$：$N$ 维概率单纯形；
- $a_i \in [0, 1]$：第 $k$ 轮策略在任务 $c_i$ 上的“任务胜任力”评估值；
- $\alpha_k \ge \alpha_{\min} > 0$：第 $k$ 轮课程向目标分布靠拢的自步调强度系数；
- $\varepsilon > 0$：相邻两轮课程分布单次更新所允许的最大 KL 散度（信赖域半径）。

在理论推导中，假设目标分布与历史课程分布均满足全支撑条件，即 $q_i > 0$ 且 $\mu_i > 0$（在工程实现中可通过引入极小的均匀探索质量满足）。

---

#### （2）任务完成度与策略胜任力的形式化定义

为了克服传统方法中利用 Critic 价值估值 $V(s)$ 易受任务时域长度、折扣因子 $\gamma^t$ 及奖励函数尺度变化干扰的缺陷，本文建立面向任务宏观目标的**任务完成度（Task Completion Degree）**形式化评价机制。

设策略 $\pi$ 在上下文任务 $c_i$（初始状态 $s_0 = g(c_i)$）下与环境交互生成的完整回合轨迹为

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T) \sim P(\cdot \mid c_i, \pi).
$$

定义轨迹任务完成度函数 $Y: \mathcal{T} \to [0, 1]$ 为

$$
Y(\tau) =
\begin{cases}
0, & \tau \in \Omega_{\mathrm{fail}} \quad (\text{轨迹发生违规截断、超出安全约束或未正常到达}),\\[2mm]
\omega_0 + \displaystyle\sum_{j=1}^m \omega_j \Phi_j(\tau), & \tau \in \Omega_{\mathrm{succ}} \quad (\text{轨迹安全到达终端目标区域}),
\end{cases}
\tag{1}
$$

其中，$\Omega_{\mathrm{fail}}$ 与 $\Omega_{\mathrm{succ}}$ 分别为失败轨迹集与成功轨迹集；$\omega_0 \in (0, 1)$ 为基础安全到达置信权重；$\Phi_j(\tau) \in [0, 1]$ 为第 $j$ 项归一化控制品质指标（如末端停站精度评分 $S_{\mathrm{stop}}$、到站准点率评分 $S_{\mathrm{time}}$ 等）；各权重满足 $\omega_0 + \sum_{j=1}^m \omega_j = 1$。

基于轨迹完成度，形式化定义策略 $\pi$ 在上下文任务 $c_i$ 上的**期望胜任力（Task Competence）**为

$$
C^\pi(c_i) = \mathbb E_{\tau \sim P(\cdot \mid c_i, \pi)}\left[ Y(\tau) \right], \qquad C^\pi(c_i) \in [0, 1].
\tag{2}
$$

**机制解耦性说明**：任务完成度 $Y(\tau)$ 与策略胜任力 $C^\pi(c_i)$ 是外层课程学习模块对智能体综合操控水平的宏观、无偏评估指标，**不直接参与内层单步 MDP 的奖励计算**。内层强化学习依然通过瞬时奖励（能耗、舒适度、安全势函数等）优化动作策略，外层课程机制则基于胜任力评估独立调度任务分布，实现策略优化与课程进度的彻底解耦。

---

#### （3）离散型 SPDL 课程优化模型

在第 $k$ 轮策略迭代后，设 $\hat C_k(c_i) \in [0, 1]$ 为策略 $\pi_k$ 在任务 $c_i$ 上的胜任力估计量，记胜任力评估向量为

$$
\mathbf a_k = \big(\hat C_k(c_1), \hat C_k(c_2), \ldots, \hat C_k(c_N)\big)^\top \in [0, 1]^N.
$$

离散型自步课程学习的目标是：在保证课程平稳演进（KL 信赖域约束）的前提下，最大化策略期望胜任力与目标分布先验的权衡目标。形式化数学优化模型构建为

$$
\begin{aligned}
p^{(k+1)} = \operatorname*{arg\,max}_{p \in \Delta_N} & \left\{ \sum_{i=1}^N p_i \hat C_k(c_i) - \alpha_k D_{\mathrm{KL}}(p \parallel \mu) \right\} \\
\text{s.t.} \quad & D_{\mathrm{KL}}(p \parallel q_k) \le \varepsilon,
\end{aligned}
\tag{3}
$$

其中，$D_{\mathrm{KL}}(p \parallel r) = \sum_{i=1}^N p_i \log \frac{p_i}{r_i}$ 为两离散概率分布间的 Kullback–Leibler 散度。

式（3）各优化项的物理内涵如下：

1. **胜任力驱动项 $\sum_{i=1}^N p_i \hat C_k(c_i)$**：驱动课程优先采样当前策略具备较高掌握度、能够持续提供有效正向学习经验的任务（体现自步学习机制）；
2. **目标导向正则项 $-\alpha_k D_{\mathrm{KL}}(p \parallel \mu)$**：作为拉动力量，引导任务采样分布平滑地向最终目标任务分布 $\mu$ 靠拢；
3. **信赖域约束 $D_{\mathrm{KL}}(p \parallel q_k) \le \varepsilon$**：限制单次更新的分布偏离幅度，抑制因有限采样估计误差带来的课程剧烈震荡。

由于胜任力天然处于有界区间 $[0, 1]$，式（3）彻底免除了传统方法中由于不同任务时域长短不一、奖励尺度动态变化导致的数值不稳定性。

---

#### （4）自适应自步调系数 $\alpha_k$

自步调系数 $\alpha_k$ 决定了课程“适应当前能力”与“向目标分布靠拢”之间的权衡。定义第 $k$ 轮训练的批次平均任务完成度 $\bar Y_k^{\mathrm{batch}}$ 及其指数移动平均（EMA）$\bar Y_k$：

$$
\bar Y_k^{\mathrm{batch}} = \frac{1}{M_k} \sum_{m=1}^{M_k} Y(\tau^{(m)}), \qquad \bar Y_k = (1 - \rho)\bar Y_{k-1} + \rho \bar Y_k^{\mathrm{batch}},
\tag{4}
$$

其中 $M_k$ 为第 $k$ 轮收集的完成回合数，$\rho \in (0, 1]$ 为平滑因子。基于当前整体胜任水平与目标分布距离，设计严格正向有界的自适应自步调系数：

$$
\alpha_k = \operatorname{clip}\left(
\zeta \frac{\max\{Y_{\mathrm{floor}}, \bar Y_k\}}{\max\{D_{\mathrm{KL}}(q_k \parallel \mu), \delta_{\mathrm{KL}}\}},
\;\alpha_{\min},\; \alpha_{\max}
\right),
\tag{5}
$$

其中，$\zeta > 0$ 为缩放因子，$Y_{\mathrm{floor}} > 0$ 为基底能力阈值，$\delta_{\mathrm{KL}} > 0$ 为目标收敛容差，$[\alpha_{\min}, \alpha_{\max}]$（$\alpha_{\min} > 0$）为安全截断区间。

当智能体整体胜任力 $\bar Y_k$ 较低时，$\alpha_k$ 处于较小值，算法专注于当前可胜任的任务；随着策略能力提升，$\bar Y_k$ 逐步增大，$\alpha_k$ 随之提升以迫使课程加速向目标分布 $\mu$ 演进；当 $D_{\mathrm{KL}}(q_k \parallel \mu) \le \delta_{\mathrm{KL}}$ 时，课程认为已充分收敛至目标分布并冻结更新。

---

### 4.3.2 课程分布的对偶求解与理论性质

为书写简洁，本小节省略轮次下标，记 $\alpha_k$ 为 $\alpha$，记胜任力评估项 $\hat C_k(c_i)$ 为 $a_i$（$a_i \in [0, 1]$），记当前课程为 $q$。将式（3）的目标函数记为

$$
F(p) = \sum_{i=1}^N p_i a_i - \alpha D_{\mathrm{KL}}(p \parallel \mu)
= \sum_{i=1}^N p_i a_i - \alpha \sum_{i=1}^N p_i \log \frac{p_i}{\mu_i}.
\tag{6}
$$

---

#### （1）目标函数严格凹性与全局唯一最优性

**定理 1（唯一全局最优性）**：对于任意 $\alpha \ge \alpha_{\min} > 0$，目标函数 $F(p)$ 在概率单纯形内部 $\operatorname{int}(\Delta_N)$ 上是严格凹函数，且式（3）的有约束凸优化问题存在唯一的全局最优解 $p^*$。

**证明**：对 $F(p)$ 求关于各分量 $p_i$ 的一阶偏导数：

$$
\frac{\partial F(p)}{\partial p_i} = a_i - \alpha (\log p_i + 1) + \alpha \log \mu_i.
$$

进而对分量求二阶偏导数：

$$
\frac{\partial^2 F(p)}{\partial p_i \partial p_j} = -\frac{\alpha}{p_i} \mathbb I\{i = j\}.
\tag{7}
$$

因此，$F(p)$ 的 Hessian 矩阵为对角矩阵：

$$
\nabla^2 F(p) = -\operatorname{diag}\left(\frac{\alpha}{p_1}, \frac{\alpha}{p_2}, \ldots, \frac{\alpha}{p_N}\right).
$$

因为对任意可行分布 $p \in \operatorname{int}(\Delta_N)$ 恒有 $p_i > 0$，且 $\alpha > 0$，所以 Hessian 矩阵的所有特征值均为负实数，即 $\nabla^2 F(p) \prec 0$（严格负定）。

又因 KL 散度约束集 $\{p \in \Delta_N : D_{\mathrm{KL}}(p \parallel q) \le \varepsilon\}$ 为非空有界闭凸集（当 $p=q$ 时散度为 $0 < \varepsilon$），由凸优化理论可知，严格凹函数在凸集上的极大值点存在且唯一。证毕。

---

#### （2）KKT 平稳性与广义 Softmax 闭式解

对 KL 信赖域不等式约束引入拉格朗日乘子 $\beta \ge 0$，对概率单纯形归一化等式约束引入乘子 $\nu \in \mathbb R$，构造拉格朗日函数：

$$
\mathcal L(p, \beta, \nu) = \sum_{i=1}^N p_i a_i - \alpha \sum_{i=1}^N p_i \log \frac{p_i}{\mu_i} - \beta \left( \sum_{i=1}^N p_i \log \frac{p_i}{q_i} - \varepsilon \right) + \nu \left( 1 - \sum_{i=1}^N p_i \right).
\tag{8}
$$

根据 Karush–Kuhn–Tucker (KKT) 最优性平稳条件，最优解满足 $\frac{\partial \mathcal L}{\partial p_i} = 0$：

$$
a_i - \alpha \left( \log p_i - \log \mu_i + 1 \right) - \beta \left( \log p_i - \log q_i + 1 \right) - \nu = 0.
\tag{9}
$$

整理式（9），将包含 $\log p_i$ 的项移至等式左端：

$$
(\alpha + \beta) \log p_i = a_i + \alpha \log \mu_i + \beta \log q_i - (\alpha + \beta + \nu).
$$

因为 $\alpha > 0$ 且 $\beta \ge 0$，分母 $\alpha + \beta > 0$ 恒成立。两边同除以 $\alpha + \beta$ 并取指数，得

$$
p_i = \exp\left( \frac{a_i + \alpha \log \mu_i + \beta \log q_i}{\alpha + \beta} \right) \cdot \exp\left( -\frac{\alpha + \beta + \nu}{\alpha + \beta} \right).
$$

利用概率单纯形归一化条件 $\sum_{i=1}^N p_i = 1$，消去与任务索引 $i$ 无关的常数项，即可推导出给定对偶乘子 $\beta$ 时的候选最优分布解析解：

$$
p_i(\beta) = \frac{\exp\left( \dfrac{a_i + \alpha \log \mu_i + \beta \log q_i}{\alpha + \beta} \right)}{\displaystyle\sum_{j=1}^N \exp\left( \dfrac{a_j + \alpha \log \mu_j + \beta \log q_j}{\alpha + \beta} \right)}, \qquad i = 1, \ldots, N.
\tag{10}
$$

式（10）呈现出**广义加权 Softmax 结构**：新课程分布是在当前任务胜任力项 $a_i$、目标先验 $\mu_i$ 和历史课程 $q_i$ 之间的几何加权平衡。当 $\beta \to \infty$ 时，$p(\beta) \to q$；当 $\beta = 0$ 时，分布完全由胜任力与目标先验决定。

---

#### （3）残差单调性与一维对偶求根

定义关于对偶乘子 $\beta$ 的 KL 距离残差函数：

$$
h(\beta) = D_{\mathrm{KL}}\big(p(\beta) \parallel q\big).
\tag{11}
$$

**定理 2（残差单调递减性）**：残差函数 $h(\beta)$ 关于 $\beta \ge 0$ 单调不增，且 $\lim_{\beta \to \infty} h(\beta) = 0$。

**证明**：令 $G(p) = \sum_{i=1}^N p_i a_i - \alpha D_{\mathrm{KL}}(p \parallel \mu)$。对于任意 $0 \le \beta_1 < \beta_2$，记 $p_1 = p(\beta_1)$，$p_2 = p(\beta_2)$。由拉格朗日最大化性质可知：

$$
\begin{aligned}
G(p_1) - \beta_1 D_{\mathrm{KL}}(p_1 \parallel q) &\ge G(p_2) - \beta_1 D_{\mathrm{KL}}(p_2 \parallel q), \\
G(p_2) - \beta_2 D_{\mathrm{KL}}(p_2 \parallel q) &\ge G(p_1) - \beta_2 D_{\mathrm{KL}}(p_1 \parallel q).
\end{aligned}
$$

两式相加并移项化简，得

$$
(\beta_2 - \beta_1) \left[ D_{\mathrm{KL}}(p_1 \parallel q) - D_{\mathrm{KL}}(p_2 \parallel q) \right] \ge 0.
$$

因为 $\beta_2 - \beta_1 > 0$，故必有 $D_{\mathrm{KL}}(p_1 \parallel q) \ge D_{\mathrm{KL}}(p_2 \parallel q)$，即 $h(\beta_1) \ge h(\beta_2)$。结合式（10），当 $\beta \to \infty$ 时 $p(\beta) \to q$，因此 $\lim_{\beta \to \infty} h(\beta) = D_{\mathrm{KL}}(q \parallel q) = 0$。证毕。

根据 KKT 互补松弛性条件 $\beta \big(h(\beta) - \varepsilon\big) = 0$ 与对偶可行性 $\beta \ge 0, h(\beta) \le \varepsilon$，对偶变量的最优解分为以下两种情况：

1. **无约束平稳情形**：若 $h(0) \le \varepsilon$，说明未激活信赖域边界，最优乘子取 $\beta^* = 0$，最优解直接为 $p^* = p(0)$；
2. **边界激活情形**：若 $h(0) > \varepsilon$，约束严格激活，必存在唯一正根 $\beta^* > 0$ 使得 $h(\beta^*) = \varepsilon$。由于 $h(\beta)$ 单调连续，可通过倍增法定界后采用**一维二分法（Bisection Search）**以极高速度精确求出数值解 $\beta^*$。

---

### 4.3.3 模块化协同训练框架与计算复杂度

离散型自步课程学习机制采用模块化、低耦合的双环协同架构。算法 1 描述了外层课程分布的精确求解过程，算法 2 描述了强化学习策略与自步课程机制的整体协同流程。

```text
Algorithm 1: Discrete-SPDL-Distribution-Update
Input: 胜任力评估向量 a[1:N], 历史课程分布 q[1:N], 目标分布 mu[1:N], 
       自步调系数 alpha (alpha >= alpha_min > 0), 信赖域半径 epsilon, 容差 tol, 最大迭代轮数 max_iter
Output: 下一轮最优任务采样分布 p*[1:N]

1: function ComputeCandidate(beta):
2:     d <- alpha + beta
3:     for i = 1 to N do
4:         logits[i] <- (a[i] + alpha * log(mu[i]) + beta * log(q[i])) / d
5:     z <- logits - max(logits)
6:     w <- exp(z)
7:     return w / sum(w)

8: p0 <- ComputeCandidate(0)
9: if KL_Divergence(p0, q) <= epsilon + tol then
10:    return p0

11: beta_low <- 0.0, beta_high <- 1.0
12: while KL_Divergence(ComputeCandidate(beta_high), q) > epsilon do
13:    beta_high <- 2.0 * beta_high

14: for t = 1 to max_iter do
15:    beta_mid <- (beta_low + beta_high) / 2.0
16:    p_mid <- ComputeCandidate(beta_mid)
17:    h_val <- KL_Divergence(p_mid, q)
18:    if abs(h_val - epsilon) <= tol then
19:        return p_mid
20:    if h_val > epsilon then
21:        beta_low <- beta_mid
22:    else
23:        beta_high <- beta_mid

24: return ComputeCandidate(beta_high)
```

```text
Algorithm 2: Competence-based Discrete-SPDL Framework
Input: 任务池 C = {c1, ..., cN}, 初始课程分布 p^(0), 目标分布 mu, 信赖域半径 epsilon,
       策略优化器 UpdatePolicy, 胜任力评估模块 EvaluateCompetence
Output: 优化后的控制策略 pi_theta

1: 初始化策略网络参数 theta_0, 胜任力评估模块参数 psi_0, 完成度滑动均值 Y_bar_0 <- Y_floor
2: for k = 0 to K-1 do
3:     q_k <- p^(k)
4:     // 1. 在当前课程分布 q_k 下采样任务并与环境交互采集轨迹数据
5:     D_k <- CollectOnlineTransitions(q_k, pi_(theta_k))
6:     
7:     // 2. 内层强化学习策略参数更新（以 PPO 为例）
8:     theta_(k+1) <- UpdatePolicy(theta_k, D_k)
9:     
10:    // 3. 计算已完成轨迹的任务完成度 Y(tau) 并更新胜任力评估模块
11:    Y_batch <- ComputeTaskCompletion(D_k)
12:    psi_(k+1) <- UpdateCompetenceEstimator(psi_k, D_k, Y_batch)
13:    
14:    // 4. 评估当前策略在全任务池上的胜任力向量
15:    a_k <- [EvaluateCompetence(c_i; psi_(k+1)) for c_i in C]
16:    
17:    // 5. 更新宏观完成度均值并计算自步调系数
18:    Y_bar_k <- (1 - rho) * Y_bar_(k-1) + rho * Mean(Y_batch)
19:    alpha_k <- ComputeSelfPacedCoefficient(Y_bar_k, q_k, mu)
20:    
21:    // 6. 求解下一轮课程采样分布
22:    if KL_Divergence(q_k, mu) <= delta_KL then
23:        p^(k+1) <- q_k   // 课程已平稳收敛至目标分布，冻结更新
24:    else
25:        p^(k+1) <- Discrete-SPDL-Distribution-Update(a_k, q_k, mu, alpha_k, epsilon)
26: return pi_(theta_K)
```

---

#### （3）胜任力评估接口的工程实现灵活性

在算法 2 中，胜任力评估模块 `EvaluateCompetence` 被设计为通用抽象接口。在工程落地时，可根据实际任务规模灵活选择具体的实现形态：

1. **统计滑动平均型**：在小规模离散任务池中，可直接统计每个上下文已完成历史轨迹完成度的指数移动平均；
2. **函数逼近型**：在复杂或连续状态上下文任务中，可构建独立的轻量级参数化网络 $\hat C_\psi(s_0) = \sigma\big(f_\psi(s_0)\big) \in (0, 1)$，以历史回合的 $(s_0, Y(\tau))$ 作为监督样本，采用二元交叉熵（BCE）或均方误差（MSE）损失进行回归拟合。

该设计保证了无论底层评估采用何种数据结构或逼近器，式（3）与算法 1 的数学求解与理论保证均保持不变。

---

#### （4）计算复杂度分析

设有限任务池大小为 $N$，二分搜索最大迭代次数为 $T_{\mathrm{iter}}$。在算法 1 中，单次计算候选分布向量及 KL 散度的复杂度均为 $O(N)$。由于残差函数单调，二分法呈指数级收敛（通常 $T_{\mathrm{iter}} \le 30$ 即可达到 $10^{-8}$ 容差），因此求解课程分布的总时间复杂度为

$$
\mathcal O\big((T_{\mathrm{iter}} + 1) N\big).
$$

对于 $N \sim 100$ 的高速磁浮列车典型工况池，单次分布更新在标准 CPU 上耗时仅为**亚毫秒级（$< 1\,\text{ms}$）**，相比于深度神经网络的前向与反向传播耗时完全可以忽略不计，具备极高的实时在线调度性能。

---

## 参考文献

1. Klink P, D'Eramo C, Peters J, et al. Self-Paced Deep Reinforcement Learning[C]//Advances in Neural Information Processing Systems (NeurIPS). 2020, 33: 9216–9227.
2. Schulman J, Wolski F, Dhariwal P, et al. Proximal Policy Optimization Algorithms[EB/OL]. arXiv:1707.06347, 2017.
3. Kumar M P, Packer B, Koller D. Self-Paced Learning for Latent Variable Models[C]//Advances in Neural Information Processing Systems (NeurIPS). 2010, 23: 1189–1197.
4. Florensa C, Held D, Wulfmeier M, et al. Reverse Curriculum Generation for Reinforcement Learning[C]//Proceedings of the 1st Annual Conference on Robot Learning (CoRL). PMLR, 2017: 482–495.
5. Portelas R, Colas C, Weng L, et al. Automatic Curriculum Learning for Intelligent Agents: A Survey[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020.
