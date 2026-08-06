# 离散型 Self-Paced Deep Reinforcement Learning：理论推导与求解算法（学位论文草稿）

## 4.3 离散型自步课程学习机制

为避免深度强化学习策略直接在完整复杂任务空间中训练而产生探索效率低、收敛不稳定等问题，本文将课程学习建模为有限任务池上的概率分布优化问题。该机制在每轮策略改进后更新任务采样分布：一方面优先选择当前策略能够有效处理的任务，另一方面逐步向目标任务分布迁移，并通过 KL 信赖域限制相邻课程之间的变化幅度。

该机制以 Actor–Critic 类深度强化学习算法为内层学习器。PPO 是本文采用的具体实例；SAC、TD3 和 DDPG 等能够提供当前策略价值估计的算法也可按相同外层课程更新规则进行耦合。需要强调的是，离散型 SPDL 的理论对象是任务采样分布，而非某一特定策略优化算法。

### 4.3.1 任务池构建与课程优化模型

#### （1）上下文任务池与基本记号

设有限上下文任务池为

$$
\mathcal C=\{c_1,c_2,\ldots,c_N\}.
$$

每个上下文 $c_i$ 对应一个具体训练任务。不同任务可具有不同的初始状态、动力学参数、约束参数或奖励参数。为方便表述，将由任务 $c_i$ 生成的完整初始状态记为 $s_i=g(c_i)$。这里的“完整”是指该状态已经包含策略网络和价值网络做出判断所需的信息。给定策略 $\pi_\theta$，其在任务 $c_i$ 下的真实期望折扣回报定义为

$$
V^{\pi_\theta}(c_i)
=
\mathbb E\left[
\sum_{t=0}^{T-1}\gamma^t r_t
\;\middle|\;
s_0=s_i,\pi_\theta
\right].
\tag{1}
$$

对于高速磁浮列车速度曲线优化问题，$c_i$ 可由候选起始状态、目标停车条件或其他可控工况参数构成；$g(c_i)$ 必须包含策略与评论家计算所需的全部状态量，而不能只包含其中的任务标签。

课程分布、目标任务分布及上一轮课程分布分别记为

$$
p=(p_1,\ldots,p_N)^\top\in\Delta_N,
\qquad
\mu=(\mu_1,\ldots,\mu_N)^\top\in\Delta_N,
\qquad
q=p^{(k)},
$$

其中

$$
\Delta_N=\left\{p\in\mathbb R_+^N:\sum_{i=1}^Np_i=1\right\}.
$$

为便于阅读，各符号的实际含义如下：

- $N$：候选任务的总数；
- $p_i$：下一轮训练抽取任务 $c_i$ 的概率，也是本节需要求解的量；
- $q_i$：当前轮训练抽取任务 $c_i$ 的概率，即更新前的课程分布；
- $\mu_i$：最终希望达到的目标任务分布中任务 $c_i$ 的概率；
- $\theta$：策略网络参数；$\phi$：评论家网络参数；
- $a_i$：课程更新时对任务 $c_i$ 的“当前可学习程度”或价值评分；
- $\alpha_k$：课程向目标分布靠拢的强度；$\varepsilon$：单次课程更新允许偏离旧课程的最大幅度。

后续推导只需保证任务池有限，且每个任务在 $q$ 和 $\mu$ 中都具有正概率，即 $q_i>0$、$\mu_i>0$。直观而言，这意味着不能把任何任务彻底排除在训练或目标分布之外；实现中可通过加入很小的平滑概率满足这一要求。概率单纯形本身只要求 $p_i\geq0$，无需预先额外规定 $p_i>0$。

#### （2）离散型 SPDL 课程优化目标

给定课程分布 $p$，策略在该课程下的性能可写为

$$
J(\theta,p)=\sum_{i=1}^Np_iV^{\pi_\theta}(c_i).
$$

第 $k$ 轮中，内层 Actor–Critic 学习器先将策略参数和评论家参数由 $(\theta_k,\phi_k)$ 更新为 $(\theta_{k+1},\phi_{k+1})$；随后固定更新后的策略，仅优化课程分布。设 $a_i$ 为上下文 $c_i$ 的任务价值估计，则离散型 SPDL 的课程更新定义为

$$
\begin{aligned}
p^{(k+1)}
&=\operatorname*{arg\,max}_{p\in\Delta_N}
\left\{
\sum_{i=1}^Np_i a_i
-\alpha_kD_{\mathrm{KL}}(p\|\mu)
\right\} \\
\text{s.t.}\quad
&D_{\mathrm{KL}}(p\|q)\leq\varepsilon,
\end{aligned}
\tag{2}
$$

其中，$\alpha_k\geq0$ 为目标分布正则系数，$\varepsilon>0$ 为相邻课程分布的 KL 信赖域半径，且

$$
D_{\mathrm{KL}}(p\|r)=\sum_{i=1}^Np_i\log\frac{p_i}{r_i}.
$$

式（2）由价值项、目标分布正则项和相邻课程信赖域约束构成。价值项使课程优先包含当前策略具有较高预期回报的任务；目标分布正则项驱动课程逐步接近最终任务分布；KL 信赖域则抑制由价值估计误差引起的单轮大幅跳变。

#### （3）经验价值项与实际估计方法

首先给出 on-policy 情形下的经验目标推导。若采用 PPO 等 on-policy 学习器，本轮采样的 $M$ 个上下文满足

$$
c^{(m)}\stackrel{\mathrm{i.i.d.}}{\sim}q,
\qquad m=1,\ldots,M.
$$

令 $\widehat V^{(m)}$ 表示更新后策略在 $c^{(m)}$ 上的价值估计，则任意候选课程分布 $p$ 的重要性采样形式为

$$
\widehat J_{k+1}^{\mathrm{IS}}(p)
=\frac{1}{M}\sum_{m=1}^M
\frac{p(c^{(m)})}{q(c^{(m)})}
\widehat V^{(m)}.
\tag{3}
$$

将同一上下文对应的项合并，可定义

$$
a_i^{\mathrm{IS}}
=\frac{1}{M}\sum_{m=1}^M
\frac{\mathbb I\{c^{(m)}=c_i\}}{q_i}
\widehat V^{(m)},
\qquad i=1,\ldots,N,
\tag{4}
$$

从而有 $\widehat J_{k+1}^{\mathrm{IS}}(p)=\sum_{i=1}^Np_ia_i^{\mathrm{IS}}$。若 $\widehat V^{(m)}$ 在给定 $c^{(m)}$ 时对更新后策略的真实价值无偏，且其构造不与同一采样过程产生额外依赖，则式（3）在期望意义下无偏。

然而，$M$ 有限而任务池可能较大。若 $c_i$ 未在本轮出现，则式（4）会机械地给出 $a_i^{\mathrm{IS}}=0$；将该随机系数代入指数型课程更新，会不合理地下调未采样任务的概率。此外，利用同一批 PPO 数据训练评论家后再评价该批样本，并不能自动满足上述无偏条件。因此，式（3）--（4）仅用于说明 on-policy 经验目标的来源，而不作为本文的实际课程价值估计器。

本文在每轮内层学习器更新后，对完整有限任务池进行批量评论家估值，并定义

$$
a_i
=\widehat V_{\phi_{k+1}}^{\pi_{\theta_{k+1}}}\bigl(g(c_i)\bigr),
\qquad i=1,\ldots,N.
\tag{5}
$$

式（5）是实际用于式（2）的 $a_i$ 构造。它保留了课程目标对 $p$ 的线性结构，同时避免了未采样上下文的零估计和由此导致的高方差。全部 $N$ 个任务价值可通过一次或若干次批量前向计算获得。

不同 Actor–Critic 算法中，式（5）的计算接口有所不同：

- 对具有状态价值评论家的 PPO、A2C 等算法，$a_i=\widehat V_{\phi_{k+1}}(s_i)$；
- 对具有随机策略动作价值评论家的算法，$a_i=\mathbb E_{a\sim\pi_{\theta_{k+1}}(\cdot\mid s_i)}[\widehat Q_{\phi_{k+1}}(s_i,a)]$；
- 对具有确定性策略动作价值评论家的算法，$a_i=\widehat Q_{\phi_{k+1}}(s_i,\pi_{\theta_{k+1}}(s_i))$。

使用式（5）时需要注意：不同任务的 $a_i$ 必须处于同一回报尺度，才能比较“哪个任务对当前策略更容易”。若内层算法使用熵正则、奖励归一化或其他附加项，则课程价值宜采用去除附加项后的评价回报，或用独立评估轨迹进行校准；不应直接比较不同尺度的评论家输出。

#### （4）自步调系数

为使课程在早期优先适应当前策略能力、在后期逐渐向目标任务分布迁移，可设置 warm-up 长度 $K_0$。本文保持训练奖励的原始尺度，不使用随训练过程变化的回报归一化，并采用如下自步调系数：

$$
\alpha_k=
\begin{cases}
0, & k\leq K_0,\\
\zeta\dfrac{\max\{0,\bar R_k\}}
{D_{\mathrm{KL}}(q\|\mu)},
& k>K_0\ \text{且}\ D_{\mathrm{KL}}(q\|\mu)>\delta_{\mathrm{KL}},
0, & k>K_0\ \text{且}\ D_{\mathrm{KL}}(q\|\mu)\leq\delta_{\mathrm{KL}}\ \text{（停止课程更新）},
\end{cases}
\tag{6}
$$

其中，$\zeta>0$ 为比例系数，$\bar R_k$ 为第 $k$ 轮策略优化期间已完成回合的原始平均折扣回报，$\delta_{\mathrm{KL}}>0$ 为目标分布接近时的更新停止阈值。$\max\{0,\bar R_k\}$ 保证 $\alpha_k\geq0$：当平均回报为负时，课程不会反向远离目标分布，而是暂时不施加向目标分布靠拢的正则。此时仍可利用全部任务的 Critic 价值在 KL 信赖域内更新课程；当 $D_{\mathrm{KL}}(q\|\mu)\leq\delta_{\mathrm{KL}}$ 时，则直接停止课程更新或保持 $p^{(k+1)}=q$，从而避免分母过小。

### 4.3.2 课程分布的对偶求解

为便于推导，本节省略迭代下标，并将 $\alpha_k$ 简记为 $\alpha$。下面的目标是在“任务价值尽可能高”“课程不要突然改变”“课程最终要靠近目标分布”这三项要求之间取得平衡。将式（2）的目标函数记为

$$
F(p)=\sum_{i=1}^Np_ia_i-\alpha D_{\mathrm{KL}}(p\|\mu).
$$

先将 KL 项展开，得到

$$
F(p)
=\sum_{i=1}^Np_ia_i
-\alpha\sum_{i=1}^Np_i\log p_i
+\alpha\sum_{i=1}^Np_i\log\mu_i.
$$

对某一个任务概率 $p_i$ 求一阶偏导数，可得

$$
\frac{\partial F}{\partial p_i}
=a_i-\alpha(\log p_i+1)+\alpha\log\mu_i
=a_i+\alpha\log\mu_i-\alpha-\alpha\log p_i.
$$

继续求二阶偏导数：当 $i\ne j$ 时，$\frac{\partial^2F}{\partial p_i\partial p_j}=0$；当 $i=j$ 时，$\frac{\partial^2F}{\partial p_i^2}=-\frac{\alpha}{p_i}$。因此 Hessian 矩阵的元素可以统一写为

$$
\left[\nabla^2F(p)\right]_{ij}
=-\frac{\alpha}{p_i}\mathbb I\{i=j\}.
\tag{7}
$$

当 $\alpha>0$ 且 $p_i>0$ 时，Hessian 的对角线元素均为负数，故 $F(p)$ 在概率单纯形内部为严格凹函数。换言之，在满足约束的分布中不存在两个不同的最优课程分布。又因为 $p=q$ 时有 $D_{\mathrm{KL}}(q\|q)=0<\varepsilon$，至少存在一个严格满足 KL 约束的可行分布。因此，可以通过拉格朗日对偶方法求解式（2），并在 $\alpha>0$ 时得到唯一的全局最优课程分布。$\alpha=0$ 的 warm-up 情形将在后文单独讨论。

对 KL 信赖域约束引入对偶变量 $\beta\geq0$，对概率归一化约束引入 $\nu\in\mathbb R$，拉格朗日函数为

$$
\begin{aligned}
\mathcal L(p,\beta,\nu)
=&\sum_{i=1}^Np_ia_i
-\alpha\sum_{i=1}^Np_i\log\frac{p_i}{\mu_i}\\
&-\beta\left(\sum_{i=1}^Np_i\log\frac{p_i}{q_i}-\varepsilon\right)
+\nu\left(1-\sum_{i=1}^Np_i\right).
\end{aligned}
\tag{8}
$$

对每个 $p_i$ 求导并令其为零，得到 KKT 平稳性条件

$$
a_i-\alpha\left(\log p_i-\log\mu_i+1\right)
-\beta\left(\log p_i-\log q_i+1\right)-\nu=0.
\tag{9}
$$

下面逐步整理式（9）。先把与 $\log p_i$ 有关的项移到等式左侧，其余项移到右侧，得到

$$
(\alpha+\beta)\log p_i
=a_i+\alpha\log\mu_i+\beta\log q_i-(\alpha+\beta+\nu).
$$

两边同除以 $\alpha+\beta$，有

$$
\log p_i
=\frac{a_i+\alpha\log\mu_i+\beta\log q_i}{\alpha+\beta}
-\frac{\alpha+\beta+\nu}{\alpha+\beta}.
$$

对两边取指数，可写成

$$
p_i
=\exp\left(
\frac{a_i+\alpha\log\mu_i+\beta\log q_i}{\alpha+\beta}
\right)
\exp\left(
-\frac{\alpha+\beta+\nu}{\alpha+\beta}
\right).
$$

第二个指数项与任务索引 $i$ 无关，可将其记为归一化常数 $C$，即

$$
p_i
=C\exp\left(
\frac{a_i+\alpha\log\mu_i+\beta\log q_i}{\alpha+\beta}
\right).
$$

再利用概率总和为 1 的条件，可得

$$
C
=\frac{1}
{\sum_{j=1}^N
\exp\left(
\dfrac{a_j+\alpha\log\mu_j+\beta\log q_j}{\alpha+\beta}
\right)}.
$$

将 $C$ 代回上式，即得到给定 $\beta$ 时的候选课程分布

$$
p_i(\beta)
=\frac{
\exp\left(
\dfrac{a_i+\alpha\log\mu_i+\beta\log q_i}{\alpha+\beta}
\right)}
{\sum_{j=1}^N
\exp\left(
\dfrac{a_j+\alpha\log\mu_j+\beta\log q_j}{\alpha+\beta}
\right)}.
\tag{10}
$$

式（10）表明，课程分布由任务价值项、目标分布先验项和历史课程先验项共同决定。$\beta$ 越大，候选分布越接近旧课程 $q$。

定义

$$
h(\beta)=D_{\mathrm{KL}}\bigl(p(\beta)\|q\bigr).
\tag{11}
$$

$h(\beta)$ 表示候选课程分布 $p(\beta)$ 与旧课程 $q$ 之间的实际 KL 距离。它直接回答了一个问题：在给定 $\beta$ 时，课程更新是否超过允许的变化范围 $\varepsilon$。为了说明可以只在一条实数轴上搜索 $\beta$，下面说明 $h(\beta)$ 随 $\beta$ 增大不会变大。令

$$
G(p)=\sum_{i=1}^Np_ia_i-\alpha D_{\mathrm{KL}}(p\|\mu).
$$

任取 $0\leq\beta_1<\beta_2$，记 $p_1=p(\beta_1)$、$p_2=p(\beta_2)$。由 $p_1$ 和 $p_2$ 分别最大化 $G(p)-\beta_1D_{\mathrm{KL}}(p\|q)$ 与 $G(p)-\beta_2D_{\mathrm{KL}}(p\|q)$，有

$$
\begin{aligned}
G(p_1)-\beta_1D_{\mathrm{KL}}(p_1\|q)
&\geq G(p_2)-\beta_1D_{\mathrm{KL}}(p_2\|q),\\
G(p_2)-\beta_2D_{\mathrm{KL}}(p_2\|q)
&\geq G(p_1)-\beta_2D_{\mathrm{KL}}(p_1\|q).
\end{aligned}
$$

两式相加可得

$$
(\beta_2-\beta_1)
\left[
D_{\mathrm{KL}}(p_1\|q)-D_{\mathrm{KL}}(p_2\|q)
\right]\geq0.
$$

故 $h(\beta)$ 单调不增；由式（10）可知，当 $\beta\to\infty$ 时 $p(\beta)\to q$，从而 $h(\beta)\to0$。直观上，$\beta$ 相当于“保持旧课程不变”的权重，取值越大，新的课程越不愿离开 $q$。在任务价值非退化且候选分布不发生平坦变化的条件下，$h(\beta)$ 严格递减；出现平坦区间时，取任一满足下述条件的 $\beta$ 即可。

由 KKT 条件，最终的 $\beta$ 还应满足

$$
\beta\geq0,
\qquad h(\beta)\leq\varepsilon,
\qquad \beta\bigl(h(\beta)-\varepsilon\bigr)=0.
\tag{12}
$$

因此，当 $\alpha>0$ 且 $h(0)\leq\varepsilon$ 时，不需要额外限制课程变化，直接取 $\beta^*=0$。当 $h(0)>\varepsilon$ 时，KL 约束被激活，需要寻找满足 $h(\beta^*)=\varepsilon$ 的正数 $\beta^*$。实际搜索分为两步：

1. 从 $\beta_{\mathrm{high}}=1$ 开始不断翻倍，直到 $h(\beta_{\mathrm{high}})\leq\varepsilon$，从而找到包含解的区间；
2. 在 $[\beta_{\mathrm{low}},\beta_{\mathrm{high}}]$ 内反复取中点。若中点的 KL 距离仍大于 $\varepsilon$，说明课程变化仍过大，应增大 $\beta$；反之则减小上界，直至达到设定容差。

在 warm-up 阶段 $\alpha=0$ 时，式（10）在 $\beta=0$ 处无定义。对任意 $\beta>0$，候选分布化为

$$
p_i(\beta)
=\frac{q_i\exp(a_i/\beta)}
{\sum_{j=1}^Nq_j\exp(a_j/\beta)}.
\tag{13}
$$

若所有任务价值在数值容差内相同，则取 $p^{(k+1)}=q$；否则从正的 $\beta$ 开始进行一维根搜索。此时目标函数中没有目标分布正则项，不能直接沿用前述 $\alpha>0$ 情形下的唯一性结论。

### 4.3.3 与 Actor–Critic 类深度强化学习算法的协同训练流程

离散型 SPDL 将任务采样分布作为外层变量，将策略优化与评论家训练交由内层 Actor–Critic 学习器完成。为保持方法的通用性，将内层更新抽象为 `UpdateActorCritic`，将任务价值估计抽象为 `EvaluateContextValues`。前者可由 PPO、SAC、TD3 或 DDPG 等算法实例化，后者按照式（5）输出完整任务池的价值向量。

```text
Algorithm 1: Discrete-SPDL-Distribution-Update
Input: a[1:N], q[1:N], mu[1:N], alpha, epsilon, tol, max_iter
Require: q_i > 0, mu_i > 0, alpha >= 0, epsilon > 0

function Softmax(logits):
    z <- logits - max(logits)
    w <- exp(z)
    return w / sum(w)

function Candidate(beta):
    d <- alpha + beta
    assert d > 0
    for i = 1,...,N do
        logits[i] <- (a[i] + alpha*log(mu[i]) + beta*log(q[i])) / d
    return Softmax(logits)

function KLToPrevious(p):
    return sum_i p[i] * (log(p[i]) - log(q[i]))

if alpha > 0 then
    p0 <- Candidate(0)
    if KLToPrevious(p0) <= epsilon + tol then
        return p0

if alpha == 0 and all a[i] are equal within tol then
    return q

beta_low <- 0
beta_high <- 1
repeat
    p_high <- Candidate(beta_high)
    if KLToPrevious(p_high) <= epsilon then
        break
    beta_high <- 2 * beta_high
    if beta_high exceeds a numerical safety limit then
        raise DualBracketingFailure

for t = 1,...,max_iter do
    beta_mid <- (beta_low + beta_high) / 2
    p_mid <- Candidate(beta_mid)
    h_mid <- KLToPrevious(p_mid)
    if abs(h_mid - epsilon) <= tol then
        return p_mid
    if h_mid > epsilon then
        beta_low <- beta_mid
    else
        beta_high <- beta_mid

return Candidate(beta_high)
```

```text
Algorithm 2: Actor-Critic Discrete-SPDL Training
Input: p^(0), mu, epsilon, K0, zeta, ActorCriticUpdate,
       EvaluateContextValues, optional replay buffer B
Require: p^(0) and mu have full support

Initialize policy parameters theta_0 and critic parameters phi_0
for k = 0,...,K-1 do
    q <- p^(k)
    D_k <- Collect online transitions by sampling task contexts from q
    B <- Append(B, D_k), when an experience replay buffer is used
    (theta_(k+1), phi_(k+1)) <- ActorCriticUpdate(theta_k, phi_k, D_k, B)
    a[1:N] <- EvaluateContextValues(C, theta_(k+1), phi_(k+1))
    alpha_k <- SelfPacedCoefficient(q, mu, K0, zeta, k)
    p^(k+1) <- Discrete-SPDL-Distribution-Update(a, q, mu, alpha_k, epsilon)
return pi_(theta_K)
```

对 PPO 等 on-policy 算法，`ActorCriticUpdate` 仅使用当前轮在线数据 $D_k$，因而式（3）的上下文采样推导直接适用。对 SAC、TD3、DDPG 等 off-policy 算法，在线新经验仍应按当前课程 $q$ 采集，但内层更新可从回放池 $B$ 的历史混合数据中抽样；此时必须在经验中保存任务或上下文标识。无论采用何种内层算法，课程更新均应使用当前策略与当前评论家对全部 $c_i$ 的估值，而不应以回放池内各任务的出现次数或历史平均回报直接构造 $a_i$。

对给定的 $a$，一次候选分布计算和一次 KL 计算的复杂度均为 $O(N)$。若二分法迭代 $T$ 次，则课程分布更新复杂度为 $O(TN)$；按式（5）批量估值全部任务的复杂度为 $O(N)$ 次价值前向计算。因此，外层课程更新总复杂度为 $O((T+1)N)$，不包括内层 Actor–Critic 策略更新开销。

实际训练中还应设置最小采样概率、KL 半径、价值估计更新周期和二分容差，并以独立评估轨迹检查课程价值排序是否与实际任务性能一致。该机制保证的是给定价值估计和 KL 信赖域下的课程分布最优性；它不保证评论家无偏、策略全局最优、训练过程安全或有限任务池外的泛化性能。

## 参考文献

Klink P, D'Eramo C, Peters J, et al. Self-Paced Deep Reinforcement Learning[C]//Advances in Neural Information Processing Systems. 2020.

Schulman J, Wolski F, Dhariwal P, et al. Proximal Policy Optimization Algorithms[EB/OL]. arXiv:1707.06347, 2017.

Haarnoja T, Zhou A, Abbeel P, et al. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor[C]//Proceedings of the 35th International Conference on Machine Learning. 2018.

Fujimoto S, van Hoof H, Meger D. Addressing Function Approximation Error in Actor-Critic Methods[C]//Proceedings of the 35th International Conference on Machine Learning. 2018.
