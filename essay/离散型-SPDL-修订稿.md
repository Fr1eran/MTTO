# 离散型 Self-Paced Deep Reinforcement Learning：理论推导与求解算法

## 1. 方法定位

本文给出 Self-Paced Deep Reinforcement Learning（SPDL；Klink et al., 2020）在有限上下文集合上的严格离散化。它适用于任务、目标、初始状态或环境参数可表示为有限候选池的情形。

离散型 SPDL 在每轮策略改进后学习新的任务采样分布。该分布同时：偏好当前策略具有较高价值的任务；逐步靠近预先给定的目标任务分布；并受相邻课程分布的 KL 信赖域限制。第三项是 SPDL 自步调性质的核心。因此，离散型 SPDL 不等价于仅施加 $\mathbb E_p[V]\ge\delta$ 的性能阈值课程。

## 2. 上下文 MDP 与记号

考虑有限上下文集合

$$
\mathcal C=\{c_1,c_2,\ldots,c_N\}.
$$

每个 $c_i$ 定义一个上下文 MDP $\mathcal M_i$。MDP 可共享状态空间、动作空间和折扣因子，但其初态分布、动力学或奖励函数允许依赖上下文。令 $\pi_\theta$ 为策略，$V_\theta(c_i)$ 为策略在上下文 $c_i$ 中的期望折扣回报。

课程分布和目标分布记为

$$
p=(p_1,\ldots,p_N)^\top\in\Delta_N,\qquad \mu=(\mu_1,\ldots,\mu_N)^\top\in\Delta_N,
$$

$$
\Delta_N=\left\{p\in\mathbb R_+^N:\sum_{i=1}^Np_i=1\right\}.
$$

其中 $p_i=P(c_i)$ 是训练时的任务采样概率，$\mu$ 是最终评价任务分布。令第 $k$ 轮更新前的课程为 $q=p^{(k)}$。以下假设

$$
q_i>0,\qquad \mu_i>0,\qquad i=1,\ldots,N.
$$

该全支持条件保证 $D_{\mathrm{KL}}(p\|q)$ 与 $D_{\mathrm{KL}}(p\|\mu)$ 有限；实践中可用极小的平滑质量满足它。

给定课程分布 $p$，策略在该课程下的性能为

$$
J(\theta,p)=\sum_{i=1}^Np_iV_\theta(c_i).
$$

## 3. 原 SPDL 的有限集合优化形式

第 $k$ 轮首先在旧课程 $q$ 下采样轨迹，并用任意合适的 RL 算法将 $\theta_k$ 更新为 $\theta_{k+1}$。然后固定策略，仅优化课程分布：

$$
p^{(k+1)}=\arg\max_{p\in\Delta_N}\left\{\widehat J_{k+1}(p)-\alpha_kD_{\mathrm{KL}}(p\|\mu)\right\},\qquad D_{\mathrm{KL}}(p\|q)\le\varepsilon.
\tag{1}
$$

其中 $\alpha_k\ge0$ 是目标分布正则系数，$\varepsilon>0$ 是相邻课程分布的 KL 信赖域半径，并且

$$
D_{\mathrm{KL}}(p\|r)=\sum_{i=1}^Np_i\log\frac{p_i}{r_i}.
$$

式（1）的第一项偏好当前有较高估计价值的任务；目标 KL 项把学习方向限定到 $\mu$；相邻 KL 约束使课程不能因有限样本价值误差而单步跳变。删除该相邻 KL 约束后，所得方法不再是原始 SPDL 的等价离散化。

### 3.1 经验价值项与重要性采样

令本轮得到 $M$ 个上下文样本 $i_m$，第 $m$ 个样本由已知行为分布 $b_m$ 产生，且 $b_m(i_m)>0$。策略更新后，在该上下文上取得的价值估计为 $\widehat V_m$。则有重要性采样估计

$$
\widehat J_{k+1}(p)=\frac1M\sum_{m=1}^M\frac{p_{i_m}}{b_m(i_m)}\widehat V_m.
\tag{2}
$$

将同一上下文的项合并，定义

$$
a_i=\frac1M\sum_{m:i_m=i}\frac{\widehat V_m}{b_m(i)}.
$$

于是，式（2）可写为线性形式

$$
\widehat J_{k+1}(p)=\sum_{i=1}^Np_ia_i.
\tag{3}
$$

若所有有限上下文均可可靠地估值，直接令 $a_i=V_{\theta_{k+1}}(c_i)$ 即可，无需采用式（2）。后续推导仅要求价值项具有式（3）的线性形式。

### 3.2 自步调系数

原 SPDL 在 warm-up 阶段首先令目标正则失效，之后逐步增强向目标分布靠拢的压力。可采用

$$
\alpha_k=0\quad(k\le K_0),\qquad \alpha_k=\zeta\frac{\bar R_k}{D_{\mathrm{KL}}(q\|\mu)}\quad(k>K_0),
\tag{4}
$$

其中 $K_0$ 是 warm-up 长度，$\zeta>0$ 是比例系数，$\bar R_k$ 为固定尺度下的平均折扣回报。若回报可为负或归一化尺度随训练改变，必须在方法中明确定义替代的、有界非负课程回报；不能无说明地沿用式（4）。

## 4. 凸性与拉格朗日函数

将式（3）代入式（1），其目标函数为

$$
F(p)=\sum_{i=1}^Np_ia_i-\alpha\sum_{i=1}^Np_i\log\frac{p_i}{\mu_i},
$$

其中为简洁起见令 $\alpha=\alpha_k$。在单纯形内部，Hessian 为

$$
\frac{\partial^2F}{\partial p_i\partial p_j}=-\frac{\alpha}{p_i}\mathbb I\{i=j\}.
$$

所以当 $\alpha>0$ 时，$F$ 严格凹；概率单纯形和 KL 信赖域均为凸集。故式（1）属于凸优化意义下的最大化凹函数问题，且最优解唯一。

对约束 $D_{\mathrm{KL}}(p\|q)\le\varepsilon$ 引入 $\beta\ge0$，对归一化约束引入 $\nu\in\mathbb R$。其拉格朗日函数为

$$
\mathcal L(p,\beta,\nu)=\sum_ip_ia_i-\alpha\sum_ip_i\log\frac{p_i}{\mu_i}-\beta\left(\sum_ip_i\log\frac{p_i}{q_i}-\varepsilon\right)+\nu\left(1-\sum_ip_i\right).
\tag{5}
$$

## 5. 闭式候选分布

对每个 $p_i$ 求一阶导数并令其为零：

$$
a_i-\alpha\left(\log p_i-\log\mu_i+1\right)-\beta\left(\log p_i-\log q_i+1\right)-\nu=0.
$$

因此

$$
(\alpha+\beta)\log p_i=a_i+\alpha\log\mu_i+\beta\log q_i-(\alpha+\beta+\nu).
\tag{6}
$$

由归一化条件消去 $\nu$，可得给定对偶变量 $\beta$ 时的广义 softmax 解：

$$
p_i(\beta)=\frac{\exp\left(\dfrac{a_i+\alpha\log\mu_i+\beta\log q_i}{\alpha+\beta}\right)}{\sum_{j=1}^N\exp\left(\dfrac{a_j+\alpha\log\mu_j+\beta\log q_j}{\alpha+\beta}\right)}.
\tag{7}
$$

式（7）是价值项、目标先验和旧课程先验的几何混合。$\beta$ 越大，更新越接近旧课程；当 $\beta\to\infty$ 时，$p(\beta)\to q$。

## 6. 对偶变量的唯一一维求解

定义

$$
h(\beta)=D_{\mathrm{KL}}\bigl(p(\beta)\|q\bigr).
$$

KKT 条件为

$$
\beta\ge0,\qquad h(\beta)\le\varepsilon,\qquad \beta\bigl(h(\beta)-\varepsilon\bigr)=0.
\tag{8}
$$

若 $\alpha>0$ 且 $h(0)\le\varepsilon$，无约束候选分布已可行，故 $\beta^*=0$。否则约束活跃，需解 $h(\beta^*)=\varepsilon$。

为证明可用一维根查找，令

$$
G(p)=\sum_ip_ia_i-\alpha D_{\mathrm{KL}}(p\|\mu).
$$

取 $0\le\beta_1<\beta_2$，并令 $p_1$、$p_2$ 分别是对应的最优候选。由两者在各自拉格朗日目标下的最优性，得到

$$
G(p_1)-\beta_1D_{\mathrm{KL}}(p_1\|q)\ge G(p_2)-\beta_1D_{\mathrm{KL}}(p_2\|q),
$$

$$
G(p_2)-\beta_2D_{\mathrm{KL}}(p_2\|q)\ge G(p_1)-\beta_2D_{\mathrm{KL}}(p_1\|q).
$$

两式相加可得

$$
(\beta_2-\beta_1)\left[D_{\mathrm{KL}}(p_1\|q)-D_{\mathrm{KL}}(p_2\|q)\right]\ge0.
$$

故 $h(\beta)$ 单调不增，并在 $\beta\to\infty$ 时趋于零。可先通过倍增搜索找到满足 $h(\beta_{\mathrm{high}})\le\varepsilon$ 的上界，再用二分法或 Brent 法求根。

在 warm-up 阶段 $\alpha=0$ 时，式（7）的 $\beta=0$ 未定义。对于 $\beta>0$，候选分布化为

$$
p_i(\beta)=\frac{q_i\exp(a_i/\beta)}{\sum_{j=1}^Nq_j\exp(a_j/\beta)}.
\tag{9}
$$

此时直接从正的 $\beta$ 开始对 KL 约束进行求根；若全部 $a_i$ 在数值容差内相同，取 $p=q$。

## 7. 数值稳定的分布更新伪代码

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

对已给定的 $a$，每次计算候选分布和 KL 值均为 $O(N)$。设二分法迭代 $T$ 次，则分布更新的复杂度为 $O(TN)$；构造式（3）的经验系数额外需要 $O(M)$。

## 8. 外层 SPDL 训练伪代码

```text
Algorithm 2: Discrete Self-Paced Deep Reinforcement Learning
Input: p^(0), mu, epsilon, K0, zeta, RL learner ImprovePolicy
Require: p^(0) and mu have full support

Initialize policy parameters theta_0
for k = 0,...,K-1 do
    q <- p^(k)
    Sample M contexts from q and collect trajectories under pi_(theta_k)
    theta_(k+1) <- ImprovePolicy(theta_k, trajectories)
    Estimate post-update values and construct a according to Eq. (3)
    Compute alpha_k according to Eq. (4), or another explicitly stated schedule
    p^(k+1) <- Discrete-SPDL-Distribution-Update(a, q, mu, alpha_k, epsilon)
return pi_(theta_K)
```

## 9. 方法边界与论文报告要求

论文应报告有限上下文池的生成机制、目标分布 $\mu$、行为分布 $b_m$、价值估计方式、$\alpha_k$ 调度、KL 半径 $\varepsilon$、全支持平滑常数及所有数值容差。若数据来自重放或多个旧课程分布，必须保存每条样本生成时的行为概率，以便在式（2）中使用正确的重要性分母。

该方法保证的是相对于估计目标和 KL 信赖域的课程分布最优性；它不保证策略全局最优、价值估计无偏、任务安全或最终性能单调提升。有限上下文池之外的泛化能力也不能由该推导直接保证，应通过独立实验验证。

## 参考文献

P. Klink, C. D'Eramo, J. Peters, and J. Pajarinen. *Self-Paced Deep Reinforcement Learning*. Advances in Neural Information Processing Systems, 2020.
