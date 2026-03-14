---
tags:
  - RL
  - LinearMDP
  - LSVI-UCB
  - Bernstein
  - RareSwitching
  - Obsidian
date: 2026-03-14
---

## 使用说明

这份错题本按如下主线组织：

1. **Jin et al. (2020)**：为什么在线性 MDP 中，`LSVI + 普通岭回归 + Hoeffding 型 bonus` 已经足够得到第一个多项式 regret bound。
2. **困难的本质**：为什么不能把“固定值函数时好用的 self-normalized / Bernstein”直接套到 `V_{k,h+1}` 上。
3. **He et al. (2023)**：如何通过 `variance-aware weighted ridge regression + 最优值/误差分解 + 单调方差上界 + rare-switching` 把主项推进到近极小极大最优。
4. **总账本**：每一个技术点到底消掉了哪一个因子：$\sqrt H$、$\sqrt d$、还是“函数类复杂度爆炸”的隐患。

## 建议先记住的符号

- 线性 MDP：对每个阶段 $h\in[H]$，存在未知测度 $\mu_h(\cdot)$ 与已知特征 $\phi:S\times A\to \mathbb R^d$，使得
  $$
  P_h(s'|s,a)=\langle \phi(s,a),\mu_h(s')\rangle,\qquad r_h(s,a)=\langle \phi(s,a),\theta_h\rangle
  $$
  且 $\|\phi(s,a)\|_2\le 1$。
- Jin 2020 的 Gram 矩阵：
  $$
  \Lambda_{k,h}=\lambda I+\sum_{\tau=1}^{k-1}\phi(s^\tau_h,a^\tau_h)\phi(s^\tau_h,a^\tau_h)^\top .
  $$
- He 2023 的加权 Gram 矩阵：
  $$
  \Sigma_{k,h}=\lambda I+\sum_{i=1}^{k-1}\bar\sigma_{i,h}^{-2}\phi(s^i_h,a^i_h)\phi(s^i_h,a^i_h)^\top .
  $$
- 转移噪声：
  $$
  \eta_{i,h}(V):=V(s^i_{h+1})-[P_hV](s^i_h,a^i_h).
  $$
- 方差算子：
  $$
  [\mathbb V_h V](s,a):=[P_h(V^2)](s,a)-([P_hV](s,a))^2.
  $$

## 一条先导提醒

不要把 **linear MDP** 和 **linear mixture MDP** 混为一谈。  
后者很多近最优结果依赖 **value-targeted regression (VTR)** 或对 base model 的积分/采样 oracle；而这份错题本的重点是：**在线性 MDP 里，如何保持“直接逼近 value function”的计算效率，同时把 regret 做到近最优。**

---

## A. 先把 Jin 2020 的逻辑链打通

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 1：固定 $V$ 时，Hoeffding / self-normalized 为什么能直接用？** >
> 设 $\{x_i\}_{i=1}^{k-1}\subset \mathbb R^d$ 是可预测向量，满足 $\|x_i\|_2\le 1$，并令
> $$
> \Lambda_k=\lambda I+\sum_{i=1}^{k-1}x_ix_i^\top.
> $$
> 设 $V:S\to[0,H]$ 为一个**固定**函数，且与 $\{s^i_{h+1}\}_{i=1}^{k-1}$ 独立于“是否由未来 LSVI 计算出来”这个问题无关；定义噪声
> $$
> \xi_i:=V(s^i_{h+1})-[P_hV](s^i_h,a^i_h).
> $$
> 证明或推出：$\{\xi_i\}$ 是有界 martingale difference sequence，且高概率下
> $$
> \left\|\sum_{i=1}^{k-1}x_i\xi_i\right\|_{\Lambda_k^{-1}}
> \lesssim H\sqrt{d\log\frac{k+\lambda}{\lambda\delta}} .
> $$
> 进一步推出，对任意 $(s,a)$,
> $$
> \left|\phi(s,a)^\top\Lambda_k^{-1}\sum_{i=1}^{k-1}x_i\xi_i\right|
> \lesssim
> H\sqrt{d\log\frac{k+\lambda}{\lambda\delta}}\cdot
> \sqrt{\phi(s,a)^\top\Lambda_k^{-1}\phi(s,a)}.
> $$

> [!success]- 核心思路（点击展开）
> 先写出条件期望：
> $$
> \mathbb E[\xi_i\mid \mathcal F_{i-1}]
> =
> \mathbb E[V(s^i_{h+1})\mid \mathcal F_{i-1}]
> -[P_hV](s^i_h,a^i_h)=0 .
> $$
> 又因为 $V\in[0,H]$，所以 $|\xi_i|\le H$。  
> 于是 $\xi_i$ 是有界 MDS，可以直接套线性 bandit 里常见的 self-normalized concentration。  
> 这一步的本质是：**只要 $V$ 是“固定靶子”，就能把 $V(s_{h+1})$ 当成普通有界噪声处理。**

> [!info]- 错误原因 / induce（点击展开）
> 常见卡点：把“值函数是固定的”与“值函数是由数据算出来的”混为一谈。  
> 本题价值：这是整条路线的出发点。后面所有麻烦，都是因为在线性 MDP 里真正需要估计的其实是 **自适应的** $V_{k,h+1}$，而不是固定的 $V$。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 2：为什么一旦换成 $V_{k,h+1}$，标准 self-normalized 就“不能直接用了”？** >
> 在 Jin 2020 的分析里，关键误差项可写成
> $$
> (\widehat P_h-\overline P_h)V_{k,h+1}(s,a)
> =
> \phi(s,a)^\top\Lambda_{k,h}^{-1}
> \sum_{\tau=1}^{k-1}\phi(s^\tau_h,a^\tau_h)\eta_{\tau,h}(V_{k,h+1}).
> $$
> 解释为什么如果 $V_{k,h+1}$ 本身是由 later-stage LSVI 用历史数据 $\{s^\tau_{h+1},a^\tau_{h+1}\}_{\tau<k}$ 算出来的，那么
> $$
> \eta_{\tau,h}(V_{k,h+1})
> =
> V_{k,h+1}(s^\tau_{h+1})-[P_hV_{k,h+1}](s^\tau_h,a^\tau_h)
> $$
> **不再是“对固定函数 $V$ 的噪声”**，从而不能直接照搬题 1 的结论。
>
> 接着，设 $\mathcal V_h$ 是一个值函数类，且 $\mathcal N_\varepsilon\subset \mathcal V_h$ 是它在 $\|\cdot\|_\infty$ 下的 $\varepsilon$-net。证明下面这种 decoupling 形式：
> $$
> \sup_{V\in\mathcal V_h}
> \left\|
> \sum_{\tau=1}^{k-1}\phi_\tau\eta_\tau(V)
> \right\|_{\Lambda_k^{-1}}
> \le
> \max_{\widetilde V\in\mathcal N_\varepsilon}
> \left\|
> \sum_{\tau=1}^{k-1}\phi_\tau\eta_\tau(\widetilde V)
> \right\|_{\Lambda_k^{-1}}
> +
> O\!\left(\frac{k\varepsilon}{\sqrt\lambda}\right).
> $$

> [!success]- 核心思路（点击展开）
> 核心分解是 $V=\widetilde V+\Delta V$，其中 $\widetilde V$ 落在 $\varepsilon$-net 上，且 $\|\Delta V\|_\infty\le\varepsilon$。  
> 对 net 上的每个 $\widetilde V$，它是“固定函数”，因此可套题 1；再对所有 $\widetilde V\in\mathcal N_\varepsilon$ 做 union bound。  
> 剩余误差由
> $$
> |\Delta V(s^\tau_{h+1})-\mathbb E[\Delta V(s^\tau_{h+1})\mid \mathcal F_{\tau-1}]|
> \le 2\varepsilon
> $$
> 控制，进而得到 $O(k\varepsilon/\sqrt\lambda)$ 项。  
> 这正是 Jin 2020 里“covering-number uniform concentration”的本质。

> [!info]- 错误原因 / induce（点击展开）
> 常见卡点：以为“有界 + martingale”就够了，忽略了**随机函数本身也依赖历史样本**。  
> 本题价值：它解释了为什么 Jin 2020 不是“不会用 self-normalized”，而是**不能直接用固定函数版的 self-normalized**。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 3：$\varepsilon$-net / covering number 到底给 regret 多带了什么？** >
> 考虑函数类
> $$
> \mathcal V_h
> =
> \left\{
> s\mapsto
> \min\!\left\{
> \max_a
> \left(
> w^\top\phi(s,a)
> +\beta\sqrt{\phi(s,a)^\top\Lambda^{-1}\phi(s,a)}
> \right),
> H
> \right\}
> \right\},
> $$
> 其中 $\|w\|_2\le L$，$\beta\in[0,B]$，$\lambda_{\min}(\Lambda)\ge \lambda$。  
> 按照 $A=\beta^2\Lambda^{-1}$ 的重参数化方法，证明一个类似下面的 covering bound：
> $$
> \log N_\varepsilon(\mathcal V_h,\|\cdot\|_\infty)
> \lesssim
> d\log\!\left(1+\frac{L}{\varepsilon}\right)
> +d^2\log\!\left(1+\frac{B^2}{\lambda\varepsilon^2}\right).
> $$
> 然后说明：如果 uniform concentration 的主项长成
> $$
> H\sqrt{d^2\log\frac{k+\lambda}{\lambda}+\log N_\varepsilon},
> $$
> 那么从数量级上看，confidence radius 会从“固定 $V$ 时的”
> $$
> \tilde O(H\sqrt d)
> $$
> 变成“自适应 $V_{k,h+1}$ 时的”
> $$
> \tilde O(dH).
> $$

> [!success]- 核心思路（点击展开）
> 先把二次型 bonus 改写成 $\sqrt{\phi^\top A\phi}$。  
> 对 $w$ 做 $d$-维球覆盖，对 $A$ 做 $d^2$-维 Frobenius 球覆盖；再利用
> $$
> \left|\sqrt{x}-\sqrt{y}\right|\le \sqrt{|x-y|}
> $$
> 与 $\max_a$、$\min\{\cdot,H\}$ 的 contraction 性质，把函数距离降到参数距离。  
> 关键是：**矩阵参数 $A$ 的自由度是 $d^2$**，从 $\log N_\varepsilon$ 进入平方根后，额外给出一个 $\sqrt{d^2}=d$。

> [!info]- 错误原因 / induce（点击展开）
> 常见卡点：只盯着“覆盖数里有个 log”，没意识到 log 里面藏着 $d^2$ 维矩阵参数。  
> 本题价值：这就是 Jin 2020 到 He 2023 之间那个 $\sqrt d$ gap 的来源之一。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 4：从单步 bonus 总和，重建 Jin 2020 的 $\tilde O(\sqrt{d^3H^4K})$ 主项** >
> 假设你已经知道 Jin 2020 的关键不等式可整理成
> $$
> \mathrm{Regret}(K)
> \lesssim
> \sum_{k=1}^{K}\sum_{h=1}^{H}
> \beta\sqrt{\phi_{k,h}^\top\Lambda_{k,h}^{-1}\phi_{k,h}}
> +
> \sum_{k=1}^{K}\sum_{h=1}^{H}\zeta_{k,h},
> $$
> 其中
> $$
> \beta=\tilde \Theta(dH),\qquad
> \sum_{k,h}\zeta_{k,h}\lesssim H\sqrt{T},
> $$
> 且
> $$
> \sum_{k=1}^{K}\sum_{h=1}^{H}\phi_{k,h}^\top\Lambda_{k,h}^{-1}\phi_{k,h}
> \lesssim dH\log(1+K/\lambda).
> $$
> 试用 Cauchy–Schwarz 与 elliptical potential lemma 推导
> $$
> \mathrm{Regret}(K)=\tilde O\!\left(\sqrt{d^3H^4K}\right)
> =\tilde O\!\left(\sqrt{d^3H^3T}\right).
> $$

> [!success]- 核心思路（点击展开）
> 先令
> $$
> x_{k,h}:=\phi_{k,h}^\top\Lambda_{k,h}^{-1}\phi_{k,h}.
> $$
> 则
> $$
> \sum_{k,h}\sqrt{x_{k,h}}
> \le
> \sqrt{KH}\cdot \sqrt{\sum_{k,h}x_{k,h}}
> \lesssim
> \sqrt{KH}\cdot \sqrt{dH\log(1+K/\lambda)}.
> $$
> 乘上 $\beta=\tilde \Theta(dH)$ 后得到
> $$
> \tilde O\!\left(dH\cdot \sqrt{dH\cdot KH}\right)
> =
> \tilde O\!\left(d^{3/2}H^2\sqrt K\right)
> =
> \tilde O\!\left(\sqrt{d^3H^4K}\right).
> $$
> 这就是 Jin 2020 主项的“总账本”。

> [!info]- 错误原因 / induce（点击展开）
> 常见卡点：只记住某个局部 lemma，却不会把它们拼成 regret。  
> 本题价值：你会更清楚地看到，**Jin 2020 的大头来自 $\beta\sim dH$**，而不是来自后面的 bonus 求和步骤。

---

## B. 再看 Bernstein / variance-aware 路线到底强在哪里

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 5：Bernstein-type self-normalized 在加权岭回归里的一般形态** >
> 设
> $$
> y_i=\langle \mu^\star,x_i\rangle+\eta_i,
> \qquad
> \mathbb E[\eta_i\mid \mathcal G_i]=0,
> \qquad
> \mathbb E[\eta_i^2\mid \mathcal G_i]\le \sigma_i^2,
> $$
> 且 $\|x_i\|_2\le 1$。取权重 $w_i=1/\bar\sigma_i$，其中 $\bar\sigma_i\ge \sigma_i$，并定义
> $$
> Z_k=\lambda I+\sum_{i=1}^{k}w_i^2x_ix_i^\top,\qquad
> \widehat\mu_k
> =
> Z_k^{-1}\sum_{i=1}^{k}w_i^2y_ix_i.
> $$
> 假设有 Bernstein-type self-normalized inequality 可用，试推出一个形如
> $$
> \|\widehat\mu_k-\mu^\star\|_{Z_k}
> \lesssim
> \sqrt d
> +
> \max_{i\le k}|\eta_i|
> \min\{1,\|w_ix_i\|_{Z_{i-1}^{-1}}\}
> +\sqrt\lambda\|\mu^\star\|_2
> $$
> 的信赖域，并解释：若 $w_i\approx 1/\sigma_i$，为什么主项的量级会从“Hoeffding 风格里的 $H\sqrt d$”下降到“Bernstein 风格里的 $\sqrt d$”。

> [!success]- 核心思路（点击展开）
> 加权以后，等价于把每个样本正规化成
> $$
> \widetilde x_i=w_ix_i,\qquad \widetilde \eta_i=w_i\eta_i,
> $$
> 于是 $\mathbb E[\widetilde\eta_i^2\mid \mathcal G_i]\le 1$。  
> 这时自归一化界的“有效噪声尺度”不再是统一的 $H$，而是被方差归一化成了常数量级。  
> 因此 leading term 从
> $$
> \tilde O(H\sqrt d)
> \quad\text{变成}\quad
> \tilde O(\sqrt d).
> $$
> 这正是 variance-aware weighting 的收益。

> [!info]- 错误原因 / induce（点击展开）
> 常见卡点：觉得“加权只是技巧性重写”。  
> 本题价值：它实际上是在把**异方差**吸收到设计矩阵里，使得高方差样本少说话、低方差样本多说话。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 6：为什么“直接把 $V_{k,h+1}$ 的方差估进去，再套 Bernstein”并不能自动解决问题？** >
> 形式上，如果直接模仿题 5，你会想把
> $$
> \eta_{i,h}(V_{k,h+1})
> =
> V_{k,h+1}(s^i_{h+1})-[P_hV_{k,h+1}](s^i_h,a^i_h)
> $$
> 看成一个条件方差不超过
> $$
> [\mathbb V_hV_{k,h+1}](s^i_h,a^i_h)
> $$
> 的噪声，然后对其加权。  
> 试说明：
>
> 1. 即使你真的引入了
>    $$
>    w_{i,h}\approx \frac1{\sqrt{[\mathbb V_hV_{k,h+1}](s^i_h,a^i_h)}},
>    $$
>    $V_{k,h+1}$ 仍然是**依赖历史样本的随机函数**；
> 2. 因而要想同时对所有可能出现的 $V_{k,h+1}$ 成立，仍然需要类似题 2 的 covering / decoupling；
> 3. 这会把你刚刚想省掉的 $\sqrt d$ 又以 $\log N_\varepsilon$ 的形式带回来。
>
> 最后用一句话总结：为什么“**加方差**”和“**摆脱自适应目标带来的函数类复杂度**”是两件不同的事情。

> [!success]- 核心思路（点击展开）
> 直接上 Bernstein 只解决了“噪声尺度太粗，只能看到 $H$”的问题；  
> 但如果被估计的目标函数 $V_{k,h+1}$ 本身依赖数据，那么**主矛盾变成：如何对一个随机函数类做统一控制**。  
> 所以：
> $$
> \text{Hoeffding} \to \text{Bernstein}
> $$
> 只能潜在消掉 $\sqrt H$；  
> 而
> $$
> \text{adaptive }V_{k,h+1}\to \text{fixed }V^\star+\text{small gap}
> $$
> 才是消掉 $\sqrt d$ 的关键。

> [!info]- 错误原因 / induce（点击展开）
> 常见卡点：把“variance-aware”误认为万能钥匙。  
> 本题价值：它会逼你把两个 gap 区分清楚：  
> - **$H$ gap**：来自 Hoeffding vs Bernstein；  
> - **$d$ gap**：来自 adaptive target vs fixed target。

---

## C. He 2023 的真正突破：把主误差拆成固定部分与小误差部分

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 7：为什么分解成 $V^\star_{h+1}$ 与 $\Delta V_{k,h+1}$ 可以去掉主要的 covering 代价？** >
> 定义
> $$
> \Delta V_{k,h+1}:=V_{k,h+1}-V^\star_{h+1}.
> $$
> 证明噪声项可写为
> $$
> \eta_{i,h}(V_{k,h+1})
> =
> \eta_{i,h}(V^\star_{h+1})
> +
> \eta_{i,h}(\Delta V_{k,h+1}).
> $$
> 因而
> $$
> \phi(s,a)^\top\Sigma_{k,h}^{-1}
> \sum_{i=1}^{k-1}\bar\sigma_{i,h}^{-2}\phi_i\eta_{i,h}(V_{k,h+1})
> = I_1+I_2,
> $$
> 其中
> $$
> I_1:=\phi^\top\Sigma^{-1}\sum_i\bar\sigma_i^{-2}\phi_i\eta_i(V^\star_{h+1}),\qquad
> I_2:=\phi^\top\Sigma^{-1}\sum_i\bar\sigma_i^{-2}\phi_i\eta_i(\Delta V_{k,h+1}).
> $$
> 然后说明：
>
> - 为什么 $I_1$ 可以直接用 Bernstein-type self-normalized 来控，而**不需要**对大函数类做 covering；
> - 为什么 $I_2$ 虽然仍然带有 adaptive target，但如果 $\Delta V_{k,h+1}$ 已经很小，它就会成为 lower-order term。

> [!success]- 核心思路（点击展开）
> 真正关键的一步不是“换了不等式”，而是**换了被控对象**。  
> 因为 $V^\star_{h+1}$ 是跨 episode 固定的，所以 $I_1$ 对应的是“固定靶子 + 方差加权”，正是题 5 的理想适用场景。  
> 只有 $I_2$ 还需要面对自适应性；但 $I_2$ 的大小与 $\Delta V_{k,h+1}$ 成正相关。  
> 所以重的 covering 成本只落在“小量”上，而不再落在整个主项上。

> [!info]- 错误原因 / induce（点击展开）
> 常见卡点：以为这是一个纯技巧性分解。  
> 本题价值：这是 He 2023 里最应该反复咀嚼的地方——**把固定的 $V^\star$ 从随机的 $V_k$ 里剥出来**，本质上是在把“最难的 uniform control”压缩到一个小误差层。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 8：如何控制参考值/误差分解本身带来的方差？** >
> 设
> $$
> 0\le \Delta V_{k,h+1}(s)\le H,\qquad \forall s.
> $$
> 证明
> $$
> [\mathbb V_h\Delta V_{k,h+1}](s,a)
> =
> [P_h(\Delta V_{k,h+1}^2)](s,a)-([P_h\Delta V_{k,h+1}](s,a))^2
> \le
> [P_h(\Delta V_{k,h+1}^2)](s,a)
> \le
> H\,[P_h\Delta V_{k,h+1}](s,a),
> $$
> 从而也可用更松但更方便的形式
> $$
> [\mathbb V_h\Delta V_{k,h+1}](s,a)\le 2H\,[P_h\Delta V_{k,h+1}](s,a).
> $$
>
> 再设存在悲观值函数 $\check V_{k,h+1}\le V^\star_{h+1}\le V_{k,h+1}$，证明
> $$
> \Delta V_{k,h+1}=V_{k,h+1}-V^\star_{h+1}
> \le
> V_{k,h+1}-\check V_{k,h+1},
> $$
> 进而
> $$
> [P_h\Delta V_{k,h+1}](s,a)\le [P_h(V_{k,h+1}-\check V_{k,h+1})](s,a).
> $$
> 解释为什么这使得“optimistic-pessimistic gap”成为控制 $D_{k,h}$ 的天然量。

> [!success]- 核心思路（点击展开）
> 因为 $0\le \Delta V\le H$，逐点有 $(\Delta V)^2\le H\Delta V$，所以
> $$
> P_h[(\Delta V)^2]\le H\,P_h[\Delta V].
> $$
> 方差总是不超过二阶矩，于是第一部分完成。  
> 第二部分只用序关系：
> $$
> \check V\le V^\star\le V_k
> \Longrightarrow
> 0\le V_k-V^\star\le V_k-\check V.
> $$
> 所以如果你能把 optimistic / pessimistic gap 压小，就同时压住了 $\Delta V$ 的方差。

> [!info]- 错误原因 / induce（点击展开）
> 常见卡点：觉得 $D_{k,h}$ 只是 proof artifact。  
> 本题价值：它告诉你，**悲观值函数不是“陪衬”**，而是把“看不见的 $\Delta V$”替换成“可估计的 optimistic-pessimistic gap”的桥梁。

---

## D. 单调方差上界与 rare-switching：不这样做，函数类复杂度会炸

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 9：如果每个 episode 都做一个 $\min$，值函数会怎样？函数类复杂度又会怎样？** >
> 设每一轮得到一个新的 optimistic candidate $\widetilde Q_{k,h}$，并定义
> $$
> Q_{k,h}(s,a)
> =
> \min\{Q_{k-1,h}(s,a),\widetilde Q_{k,h}(s,a),H\},
> \qquad
> V_{k,h}(s)=\max_a Q_{k,h}(s,a).
> $$
> 请证明：
>
> 1. $Q_{k,h}(s,a)$ 关于 $k$ 点态单调不增；
> 2. 因而 $V_{k,h}(s)$ 也点态单调不增；
> 3. 但如果每个 episode 都更新一次，那么 $Q_{k,h}$ 属于“**$k$ 个二次型 / 椭球 bonus 函数的点态最小值**”这一函数类；
> 4. 若对每个分支都做一个 $\varepsilon$-cover，再对所有分支组合取积，解释为什么 covering number 会至少呈现
>    $$
>    \log N_\varepsilon \propto k
>    $$
>    这样的线性增长，从而对 $k=K$ 时会把 uniform concentration 彻底拖垮。

> [!success]- 核心思路（点击展开）
> 前两问只用序关系：
> $$
> Q_k=\min\{Q_{k-1},\widetilde Q_k,H\}\le Q_{k-1}.
> $$
> 再对 action 取 $\max$ 即得 $V_k\le V_{k-1}$。  
> 第三、四问是 rare-switching 的根源：  
> 若每轮都更新，最后的 $Q_k$ 其实是
> $$
> Q_k = \min\{\widetilde Q_1,\widetilde Q_2,\dots,\widetilde Q_k,H\},
> $$
> 于是函数类复杂度会随着“分支数”线性累积到 $\log N_\varepsilon$ 里。  
> 这不是 regret 的局部问题，而是**uniform concentration 失效**的问题。

> [!info]- 错误原因 / induce（点击展开）
> 常见卡点：只看到“取 $\min$”能保证单调，却没看到它同时在制造一个越来越复杂的函数类。  
> 本题价值：你会明白 rare-switching 不是为省计算量而已，更是为省掉 **class complexity explosion**。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 10：rare-switching 如何把更新次数压到 $O(dH\log(1+K/\lambda))$？** >
> 设对每个 stage $h$，只在满足
> $$
> \det(\Sigma_{k,h})\ge 2\det(\Sigma_{k_{\mathrm{last}},h})
> $$
> 时才触发更新 $Q_{k,h},\check Q_{k,h}$。  
> 证明固定 $h$ 时，触发更新的总次数不超过
> $$
> \log_2 \frac{\det(\Sigma_{K+1,h})}{\det(\lambda I)}.
> $$
> 再结合 AM-GM 或 trace-det 不等式，推出
> $$
> \det(\Sigma_{K+1,h})
> \le
> \left(\lambda+\frac{\mathrm{Tr}(\Sigma_{K+1,h}-\lambda I)}{d}\right)^d,
> $$
> 从而得到一个数量级结论
> $$
> \#\{\text{stage }h\text{ 的更新}\}
> \lesssim
> d\log(1+K/\lambda).
> $$
> 最后说明总更新次数为
> $$
> O(dH\log(1+K/\lambda)).
> $$

> [!success]- 核心思路（点击展开）
> 每触发一次，det 至少翻倍；所以更新次数就是最终 $\log\det$ 增量。  
> 又因为 $\Sigma_{k,h}$ 是正定矩阵，可以用
> $$
> \det(M)\le \left(\frac{\mathrm{Tr}(M)}{d}\right)^d
> $$
> 或等价的 AM-GM 估计，把 $\log\det$ 转成 $d\log(\cdot)$。  
> 于是 rare-switching 把“每个 episode 一次更新”压缩成“每个 stage 只有 $O(d\log K)$ 次有效更新”，从而把题 9 里的函数类复杂度从依赖 $K$ 压到依赖 $dH\log K$。

> [!info]- 错误原因 / induce（点击展开）
> 常见卡点：以为 rare-switching 只是经验 trick。  
> 本题价值：其实它和 elliptical potential lemma 是同一类 log-det 思想在“更新次数控制”上的另一种用法。

---

## E. 最后一题：把整条路线合起来

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 11：在“理想化主项”下，重建 LSVI-UCB++ 的 $\tilde O(d\sqrt{H^3K})$** >
> 假设你已经得到了如下三个输入：
>
> 1. **Bernstein + 加权回归** 给出更紧的 bonus 尺度：
>    $$
>    \beta=\tilde O(\sqrt d).
>    $$
> 2. **总估计方差** 可控：
>    $$
>    \sum_{k=1}^{K}\sum_{h=1}^{H}\sigma_{k,h}^2
>    =
>    \tilde O(H^2K).
>    $$
> 3. **加权椭球势** 可控：
>    $$
>    \sum_{k=1}^{K}\sum_{h=1}^{H}
>    \bar\sigma_{k,h}^{-2}\phi_{k,h}^\top\Sigma_{k,h}^{-1}\phi_{k,h}
>    =
>    \tilde O(dH).
>    $$
>
> 证明主 bonus 和满足
> $$
> \sum_{k,h}
> \beta\,\bar\sigma_{k,h}\,
> \sqrt{\phi_{k,h}^\top\Sigma_{k,h}^{-1}\phi_{k,h}}
> \le
> \beta\,
> \sqrt{
> \left(\sum_{k,h}\sigma_{k,h}^2\right)
> \left(
> \sum_{k,h}\bar\sigma_{k,h}^{-2}\phi_{k,h}^\top\Sigma_{k,h}^{-1}\phi_{k,h}
> \right)
> }.
> $$
> 从而推出
> $$
> \mathrm{Regret}(K)=\tilde O(d\sqrt{H^3K})
> $$
> 的主量级。  
>
> 最后，请你写一句“技术因子对照表”，明确指出：
>
> - 哪一步消掉了 $\sqrt H$；
> - 哪一步消掉了 $\sqrt d$；
> - 哪一步保证单调方差上界可用但又不至于让函数类复杂度失控。

> [!success]- 核心思路（点击展开）
> 直接用 Cauchy–Schwarz：
> $$
> \sum_{k,h}\bar\sigma_{k,h}\sqrt{x_{k,h}}
> \le
> \sqrt{\sum_{k,h}\sigma_{k,h}^2}
> \cdot
> \sqrt{\sum_{k,h}\bar\sigma_{k,h}^{-2}x_{k,h}},
> \qquad x_{k,h}:=\phi_{k,h}^\top\Sigma_{k,h}^{-1}\phi_{k,h}.
> $$
> 代入三个输入得到
> $$
> \tilde O(\sqrt d)\cdot \sqrt{H^2K}\cdot \sqrt{dH}
> =
> \tilde O(d\sqrt{H^3K}).
> $$
> 因子对照表可写成：
> - **Hoeffding $\to$ Bernstein + variance-aware weighting**：消掉主要的 $\sqrt H$；
> - **直接估 $V_k$ $\to$ 先抽出固定的 $V^\star$**：消掉主要的 $\sqrt d$；
> - **每轮更新 $\to$ rare-switching**：保住单调方差上界，同时控制函数类复杂度。

> [!info]- 错误原因 / induce（点击展开）
> 常见卡点：学了很多 lemma，但脑中没有“谁消掉了哪个因子”的总图。  
> 本题价值：做完这一题，你就能把 Jin 2020 与 He 2023 的技术差异压缩成一句干净的话。

---

## 你现在的脉络，还建议额外补一块

如果你想把这套题真正做成“能反复复习的研究型错题本”，建议再单独加一页：

### 额外桥接页：linear MDP vs linear mixture MDP

至少回答下面两个问题：

1. 为什么 linear mixture MDP 的 near-optimal 方法里，`value-targeted regression / base-model oracle` 是自然的，而在线性 MDP 里这会破坏“直接逼近 value function”的计算优势？
2. 为什么 He 2023 必须额外构造 pessimistic value $\check V_{k,h}$，而不是简单照搬 linear mixture MDP 的 variance estimator？

这页不一定写成错题，但最好写成“概念对照表”。

---

## 参考文献（便于你继续扩展）

1. Chi Jin, Zhuoran Yang, Zhaoran Wang, Michael I. Jordan. **Provably Efficient Reinforcement Learning with Linear Function Approximation.** COLT 2020.  
2. Jiafan He, Heyang Zhao, Dongruo Zhou, Quanquan Gu. **Nearly Minimax Optimal Reinforcement Learning for Linear Markov Decision Processes.** ICML 2023.  
3. Mohammad Gheshlaghi Azar, Ian Osband, Rémi Munos. **Minimax Regret Bounds for Reinforcement Learning.** ICML 2017.  
4. Dongruo Zhou, Quanquan Gu, Csaba Szepesvári. **Nearly Minimax Optimal Reinforcement Learning for Linear Mixture MDPs.** COLT 2021.  
5. Zihan Zhang, Jiaqi Yang, Xiangyang Ji, Simon S. Du. **Improved Variance-Aware Confidence Sets for Linear Bandits and Linear Mixture MDP.** NeurIPS 2021.  
6. Zihan Zhang, Yuan Zhou, Xiangyang Ji. **Almost Optimal Model-Free Reinforcement Learning via Reference-Advantage Decomposition.** NeurIPS 2020.