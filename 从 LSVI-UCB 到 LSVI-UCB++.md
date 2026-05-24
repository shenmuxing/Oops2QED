---
tags:
  - RL
  - LinearMDP
  - LSVI-UCB
  - Bernstein
  - RareSwitching
date: 2026-03-14
---

## 使用说明

这是一份研究型错题本，用这个方式将往年的技术突破用错题的方式组织起来，方便一个人浓缩学习人类的进步

这份错题本按如下主线组织：

1. **Jin et al. (2020)**：为什么在线性 MDP 中，`LSVI + 普通岭回归 + Hoeffding 型 bonus` 已经足够得到第一个多项式 regret bound。
2. **困难的本质**：为什么不能把“固定值函数时好用的 self-normalized / Bernstein”直接套到 `V_{k,h+1}` 上。
3. **He et al. (2023)**：如何通过 `variance-aware weighted ridge regression + 最优值/误差分解 + 单调方差上界 + rare-switching` 把主项推进到近极小极大最优。
4. **总账本**：每一个技术点到底消掉了哪一个因子：$\sqrt H$、$\sqrt d$、还是“函数类复杂度爆炸”的隐患。

## 符号和本错题本背景

- 线性 MDP：对每个阶段 $h\in[H]$，存在未知测度 $\mu_h(\cdot)$ 与已知特征 $\phi:S\times A\to \mathbb R^d$，使得
  $$P_h(s'|s,a)=\langle \phi(s,a),\mu_h(s')\rangle,\qquad r_h(s,a)=\langle \phi(s,a),\theta_h\rangle$$
  且 $\|\phi(s,a)\|_2\le 1$。
- Jin 2020 的 Gram 矩阵：
  $$\Lambda_{k,h}=\lambda I+\sum_{\tau=1}^{k-1}\phi(s^\tau_h,a^\tau_h)\phi(s^\tau_h,a^\tau_h)^\top .$$
- He 2023 的加权 Gram 矩阵：
  $$\Sigma_{k,h}=\lambda I+\sum_{i=1}^{k-1}\bar\sigma_{i,h}^{-2}\phi(s^i_h,a^i_h)\phi(s^i_h,a^i_h)^\top .$$
- 转移噪声：
  $$\eta_{i,h}(V):=V(s^i_{h+1})-[P_hV](s^i_h,a^i_h).$$
- 方差算子：
  $$[\mathbb V_h V](s,a):=[P_h(V^2)](s,a)-([P_hV](s,a))^2.$$


> [!info]- 什么是 $\varepsilon$-net（点击展开）
> 设 $(\mathcal F,d)$ 是一个度量空间。一个有限集合 $\mathcal N_\varepsilon\subset\mathcal F$ 称为 $\mathcal F$ 在距离 $d$ 下的一个 $\varepsilon$-net，如果对任意 $f\in\mathcal F$，都存在某个 $\widetilde f\in\mathcal N_\varepsilon$ 使得
> $$
> d(f,\widetilde f)\le \varepsilon.
> $$
> 当 $d(f,g)=\|f-g\|_\infty$ 时，这就是说：对函数类里的每个函数，都能在有限集合 $\mathcal N_\varepsilon$ 里找到一个函数，在所有状态上都与它逐点相差不超过 $\varepsilon$。
>
> 最小可能的 net 大小记为 covering number：
> $$
> N_\varepsilon(\mathcal F,d)
> :=
> \min\left\{|\mathcal N|:\mathcal N\subset\mathcal F,\ \forall f\in\mathcal F,\ \exists \widetilde f\in\mathcal N,\ d(f,\widetilde f)\le\varepsilon\right\}.
> $$
> 也常记
> $$
> N_\varepsilon(\mathcal F,\|\cdot\|_\infty).
> $$
>
> 在这份错题本里，$\varepsilon$-net 的作用是：先把“对一个可能无限的函数类做 uniform concentration”转成“对一个有限集合做 union bound”，再把近似误差单独控制。例如题 2 里的标准套路就是
> $$
> \sup_{V\in\mathcal V_h}
> \left\|
> \sum_{i=1}^{k-1}x_i\eta_i(V)
> \right\|_{\Lambda_k^{-1}}
> \le
> \max_{\widetilde V\in\mathcal N_\varepsilon}
> \left\|
> \sum_{i=1}^{k-1}x_i\eta_i(\widetilde V)
> \right\|_{\Lambda_k^{-1}}
> +
> O\!\left(\frac{k\varepsilon}{\sqrt\lambda}\right).
> $$
> 其中第一项现在只需对有限个固定函数做 concentration，第二项是把原函数替换成 net 点所付出的近似代价。
>
> 直觉上，$\varepsilon$-net 是把一个连续函数类“离散化”成有限个代表元；covering number 则衡量这个函数类到底有多复杂。

不要把 **linear MDP** 和 **linear mixture MDP** 混为一谈。  
后者很多近最优结果依赖 **value-targeted regression (VTR)** 或对 base model 的积分/采样 oracle；而这份错题本的重点是：**在线性 MDP 里，如何保持“直接逼近 value function”的计算效率，同时把 regret 做到近最优。**

---
## A. 先把 Jin 2020 的逻辑链打通

> [!question] (2026-03-14) 第1次做，正确总0次，上次错误  
> **题目 1：固定 $V$ 时，Hoeffding / self-normalized 为什么能直接用？**  
> 固定某个阶段 $h$。对 $i=1,\dots,k-1$，记
> $$
> x_i:=\phi(s_h^i,a_h^i),\qquad 
> \Lambda_k=\lambda I+\sum_{i=1}^{k-1}x_ix_i^\top.
> $$
> 设 $\{\mathcal F_i\}$ 为如下 filtration：$\mathcal F_{i-1}$ 表示在第 $i$ 个样本的下一状态 $s_{h+1}^i$ 被观测到之前的全部历史信息，因此 $x_i$ 是 $\mathcal F_{i-1}$-可测，并且条件于 $\mathcal F_{i-1}$，
> $$
> s_{h+1}^i\sim P_h(\cdot\mid s_h^i,a_h^i).
> $$
> 设 $V:S\to[0,H]$ 为一个固定函数，定义噪声
> $$
> \xi_i:=V(s_{h+1}^i)-[P_hV](s_h^i,a_h^i).
> $$
> 证明：$\{\xi_i\}$ 是有界 martingale difference sequence，且高概率下
> $$
> \left\|\sum_{i=1}^{k-1}x_i\xi_i\right\|_{\Lambda_k^{-1}}
> \lesssim H\sqrt{d\log\frac{k+\lambda}{\lambda\delta}}.
> $$
> 进一步证明，对任意 $(s,a)$，
> $$
> \left|\phi(s,a)^\top\Lambda_k^{-1}\sum_{i=1}^{k-1}x_i\xi_i\right|
> \lesssim
> H\sqrt{d\log\frac{k+\lambda}{\lambda\delta}}
> \cdot
> \sqrt{\phi(s,a)^\top\Lambda_k^{-1}\phi(s,a)}.
> $$

> [!success]- 核心思路（点击展开）
> 关键在于：当 $V$ 是固定靶子时，
> $$
> \xi_i=V(s_{h+1}^i)-[P_hV](s_h^i,a_h^i)
> $$
> 就是一个标准的条件零均值噪声。
>
> 先证明 $\{\xi_i\}$ 是 martingale difference sequence。因为 $x_i=\phi(s_h^i,a_h^i)$ 在观察到 $s_{h+1}^i$ 之前已经由历史决定，所以 $x_i$ 是 $\mathcal F_{i-1}$-可测。又由于 $V$ 是固定函数，条件于 $\mathcal F_{i-1}$，唯一的随机性来自
> $$
> s_{h+1}^i\sim P_h(\cdot\mid s_h^i,a_h^i).
> $$
> 因此
> $$
> \mathbb E[V(s_{h+1}^i)\mid \mathcal F_{i-1}]
> =
> [P_hV](s_h^i,a_h^i),
> $$
> 从而
> $$
> \mathbb E[\xi_i\mid \mathcal F_{i-1}]
> =
> \mathbb E[V(s_{h+1}^i)\mid \mathcal F_{i-1}]
> -[P_hV](s_h^i,a_h^i)
> =0.
> $$
> 所以 $\{\xi_i\}$ 是相对于 $\{\mathcal F_i\}$ 的 martingale difference sequence。
>
> 再看有界性。由于 $V:S\to[0,H]$，所以
> $$
> 0\le V(s_{h+1}^i)\le H,\qquad 0\le [P_hV](s_h^i,a_h^i)\le H,
> $$
> 因而
> $$
> |\xi_i|
> =
> \left|V(s_{h+1}^i)-[P_hV](s_h^i,a_h^i)\right|
> \le H.
> $$
>
> 更严格地说，设
> $$
> Y_i:=V(s_{h+1}^i)\in[0,H].
> $$
> 那么
> $$
> \xi_i=Y_i-\mathbb E[Y_i\mid \mathcal F_{i-1}].
> $$
> 由条件版 Hoeffding 引理，对任意 $t\in\mathbb R$，
> $$
> \mathbb E\!\left[e^{t\xi_i}\mid \mathcal F_{i-1}\right]
> \le e^{t^2H^2/8}.
> $$
> 这说明 $\xi_i$ 是条件次高斯的，因此可以直接套用线性 bandit 中标准的 self-normalized concentration。
>
> 于是，以至少 $1-\delta$ 的概率，
> $$
> \left\|\sum_{i=1}^{k-1}x_i\xi_i\right\|_{\Lambda_k^{-1}}
> \le
> \frac H2
> \sqrt{
> 2\log\frac{\det(\Lambda_k)^{1/2}}{\det(\lambda I)^{1/2}\delta}
> }.
> $$
> 又因为
> $$
> \det(\lambda I)=\lambda^d,
> $$
> 且由 $\|x_i\|_2\le 1$ 可得
> $$
> x_ix_i^\top\preceq I,
> \qquad
> \sum_{i=1}^{k-1}x_ix_i^\top\preceq (k-1)I,
> $$
> 所以
> $$
> \Lambda_k
> =
> \lambda I+\sum_{i=1}^{k-1}x_ix_i^\top
> \preceq
> (\lambda+k-1)I.
> $$
> 从而
> $$
> \det(\Lambda_k)\le (\lambda+k-1)^d.
> $$
> 代回上式，得到
> $$
> \left\|\sum_{i=1}^{k-1}x_i\xi_i\right\|_{\Lambda_k^{-1}}
> \le
> \frac H2
> \sqrt{
> d\log\frac{\lambda+k-1}{\lambda}
> +
> 2\log\frac1\delta
> }.
> $$
> 将常数吸收到 $\lesssim$ 记号中，就得到
> $$
> \left\|\sum_{i=1}^{k-1}x_i\xi_i\right\|_{\Lambda_k^{-1}}
> \lesssim
> H\sqrt{d\log\frac{k+\lambda}{\lambda\delta}}.
> $$
>
> 接着证明第二个结论。令
> $$
> u:=\phi(s,a),\qquad z:=\sum_{i=1}^{k-1}x_i\xi_i.
> $$
> 则
> $$
> u^\top\Lambda_k^{-1}z
> =
> (\Lambda_k^{-1/2}u)^\top(\Lambda_k^{-1/2}z).
> $$
> 由 Cauchy--Schwarz 不等式，
> $$
> |u^\top\Lambda_k^{-1}z|
> \le
> \|\Lambda_k^{-1/2}u\|_2\,
> \|\Lambda_k^{-1/2}z\|_2.
> $$
> 也就是
> $$
> |u^\top\Lambda_k^{-1}z|
> \le
> \sqrt{u^\top\Lambda_k^{-1}u}\,
> \|z\|_{\Lambda_k^{-1}}.
> $$
> 将 $u=\phi(s,a)$ 和 $z=\sum_{i=1}^{k-1}x_i\xi_i$ 代回去，再结合上面的高概率界，就得到
> $$
> \left|\phi(s,a)^\top\Lambda_k^{-1}\sum_{i=1}^{k-1}x_i\xi_i\right|
> \lesssim
> H\sqrt{d\log\frac{k+\lambda}{\lambda\delta}}
> \cdot
> \sqrt{\phi(s,a)^\top\Lambda_k^{-1}\phi(s,a)}.
> $$
>
> 因此结论成立。

> [!info]- 必要说明（点击展开）
> 无

---

> [!question] (2026-03-19) 第1次做，正确总0次，上次错误
> **题目 2：为什么一旦换成 $V_{k,h+1}$，题 1 的 self-normalized 结论就不能直接套用？**
>
> 固定某个阶段 $h$。沿用题 1 的记号：对 $i=1,\dots,k-1$，记
> $$
> x_i:=\phi(s_h^i,a_h^i),\qquad
> \Lambda_k=\lambda I+\sum_{i=1}^{k-1}x_ix_i^\top.
> $$
> 设 $\{\mathcal F_i\}$ 为题 1 中的 filtration：$\mathcal F_{i-1}$ 表示在第 $i$ 个样本的下一状态 $s_{h+1}^i$ 被观测到之前的全部历史信息，因此 $x_i$ 是 $\mathcal F_{i-1}$-可测，并且条件于 $\mathcal F_{i-1}$，
> $$
> s_{h+1}^i\sim P_h(\cdot\mid s_h^i,a_h^i).
> $$
>
> 在第 $k$ 个 episode 的 backward update 中，考虑误差项
> $$
> (\widehat P_h-\overline P_h)V_{k,h+1}(s,a)
> =
> \phi(s,a)^\top\Lambda_k^{-1}
> \sum_{i=1}^{k-1}x_i\,\eta_i(V_{k,h+1}),
> $$
> 其中对任意函数 $V:S\to\mathbb R$ 定义
> $$
> \eta_i(V):=V(s_{h+1}^i)-[P_hV](s_h^i,a_h^i).
> $$
>
> 这里 $V_{k,h+1}$ 不再是预先固定的函数，而是由 later-stage LSVI 利用前 $k-1$ 个 episode 的数据算出来的。
>
> 1. 在题 1 中，如果 $V$ 是一个固定函数，那么
>    $$
>    \eta_i(V)=V(s_{h+1}^i)-[P_hV](s_h^i,a_h^i)
>    $$
>    是相对于 $\{\mathcal F_i\}$ 的 martingale difference sequence，因此题 1 的 self-normalized concentration 可以直接使用。
>
> 说明为什么一旦把 $V$ 换成 $V_{k,h+1}$，上述论证一般就失效。（提示：解释为什么
>    $$
>    V_{k,h+1}(s_{h+1}^i)
>    $$
>    中的函数 $V_{k,h+1}$ 本身依赖历史数据，通常既不是预先固定的，也未必是 $\mathcal F_{i-1}$-可测，因此
>    $$
>    \eta_i(V_{k,h+1})
>    =
>    V_{k,h+1}(s_{h+1}^i)-[P_hV_{k,h+1}](s_h^i,a_h^i)
>    $$
>    不再是“对固定函数 $V$ 的条件零均值噪声”，从而不能直接照搬题 1 的结论。）
>
> 2. 设 $\mathcal V_h$ 是一类候选值函数，$\mathcal N_\varepsilon\subset\mathcal V_h$ 是它在 $\|\cdot\|_\infty$ 下的一个 $\varepsilon$-net。证明下面的 decoupling 不等式：
>    $$
>    \sup_{V\in\mathcal V_h}
>    \left\|
>    \sum_{i=1}^{k-1}x_i\eta_i(V)
>    \right\|_{\Lambda_k^{-1}}
>    \le
>    \max_{\widetilde V\in\mathcal N_\varepsilon}
>    \left\|
>    \sum_{i=1}^{k-1}x_i\eta_i(\widetilde V)
>    \right\|_{\Lambda_k^{-1}}
>    +
>    O\!\left(\frac{k\varepsilon}{\sqrt\lambda}\right).
>    $$
>
> 3. 结合前两问解释：这一步虽然把“随机靶函数”问题转化成了“有限 net 上的固定函数问题”，但代价是最终 bound 会显式依赖 covering number，而这正是 Jin 2020 里 uniform concentration 的来源。

> [!success]- 核心思路（点击展开）
> 这题真正要抓住的是：**题 1 能用 self-normalized，是因为靶函数 $V$ 是固定的；一旦靶函数本身由数据算出来，martingale difference 结构就不再自动成立。**
>
> 先看固定函数的情形。若 $V$ 是固定函数，那么条件于 $\mathcal F_{i-1}$，唯一的随机性来自
> $$
> s_{h+1}^i\sim P_h(\cdot\mid s_h^i,a_h^i),
> $$
> 而 $V$ 本身不是随机对象，因此
> $$
> \mathbb E[V(s_{h+1}^i)\mid \mathcal F_{i-1}]
> =
> [P_hV](s_h^i,a_h^i).
> $$
> 从而
> $$
> \mathbb E[\eta_i(V)\mid \mathcal F_{i-1}]=0.
> $$
> 所以 $\eta_i(V)$ 是标准的 martingale difference noise，这正是题 1 可以直接套用 self-normalized concentration 的原因。
>
> 但对 $V_{k,h+1}$，情况不同。因为 $V_{k,h+1}$ 是由前 $k-1$ 个 episode 在第 $h+1$ 层的数据通过 LSVI 算出来的，而这些数据本身就包含
> $$
> s_{h+1}^1,\dots,s_{h+1}^{k-1}.
> $$
> 因此对于某个固定的 $i<k$，$V_{k,h+1}$ 往往依赖于 $s_{h+1}^i$ 本身，甚至还依赖于比 $i$ 更晚的样本。于是它一般不是题 1 中那种“先固定、后取条件期望”的函数对象，也通常不能当作 $\mathcal F_{i-1}$-可测量来处理。
>
> 换句话说，
> $$
> V_{k,h+1}(s_{h+1}^i)
> $$
> 不再是“固定函数作用在随机下一状态上”，而是“由数据构造出来的随机函数，再作用在同一批数据中的随机样本点上”。因此通常不能直接写出
> $$
> \mathbb E[V_{k,h+1}(s_{h+1}^i)\mid \mathcal F_{i-1}]
> =
> [P_hV_{k,h+1}](s_h^i,a_h^i),
> $$
> 也就不能直接断言
> $$
> \mathbb E[\eta_i(V_{k,h+1})\mid \mathcal F_{i-1}]=0.
> $$
> 所以题 1 的 fixed-target self-normalized concentration 不能直接照搬到这里。
>
> 接下来证明 decoupling 不等式。任取 $V\in\mathcal V_h$。由 $\varepsilon$-net 的定义，存在某个 $\widetilde V\in\mathcal N_\varepsilon$ 使得
> $$
> \|V-\widetilde V\|_\infty\le\varepsilon.
> $$
> 记
> $$
> \Delta:=V-\widetilde V,
> \qquad \|\Delta\|_\infty\le\varepsilon.
> $$
> 由于 $\eta_i(\cdot)$ 关于函数是线性的，
> $$
> \eta_i(V)=\eta_i(\widetilde V)+\eta_i(\Delta).
> $$
> 因而
> $$
> \sum_{i=1}^{k-1}x_i\eta_i(V)
> =
> \sum_{i=1}^{k-1}x_i\eta_i(\widetilde V)
> +
> \sum_{i=1}^{k-1}x_i\eta_i(\Delta).
> $$
> 对两边取 $\Lambda_k^{-1}$-范数并用三角不等式，得到
> $$
> \left\|
> \sum_{i=1}^{k-1}x_i\eta_i(V)
> \right\|_{\Lambda_k^{-1}}
> \le
> \left\|
> \sum_{i=1}^{k-1}x_i\eta_i(\widetilde V)
> \right\|_{\Lambda_k^{-1}}
> +
> \left\|
> \sum_{i=1}^{k-1}x_i\eta_i(\Delta)
> \right\|_{\Lambda_k^{-1}}.
> $$
>
> 下面控制第二项。对每个 $i$，
> $$
> \eta_i(\Delta)
> =
> \Delta(s_{h+1}^i)-[P_h\Delta](s_h^i,a_h^i).
> $$
> 由 $\|\Delta\|_\infty\le\varepsilon$，
> $$
> |\Delta(s_{h+1}^i)|\le\varepsilon,
> \qquad
> |[P_h\Delta](s_h^i,a_h^i)|\le\varepsilon,
> $$
> 所以
> $$
> |\eta_i(\Delta)|\le 2\varepsilon.
> $$
> 从而
> $$
> \left\|
> \sum_{i=1}^{k-1}x_i\eta_i(\Delta)
> \right\|_{\Lambda_k^{-1}}
> \le
> \sum_{i=1}^{k-1}
> |\eta_i(\Delta)|\,\|x_i\|_{\Lambda_k^{-1}}
> \le
> 2\varepsilon\sum_{i=1}^{k-1}\|x_i\|_{\Lambda_k^{-1}}.
> $$
> 又因为
> $$
> \Lambda_k\succeq \lambda I,
> $$
> 所以
> $$
> \Lambda_k^{-1}\preceq \lambda^{-1}I.
> $$
> 再结合 $\|x_i\|_2\le 1$，
> $$
> \|x_i\|_{\Lambda_k^{-1}}
> \le
> \frac1{\sqrt\lambda}.
> $$
> 于是
> $$
> \left\|
> \sum_{i=1}^{k-1}x_i\eta_i(\Delta)
> \right\|_{\Lambda_k^{-1}}
> \le
> \frac{2(k-1)\varepsilon}{\sqrt\lambda}
> =
> O\!\left(\frac{k\varepsilon}{\sqrt\lambda}\right).
> $$
> 代回可得，对任意 $V\in\mathcal V_h$，
> $$
> \left\|
> \sum_{i=1}^{k-1}x_i\eta_i(V)
> \right\|_{\Lambda_k^{-1}}
> \le
> \max_{\widetilde V\in\mathcal N_\varepsilon}
> \left\|
> \sum_{i=1}^{k-1}x_i\eta_i(\widetilde V)
> \right\|_{\Lambda_k^{-1}}
> +
> O\!\left(\frac{k\varepsilon}{\sqrt\lambda}\right).
> $$
> 再对左侧取 $\sup_{V\in\mathcal V_h}$，结论成立。
>
> 这说明：面对随机靶函数 $V_{k,h+1}$，一个自然补救办法是先用 covering 把问题 decouple 到有限个固定函数上；对每个固定的 $\widetilde V\in\mathcal N_\varepsilon$，题 1 的 concentration 可以使用；最后再做 union bound。其代价就是 bound 中显式出现 $\log |\mathcal N_\varepsilon|$，也就是函数类复杂度。

> [!info]- 必要说明（点击展开）
> 这题的重点不是“会不会套 concentration”，而是识别 **fixed-target 分析的适用边界**：
>
> 4. 题 1 之所以成立，是因为 $V$ 固定，所以 $\eta_i(V)$ 是标准的 martingale difference。
> 5. 题 2 的困难在于：$V_{k,h+1}$ 也是从数据里学出来的，因此会和同一批样本发生耦合。
> 6. Jin 2020 的处理方式不是“不会用 self-normalized”，而是先用 covering 把随机函数问题转成有限多个 fixed-target 问题，再统一做 concentration。
> 7. 这一步的代价就是显式的函数类复杂度，而这也正是后续更精细技术要继续消掉的对象。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 3：$\varepsilon$-net / covering number 到底给 regret 多带了什么？**  
> 固定某个阶段 $h$。考虑函数类
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
> :
> \|w\|_2\le L,\ \beta\in[0,B],\ \lambda_{\min}(\Lambda)\ge \lambda
> \right\}.
> $$
> 假设对每个 $s$，$\max_a$ 都能取到（例如动作集合有限），并且 $\|\phi(s,a)\|_2\le 1$。
>
> 1. 令
>    $$
>    A:=\beta^2\Lambda^{-1}.
>    $$
>    证明对任意允许的 $\beta,\Lambda$，都有
>    $$
>    A\succeq 0,
>    \qquad
>    \|A\|_{\mathrm{op}}\le \frac{B^2}{\lambda}.
>    $$
>    从而可把 bonus 写成
>    $$
>    \beta\sqrt{\phi(s,a)^\top\Lambda^{-1}\phi(s,a)}
>    =
>    \sqrt{\phi(s,a)^\top A\phi(s,a)}.
>    $$
>
> 2. 设
>    $$
>    f_{w,A}(s)
>    :=
>    \min\!\left\{
>    \max_a\bigl(w^\top\phi(s,a)+\sqrt{\phi(s,a)^\top A\phi(s,a)}\bigr),
>    H
>    \right\}.
>    $$
>    证明：若 $\|w-\widetilde w\|_2\le \varepsilon/2$ 且 $\|A-\widetilde A\|_F\le \varepsilon^2/4$，那么
>    $$
>    \|f_{w,A}-f_{\widetilde w,\widetilde A}\|_\infty\le \varepsilon.
>    $$
>
> 3. 由此推出一个 covering bound：
>    $$
>    \log N_\varepsilon(\mathcal V_h,\|\cdot\|_\infty)
>    \lesssim
>    d\log\!\left(1+\frac{L}{\varepsilon}\right)
>    +
>    d^2\log\!\left(1+\frac{B^2}{\lambda\varepsilon^2}\right).
>    $$
>    这里允许把维度常数吸收到对数项里。
>
> 4. 最后解释：如果某个 uniform concentration 的主项长成
>    $$
>    H\sqrt{d\log\frac{k+\lambda}{\lambda}+\log N_\varepsilon(\mathcal V_h,\|\cdot\|_\infty)},
>    $$
>    那么为什么从数量级上看，confidence radius 会从“固定 $V$ 时的”
>    $$
>    \tilde O(H\sqrt d)
>    $$
>    变成“自适应 $V_{k,h+1}$ 时的”
>    $$
>    \tilde O(dH).
>    $$

> [!success]- 核心思路（点击展开）
> 先做重参数化。因为 $\Lambda\succeq \lambda I$，所以
> $$
> \Lambda^{-1}\preceq \lambda^{-1}I.
> $$
> 再由 $\beta\in[0,B]$，可得
> $$
> A=\beta^2\Lambda^{-1}\succeq 0,
> \qquad
> \|A\|_{\mathrm{op}}
> \le
> \beta^2\|\Lambda^{-1}\|_{\mathrm{op}}
> \le
> \frac{B^2}{\lambda}.
> $$
> 并且
> $$
> \beta\sqrt{\phi^\top\Lambda^{-1}\phi}
> =
> \sqrt{\phi^\top(\beta^2\Lambda^{-1})\phi}
> =
> \sqrt{\phi^\top A\phi}.
> $$
> 所以函数类实质上由参数 $(w,A)$ 决定，其中 $w$ 是一个 $d$ 维向量，$A$ 是一个 $d\times d$ 矩阵。
>
> 接着证明参数扰动会怎样传到函数值上。任取 $(s,a)$，有
> $$
> \bigl|(w-\widetilde w)^\top\phi(s,a)\bigr|
> \le
> \|w-\widetilde w\|_2\,\|\phi(s,a)\|_2
> \le
> \frac{\varepsilon}{2}.
> $$
> 对二次型项，用
> $$
> \left|\sqrt{x}-\sqrt{y}\right|
> \le
> \sqrt{|x-y|}
> $$
> 得到
> $$
> \left|
> \sqrt{\phi^\top A\phi}
> -
> \sqrt{\phi^\top\widetilde A\phi}
> \right|
> \le
> \sqrt{\left|\phi^\top(A-\widetilde A)\phi\right|}.
> $$
> 又因为
> $$
> \phi^\top(A-\widetilde A)\phi
> =
> \langle A-\widetilde A,\phi\phi^\top\rangle,
> $$
> 且
> $$
> \|\phi\phi^\top\|_F=\|\phi\|_2^2\le 1,
> $$
> 所以
> $$
> \left|\phi^\top(A-\widetilde A)\phi\right|
> \le
> \|A-\widetilde A\|_F.
> $$
> 因而
> $$
> \left|
> \sqrt{\phi^\top A\phi}
> -
> \sqrt{\phi^\top\widetilde A\phi}
> \right|
> \le
> \sqrt{\|A-\widetilde A\|_F}
> \le
> \frac{\varepsilon}{2}.
> $$
> 于是对每个 $(s,a)$ 都有
> $$
> \left|
> w^\top\phi(s,a)+\sqrt{\phi^\top A\phi}
> -
> \widetilde w^\top\phi(s,a)-\sqrt{\phi^\top\widetilde A\phi}
> \right|
> \le
> \varepsilon.
> $$
>
> 现在再利用两个 contraction：
>
> 1. 对任意两组实数 $\{u_a\},\{v_a\}$，
>    $$
>    \left|\max_a u_a-\max_a v_a\right|
>    \le
>    \max_a |u_a-v_a|;
>    $$
> 2. 截断映射 $x\mapsto \min\{x,H\}$ 是 $1$-Lipschitz。
>
> 因此
> $$
> \|f_{w,A}-f_{\widetilde w,\widetilde A}\|_\infty
> \le
> \varepsilon.
> $$
>
> 下面构造 covering。对向量参数 $w$，$d$ 维欧氏球 $\{w:\|w\|_2\le L\}$ 有一个半径 $\varepsilon/2$ 的 net，其大小满足
> $$
> \log N_{\,\varepsilon/2}\bigl(B_2(L),\|\cdot\|_2\bigr)
> \lesssim
> d\log\!\left(1+\frac{L}{\varepsilon}\right).
> $$
> 对矩阵参数 $A$，由于
> $$
> \|A\|_F\le \sqrt d\,\|A\|_{\mathrm{op}}
> \le
> \sqrt d\,\frac{B^2}{\lambda},
> $$
> 所以在 $d^2$ 维 Frobenius 空间里，半径为 $O(B^2/\lambda)$ 的球有一个半径 $\varepsilon^2/4$ 的 net，其大小满足
> $$
> \log N_{\,\varepsilon^2/4}\bigl(\mathcal A,\|\cdot\|_F\bigr)
> \lesssim
> d^2\log\!\left(1+\frac{B^2}{\lambda\varepsilon^2}\right),
> $$
> 这里把 $\sqrt d$ 等温和维度因子吸收到对数中。
>
> 将两个 net 取笛卡尔积，再用上面已经证明的参数到函数的误差传递，就得到
> $$
> \log N_\varepsilon(\mathcal V_h,\|\cdot\|_\infty)
> \lesssim
> d\log\!\left(1+\frac{L}{\varepsilon}\right)
> +
> d^2\log\!\left(1+\frac{B^2}{\lambda\varepsilon^2}\right).
> $$
>
> 最后解释它对 confidence radius 的影响。固定一个目标函数 $V$ 时，题 1 的 self-normalized 主项只有
> $$
> \tilde O(H\sqrt d).
> $$
> 但一旦要对整个函数类 $\mathcal V_h$ 同时成立，uniform concentration 里要多出
> $$
> \log N_\varepsilon(\mathcal V_h,\|\cdot\|_\infty).
> $$
> 上面已经看到，这里面的主导自由度来自矩阵参数 $A$ 的 $d^2$ 维复杂度；因此
> $$
> \sqrt{\log N_\varepsilon}
> \asymp
> \tilde O(d).
> $$
> 放回到前面的主项
> $$
> H\sqrt{d+\log N_\varepsilon}
> $$
> 之后，数量级就从
> $$
> \tilde O(H\sqrt d)
> $$
> 抬到
> $$
> \tilde O(dH).
> $$
> 这正是 covering number 在 Jin 2020 路线里“多带出来”的那一个 $\sqrt d$。

> [!info]- 必要说明（点击展开）
> 这题真正要记住的不是某个具体 covering 常数，而是：
>
> 1. 自适应值函数带来的 uniform concentration，最后不是只多一个“对数”那么简单；
> 2. 这个对数后面站着的是一个 $d^2$ 维矩阵参数；
> 3. 一旦把它放进平方根，额外出来的就是那个 $\sqrt{d^2}=d$，也就是相对 fixed-target 分析多出来的 $\sqrt d$。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 4：从单步 bonus 总和，重建 Jin 2020 的 $\tilde O(\sqrt{d^3H^4K})$ 主项**  
> 设总时间步数 $T:=KH$。假设你已经知道 Jin 2020 的 regret 分析可以整理成
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
> \beta=\tilde \Theta(dH),
> \qquad
> \sum_{k=1}^{K}\sum_{h=1}^{H}\zeta_{k,h}\lesssim H\sqrt{T},
> $$
> 并且椭球势满足
> $$
> \sum_{k=1}^{K}\sum_{h=1}^{H}\phi_{k,h}^\top\Lambda_{k,h}^{-1}\phi_{k,h}
> \lesssim
> dH\log(1+K/\lambda).
> $$
> 试完成以下推导：
>
> 1. 用 Cauchy--Schwarz 证明
>    $$
>    \sum_{k=1}^{K}\sum_{h=1}^{H}
>    \sqrt{\phi_{k,h}^\top\Lambda_{k,h}^{-1}\phi_{k,h}}
>    \lesssim
>    \sqrt{KH}\,
>    \sqrt{dH\log(1+K/\lambda)}.
>    $$
>
> 2. 将其乘上 $\beta=\tilde\Theta(dH)$，推出
>    $$
>    \sum_{k,h}
>    \beta\sqrt{\phi_{k,h}^\top\Lambda_{k,h}^{-1}\phi_{k,h}}
>    =
>    \tilde O\!\left(d^{3/2}H^2\sqrt K\right).
>    $$
>
> 3. 最后说明为什么这正好等于
>    $$
>    \tilde O\!\left(\sqrt{d^3H^4K}\right)
>    =
>    \tilde O\!\left(\sqrt{d^3H^3T}\right),
>    $$
>    并解释为什么 $\sum_{k,h}\zeta_{k,h}$ 是 lower-order term。

> [!success]- 核心思路（点击展开）
> 令
> $$
> x_{k,h}:=
> \phi_{k,h}^\top\Lambda_{k,h}^{-1}\phi_{k,h}.
> $$
> 那么 bonus 总和就是
> $$
> \sum_{k,h}\beta\sqrt{x_{k,h}}.
> $$
>
> 先看不带 $\beta$ 的部分。由 Cauchy--Schwarz，
> $$
> \sum_{k=1}^{K}\sum_{h=1}^{H}\sqrt{x_{k,h}}
> \le
> \sqrt{KH}\,
> \sqrt{\sum_{k=1}^{K}\sum_{h=1}^{H}x_{k,h}}.
> $$
> 再代入椭球势界
> $$
> \sum_{k,h}x_{k,h}
> \lesssim
> dH\log(1+K/\lambda),
> $$
> 得
> $$
> \sum_{k,h}\sqrt{x_{k,h}}
> \lesssim
> \sqrt{KH}\,
> \sqrt{dH\log(1+K/\lambda)}.
> $$
>
> 乘上
> $$
> \beta=\tilde\Theta(dH)
> $$
> 后得到
> $$
> \sum_{k,h}\beta\sqrt{x_{k,h}}
> \lesssim
> dH\cdot
> \sqrt{KH}\cdot
> \sqrt{dH\log(1+K/\lambda)}.
> $$
> 把 polylog 吸收到 $\tilde O(\cdot)$ 记号中，就有
> $$
> \sum_{k,h}\beta\sqrt{x_{k,h}}
> =
> \tilde O\!\left(d^{3/2}H^2\sqrt K\right).
> $$
>
> 这可以直接重写成
> $$
> d^{3/2}H^2\sqrt K
> =
> \sqrt{d^3H^4K}.
> $$
> 又因为 $T=KH$，所以
> $$
> \sqrt{d^3H^4K}
> =
> \sqrt{d^3H^3(KH)}
> =
> \sqrt{d^3H^3T}.
> $$
>
> 最后看余项。题设给出
> $$
> \sum_{k,h}\zeta_{k,h}
> \lesssim
> H\sqrt T
> =
> H\sqrt{KH}
> =
> H^{3/2}\sqrt K.
> $$
> 与主项
> $$
> d^{3/2}H^2\sqrt K
> $$
> 相比，它少了一个大约 $\sqrt{dH}$ 的因子，因此在主量级上是 lower-order term。于是
> $$
> \mathrm{Regret}(K)
> =
> \tilde O\!\left(\sqrt{d^3H^4K}\right)
> =
> \tilde O\!\left(\sqrt{d^3H^3T}\right).
> $$

> [!info]- 必要说明（点击展开）
> 这题是 Jin 2020 的“总账本”：
>
> 4. 真正的大头不是 bonus 求和本身，而是
>    $$
>    \beta\sim dH;
>    $$
> 5. bonus 求和只负责把单步不确定性累加成一个
>    $$
>    \sqrt{KH}\cdot \sqrt{dH}
>    $$
>    量级；
> 6. 两者一乘，才得到
>    $$
>    d^{3/2}H^2\sqrt K.
>    $$

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 5：Bernstein-type self-normalized 在加权岭回归里的一般形态**  
> 设 $\{\mathcal G_i\}$ 是一列 filtration，并且对每个 $i$，$x_i\in\mathbb R^d$ 是 $\mathcal G_{i-1}$-可测，满足
> $$
> y_i=\langle \mu^\star,x_i\rangle+\eta_i,
> \qquad
> \mathbb E[\eta_i\mid \mathcal G_{i-1}]=0,
> \qquad
> \mathbb E[\eta_i^2\mid \mathcal G_{i-1}]\le \sigma_i^2,
> \qquad
> \|x_i\|_2\le 1.
> $$
> 取权重
> $$
> w_i:=\frac1{\bar\sigma_i},
> \qquad
> \bar\sigma_i\ge \sigma_i,
> $$
> 并定义加权 Gram 矩阵与加权岭回归估计
> $$
> Z_k=\lambda I+\sum_{i=1}^{k}w_i^2x_ix_i^\top,
> \qquad
> \widehat\mu_k
> =
> Z_k^{-1}\sum_{i=1}^{k}w_i^2y_ix_i.
> $$
>
> 现在假设你已经知道如下 Bernstein-type self-normalized 结论可用：对任意满足
> $$
> \widetilde Z_k=\lambda I+\sum_{i=1}^{k}\widetilde x_i\widetilde x_i^\top,
> \qquad
> \mathbb E[\widetilde\eta_i\mid \mathcal G_{i-1}]=0,
> \qquad
> \mathbb E[\widetilde\eta_i^2\mid \mathcal G_{i-1}]\le 1,
> $$
> 的可预测向量 $\widetilde x_i$ 与噪声 $\widetilde\eta_i$，高概率下有
> $$
> \left\|
> \sum_{i=1}^{k}\widetilde x_i\widetilde\eta_i
> \right\|_{\widetilde Z_k^{-1}}
> \lesssim
> \sqrt d
> +
> \max_{i\le k}
> |\widetilde\eta_i|
> \min\{1,\|\widetilde x_i\|_{\widetilde Z_{i-1}^{-1}}\},
> $$
> 其中对数因子隐含在 $\lesssim$ 中。
>
> 请完成下面三步：
>
> 1. 令
>    $$
>    \widetilde x_i:=w_ix_i,
>    \qquad
>    \widetilde\eta_i:=w_i\eta_i.
>    $$
>    证明
>    $$
>    Z_k=\lambda I+\sum_{i=1}^{k}\widetilde x_i\widetilde x_i^\top,
>    \qquad
>    \mathbb E[\widetilde\eta_i^2\mid \mathcal G_{i-1}]\le 1.
>    $$
>
> 2. 证明参数误差满足
>    $$
>    \widehat\mu_k-\mu^\star
>    =
>    Z_k^{-1}\sum_{i=1}^{k}w_i^2x_i\eta_i
>    -
>    \lambda Z_k^{-1}\mu^\star.
>    $$
>    从而先推出精确形式
>    $$
>    \|\widehat\mu_k-\mu^\star\|_{Z_k}
>    \lesssim
>    \sqrt d
>    +
>    \max_{i\le k}|w_i\eta_i|
>    \min\{1,\|w_ix_i\|_{Z_{i-1}^{-1}}\}
>    +
>    \sqrt\lambda\|\mu^\star\|_2.
>    $$
>    再说明：若额外有 $\bar\sigma_i\ge 1$（因此 $w_i\le 1$），则上式可进一步简化成
>    $$
>    \|\widehat\mu_k-\mu^\star\|_{Z_k}
>    \lesssim
>    \sqrt d
>    +
>    \max_{i\le k}|\eta_i|
>    \min\{1,\|w_ix_i\|_{Z_{i-1}^{-1}}\}
>    +
>    \sqrt\lambda\|\mu^\star\|_2.
>    $$
>
> 3. 最后解释：为什么当 $w_i\approx 1/\sigma_i$ 时，leading term 的量级会从 Hoeffding 风格的
>    $$
>    \tilde O(H\sqrt d)
>    $$
>    降到 Bernstein 风格的
>    $$
>    \tilde O(\sqrt d).
>    $$

> [!success]- 核心思路（点击展开）
> 第一步是把问题改写成“单位方差噪声”的形式。定义
> $$
> \widetilde x_i:=w_ix_i,
> \qquad
> \widetilde\eta_i:=w_i\eta_i.
> $$
> 那么
> $$
> Z_k
> =
> \lambda I+\sum_{i=1}^{k}w_i^2x_ix_i^\top
> =
> \lambda I+\sum_{i=1}^{k}\widetilde x_i\widetilde x_i^\top.
> $$
> 同时
> $$
> \mathbb E[\widetilde\eta_i^2\mid \mathcal G_{i-1}]
> =
> w_i^2\,\mathbb E[\eta_i^2\mid \mathcal G_{i-1}]
> \le
> w_i^2\sigma_i^2
> =
> \frac{\sigma_i^2}{\bar\sigma_i^2}
> \le
> 1.
> $$
> 所以归一化以后，噪声的条件方差已经变成常数量级。
>
> 第二步是把估计误差拆开。由定义
> $$
> \widehat\mu_k
> =
> Z_k^{-1}\sum_{i=1}^{k}w_i^2y_ix_i
> =
> Z_k^{-1}\sum_{i=1}^{k}w_i^2(\langle \mu^\star,x_i\rangle+\eta_i)x_i.
> $$
> 把线性部分与噪声部分分开：
> $$
> \widehat\mu_k
> =
> Z_k^{-1}\left(\sum_{i=1}^{k}w_i^2x_ix_i^\top\right)\mu^\star
> +
> Z_k^{-1}\sum_{i=1}^{k}w_i^2x_i\eta_i.
> $$
> 又因为
> $$
> \sum_{i=1}^{k}w_i^2x_ix_i^\top
> =
> Z_k-\lambda I,
> $$
> 所以
> $$
> \widehat\mu_k
> =
> Z_k^{-1}(Z_k-\lambda I)\mu^\star
> +
> Z_k^{-1}\sum_{i=1}^{k}w_i^2x_i\eta_i
> =
> \mu^\star
> -
> \lambda Z_k^{-1}\mu^\star
> +
> Z_k^{-1}\sum_{i=1}^{k}w_i^2x_i\eta_i.
> $$
> 因而
> $$
> \widehat\mu_k-\mu^\star
> =
> Z_k^{-1}\sum_{i=1}^{k}w_i^2x_i\eta_i
> -
> \lambda Z_k^{-1}\mu^\star.
> $$
>
> 对两边取 $Z_k$-范数并用三角不等式：
> $$
> \|\widehat\mu_k-\mu^\star\|_{Z_k}
> \le
> \left\|
> \sum_{i=1}^{k}w_i^2x_i\eta_i
> \right\|_{Z_k^{-1}}
> +
> \lambda\|Z_k^{-1}\mu^\star\|_{Z_k}.
> $$
> 第二项可化简为
> $$
> \lambda\|Z_k^{-1}\mu^\star\|_{Z_k}
> =
> \sqrt\lambda\,\|\mu^\star\|_{\lambda Z_k^{-1}}
> \le
> \sqrt\lambda\,\|\mu^\star\|_2,
> $$
> 因为 $Z_k\succeq \lambda I$，故 $\lambda Z_k^{-1}\preceq I$。
>
> 再看第一项。注意
> $$
> \sum_{i=1}^{k}w_i^2x_i\eta_i
> =
> \sum_{i=1}^{k}\widetilde x_i\widetilde\eta_i.
> $$
> 所以可直接套题设的 Bernstein-type self-normalized 结论，得到
> $$
> \left\|
> \sum_{i=1}^{k}w_i^2x_i\eta_i
> \right\|_{Z_k^{-1}}
> \lesssim
> \sqrt d
> +
> \max_{i\le k}
> |\widetilde\eta_i|
> \min\{1,\|\widetilde x_i\|_{Z_{i-1}^{-1}}\}.
> $$
> 将 $\widetilde\eta_i=w_i\eta_i$、$\widetilde x_i=w_ix_i$ 代回，有
> $$
> \left\|
> \sum_{i=1}^{k}w_i^2x_i\eta_i
> \right\|_{Z_k^{-1}}
> \lesssim
> \sqrt d
> +
> \max_{i\le k}
> |w_i\eta_i|
> \min\{1,\|w_ix_i\|_{Z_{i-1}^{-1}}\}.
> $$
> 因为 $w_i\le 1$ 或者至少可把 $w_i$ 吸收到隐藏常数与截断项中，通常写成更简洁的形式
> $$
> \left\|
> \sum_{i=1}^{k}w_i^2x_i\eta_i
> \right\|_{Z_k^{-1}}
> \lesssim
> \sqrt d
> +
> \max_{i\le k}
> |\eta_i|
> \min\{1,\|w_ix_i\|_{Z_{i-1}^{-1}}\}.
> $$
> 故最终
> $$
> \|\widehat\mu_k-\mu^\star\|_{Z_k}
> \lesssim
> \sqrt d
> +
> \max_{i\le k}|\eta_i|
> \min\{1,\|w_ix_i\|_{Z_{i-1}^{-1}}\}
> +
> \sqrt\lambda\|\mu^\star\|_2.
> $$
>
> 最后解释量级改进。Hoeffding 风格分析把每个噪声都粗暴看成 $|\eta_i|\le H$ 的有界噪声，因此主项自然带着一个统一的 $H$，形成
> $$
> \tilde O(H\sqrt d).
> $$
> 但现在做了加权归一化之后，
> $$
> \widetilde\eta_i=w_i\eta_i\approx \eta_i/\sigma_i
> $$
> 的条件方差已经被压到 $1$ 的量级，所以 leading term 不再看见统一的 $H$，而只剩下“维度复杂度”
> $$
> \tilde O(\sqrt d).
> $$
> 这就是 Bernstein + variance-aware weighting 的核心收益。

> [!info]- 必要说明（点击展开）
> 这题真正要记住的是：加权并不是形式重写，而是在把异方差直接吸收到设计矩阵里。
>
> - 大方差样本对应较小权重，少说话；
> - 小方差样本对应较大权重，多说话；
> - 归一化后，self-normalized inequality 看到的是“单位方差噪声”，因此主项不再被统一的 $H$ 支配。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 6：为什么“直接把 $V_{k,h+1}$ 的方差估进去，再套 Bernstein”并不能自动解决问题？**  
> 固定某个阶段 $h$。对任意函数 $V:S\to\mathbb R$，记
> $$
> \eta_{i,h}(V)
> :=
> V(s^i_{h+1})-[P_hV](s^i_h,a^i_h),
> \qquad
> \phi_i:=\phi(s^i_h,a^i_h).
> $$
> 形式上你也许会想：如果对每个 $V$ 都定义
> $$
> \bar\sigma_{i,h}(V)^2
> \gtrsim
> [\mathbb V_hV](s^i_h,a^i_h),
> \qquad
> w_{i,h}(V):=\frac1{\bar\sigma_{i,h}(V)},
> $$
> 那么是不是就可以像题 5 那样，把
> $$
> \eta_{i,h}(V_{k,h+1})
> $$
> 当成一个可加权的 Bernstein 型噪声来处理？
>
> 请回答下面三个问题：
>
> 1. 为什么即使你真的引入了
>    $$
>    w_{i,h}(V_{k,h+1})
>    \approx
>    \frac1{\sqrt{[\mathbb V_hV_{k,h+1}](s^i_h,a^i_h)}},
>    $$
>    也不能自动把
>    $$
>    \eta_{i,h}(V_{k,h+1})
>    =
>    V_{k,h+1}(s^i_{h+1})-[P_hV_{k,h+1}](s^i_h,a^i_h)
>    $$
>    看成题 5 里那种“固定目标 + 条件零均值 + 可预测权重”的噪声？
>
> 2. 为什么如果想让高概率结论对所有可能出现的 $V_{k,h+1}$ 同时成立，仍然需要类似题 2 的 covering / decoupling？
>
> 3. 为什么这又会把你刚刚想省掉的那个 $\sqrt d$，以 $\log N_\varepsilon$ 的形式带回来？
>
> 最后请用一句话总结：为什么“加方差”和“摆脱自适应目标带来的函数类复杂度”是两件不同的事情。

> [!success]- 核心思路（点击展开）
> 第一问的关键是：Bernstein 只改善“固定目标时的噪声尺度”，并不自动消除“目标函数本身依赖数据”这一耦合。
>
> 若 $V$ 是预先固定的，那么条件于过去历史，$V$ 不是随机对象，且
> $$
> \mathbb E[\eta_{i,h}(V)\mid \mathcal F_{i-1}]=0,
> \qquad
> \mathbb E[\eta_{i,h}(V)^2\mid \mathcal F_{i-1}]
> \le
> [\mathbb V_hV](s^i_h,a^i_h).
> $$
> 这正是题 5 能工作的前提。
>
> 但对 $V_{k,h+1}$，情况不同。它是由前 $k-1$ 个 episode 在 stage $h+1$ 的数据通过 LSVI 算出来的，因此通常依赖
> $$
> s^1_{h+1},\dots,s^{k-1}_{h+1}.
> $$
> 对某个固定的 $i<k$ 来说，$V_{k,h+1}$ 往往依赖于样本点 $s^i_{h+1}$ 本身，甚至还依赖于比 $i$ 更晚的样本。因此：
>
> 4. $V_{k,h+1}$ 不是预先固定的函数；
> 5. $V_{k,h+1}$ 通常也不是 $\mathcal F_{i-1}$-可测；
> 6. 连带着
>    $$
>    w_{i,h}(V_{k,h+1})
>    $$
>    也未必是可预测的。
>
> 所以你不能直接写
> $$
> \mathbb E[\eta_{i,h}(V_{k,h+1})\mid \mathcal F_{i-1}]=0,
> $$
> 也不能直接把它放进题 5 的 Bernstein 型 self-normalized 模板里。
>
> 第二问：如果你想让结论对所有可能出现的 $V_{k,h+1}$ 都成立，本质上还是在做
> $$
> \sup_{V\in\mathcal V_{h+1}}
> \left\|
> \sum_i w_{i,h}(V)\phi_i\eta_{i,h}(V)
> \right\|_{\cdot}.
> $$
> 这依然是一个“随机目标函数类”的 uniform concentration 问题。最自然的补救仍然是题 2 的套路：先用一个 $\varepsilon$-net 把函数类离散化，再对有限个 net 点分别做 fixed-target concentration，最后 union bound。
>
> 第三问：一旦这么做，就会显式引入
> $$
> \log N_\varepsilon(\mathcal V_{h+1},\|\cdot\|_\infty).
> $$
> 而题 3 已经说明，在 Jin 2020 这类函数类里，$\log N_\varepsilon$ 的主导自由度来自一个 $d^2$ 维矩阵参数；因此放进平方根以后，会重新贡献一个数量级上的
> $$
> d,
> $$
> 也就是相对 fixed-target 情形多出来的那个 $\sqrt d$。
>
> 用一句话总结就是：
> $$
> \text{“加方差”解决的是噪声尺度问题；“摆脱 adaptive target 的类复杂度”解决的是 uniform control 问题。}
> $$
> 前者主要针对 $H$，后者主要针对 $d$，它们不是同一件事。

> [!info]- 必要说明（点击展开）
> 这题最容易犯的错误，是把“variance-aware”误以为万能钥匙。实际上：
>
> - Hoeffding $\to$ Bernstein：解决的是“单个 fixed target 的噪声太粗，只看到 $H$”；
> - adaptive $V_k$ $\to$ fixed reference + small error：解决的是“目标函数类太大，uniform concentration 付出 covering 代价”。
>
> 两个 gap 对应两种不同的技术。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 7：为什么分解成 $V^\star_{h+1}$ 与 $\Delta V_{k,h+1}$ 可以去掉主要的 covering 代价？**  
> 固定某个阶段 $h$。记
> $$
> \Delta V_{k,h+1}:=V_{k,h+1}-V^\star_{h+1}.
> $$
> 设加权 Gram 矩阵为
> $$
> \Sigma_{k,h}
> =
> \lambda I+\sum_{i=1}^{k-1}\bar\sigma_{i,h}^{-2}\phi_i\phi_i^\top,
> \qquad
> \phi_i:=\phi(s^i_h,a^i_h).
> $$
> 考虑转移估计误差项
> $$
> \phi(s,a)^\top\Sigma_{k,h}^{-1}
> \sum_{i=1}^{k-1}\bar\sigma_{i,h}^{-2}\phi_i\eta_{i,h}(V_{k,h+1}),
> $$
> 其中
> $$
> \eta_{i,h}(V)
> :=
> V(s^i_{h+1})-[P_hV](s^i_h,a^i_h).
> $$
>
> 1. 证明线性分解
>    $$
>    \eta_{i,h}(V_{k,h+1})
>    =
>    \eta_{i,h}(V^\star_{h+1})
>    +
>    \eta_{i,h}(\Delta V_{k,h+1}).
>    $$
>
> 2. 为简记，下面写
>    $$
>    \phi:=\phi(s,a),
>    \qquad
>    \Sigma:=\Sigma_{k,h},
>    \qquad
>    \bar\sigma_i:=\bar\sigma_{i,h},
>    \qquad
>    \eta_i(\cdot):=\eta_{i,h}(\cdot).
>    $$
>    证明原式可写成
>    $$
>    I_1+I_2,
>    $$
>    其中
>    $$
>    I_1
>    :=
>    \phi^\top\Sigma^{-1}
>    \sum_i\bar\sigma_i^{-2}\phi_i\eta_i(V^\star_{h+1}),
>    \qquad
>    I_2
>    :=
>    \phi^\top\Sigma^{-1}
>    \sum_i\bar\sigma_i^{-2}\phi_i\eta_i(\Delta V_{k,h+1}).
>    $$
>    说明为什么：
>
>    - $I_1$ 可以直接用 Bernstein-type self-normalized 控制，而**不需要**对大函数类做 covering；
>    - $I_2$ 虽然仍然带有 adaptive target，但如果 $\Delta V_{k,h+1}$ 已经很小，它就会成为 lower-order term。
>
> 3. 用一句话解释：这一步为什么是 He 2023 相对 Jin 2020 的关键转折点。

> [!success]- 核心思路（点击展开）
> 第一问只用线性性。由定义
> $$
> \Delta V_{k,h+1}=V_{k,h+1}-V^\star_{h+1},
> $$
> 所以
> $$
> V_{k,h+1}=V^\star_{h+1}+\Delta V_{k,h+1}.
> $$
> 又因为 $\eta_{i,h}(\cdot)$ 对函数是线性的，
> $$
> \eta_{i,h}(f+g)=\eta_{i,h}(f)+\eta_{i,h}(g),
> $$
> 于是
> $$
> \eta_{i,h}(V_{k,h+1})
> =
> \eta_{i,h}(V^\star_{h+1})
> +
> \eta_{i,h}(\Delta V_{k,h+1}).
> $$
>
> 代回原式后，自然得到
> $$
> \phi^\top\Sigma^{-1}
> \sum_i\bar\sigma_i^{-2}\phi_i\eta_i(V_{k,h+1})
> =I_1+I_2.
> $$
>
> 现在解释为什么 $I_1$ 好控。关键点在于 $V^\star_{h+1}$ 是跨 episode 固定的最优值函数，不依赖当前收集到的随机数据。因此对每个 $i$，
> $$
> \eta_{i,h}(V^\star_{h+1})
> =
> V^\star_{h+1}(s^i_{h+1})-[P_hV^\star_{h+1}](s^i_h,a^i_h)
> $$
> 就是一个标准的 fixed-target martingale difference。
> 如果权重满足
> $$
> \bar\sigma_{i,h}^2
> \gtrsim
> [\mathbb V_hV^\star_{h+1}](s^i_h,a^i_h),
> $$
> 那么
> $$
> \bar\sigma_{i,h}^{-1}\eta_{i,h}(V^\star_{h+1})
> $$
> 的条件方差就是常数量级，于是可以直接套 Bernstein-type self-normalized。不需要对整个大函数类做 covering，原因很简单：这里根本没有“大函数类”，只有一个固定函数 $V^\star_{h+1}$。
>
> 再看 $I_2$。它的确仍然带有随机目标
> $$
> \Delta V_{k,h+1}=V_{k,h+1}-V^\star_{h+1},
> $$
> 所以自适应性问题没有完全消失；但它现在只落在“误差函数”这一层，而不再落在整个主项上。粗糙地说，如果
> $$
> \|\Delta V_{k,h+1}\|_\infty
> $$
> 已经很小，那么对每个 $i$，
> $$
> |\eta_{i,h}(\Delta V_{k,h+1})|
> \le
> |\Delta V_{k,h+1}(s^i_{h+1})|
> +
> |[P_h\Delta V_{k,h+1}](s^i_h,a^i_h)|
> \le
> 2\|\Delta V_{k,h+1}\|_\infty,
> $$
> 从而 $I_2$ 至少在量级上与 $\Delta V_{k,h+1}$ 成正比。更精细的控制会在下一题里通过方差上界完成，但这里已经足够说明：一旦误差层很小，$I_2$ 就会成为 lower-order term。
>
> 因此，He 2023 的关键转折点不是“换了一个 concentration inequality”，而是：
> $$
> \text{先把固定的主参考项 }V^\star\text{ 从随机的 }V_k\text{ 里剥出来。}
> $$
> 这样大部分质量都落回到 fixed-target + Bernstein 的理想情形，只有小误差层才需要继续付出自适应控制的代价。

> [!info]- 必要说明（点击展开）
> 这题最值得反复咀嚼的一点是：
>
> - Jin 2020 的 uniform concentration，是把“大函数类复杂度”直接压在整个目标 $V_{k,h+1}$ 上；
> - He 2023 的 reference-error decomposition，是把最重的那部分质量先挪到固定的 $V^\star_{h+1}$ 上；
> - 于是 covering 成本不再打在主项上，而只打在一个已经很小的误差项上。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 8：如何控制参考值/误差分解本身带来的方差？**  
> 固定某个阶段 $h$，设
> $$
> \Delta V_{k,h+1}:=V_{k,h+1}-V^\star_{h+1},
> \qquad
> 0\le \Delta V_{k,h+1}(s)\le H\quad \forall s.
> $$
> 再设存在悲观值函数 $\check V_{k,h+1}$，满足
> $$
> \check V_{k,h+1}\le V^\star_{h+1}\le V_{k,h+1}.
> $$
> 定义 optimistic-pessimistic gap 的一步转移版本
> $$
> D_{k,h}(s,a)
> :=
> [P_h(V_{k,h+1}-\check V_{k,h+1})](s,a).
> $$
>
> 请证明：
>
> 1. 方差总是不超过二阶矩，因此
>    $$
>    [\mathbb V_h\Delta V_{k,h+1}](s,a)
>    \le
>    [P_h(\Delta V_{k,h+1}^2)](s,a).
>    $$
>
> 2. 由 $0\le \Delta V_{k,h+1}\le H$，推出
>    $$
>    [P_h(\Delta V_{k,h+1}^2)](s,a)
>    \le
>    H\,[P_h\Delta V_{k,h+1}](s,a).
>    $$
>    因而
>    $$
>    [\mathbb V_h\Delta V_{k,h+1}](s,a)
>    \le
>    H\,[P_h\Delta V_{k,h+1}](s,a).
>    $$
>    并说明为什么在后续推导里也常把它写成更松的
>    $$
>    [\mathbb V_h\Delta V_{k,h+1}](s,a)\le 2H\,[P_h\Delta V_{k,h+1}](s,a).
>    $$
>
> 3. 证明
>    $$
>    0\le \Delta V_{k,h+1}
>    =
>    V_{k,h+1}-V^\star_{h+1}
>    \le
>    V_{k,h+1}-\check V_{k,h+1}.
>    $$
>    从而推出
>    $$
>    [P_h\Delta V_{k,h+1}](s,a)
>    \le
>    D_{k,h}(s,a).
>    $$
>
> 4. 最后解释：为什么这说明 $D_{k,h}$ 是控制误差项方差的天然量？

> [!success]- 核心思路（点击展开）
> 第一问只用“方差不超过二阶矩”：
> $$
> [\mathbb V_h\Delta V](s,a)
> =
> [P_h(\Delta V^2)](s,a)-([P_h\Delta V](s,a))^2
> \le
> [P_h(\Delta V^2)](s,a).
> $$
> 把 $\Delta V$ 换回 $\Delta V_{k,h+1}$ 即得
> $$
> [\mathbb V_h\Delta V_{k,h+1}](s,a)
> \le
> [P_h(\Delta V_{k,h+1}^2)](s,a).
> $$
>
> 第二问利用
> $$
> 0\le \Delta V_{k,h+1}(s)\le H.
> $$
> 因为对每个状态 $s'$ 都有
> $$
> (\Delta V_{k,h+1}(s'))^2
> \le
> H\,\Delta V_{k,h+1}(s'),
> $$
> 所以对转移核 $P_h(\cdot\mid s,a)$ 取期望，得到
> $$
> [P_h(\Delta V_{k,h+1}^2)](s,a)
> \le
> H\,[P_h\Delta V_{k,h+1}](s,a).
> $$
> 与第一步合并，就有
> $$
> [\mathbb V_h\Delta V_{k,h+1}](s,a)
> \le
> H\,[P_h\Delta V_{k,h+1}](s,a).
> $$
> 在很多后续推导里，人们也会写成
> $$
> [\mathbb V_h\Delta V_{k,h+1}](s,a)
> \le
> 2H\,[P_h\Delta V_{k,h+1}](s,a),
> $$
> 这不是更强的结论，而只是更松、更方便吸收常数的形式。
>
> 第三问只用序关系。由
> $$
> \check V_{k,h+1}\le V^\star_{h+1}\le V_{k,h+1},
> $$
> 逐点得到
> $$
> 0\le V_{k,h+1}-V^\star_{h+1}\le V_{k,h+1}-\check V_{k,h+1}.
> $$
> 也就是
> $$
> 0\le \Delta V_{k,h+1}\le V_{k,h+1}-\check V_{k,h+1}.
> $$
> 两边对转移核 $P_h(\cdot\mid s,a)$ 取期望，便有
> $$
> [P_h\Delta V_{k,h+1}](s,a)
> \le
> [P_h(V_{k,h+1}-\check V_{k,h+1})](s,a)
> =
> D_{k,h}(s,a).
> $$
>
> 把这两步串起来，就得到
> $$
> [\mathbb V_h\Delta V_{k,h+1}](s,a)
> \le
> H\,[P_h\Delta V_{k,h+1}](s,a)
> \le
> H\,D_{k,h}(s,a),
> $$
> 或者更松地写成
> $$
> [\mathbb V_h\Delta V_{k,h+1}](s,a)
> \le
> 2H\,D_{k,h}(s,a).
> $$
> 这说明：误差项 $\Delta V_{k,h+1}$ 的方差，可以被 optimistic-pessimistic gap 的一步传播量 $D_{k,h}$ 所控制。
>
> 因而 $D_{k,h}$ 不是 proof artifact，而是一个非常自然的桥梁：
>
> - 它把“看不见的误差方差”
>   $$
>   [\mathbb V_h\Delta V_{k,h+1}]
>   $$
>   变成了“可以由上下界值函数差来控制”的量；
> - 一旦你能把 optimistic / pessimistic gap 压小，误差项方差也会一起被压小。

> [!info]- 必要说明（点击展开）
> 这题告诉你：悲观值函数 $\check V_{k,h+1}$ 在 He 2023 里不是陪衬。
>
> 它的作用是把
> $$
> \Delta V_{k,h+1}=V_{k,h+1}-V^\star_{h+1}
> $$
> 这种“看不见的误差”替换成
> $$
> V_{k,h+1}-\check V_{k,h+1}
> $$
> 这种算法内部可控的 gap，再把方差估计压到这个可控量上。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 9：如果每个 episode 都做一个 $\min$，值函数会怎样？函数类复杂度又会怎样？**  
> 固定某个阶段 $h$。设每一轮都会产生一个新的 optimistic candidate $\widetilde Q_{k,h}$，并递推定义
> $$
> Q_{k,h}(s,a)
> =
> \min\{Q_{k-1,h}(s,a),\widetilde Q_{k,h}(s,a),H\},
> \qquad
> V_{k,h}(s)=\max_a Q_{k,h}(s,a).
> $$
> 假设每个 candidate 都来自同一个基础函数类 $\mathcal G_h$。
>
> 1. 证明 $Q_{k,h}(s,a)$ 关于 $k$ 点态单调不增。
>
> 2. 推出 $V_{k,h}(s)$ 关于 $k$ 也点态单调不增。
>
> 3. 将递推展开，证明
>    $$
>    Q_{k,h}(s,a)
>    =
>    \min\{\widetilde Q_{1,h}(s,a),\widetilde Q_{2,h}(s,a),\dots,\widetilde Q_{k,h}(s,a),H\}.
>    $$
>    从而说明：若每个 $\widetilde Q_{j,h}$ 都带一个椭球 bonus，那么 $Q_{k,h}$ 属于“$k$ 个 bonus 函数逐点最小值”的复合函数类。
>
> 4. 设 $\widehat{\mathcal G}_{h,\varepsilon}$ 是 $\mathcal G_h$ 在 $\|\cdot\|_\infty$ 下的一个 $\varepsilon$-net，大小为 $N_\varepsilon(\mathcal G_h)$。证明基于逐分支 covering 的自然做法会给出
>    $$
>    N_\varepsilon(\mathcal F_{k,h},\|\cdot\|_\infty)
>    \le
>    N_\varepsilon(\mathcal G_h,\|\cdot\|_\infty)^k,
>    $$
>    其中
>    $$
>    \mathcal F_{k,h}
>    :=
>    \left\{
>    \min\{g_1,\dots,g_k,H\}:g_j\in\mathcal G_h
>    \right\}.
>    $$
>    因而
>    $$
>    \log N_\varepsilon(\mathcal F_{k,h},\|\cdot\|_\infty)
>    \le
>    k\log N_\varepsilon(\mathcal G_h,\|\cdot\|_\infty).
>    $$
>
> 5. 结合第四问解释：为什么如果每个 episode 都更新一次，那么到 $k=K$ 时，uniform concentration 会被函数类复杂度拖垮？

> [!success]- 核心思路（点击展开）
> 第一问只用递推定义：
> $$
> Q_{k,h}(s,a)
> =
> \min\{Q_{k-1,h}(s,a),\widetilde Q_{k,h}(s,a),H\}
> \le
> Q_{k-1,h}(s,a).
> $$
> 所以对每个 $(s,a)$，序列 $Q_{k,h}(s,a)$ 都是单调不增的。
>
> 第二问对 action 取最大值即可。因为对所有 $a$ 都有
> $$
> Q_{k,h}(s,a)\le Q_{k-1,h}(s,a),
> $$
> 所以
> $$
> V_{k,h}(s)
> =
> \max_a Q_{k,h}(s,a)
> \le
> \max_a Q_{k-1,h}(s,a)
> =
> V_{k-1,h}(s).
> $$
> 因而 $V_{k,h}(s)$ 也点态单调不增。
>
> 第三问把递推展开即可。对 $k=2$ 有
> $$
> Q_{2,h}
> =
> \min\{Q_{1,h},\widetilde Q_{2,h},H\}.
> $$
> 若再把 $Q_{1,h}$ 本身展开，就得到
> $$
> Q_{2,h}
> =
> \min\{\widetilde Q_{1,h},\widetilde Q_{2,h},H\}.
> $$
> 继续归纳下去，可得
> $$
> Q_{k,h}
> =
> \min\{\widetilde Q_{1,h},\widetilde Q_{2,h},\dots,\widetilde Q_{k,h},H\}.
> $$
> 所以到第 $k$ 轮为止，$Q_{k,h}$ 并不是“某一个简单 bonus 函数”，而是 $k$ 个候选 bonus 函数做逐点最小值后的复合对象。
>
> 第四问要说明 covering 为什么会炸。令
> $$
> \widehat{\mathcal G}_{h,\varepsilon}
> $$
> 是基础类 $\mathcal G_h$ 的一个 $\varepsilon$-net。考虑
> $$
> \widehat{\mathcal F}_{k,h,\varepsilon}
> :=
> \left\{
> \min\{\widehat g_1,\dots,\widehat g_k,H\}:
> \widehat g_j\in\widehat{\mathcal G}_{h,\varepsilon}
> \right\}.
> $$
> 这个集合的大小显然至多是
> $$
> |\widehat{\mathcal G}_{h,\varepsilon}|^k
> =
> N_\varepsilon(\mathcal G_h,\|\cdot\|_\infty)^k.
> $$
> 下面说明它确实能 cover $\mathcal F_{k,h}$。任取
> $$
> f=\min\{g_1,\dots,g_k,H\}\in\mathcal F_{k,h},
> \qquad
> g_j\in\mathcal G_h.
> $$
> 对每个 $j$ 选取近似点 $\widehat g_j\in\widehat{\mathcal G}_{h,\varepsilon}$，使得
> $$
> \|g_j-\widehat g_j\|_\infty\le \varepsilon.
> $$
> 定义
> $$
> \widehat f
> :=
> \min\{\widehat g_1,\dots,\widehat g_k,H\}.
> $$
> 由于逐点最小值映射是 $1$-Lipschitz 的，
> $$
> \|f-\widehat f\|_\infty
> \le
> \max_{1\le j\le k}\|g_j-\widehat g_j\|_\infty
> \le
> \varepsilon.
> $$
> 所以
> $$
> N_\varepsilon(\mathcal F_{k,h},\|\cdot\|_\infty)
> \le
> N_\varepsilon(\mathcal G_h,\|\cdot\|_\infty)^k,
> $$
> 进而
> $$
> \log N_\varepsilon(\mathcal F_{k,h},\|\cdot\|_\infty)
> \le
> k\log N_\varepsilon(\mathcal G_h,\|\cdot\|_\infty).
> $$
>
> 这说明：哪怕基础类 $\mathcal G_h$ 的复杂度本来还可以接受，一旦每个 episode 都新增一个分支，再对所有分支组合做 covering，分析里的类复杂度就会线性累加到 $k$。当 $k=K$ 时，这个复杂度会直接进入 uniform concentration 或 union bound，从而把整体 regret 分析拖垮。
>
> 所以 rare-switching 的意义不只是“少算几次”，更是“别让复合函数类的分支数一路涨到 $K$”。

> [!info]- 必要说明（点击展开）
> 这里要抓住一个常被忽略的点：
>
> - 单调性本身是好事，因为它能给你稳定的方差上界；
> - 但“每轮都取一次 $\min$”会不断增加函数表达式的分支数；
> - 如果不限制更新次数，单调性带来的收益会被函数类复杂度爆炸反噬掉。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 10：rare-switching 如何把更新次数压到 $O(dH\log(1+K/\lambda))$？**  
> 固定某个阶段 $h$。设加权 Gram 矩阵按
> $$
> \Sigma_{k,h}
> =
> \lambda I+\sum_{i=1}^{k-1}\bar\sigma_{i,h}^{-2}\phi_i\phi_i^\top
> $$
> 演化，并假设
> $$
> \|\phi_i\|_2\le 1,
> \qquad
> 0<\bar\sigma_{i,h}^{-2}\le 1.
> $$
> 对 stage $h$，只在满足
> $$
> \det(\Sigma_{k,h})\ge 2\det(\Sigma_{k_{\mathrm{last}},h})
> $$
> 时才触发一次新的更新，其中 $k_{\mathrm{last}}$ 表示上一次更新发生时的 episode，初始时 $\Sigma_{1,h}=\lambda I$。
>
> 请证明：
>
> 1. 若第 $h$ 层一共触发了 $m_h$ 次更新，则
>    $$
>    m_h
>    \le
>    \log_2\frac{\det(\Sigma_{K+1,h})}{\det(\lambda I)}.
>    $$
>
> 2. 用特征值的 AM--GM 不等式或 trace-det 不等式证明
>    $$
>    \det(\Sigma_{K+1,h})
>    \le
>    \left(
>    \lambda+\frac{\mathrm{Tr}(\Sigma_{K+1,h}-\lambda I)}{d}
>    \right)^d.
>    $$
>
> 3. 再用题设的 $\bar\sigma_{i,h}^{-2}\le 1$ 与 $\|\phi_i\|_2\le 1$ 推出
>    $$
>    \mathrm{Tr}(\Sigma_{K+1,h}-\lambda I)
>    \le
>    K.
>    $$
>    从而得到
>    $$
>    m_h
>    \lesssim
>    d\log(1+K/\lambda).
>    $$
>
> 4. 最后说明总更新次数满足
>    $$
>    \sum_{h=1}^{H}m_h
>    =
>    O(dH\log(1+K/\lambda)).
>    $$

> [!success]- 核心思路（点击展开）
> 先证明第一问。设第 $h$ 层的更新时刻为
> $$
> 1<k_1<k_2<\cdots<k_{m_h}\le K+1.
> $$
> 由触发条件，每发生一次更新，det 至少翻倍，所以
> $$
> \det(\Sigma_{k_j,h})
> \ge
> 2\det(\Sigma_{k_{j-1},h})
> \qquad (j=1,\dots,m_h),
> $$
> 其中把初始矩阵
> $$
> \Sigma_{1,h}=\lambda I
> $$
> 看作第 $0$ 次参考点。递推下去得到
> $$
> \det(\Sigma_{k_{m_h},h})
> \ge
> 2^{m_h}\det(\lambda I).
> $$
> 又由于 $\Sigma_{k,h}$ 随 $k$ 单调增大（每一步都在加半正定矩阵），故
> $$
> \det(\Sigma_{K+1,h})
> \ge
> \det(\Sigma_{k_{m_h},h}).
> $$
> 因而
> $$
> \det(\Sigma_{K+1,h})
> \ge
> 2^{m_h}\det(\lambda I),
> $$
> 也就是
> $$
> m_h
> \le
> \log_2\frac{\det(\Sigma_{K+1,h})}{\det(\lambda I)}.
> $$
>
> 第二问用特征值记号更清楚。设 $\Sigma_{K+1,h}$ 的特征值为 $\gamma_1,\dots,\gamma_d$，则
> $$
> \det(\Sigma_{K+1,h})=\prod_{j=1}^{d}\gamma_j,
> \qquad
> \mathrm{Tr}(\Sigma_{K+1,h})=\sum_{j=1}^{d}\gamma_j.
> $$
> 由 AM--GM，
> $$
> \left(\prod_{j=1}^{d}\gamma_j\right)^{1/d}
> \le
> \frac1d\sum_{j=1}^{d}\gamma_j
> =
> \frac{\mathrm{Tr}(\Sigma_{K+1,h})}{d}.
> $$
> 故
> $$
> \det(\Sigma_{K+1,h})
> \le
> \left(\frac{\mathrm{Tr}(\Sigma_{K+1,h})}{d}\right)^d
> =
> \left(
> \lambda+\frac{\mathrm{Tr}(\Sigma_{K+1,h}-\lambda I)}{d}
> \right)^d.
> $$
>
> 第三问估计 trace 增量：
> $$
> \Sigma_{K+1,h}-\lambda I
> =
> \sum_{i=1}^{K}\bar\sigma_{i,h}^{-2}\phi_i\phi_i^\top.
> $$
> 因此
> $$
> \mathrm{Tr}(\Sigma_{K+1,h}-\lambda I)
> =
> \sum_{i=1}^{K}\bar\sigma_{i,h}^{-2}\mathrm{Tr}(\phi_i\phi_i^\top)
> =
> \sum_{i=1}^{K}\bar\sigma_{i,h}^{-2}\|\phi_i\|_2^2
> \le
> \sum_{i=1}^{K}1
> =
> K.
> $$
> 代回上一式，
> $$
> \det(\Sigma_{K+1,h})
> \le
> \left(\lambda+\frac{K}{d}\right)^d
> \le
> (\lambda+K)^d.
> $$
> 再除以
> $$
> \det(\lambda I)=\lambda^d
> $$
> 后得到
> $$
> \frac{\det(\Sigma_{K+1,h})}{\det(\lambda I)}
> \le
> \left(1+\frac{K}{\lambda}\right)^d.
> $$
> 于是
> $$
> m_h
> \le
> \log_2\frac{\det(\Sigma_{K+1,h})}{\det(\lambda I)}
> \lesssim
> d\log(1+K/\lambda).
> $$
>
> 最后对所有 $h=1,\dots,H$ 求和，就得到总更新次数
> $$
> \sum_{h=1}^{H}m_h
> =
> O(dH\log(1+K/\lambda)).
> $$
>
> 这说明 rare-switching 把“每个 episode 每个 stage 都可能更新一次”的最坏情况
> $$
> O(KH)
> $$
> 压缩成了
> $$
> O(dH\log K)
> $$
> 量级。

> [!info]- 必要说明（点击展开）
> rare-switching 和 elliptical potential lemma 本质上是同一种 log-det 思想的两种用法：
>
> - 在 regret 求和里，它控制“探索量总账”；
> - 在这里，它控制“真正发生函数切换的次数”。
>
> 这也是为什么它不只是工程 trick，而是证明中不可替代的一环。

---

> [!question] (2026-03-14) 第1次做，正确总0次，上次无  
> **题目 11：在理想化主项下，重建 LSVI-UCB++ 的 $\tilde O(d\sqrt{H^3K})$**  
> 为了让这一题的推导直接闭合，下面把第二个输入写成对 proxy variance $\bar\sigma_{k,h}$ 的总量控制；在实际论文里，这一步通常由真方差总和与单调方差上界进一步推出。
>
> 假设你已经得到了如下三个输入：
>
> 1. **Bernstein + 加权回归** 给出更紧的 bonus 尺度：
>    $$
>    \beta=\tilde O(\sqrt d).
>    $$
>
> 2. **总 proxy variance 可控：**
>    $$
>    \sum_{k=1}^{K}\sum_{h=1}^{H}\bar\sigma_{k,h}^2
>    =
>    \tilde O(H^2K).
>    $$
>
> 3. **加权椭球势可控：**
>    $$
>    \sum_{k=1}^{K}\sum_{h=1}^{H}
>    \bar\sigma_{k,h}^{-2}\phi_{k,h}^\top\Sigma_{k,h}^{-1}\phi_{k,h}
>    =
>    \tilde O(dH).
>    $$
>
> 设主 bonus 和为
> $$
> \mathcal B
> :=
> \sum_{k=1}^{K}\sum_{h=1}^{H}
> \beta\,\bar\sigma_{k,h}\,
> \sqrt{
> \bar\sigma_{k,h}^{-2}
> \phi_{k,h}^\top\Sigma_{k,h}^{-1}\phi_{k,h}
> }.
> $$
> 请完成下面三步：
>
> 4. 用 Cauchy--Schwarz 证明
>    $$
>    \mathcal B
>    \le
>    \beta\,
>    \sqrt{
>    \left(\sum_{k,h}\bar\sigma_{k,h}^2\right)
>    \left(
>    \sum_{k,h}\bar\sigma_{k,h}^{-2}\phi_{k,h}^\top\Sigma_{k,h}^{-1}\phi_{k,h}
>    \right)
>    }.
>    $$
>
> 5. 将三个输入代入，推出
>    $$
>    \mathcal B
>    =
>    \tilde O(d\sqrt{H^3K}).
>    $$
>
> 6. 最后写一句“技术因子对照表”，明确指出：
>
>    - 哪一步消掉了主要的 $\sqrt H$；
>    - 哪一步消掉了主要的 $\sqrt d$；
>    - 哪一步保证单调方差上界可用、但又不至于让函数类复杂度失控。

> [!success]- 核心思路（点击展开）
> 先把 Cauchy--Schwarz 写得机械一点。令
> $$
> a_{k,h}:=\bar\sigma_{k,h},
> \qquad
> b_{k,h}:=
> \sqrt{
> \bar\sigma_{k,h}^{-2}
> \phi_{k,h}^\top\Sigma_{k,h}^{-1}\phi_{k,h}
> }.
> $$
> 则
> $$
> \mathcal B
> =
> \beta\sum_{k,h}a_{k,h}b_{k,h}.
> $$
> 由 Cauchy--Schwarz，
> $$
> \sum_{k,h}a_{k,h}b_{k,h}
> \le
> \sqrt{\sum_{k,h}a_{k,h}^2}\,
> \sqrt{\sum_{k,h}b_{k,h}^2}.
> $$
> 代回 $a_{k,h},b_{k,h}$ 的定义，就得到
> $$
> \mathcal B
> \le
> \beta\,
> \sqrt{
> \left(\sum_{k,h}\bar\sigma_{k,h}^2\right)
> \left(
> \sum_{k,h}\bar\sigma_{k,h}^{-2}\phi_{k,h}^\top\Sigma_{k,h}^{-1}\phi_{k,h}
> \right)
> }.
> $$
>
> 现在代入三个输入：
> $$
> \beta=\tilde O(\sqrt d),
> \qquad
> \sum_{k,h}\bar\sigma_{k,h}^2=\tilde O(H^2K),
> \qquad
> \sum_{k,h}\bar\sigma_{k,h}^{-2}\phi_{k,h}^\top\Sigma_{k,h}^{-1}\phi_{k,h}
> =\tilde O(dH).
> $$
> 于是
> $$
> \mathcal B
> =
> \tilde O(\sqrt d)\cdot
> \sqrt{H^2K}\cdot
> \sqrt{dH}
> =
> \tilde O(d\sqrt{H^3K}).
> $$
> 这就是理想化主项下 LSVI-UCB++ 的目标量级。
>
> 最后的“技术因子对照表”可以写成一句很紧的总括：
>
> - **Hoeffding $\to$ Bernstein + variance-aware weighting**：把单个 fixed target 的噪声尺度从统一的 $H$ 降到方差尺度，消掉主要的 $\sqrt H$；
> - **直接 uniform 控制自适应的 $V_{k,h+1}$ $\to$ 先抽出固定的 $V^\star_{h+1}$**：把大函数类的 covering 代价从主项上挪走，消掉主要的 $\sqrt d$；
> - **每轮都更新 $\to$ rare-switching**：既保住单调方差上界，又把函数类分支数压到 $O(dH\log K)$，避免 class complexity explosion。
>
> 因而，从 Jin 2020 到 He 2023，最核心的三件事分别对应三个技术动作：
> $$
> \sqrt H,\ \sqrt d,\ \text{以及函数类复杂度爆炸}
> $$
> 各自被不同的机制处理掉，而不是靠同一个技巧“一把梭”。

> [!info]- 必要说明（点击展开）
> 这题最好作为整份错题本的总收束来反复看。因为它把所有局部 lemma 都压缩成了一个三行总账：
>
> 1. bonus 尺度为什么从 $H\sqrt d$ 变成 $\sqrt d$；
> 2. 为什么总方差给出一个 $\sqrt{H^2K}$；
> 3. 为什么加权椭球势只给出一个 $\sqrt{dH}$。
>
> 三者一乘，就是
> $$
> d\sqrt{H^3K}.
> $$


---

## 参考文献（便于你继续扩展）

1. Chi Jin, Zhuoran Yang, Zhaoran Wang, Michael I. Jordan. **Provably Efficient Reinforcement Learning with Linear Function Approximation.** COLT 2020.  
2. Jiafan He, Heyang Zhao, Dongruo Zhou, Quanquan Gu. **Nearly Minimax Optimal Reinforcement Learning for Linear Markov Decision Processes.** ICML 2023.  
3. Mohammad Gheshlaghi Azar, Ian Osband, Rémi Munos. **Minimax Regret Bounds for Reinforcement Learning.** ICML 2017.  
4. Dongruo Zhou, Quanquan Gu, Csaba Szepesvári. **Nearly Minimax Optimal Reinforcement Learning for Linear Mixture MDPs.** COLT 2021.  
5. Zihan Zhang, Jiaqi Yang, Xiangyang Ji, Simon S. Du. **Improved Variance-Aware Confidence Sets for Linear Bandits and Linear Mixture MDP.** NeurIPS 2021.  
6. Zihan Zhang, Yuan Zhou, Xiangyang Ji. **Almost Optimal Model-Free Reinforcement Learning via Reference-Advantage Decomposition.** NeurIPS 2020.