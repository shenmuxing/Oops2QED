Practise thou each morn, till thou hast it at thy fingers' ends.

---

MDP:

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},r,\mathbb{P},\gamma),
$$

where $\mathcal{S}$ and $\mathcal{A}$ are finite, and $\gamma\in[0,1)$.

We consider a policy $\pi$.

The value function is

$$
V^\pi(s)
=
\mathbb{E}_{a_t\sim \pi(\cdot\mid s_t),\ s_{t+1}\sim \mathbb{P}(\cdot\mid s_t,a_t)}
\left[
\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)
\mid s_0=s
\right].
$$

For an initial distribution $\mu$, we define

$$
V^\pi(\mu)
=
\mathbb{E}_{s_0\sim \mu}
\left[
V^\pi(s_0)
\right].
$$

The action-value function is

$$
Q^\pi(s,a)
=
\mathbb{E}_{a_t\sim \pi(\cdot\mid s_t),\ s_{t+1}\sim \mathbb{P}(\cdot\mid s_t,a_t)}
\left[
\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)
\mid s_0=s,\ a_0=a
\right].
$$

Relation of $V^\pi$ and $Q^\pi$:

$$
V^\pi(s)
=
\sum_{a\in\mathcal{A}}
\pi(a\mid s)Q^\pi(s,a).
$$

Also,

$$
Q^\pi(s,a)
=
r(s,a)
+
\gamma
\sum_{s'\in\mathcal{S}}
\mathbb{P}(s'\mid s,a)V^\pi(s').
$$

Bellman expectation equations:
$$
V^\pi(s)
=
\sum_{a\in\mathcal{A}}
\pi(a\mid s)
\left[
r(s,a)
+
\gamma
\sum_{s'\in\mathcal{S}}
\mathbb{P}(s'\mid s,a)V^\pi(s')
\right].
$$

The optimal value function is
$$V^\star(s)=\sup_{\pi} V^\pi(s).$$

The optimal action-value function is
$$ Q^\star(s,a) = \sup_{\pi} Q^\pi(s,a).$$

The Bellman optimality equation for $V^\star$ is

$$
V^\star(s)
=
\max_{a\in\mathcal{A}}
\left[
r(s,a)
+
\gamma
\sum_{s'\in\mathcal{S}}
\mathbb{P}(s'\mid s,a)V^\star(s')
\right].
$$

The Bellman optimality equation for $Q^\star$ is

$$
Q^\star(s,a)
=
r(s,a)
+
\gamma
\sum_{s'\in\mathcal{S}}
\mathbb{P}(s'\mid s,a)
\max_{a'\in\mathcal{A}}Q^\star(s',a').
$$

An optimal policy can be obtained greedily from $Q^\star$:

$$
\pi^\star(s)
\in
\arg\max_{a\in\mathcal{A}}Q^\star(s,a).
$$

## Value Iteration

Value iteration directly applies the Bellman optimality update.

Initialize $V_0(s)$ arbitrarily for all $s\in\mathcal{S}$.

For $k=0,1,2,\dots$, update

$$
V_{k+1}(s)
=
\max_{a\in\mathcal{A}}
\left[
r(s,a)
+
\gamma
\sum_{s'\in\mathcal{S}}
\mathbb{P}(s'\mid s,a)V_k(s')
\right],
\qquad \forall s\in\mathcal{S}.
$$

After enough iterations, define the greedy policy

$$
\pi_k(s)
\in
\arg\max_{a\in\mathcal{A}}
\left[
r(s,a)
+
\gamma
\sum_{s'\in\mathcal{S}}
\mathbb{P}(s'\mid s,a)V_k(s')
\right].
$$

When $k$ is large enough, $V_k$ approximates $V^\star$, and the greedy policy $\pi_k$ approximates an optimal policy.

Main known-model planning facts:

- VI has no learning regret by itself; it assumes $P,r$ are known.
- The Bellman optimality operator is a $\gamma$-contraction.
- For $Q$-value iteration $Q^{(k+1)}=\mathcal{T}Q^{(k)}$ with $Q^{(0)}=0$, the greedy policy is $\epsilon$-optimal once

$$
k
\ge
\frac{\log\frac{2}{(1-\gamma)^2\epsilon}}{1-\gamma}.
$$

- Per iteration, the usual tabular update costs about $|\mathcal{S}|^2|\mathcal{A}|$.
- For exact optimal planning with rational $(P,r,\gamma)$, the book's table records VI complexity roughly

$$
|\mathcal{S}|^2|\mathcal{A}|\,L(P,r,\gamma)
\log\frac{1}{1-\gamma},
$$

and VI is not strongly polynomial.

### UCBVI

UCBVI is the online-learning version of finite-horizon value iteration for an unknown tabular MDP.

Setting:

- episode length $H$,
- $K$ episodes,
- finite state/action sizes $S,A$,
- known rewards $r_h(s,a)\in[0,1]$,
- unknown transition kernels $P_h^\star$.

The regret is

$$
\operatorname{Regret}
=
\mathbb{E}
\left[
\sum_{k=0}^{K-1}
\left(
V_0^\star(s_0)
-
V_0^{\pi^k}(s_0)
\right)
\right].
$$

At the start of episode $k$, form counts from previous episodes:

$$
N_h^k(s,a,s')
=
\sum_{i=0}^{k-1}
\mathbf{1}\{(s_h^i,a_h^i,s_{h+1}^i)=(s,a,s')\},
$$

$$
N_h^k(s,a)
=
\sum_{i=0}^{k-1}
\mathbf{1}\{(s_h^i,a_h^i)=(s,a)\}.
$$

The empirical transition model is

$$
\widehat P_h^k(s'\mid s,a)
=
\frac{N_h^k(s,a,s')}{N_h^k(s,a)}.
$$

For unvisited $(h,s,a)$, use an initialization convention or an infinite bonus; the displayed formulas are the visited-pair formulas.

With

$$
L=\ln(SAHK/\delta),
$$

use the optimistic bonus

$$
b_h^k(s,a)
=
2H\sqrt{\frac{L}{N_h^k(s,a)}}.
$$

Then run backward finite-horizon VI on the empirical model with reward $r+b$:

$$
\widehat V_H^k(s)=0,
$$

$$
\widehat Q_h^k(s,a)
=
\min\left\{
r_h(s,a)
+
b_h^k(s,a)
+
\widehat P_h^k(\cdot\mid s,a)^\top
\widehat V_{h+1}^k,
H
\right\},
$$

$$
\widehat V_h^k(s)
=
\max_a \widehat Q_h^k(s,a),
\qquad
\pi_h^k(s)
\in
\arg\max_a \widehat Q_h^k(s,a).
$$

Execute $\pi^k$ for one episode, collect a trajectory, update the counts, and repeat.

Main regret facts:

- The simple analysis in Chapter 7 gives

$$
\operatorname{Regret}
\le
10H^2S
\sqrt{
AK\ln(SAH^2K^2)
}
=
\widetilde O(H^2S\sqrt{AK}).
$$

- The refined analysis improves the leading $S$ dependence:

$$
\operatorname{Regret}
=
\widetilde O
\left(
H^2\sqrt{SAK}
+
H^3S^2A
\right).
$$

- The proof skeleton is optimism from the bonus, then simulation-lemma regret decomposition, then count summation

$$
\sum_{k,h}
\frac{1}{\sqrt{N_h^k(s_h^k,a_h^k)}}
\lesssim
H\sqrt{SAK}.
$$

## Policy Iteration

Policy iteration alternates between policy evaluation and policy improvement.

Initialize a policy $\pi_0$ arbitrarily.

For $k=0,1,2,\dots$, repeat the following two steps.

### Policy Evaluation

Given the current policy $\pi_k$, compute its value function $V^{\pi_k}$ by solving

$$
V^{\pi_k}(s)
=
\sum_{a\in\mathcal{A}}
\pi_k(a\mid s)
\left[
r(s,a)
+
\gamma
\sum_{s'\in\mathcal{S}}
\mathbb{P}(s'\mid s,a)V^{\pi_k}(s')
\right],
\qquad \forall s\in\mathcal{S}.
$$

### Policy Improvement

First define

$$
Q^{\pi_k}(s,a)
=
r(s,a)
+
\gamma
\sum_{s'\in\mathcal{S}}
\mathbb{P}(s'\mid s,a)V^{\pi_k}(s'),
$$

then update

$$
\pi_{k+1}(s)
\in
\arg\max_{a\in\mathcal{A}}Q^{\pi_k}(s,a).
$$

Policy iteration stops when the policy no longer changes:

$$
\pi_{k+1}=\pi_k.
$$

At this point, the policy is optimal:

$$
\pi_k=\pi^\star,
\qquad
V^{\pi_k}=V^\star.
$$

Main known-model planning facts:

- PI also has no learning regret by itself; it assumes $P,r$ are known.
- Policy improvement gives monotonic improvement:

$$
Q^{\pi_{k+1}}
\ge
\mathcal{T}Q^{\pi_k}
\ge
Q^{\pi_k}.
$$

- The policy sequence contracts toward optimality:

$$
\|Q^{\pi_{k+1}}-Q^\star\|_\infty
\le
\gamma
\|Q^{\pi_k}-Q^\star\|_\infty.
$$

- The $k$-th policy satisfies $Q^{\pi_k}\ge Q^\star-\epsilon\mathbf{1}$ once

$$
k
\ge
\frac{\log\frac{1}{(1-\gamma)\epsilon}}{1-\gamma}.
$$

- Per iteration, policy evaluation plus improvement costs about

$$
|\mathcal{S}|^3
+
|\mathcal{S}|^2|\mathcal{A}|,
$$

using direct linear-system solving for evaluation.

- For exact optimal planning with rational $(P,r,\gamma)$, the book's table records PI complexity roughly

$$
\left(
|\mathcal{S}|^3
+
|\mathcal{S}|^2|\mathcal{A}|
\right)
L(P,r,\gamma)
\log\frac{1}{1-\gamma}.
$$

- PI is strongly polynomial when $\gamma$ is fixed; one exact-iteration bound is

$$
\frac{
|\mathcal{S}|^2|\mathcal{A}|
\log\frac{|\mathcal{S}|^2}{1-\gamma}
}{
1-\gamma
}.
$$

## Difference Between Value Iteration and Policy Iteration

In value iteration, the max operator appears inside every value update.

In policy iteration, the max operator appears only in the policy improvement step.
