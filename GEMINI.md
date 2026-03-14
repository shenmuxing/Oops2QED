这个是一个obsidian仓库，名字叫`错题本`. 用于记录用户在学术上公式推导中遇到卡壳、不太会的地方，并帮助用户用闪卡的思想复习经常犯错、不熟悉的点帮助用户提升能力。

Gemini每次创作题卡，应该单独创建一个新的markdown 文件，以`gemini_[日期].md` 作为文件标题，每个题目按照下面的模板来写：

> [!question] (记录日期年月日) 第n次做，正确总x次，上次正确/错误
> 问题正文

> [!success]- 核心思路（点击展开）
>  这里利用了 $\nabla_\theta p(x) = p(x) \nabla_\theta \log p(x)$ 这一恒等式。
>  $$\begin{aligned}\nabla J(\theta) &= \int \nabla p(x) f(x) dx \\&= \int p(x) \frac{\nabla p(x)}{p(x)} f(x) dx \\&= \mathbb{E}_{x \sim p} [\nabla \log p(x) f(x)]\end{aligned}$$


> [!info]- 错误原因（点击展开）
> 应具体指出缺失的知识点或思维断层（例如：“未掌握利用生成类证明随机向量可测性的方法”）。


题目需要保证是一个完整的题目，尽量给出符号和必要的前提条件。例如：

> [!question] (2026-01-29) 第2次做，正确总0次，上次错误
> **题目：Softmax Policy Gradient Lemma 12.2 推导**
> 
> 给定 Softmax 策略：
> $$ \pi_ {\theta} (a | s) = \frac {\exp (\theta_ {s , a})}{\sum_ {a ^ {\prime}} \exp (\theta_ {s , a ^ {\prime}})} $$ 
> 已知其对数梯度为：
> $$ \frac {\partial \log \pi_ {\theta} (a | s)}{\partial \theta_ {s ^ {\prime} , a ^ {\prime}}} = \mathbb {1} [ s = s ^ {\prime} ] \left(\mathbb {1} [ a = a ^ {\prime} ] - \pi_ {\theta} (a ^ {\prime} | s)\right) $$ 
> 
> 请证明：
> $$ \frac {\partial V ^ {\pi_ {\theta}} (\mu)}{\partial \theta_ {s , a}} = \frac {1}{1 - \gamma} d _ {\mu} ^ {\pi_ {\theta}} (s) \pi_ {\theta} (a | s) A ^ {\pi_ {\theta}} (s, a) $$ 
>
> 其中 $\mu$ 为初始状态分布，$d_{\mu}^{\pi_{\theta}}(s)$ 为 初始状态为 $\mu$ 时的占用度量。已知：
> $$ \operatorname{Pr}_{\mu}^{\pi}(\tau)=\mu\left(s_{0}\right) \pi\left(a_{0} \mid s_{0}\right) P\left(s_{1} \mid s_{0}, a_{0}\right) \pi\left(a_{1} \mid s_{1}\right) \cdots .$$
> It is convenient to define the discounted total reward of a trajectory as:
> $$R(\tau):=\sum_{t=0}^{\infty} \gamma^{t} r\left(s_{t}, a_{t}\right)$$
> where $s_{t}, a_{t}$ are the state-action pairs in $\tau$. Observe that:
> $$V^{\pi_{\theta}}(\mu)=\mathbb{E}_{\tau \sim \operatorname{Pr}_{\mu}^{\pi_{\theta}}}[R(\tau)] .$$
> Policy Gradient in Advantage expression:
> $$\nabla V^{\pi_{\theta}}(\mu)=\frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi_{\theta}}} \mathbb{E}_{a \sim \pi_{\theta}(\cdot \mid s)}\left[A^{\pi_{\theta}}(s, a) \nabla \log \pi_{\theta}(a \mid s)\right]$$


Gemini的职责是根据用户需求写问题、正确答案和错误原因，以及其它可能的需求。

Gemini可以使用中文/英文写题卡。