# Book Sources

Use this index when a card request names a book, paper, or shorthand source. Prefer the local file first; use the fallback URL only when the local source is unavailable or cannot be read.

Default placement rule: only sources with an explicit `Note folder` below, or sources the user explicitly says should be a standalone体系, get a dedicated folder. Standalone papers without such a folder belong to the repository-root regular dated-card system.

## regular-papers

- Default note folder: repository root, using regular dated cards like `YYYY-MM-DD.md`.
- Title: A Diffusion Analysis of Policy Gradient for Stochastic Bandits
- Author: Tor Lattimore
- Local path (absolute): `D:\ProgramData\OnedriveData\OneDrive\文档\zotero\storage\DLC6PFM7\Lattimore - 2026 - A Diffusion Analysis of Policy Gradient for Stochastic Bandits.pdf`
- Zotero item: `HU7I7KHR`
- Zotero attachment: `DLC6PFM7`
- Fallback URL: `https://arxiv.org/pdf/2603.10219`
- Existing cards: `2026-06-19.md` for Algorithm 2 and Lemma 1; `2026-06-22.md` for Lemma 2.
- Aliases: `lattimore diffusion`, `diffusion bandits`, `A diffusion analysis of policy gradient for stochastic bandits`, `arXiv:2603.10219`
- Notes: Use this source for continuous-time softmax policy-gradient bandit diffusion, conservation of softmax parameters, and Lemma 2-style lower-tail control of theta coordinates. Keep cards in the root regular system unless the user explicitly asks for a dedicated folder.

## rltheorybook

- Title: Reinforcement Learning Theory
- Local path: `local\rltheorybook.pdf`
- Fallback URL: `https://rltheorybook.github.io/rltheorybook_ABJKS.pdf`
- Notes: Use this source for requests that mention `rltheorybook`, `RL theory book`, or related reinforcement-learning theory exercises.

## stochastic-process

- Scope: Umbrella collection for stochastic-process courses, books, papers, and future related materials.
- Local folder: `local\stochastic-process\`
- Default note folder: `stochastic-process\`
- Aliases: `stochastic process`, `随机过程`
- Organization: Local materials may use one source subfolder per course, book, or paper. Keep all stochastic-process cards directly in `stochastic-process\` without source, lecture, chapter, or topic subfolders. Name cards only by date following the normal `YYYY-MM-DD.md` rule, append same-day cards to the same file, and distinguish sources inside each card's `出处`. When the user names only the general topic, use the primary source below unless they specify another source.

### zhang-chihao-ai2613

- Title: AI2613 随机过程
- Author: 张驰豪
- Status: Primary source for the current stochastic-process study plan.
- Local folder: `local\stochastic-process\zhang-chihao-ai2613\`
- Note folder: `stochastic-process\` using flat dated cards shared with all other stochastic-process sources.
- Aliases: `AI2613`, `AI2613 随机过程`, `张驰豪随机过程`, `驰豪随机过程`, `驰豪的随机过程`
- Lectures:
  - Lecture 1: `local\stochastic-process\zhang-chihao-ai2613\lec1.pdf` — 独立随机变量
  - Lecture 2: `local\stochastic-process\zhang-chihao-ai2613\lec2.pdf` — 有限状态马尔可夫链
  - Lecture 3: `local\stochastic-process\zhang-chihao-ai2613\lec3.pdf` — 马尔可夫链基本定理与耦合法
  - Lecture 4: `local\stochastic-process\zhang-chihao-ai2613\lec4.pdf` — 可数无穷状态马尔可夫链
  - Lecture 5: `local\stochastic-process\zhang-chihao-ai2613\lec5.pdf` — 马尔可夫链基本定理（一般版本）及应用
  - Lecture 6: `local\stochastic-process\zhang-chihao-ai2613\lec6.pdf` — 泊松过程
  - Lecture 7: `local\stochastic-process\zhang-chihao-ai2613\lec7.pdf` — 泊松近似
  - Lecture 8: `local\stochastic-process\zhang-chihao-ai2613\lec8.pdf` — 连续时间离散空间马尔可夫过程
  - Lecture 9: `local\stochastic-process\zhang-chihao-ai2613\lec9.pdf` — $\sigma$-代数与条件期望
  - Lecture 10: `local\stochastic-process\zhang-chihao-ai2613\lec10.pdf` — 离散鞅与可选停时定理
  - Lecture 11: `local\stochastic-process\zhang-chihao-ai2613\lec11.pdf` — 鞅与集中不等式
  - Lecture 12: `local\stochastic-process\zhang-chihao-ai2613\lec12.pdf` — 布朗运动
  - Lecture 13: `local\stochastic-process\zhang-chihao-ai2613\lec13.pdf` — 扩散过程和伊藤积分
  - Lecture 14: `local\stochastic-process\zhang-chihao-ai2613\lec14.pdf` — 马尔可夫过程、朗之万蒙特卡洛与扩散模型
- Notes: Use this course as the primary source for future stochastic-process study, derivation cards, reviews, and variants until the user selects another source. Do not create a course or content subfolder for its cards. In each card's `出处`, include `AI2613 随机过程`, the lecture number, and the relevant section or page when available.

## policy-gradient-policy-iteration

- Title: New Advances in Policy Optimization
- Local folder: `local\policy-gradient-policy-iteration\`
- Primary thesis: `local\policy-gradient-policy-iteration\phd-thesis-new-advances-in-policy-optimization.pdf`
- Note folder: `policy-gradient-policy-iteration\`
- Aliases: `jiacai, 2026`, `Jiacai 2026`, `Liu Jiacai thesis`, `phd-thesis`
- Related papers:
  - Hadamard PG: `local\policy-gradient-policy-iteration\hadamard-policy-gradient-linear-convergence-2025.pdf`
  - Projected PG: `local\policy-gradient-policy-iteration\projected-policy-gradient-any-constant-stepsizes-2025.pdf`
  - Elementary PG analysis: `local\policy-gradient-policy-iteration\elementary-analysis-policy-gradient-methods-2024.pdf`
  - Phi-update: `local\policy-gradient-policy-iteration\phi-update-policy-convergence-iclr-2025.pdf`
  - DAPO: `local\policy-gradient-policy-iteration\dapo-direct-advantage-based-policy-optimization-neurips-2025.pdf`
- Fallback URLs:
  - Hadamard PG: `https://arxiv.org/pdf/2305.19575`
  - Projected PG: `https://www.jmlr.org/papers/volume26/24-1530/24-1530.pdf`
  - Elementary PG analysis: `https://arxiv.org/pdf/2404.03372`
  - Phi-update: `https://openreview.net/pdf?id=fh7GYa7cjO`
  - DAPO: `https://openreview.net/pdf?id=77eEDRhPkQ`
- Notes: Use this source for requests about Liu Jiacai's thesis series, new advances in policy optimization, policy gradient versus policy iteration, direct/simplex parameterization, softmax PG/NPG, entropy-regularized soft policy iteration, Hadamard PG, phi-update, or DAPO. Prefer the thesis for chapter-level orientation, then use the related papers for exact theorem statements and proofs.
- Alias rule: When the user says `jiacai, 2026`, resolve it to the primary thesis above, not to the related papers, unless the user explicitly asks for a paper-level theorem or proof cross-check.
