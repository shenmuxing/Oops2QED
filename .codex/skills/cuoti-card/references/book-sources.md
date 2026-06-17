# Book Sources

Use this index when a card request names a book, paper, or shorthand source. Prefer the local file first; use the fallback URL only when the local source is unavailable or cannot be read.

## rltheorybook

- Title: Reinforcement Learning Theory
- Local path: `local\rltheorybook.pdf`
- Fallback URL: `https://rltheorybook.github.io/rltheorybook_ABJKS.pdf`
- Notes: Use this source for requests that mention `rltheorybook`, `RL theory book`, or related reinforcement-learning theory exercises.

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
