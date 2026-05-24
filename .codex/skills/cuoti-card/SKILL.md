---
name: cuoti-card
description: Create, revise, and review Obsidian mistake-notebook cards for this repository. Use when the user asks to 出题, 制作题卡, 讲解, 复盘, 变式, 复习, or update statistics for academic derivation practice in the 错题本 project.
---

# Cuoti Card

## Scope

Use this skill for the Obsidian repository `错题本`, whose purpose is to record formula-derivation sticking points, common mistakes, and unfamiliar techniques as reusable practice cards.

Write in Chinese by default. Switch to bilingual or English only when the user asks.

Before and after modifying any Markdown file, run a strict UTF-8 decode check and report the result. Prefer manual edits. Use scripted bulk replacement only when the replacement is low-risk and tightly scoped.

## File Rules

- Create new Codex-authored card files as `codex_YYYY-MM-DD.md`.
- Append multiple cards to the same dated file in chronological order.
- Do not rewrite historical card text unless the user explicitly asks. For substantive corrections, add a dated "修订版" card.
- Preserve compatibility with existing Obsidian callout cards.

## Card Template

Use this default structure:

```markdown
> [!question] (YYYY-MM-DD) 第n次做，正确总x次，上次正确/错误/新题
> **题目：**（完整题干，含符号定义与必要前提）

> [!success]- 核心思路（点击展开）
> 给出可复现的推导链条：关键恒等式 -> 中间步骤 -> 最终结论。
> 如有多解法，先主线再补充替代思路。

> [!info]- 错误原因（点击展开）
> 定位到具体缺口：知识点缺失 / 步骤跳跃 / 符号误用 / 计算粗心。
```

Optional metadata:

```markdown
出处：书名 + 章节 + 题号
标签：#policy-gradient #measure-theory #bandit
```

## Quality Bar

- Make every problem self-contained: define symbols, assumptions, and the target statement.
- Make every solution checkable: include the key identities and intermediate transformations, not only the final result.
- Make every error reason actionable: identify exactly where the user's reasoning broke and what to check next time.
- If the user has not provided an answer result, use `上次新题` for new cards or leave existing statistics unchanged and ask for confirmation.

## Interaction Modes

Handle common commands as follows:

- `出题`: create a self-contained card matching the requested source, knowledge point, difficulty, language, and hint preference.
- `讲解`: rewrite the `核心思路` in the requested style, such as intuition-first then rigorous proof.
- `复盘`: update statistics if the result is provided, rewrite `错误原因`, and add a concrete next-review suggestion.
- `变式`: generate requested variants from the original card, clearly marking lower or higher difficulty when relevant.
- `复习`: select cards from the requested time window and return practice prompts before revealing solutions unless the user asks for full answers.

## Statistics

- `第n次做`: increment by 1 each time the same card is redone.
- `正确总x次`: increment only when the user reports a correct answer.
- `上次正确/错误/新题`: reflect the immediately previous attempt.
