---
name: weekly-report
 description: "Generate a weekly report for an AI framework engineer. Converts raw weekly notes (fragmented, mixed Chinese/English, shorthand) into a concise, manager-friendly weekly report. Produces Chinese draft first for review, then outputs both an English final version and a manager report version. Use when the user provides weekly notes or asks to generate a weekly report."
argument-hint: "Paste raw weekly notes"
---

# Weekly Report Generation

Convert raw engineering notes into a concise, realistic, manager-friendly weekly report.

## When to Use
- User provides raw weekly notes (Chinese, English, or mixed)
- User asks to generate or format a weekly report
- User wants to summarize weekly progress for managers

## Workflow

Always follow this workflow unless the user explicitly requests otherwise:

### Step 1 -- Chinese Draft

Generate a Chinese draft first for review.

Purpose:
- Help user quickly verify technical correctness
- Make edits efficiently
- Reduce misunderstanding

### Step 2 -- English Final Version

After user confirms or provides edits, generate concise English weekly report.

### Step 3 -- Manager Report

Along with the English final version, also generate a shorter manager scan version for quick reading.

### Step 4 -- Data Tables (Optional)

When notes contain quantitative results (performance numbers, kernel counts, before/after comparisons), append a compact markdown table as a "Data Appendix" section after Risk / Dependency.

Rules:
- Only include when there are concrete numbers worth comparing (e.g. latency, kernel count, speedup)
- Keep table compact: 3-6 rows max, only columns that matter
- Use aligned columns and consistent units (ms, µs, ×)
- Table title should be a single descriptive line
- Do not duplicate prose already in the bullets; the table supplements, not replaces
- If no quantitative data in the notes, skip this section entirely
- Include in all versions (Chinese draft, English final, manager report) when present

Example:

```
Data Appendix

MXFP8 quant kernel performance (M=1024, BMG B60):

| Model | Compiled (ms) | Handwritten C (ms) | Speedup |
|---|---|---|---|
| Llama-3.1 K=8192 | 0.149 | 0.344 | 2.3× |
| Llama-4 K=5120 | 0.106 | 0.208 | 2.0× |
```

### Step 5 -- Concise Version (Optional)

If requested, further compress either final output.

## Chinese Draft Rules

Requirements:
- Concise, technically accurate, easy to review
- Preserve technical meaning
- Natural engineer tone
- No excessive wording, no literal translation style
- Default to grouped workstreams instead of item-by-item rewrite
- Each delivery bullet should summarize progress plus conclusion, impact, blocker, or next decision when relevant
- If the source notes are already well-grouped, prefer direct summary bullets over forcing extra abstraction

Structure:

```
Last Week Delivery

- <category>
  - ...

Next Week Plan

- ...

Risk / Dependency

- ...
```

Section titles in the draft may be Chinese or English. The draft content itself should be in Chinese.

Use sub-categories when useful: UT, Feature, Investigation, Infra / Efficiency, Performance, Validation, Tooling.

Prefer 2-4 grouped workstreams for Last Week Delivery when notes are fragmented, but allow more bullets when there are several distinct meaningful updates.

## English Final Version Rules

Structure:

```
Last Week Delivery

- <category>
  - ...

Next Week Plan

- ...

Risk / Dependency

- ...
```

This is the standard shareable version and must follow the formatting rules exactly.

Requirements:
- Default to grouped workstreams instead of item-by-item rewrite
- Each delivery bullet should summarize progress plus conclusion, impact, blocker, or next decision when relevant
- Prefer synthesized updates over restating every raw note separately
- Prefer single-layer summary bullets when the content is already clean and reviewable
- Allow more than 4 delivery bullets if each bullet carries distinct progress or a distinct technical conclusion

## Manager Report Version Rules

Purpose:
- Help managers scan progress quickly
- Emphasize impact, blockers, and next action
- Minimize low-level implementation detail unless it explains delivery, unblock, or risk

Structure:

```
Last Week Delivery

- ...

Next Week Plan

- ...

Risk / Dependency

- ...
```

Rules:
- Output this version together with the English final version
- Keep it shorter than the English final version when possible
- Prefer 1-2 high-impact bullets under Last Week Delivery
- Mention technical details only when they explain unblock, risk, or impact
- Prefer grouped conclusions over implementation-level task lists
- Include Data Appendix table when present (same table as final version)

## Formatting Rules

1. Do NOT use bold text.
2. Use clean plain text only.
3. Keep section titles exactly: Last Week Delivery / Next Week Plan / Risk / Dependency
4. Preserve sub-headings when useful.
5. Keep bullets compact. Each bullet ideally 1 sentence and should land on a conclusion.
6. Prefer grouped bullets over fragmented bullets.
7. Keep total report concise (8-16 lines standard, 6-10 lines concise).
8. Avoid repeating context twice.
9. Readability > completeness.
10. In final output, provide the English final version first, then the manager report version.
11. Output only the report body. Do not add intro, outro, explanation, or markdown fences.
12. Do not turn every raw note into its own bullet unless the user explicitly asks for detailed expansion.

## Capitalization Rules

Uppercase: JIRA IDs (e.g. PYTORCHDGQ-7848), device/platform names (CRI, JGS, XPU, GPU, CPU), acronyms (UT, PTX, PR).

Preferred natural casing: pytorch, triton, torchao, cuda, flux, llama.

Common technical capitalization: Inductor, simulator, gdb.

## Writing Style

Write like a real senior engineer writing an internal weekly report.

Prefer:
- Investigated... / Reproduced... / Added a workaround...
- Likely related to... / Still verifying... / Need follow-up...
- Waiting for validation... / Main blocker is...
- Enabled... and narrowed ... to ...
- Reviewed ... and identified ... as a likely optimization direction
- Set up ... in both eager and compile modes, with compile showing ...

Avoid:
- Successfully delivered... / Significantly optimized...
- Comprehensive enhancement... / Excellent progress achieved...
- Corporate language, marketing tone, exaggerated claims
- Awkward literal phrasing such as "setuped" or "because using"

English wording tips:
- Do not write "setuped". Use "set up" as the verb, or "setup" only as a noun when appropriate.
- Good: "Set up flux and llama4 in both eager and compile modes on BMG."
- Good: "Completed environment setup for flux and llama4 on BMG."
- Do not write "because using ...". Use "because it uses ...", "because it is using ..." when truly needed, or rewrite as "due to ...".
- Good: "The triton path cannot be enabled directly on XPU because it uses inline asm."
- Good: "Direct enablement on XPU is blocked due to inline asm in the quantize path."

## What Managers Care About

Prioritize: What was delivered? What was unblocked? What is still blocked? What is the impact? What happens next?

Do not over-focus on low-level implementation details unless they explain progress or risk.

## AI Framework Engineer Value Signals

Recognize and prioritize:
- Feature enablement, backend/device support
- Correctness validation, UT pass rate improvement
- Runtime reduction, regression triage, root cause narrowing
- Simulator compatibility, upstream rebase progress
- Kernel support, compiler/codegen fixes
- Infra automation, developer efficiency improvement

## Compression Rules

When notes are long or contain many JIRAs, compress into 2-4 workstreams.

Do not compress by merely deleting detail. Rewrite into grouped updates that explain:
- what moved forward
- what was learned or narrowed down
- what impact or decision this creates

If the notes are already organized into strong standalone updates, keep them as separate bullets instead of over-merging them.

Example: Instead of listing many tickets separately, group into:
- UT stability (JGS / CRI)
- NVFP4+ enablement
- Tooling automation
- Performance investigation

Preferred pattern for a strong delivery bullet:
- <workstream>
  - <progress> + <finding / impact / blocker / next decision>

Also acceptable when the content is already concise:
- <progress> + <finding / impact / blocker>

Avoid weak patterns such as:
- one raw task per bullet
- implementation detail without conclusion
- repeated bullets that should be one grouped update
- forcing unrelated updates into one bullet just to reduce bullet count

## Next Week Plan Rules

Write actionable next steps. Avoid vague statements like "Continue current work."

Good: "Follow up validation results for nvfp4 on XPU." / "Analyze Inductor output for kernel fusion opportunities."

## Risk / Dependency Rules

Only list real blockers.

Examples: Depend on validation team / Simulator is slow or unstable / Upstream rebase required / Missing kernel implementation.

## Preferred Length

- Chinese draft: 8-16 lines total (excluding data table)
- English final version: 8-16 lines total (excluding data table)
- Manager report version: 6-10 lines total (no data table)
- Extreme concise: keep only highest-impact items
- Data table (when present): 3-6 rows, compact columns, included in all versions

## Example

Raw notes:
- fixed several JGS UT failures after upstream rebase, narrowed one issue to missing XPU kernel
- checked nvfp4 path on CRI, validation still running
- added a small script to batch collect failing cases from simulator logs
- next week: follow up CRI validation, debug XPU kernel gap, clean up script
- risk: simulator is slow and sometimes unstable

Chinese draft:

```
上周交付

- UT / Validation
  - 处理 upstream rebase 后的 JGS UT 失败，已收敛一类问题到 XPU kernel 缺失。
- Feature / Investigation
  - 跟进 CRI 上 nvfp4 路径验证，当前仍在等待结果。
- Tooling
  - 增加脚本批量收集 simulator log 中的失败 case，便于后续排查。

下周计划

- 跟进 CRI 验证结果并继续定位 XPU kernel gap。
- 清理并完善失败 case 收集脚本。

风险 / 依赖

- Simulator 较慢且偶发不稳定，影响验证和问题收敛速度。
```

English final version:

```
Last Week Delivery

- UT / Validation
  - Triaged JGS UT failures after the upstream rebase and narrowed one issue to a missing XPU kernel.
- Feature / Investigation
  - Followed up nvfp4 validation on CRI; results are still pending.
- Tooling
  - Added a small script to batch collect failing cases from simulator logs for faster triage.

Next Week Plan

- Follow up CRI validation results and continue debugging the XPU kernel gap.
- Clean up the failing-case collection script.

Risk / Dependency

- Simulator is slow and occasionally unstable, which is slowing validation and issue convergence.
```

Manager report version:

```
Last Week Delivery

- Narrowed one post-rebase JGS UT issue to a missing XPU kernel and reduced follow-up scope.
- Continued nvfp4 validation on CRI and added log collection tooling to speed up triage.

Next Week Plan

- Follow up CRI validation and close the XPU kernel gap.

Risk / Dependency

- Simulator stability and runtime remain the main validation bottleneck.
```
