# Plan: notyet, a completion gate for AI coding agents

When an agent says it's done, **notyet** checks whether it is. It runs the checks itself,
compares against where the session started, and hands the agent a short list to fix before it
may stop. The human gets a receipt of what was checked, what passed, and what wasn't checked.

Evidence: [RESEARCH.md](RESEARCH.md), and the source notes in [research/](research/). This plan
incorporates an independent review of the earlier draft; §9 lists what changed and why.

## 1. The problem

- **False "done".** This is the most frequently *reported* failure (F1). The agent stops when the
  work looks finished: tests never ran or ran wrong, or the agent verified with a proxy.
- **Test gaming** removes the safety net (F3): tests weakened, skipped, deleted, their denominators
  changed.
- **Undone work** drives the re-prompting loop (F5): earlier fixes and the user's own edits
  reverted within a session.
- **Missing tests.** Test-passing agent PRs had coverage gaps in 100% of METR's reviews, and only
  27% of changed Python lines were covered (arXiv 2607.18057).
- **Incomplete propagation** (F2) is among the costly failures, but type checkers already catch
  the exact cases in typed code.
- **Rules don't fix any of this.** Agents skip checks that are sitting in their context.
  Enforcement has to live in the harness.

## 2. What notyet is, and isn't

**It is** a deterministic, execution-grounded verdict on the agent's change: relative to the
session's starting point, auditable, and portable across agents and CI. That is the differentiator
most likely to survive Claude Code adding its own verification features. **It isn't** an AI
reviewer (no LLM bug-finding), a sandbox (no guarding against destructive commands), or a
replacement for the type checker (it runs the repo's own).

## 3. Core design decisions

1. **Baseline is the session start, not HEAD.**
   - At `SessionStart`, and at `UserPromptSubmit` if none exists yet, notyet snapshots the whole
     working tree, including uncommitted and untracked files.
   - It uses a temporary git index and `write-tree`, so it touches neither the user's index nor
     their files.
   - The change being checked is session baseline → now. It survives the agent committing
     mid-session, and it excludes the user's pre-existing edits.
   - HEAD is only used to decide whether a failure was already there before the session.
   - Limit: edits the user makes *during* the session count as part of the change. The receipt says
     so.
2. **Run only when something changed.** The gate runs at Stop only if the tree differs from the
   last tree it checked, so conversation-only turns cost nothing. Results are cached per tree hash.
3. **The gate runs the checks itself.**
   - It uses its own test runs with a user-configured command, and reads junit/coverage reports
     rather than exit codes.
   - Baseline runs happen in an isolated checkout of the baseline tree, with an import canary that
     proves the baseline code (not the working tree) was imported.
   - Failures are re-run on the new tree before anything blocks, so flaky tests don't cause false
     blocks.
4. **Block only on execution evidence:**
   - tests that passed at baseline and fail now;
   - tests dropped, skipped or no longer collected;
   - a baseline version of a modified test that fails on the new code.

   Everything else is a finding to resolve (fix or justify) or a note.
5. **Resolving findings:**
   - **Block-tier findings** allow only a fix or `needs-human`. A needs-human item is shown at the
     top of the receipt as a failed check, never as a pass.
   - **Resolve-tier findings** also accept `acknowledge(id, reason)`. The reason is recorded and
     shown to the human.
6. **Loop guard.**
   - Findings are fingerprinted, and the gate never re-blocks an identical set.
   - If there's no progress for two iterations, it allows the stop, but the receipt leads with
     **"unresolved: N"**.
   - It stays at or below 6 continuations (Claude Code's cap is 8), and allows the stop while
     `background_tasks` is non-empty.
7. **No automatic rule demotion.** Agent acknowledgments are logged and shown, never used to change
   rules; the agent has a conflict of interest. Rules change only when a human marks a false
   positive from the receipt. Rule-health statistics wait until there are real users.
8. **Structured claims instead of parsing prose.** An MCP tool `declare(intent, tests_run, claims)`
   lets the agent state what it did, and the gate checks it against its own evidence. There's no
   regex over the agent's final message. This comes after the MVP.
9. **Undone work comes from session snapshots, not git blame.** Most undone work happens in-session
   or in uncommitted edits, which blame can't see. Each gate run stores the tree it checked. Lines
   that were added in an earlier checked tree and later removed are findings.
10. **Tamper resistance is realistic, not absolute.**
    - Test edits are findings in their own right.
    - The gate's config and scripts sit outside the repo's working tree, under `.git/notyet/` and
      the user config.
    - `.claude/settings.json` edits are guarded with a `ConfigChange` hook.
    - A PreToolUse deny can't stop `Bash` (`sed`, `python -c`), so the receipt records whether the
      gate's config changed during the session.
11. **Negative space.** The receipt always lists what wasn't checked:
    - no test command configured;
    - the budget was exceeded;
    - coverage wasn't available;
    - baseline isolation failed;
    - non-Python files changed.
12. **History as advice.** Co-change counts from main-line history add up to three advisory lines
    ("usually changes with this: X, in 5 of 7 changes, e.g. a1b2c3"). It is never a block, and it
    costs almost nothing (cached per HEAD). The old calibrated probability model was dropped with
    the rest of codebase_intel: advice lines need counts and an example, not a probability.
13. **Repos with no tests or slow suites** (decision pending, §8). Default: report "not checked",
    don't block; configurable.

## 4. MVP (about 4 weeks, Python, Claude Code)

A new `notyet/` package in this repo. The pieces of the old `codebase_intel` package worth keeping
(Python import resolution, git co-change history) were ported into `notyet/`, and the rest removed.

| Week | Build | Checks at the end |
|---|---|---|
| **1. Skeleton, report-only** | `notyet init` writes the config and detects the test command, confirming it with the user. Session store in `.git/notyet/`. Baseline snapshot at SessionStart/UserPromptSubmit. Stop hook that runs only on changes, with the loop guard. Receipt saved to a file, with a short summary shown to the user. `notyet install claude`, which shows the settings diff and requires confirmation. `notyet status`, `notyet receipt`. | Unit tests on recorded hook payloads. Baseline correctness: the agent commits, the user has pre-existing edits, untracked files. End-to-end in a scratch repo |
| **2. Execution evidence** | Test selection (changed and new tests, tests importing changed modules via the reverse import graph). Budgeted pytest run with junit. Isolated baseline run with the import canary. Classification: regression, pre-existing or flaky. New-tests-collected and total-collected-count checks against the baseline. Diffs of ruff/pyright output against the baseline, when configured | Seeded regressions in real repos are caught. Zero false blocks on clean changes. Latency p50/p95 measured |
| **3. Integrity + coverage** | Test-edit delta: deleted tests, `skip`/`xfail`, assertions removed or loosened. Baseline versions of modified tests run against the new code. Changed lines that no test executed (coverage on the selected run). Undone work from session snapshots. Enforce mode: block per §3.4 | Tamper cases caught on ImpossibleBench-style seeds. No false blocks on legitimate test updates mined from real commits |
| **4. Evidence** | Dogfood 40–50 real Claude Code sessions across 3 or more Python repos, hand-labeling every finding. A free, seeded tamper A/B (§5) | Catches where the agent claimed done; false blocks per session; latency; a written report |

Deferred until after the MVP: `declare()` claims, rule health, other agents, CI Action, reviewer
ingestion, completeness checks, TypeScript.

## 5. Evaluation: free first, and explicit about when money is needed

**Free (the MVP and right after):**
- **Unit and scenario tests** on recorded hook payloads and scratch git repos: baseline semantics,
  the loop guard, snapshot isolation.
- **Seeded-fault suite.** In our six benchmark repos (requests, flask, httpx, click, rich, attrs),
  script realistic agent-style changes: break a function; drop, skip or weaken a test; revert an
  earlier edit; leave new code untested. Also script clean changes, including legitimate test
  updates mined from real commits where source and assertions changed together. Measure catch rate
  and false blocks. Hold out repos for any tuning.
- **Test-selection safety** on NameRTS (500 Python commits with ground truth). Its repos (sympy,
  scikit-learn, matplotlib, …) must not also be used to tune against SWE-smith.
- **Tamper detection:**
  - positives: seeded tampers on real bugs in real repos (`eval/notyet_tamper.py`). ImpossibleBench
    publishes its tasks (HF `fjzzq2002/impossible_swebench`) and harness, but not agent
    transcripts or patches (checked 2026-09-28), so agent-made positives need agent runs: the
    paid A/B below, or dogfooding;
  - negatives: legitimate test edits from real merged commits, replayed with execution
    (`eval/notyet_replay.py`).
- **Dogfooding.** Our own Claude Code sessions, on your subscription: 40–50 sessions, every finding
  hand-labeled.
- **Case study.** About 30 real cross-file follow-up fixes to merged agent PRs (AIDev), hand-labeled
  blind to the gate: would notyet's static and history checks have flagged them? Qualitative, and
  static-only, because execution can't be reproduced on historical repos.

**Costs money (only with your go-ahead, when the free evidence runs out):**
- **A controlled A/B on SWE-bench Live:** headless Claude Code with and without notyet, measuring
  regressions of previously passing tests, full-suite regressions, test edits, resolve rate and
  cost. About $1–2 per instance per run, so 50 instances × 3 runs × 2 arms is about $300–600.
- Smaller pilot: 20 instances × 1 run × 2 arms is about $40–80. That's where I'd propose starting,
  once the MVP is stable.

## 6. After the MVP (in order, each gated on MVP results)

1. **Structured claims:** `declare()` over MCP, checked against evidence. Plus plan-time `brief()`
   and `explain(id)`.
2. **Completeness, narrowed to what nobody else does:**
   - old names and patterns left in strings, config and docs;
   - imports and references that don't resolve;
   - new code with no callers ("never wired").

   Typed-code caller breakage stays with the repo's pyright/mypy, diffed against the baseline.
3. **Human feedback loop:** mark false positives from the receipt, and rule health built on those
   human verdicts.
4. **Portability:** git pre-commit, GitHub Action (receipt as a PR comment), and adapters for Codex,
   Cursor and Copilot.
5. **Reviewer ingestion:** high-severity AI-review findings become resolve items, and test evidence
   down-ranks the ones it disproves.
6. **TypeScript:** tsc and jest/vitest. AIDev has more TypeScript repos than Python ones, so this
   matters for reach.
7. **Restructure and rename:** done. The old package was removed, and the GitHub repo is
   `saarthakg/notyet` (renamed 2026-09-28).

## 7. Architecture (MVP)

```
notyet/
  config.py        .notyet.toml: test command, budget, mode (report|enforce), linters, paths
  store.py         .git/notyet/: sessions, snapshots (tree ids), gate runs, findings, acks, receipts
  snapshot.py      tree snapshot via a temp index; checkout of a tree to an isolated dir
  change.py        baseline→now diff: files, hunks, test-edit delta, removed-since-checked lines
  engines/
    execution.py   selection, run, junit, baseline comparison, collection counts, linter diffs
    integrity.py   test-edit findings, baseline-test-on-new-code, undone work
    coverage.py    changed lines not executed by selected tests
    history.py     advisory co-change (built on notyet/history.py)
  gate.py          findings → decision; fingerprints; loop guard
  receipt.py       receipt (markdown + JSON), short summary for the transcript
  hooks/claude.py  SessionStart / UserPromptSubmit / Stop / ConfigChange entry points; install
  cli.py           init, install, status, receipt, check, ack, hook
```

## 8. Decisions

| Decision | Status |
|---|---|
| Name | **notyet** (PyPI and npm free, Sept 2026) |
| Language | **Python first**; TypeScript after the MVP |
| End-to-end study budget | **None for now.** Free evaluation first; I'll say plainly when paid evidence is needed, starting with the $40–80 pilot |
| Repos without tests or with slow suites | **Open.** Default "not checked, don't block", configurable |
| Installing hooks in your real repos for dogfooding (week 4) | Needs your approval, per repo |

## 9. What changed after the review

| Review finding | Change |
|---|---|
| HEAD is the wrong baseline (Stop fires every turn; commits empty the diff) | Session-start snapshot baseline (§3.1); HEAD only for pre-existing failures |
| Rule demotion can be gamed by the agent | No automatic demotion; only human-marked false positives count (§3.7) |
| Escape hatches are the easy way out | Block-tier allows only a fix or needs-human; unresolved and needs-human lead the receipt as failures (§3.5–6) |
| The strongest tamper signal was missing | Baseline versions of modified tests run on the new code, plus collected-count comparison (§3.4) |
| Baseline test execution is the riskiest bet | Isolated checkout, import canary, flaky re-runs, user-configured command, caching (§3.3) |
| MVP gated behind a restructure and infrastructure | No phase 0; new package, reuse old code; a 4-week MVP (§4) |
| Evaluation bars misleading (additive gold patches, contamination, leakage, cost) | Mined legitimate test edits as negatives, held-out repos, SWE-bench Live, costs stated (§5) |
| AIDev flagship not feasible | Downgraded to a hand-labeled case study of about 30 fixes (§5) |
| Missing: uncovered new code | Coverage of changed lines is in the MVP (§4, week 3) |
| F2 overclaimed; completeness adds little beyond pyright | Moved after the MVP, narrowed to what nobody else does (§6.2); research wording corrected |
| Regex claim parsing is brittle | `declare()` over MCP instead (§3.8) |
| Blame can't see in-session undone work | Session-snapshot based (§3.9) |
| Gate runs on every turn | Runs only when the tree changed (§3.2) |
| Tamper protection overstated | Stated honestly; config changes recorded on the receipt (§3.10) |
| (Kept, against the review's suggestion to cut all "fit" checks) | Co-change history stays in the MVP as advisory lines. It's already built, and it's the repo-history signal nobody else enforces (§3.12) |
