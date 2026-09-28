# Track 6: what human reviewers catch (subagent, 2026-09-27)

## Classic research
- Mäntylä & Lassenius (TSE 2009): "75 percent of defects found during the review do not affect the visible functionality"; industrial 77/23 evolvability/functional.
- Beller et al. MSR 2014: 75:25 maintainability:functional; 7–35% of comments discarded.
- Bacchelli & Bird ICSE 2013 (Microsoft):
  - top motivation is finding defects (44%);
  - but actual comments: code improvements 29% (incl. 55 "removing not necessary or unused code"), defects only 14% (4th of 9);
  - understanding the change is the main challenge.
- Sadowski et al. ICSE-SEIP 2018 (Google):
  - expectations: education, maintaining norms, gatekeeping, accident prevention;
  - median change 24 lines, ~90% touch <10 files, median 1 reviewer, <4h latency.
- Bosu et al. MSR 2015: 64–68% of comments useful.
  - Useful: defects, corner cases/validation, API/design/conventions for newcomers.
  - Nits somewhat useful.
  - Experienced reviewers 65–71% useful vs 32–37% for first-timers; more files → less useful.

## Agent PRs
- METR 2025 (Claude 3.7, 15 PRs):
  - test-passing PRs: coverage gaps 100%, docs 75%, lint 75%, other quality 50%, core 25%;
  - 26–42 min fix time;
  - "doesn't make use of the existing numGraphemeClusters function".
- METR 2026:
  - golden patches merged 68%; grader −24.2pp vs maintainers;
  - categories: code quality (verbose, non-conforming), breaks other code (touches unrelated code), core;
  - among grader-passing rejects, code quality ≥ core;
  - Sonnet 4.5 time horizon 50 min by grader vs ~8 min by maintainers.
- 2601.15195: merge rates Codex 82.59%, Cursor 65.22%, Claude Code 59.04%, Devin 53.76%, Copilot 43.04%.
- 2602.04226: 67.9% of rejected PRs have no explicit feedback; agent-only modes include too large, no added value, context limitation, increased complexity.
- SWE-Gate (2609.04167): of 644 functionally passing repairs, 221 (34%) violated review constraints mined from real review comments.
  - Categories: error semantics 152, schema/typing 143, ordering/argument preservation 86, encoding 74, scope generalization 62, compatibility/deprecation 55, missing-vs-empty 51, perf 41, idempotence 30, lifecycle 19.
- 2607.18057 (ICSME 2026): agents changed tests in only 49.6% of relevant PRs; changed-line coverage Java 61.5% / Python 27.0%; error-handling lines missed 86%/81%.
- 2605.02273: 61.38% of AI PRs get no review activity; 71.58% of review comments on them are by agents; 25.92% of human comments steer agents.
- 2601.13754: ~80% merged without explicit review.

## Examples (verified)
- METR pytest: "This exact code is used elsewhere… should be a reusable function"; "useless AI slop comment"; logic "complicated for no valid reason".
- Sphinx: wrong class/layer modified.
- tldraw (Steve Ruiz 2026-01-17):
  - "formally correct. Tests and checks passed" yet "ignored existing patterns";
  - fix is "use this helper, use our existing UI components";
  - now auto-closes external PRs.
- Excalidraw 2× PRs in Q4 2025.
- Ghostty AI_POLICY: burden of validation on the maintainer; "AI is very good at being overly verbose".
- curl: AI slop ~20% of submissions; bounty ended Jan 2026; each report engages 3–4 people.
- LLVM: "extractive contribution"; worth more than review time.
- Jellyfin: touching unrelated Y and Z → rejected; excessive comments.
- QEMU, Gentoo, NetBSD, Zig ban AI; RPCS3.
- Discourse (Saffron): "deciphering alien intelligence".
- VisiData: fake "How I tested".
- HN: defensive code to a fault; mock everything.
- Linux: Assisted-by tag.

## Review burden
- Faros (vendor): review time +91%, PR size +154%.
- Habituation (2606.22721): approvals 30.1→36.8%, latency 3.5×, inline comments −22%.
- Vibe coders (2602.23905): 4.52× review comments, 5.16× longer to resolve, 31% lower acceptance.
- CodeRabbit: 1.7× issues.

## Automatable before the agent finishes
- Duplication/reuse (clone detection vs repo).
- Scope creep (files/symbols outside intent, base-class touches, size budget).
- Tests present, diff coverage (error branches), mutation of new tests, fails-on-revert.
- The repo's own lint/format/type/docs/changelog.
- Comment and defensive-code hygiene vs repo baseline.
- Pattern conformance (mined past review comments → constraints, SWE-Gate).
- PR hygiene (template, claims vs diff).

## Needs a human
Whether the change is wanted; design approach; underspecified semantics; trust; licensing; knowledge transfer.
