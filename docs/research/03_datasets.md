# Track 3: datasets of agent PR fallout (verified by subagent)

## Datasets
- AIDev (Hao Li, Haoxiang Zhang, Hassan; arXiv 2507.15003 / MSR'26 2602.09185)
  - hf: hao-li/AIDev, CC-BY-4.0.
  - v4: 2,743,854 PRs; AIDev-pop v4 (>100 stars): 71,677 PRs, 6,673 repos, through Oct 24 2025. Claude Code 1,942 PRs (1,271 merged); Copilot 23,496; Codex 35,521; Devin 6,172; Cursor 4,059; Jules 487.
  - AIDev-pop v3: 33,596 PRs, Claude Code 459 (271 merged).
  - Tables: pr_commit_details (with patch; large patches missing), pr_reviews, pr_review_comments (diff_hunk, path), pr_comments.
  - v3 only: pr_timeline, pr_task_type (LLM conventional-commit label), issue, human_pull_request.
  - No CI status, no post-merge commits.
  - v3 repo languages: TS 650, Py 530, Go 242, C# 220, JS 190.
- Who Finishes the Job? (Takerngsaksiri, Duong, Barnett; arXiv 2609.26847); github wannita901/pr-fix-authorship
  - 6,774 merged agent PRs (Claude Code 130); 5,044 human PRs.
  - Fix link: fix-tagged PR within 30 days, co-edits ≥1 non-boilerplate file of P. Human κ=0.77; LLM judge Direct-fix precision 90%.
  - Verified fix rates: Codex 5.5%, Cursor 5.2%, Devin 3.7%, Copilot 3.5%, Claude Code 3.2%.
  - OR 1.62 (CI 1.10–2.39) vs humans. 30-day incidence 4.5% vs 2.6%; half of it in week 1.
  - 69.6% of fixes by the same agent.
  - REQUIRES file overlap, so it can't see fixes landing only in other files.
- Violent Delights (Xia & Miller; arXiv 2607.09902); github post-merge-reality, Apache-2.0
  - 182 repos, May 2025–May 2026, commit-level with 15 tools.
  - Corrective maintenance +49% on agentic lines; bug-fix termination +51%; 4.0% vs 2.7% of lines hit by a bugfix within 180 days.
  - Semgrep findings 1.14× (1.51× high severity).
  - +10pp no-review rate → ~6% more maintenance.
  - Only counts fixes that edit the agent's own lines.
- MOSAIC-agentic-3m (arXiv 2604.00917): 111,969 PRs, Claude Code 19,148; license conflict (GPL-3.0 vs CC BY-NC-SA); 40–75% of PRs in zero-star repos.
- 2509.14745 On the Use of Agentic Coding: 567 Claude Code PRs.
  - 83.77% merged vs humans 91.01%; 54.95% merged as-is.
  - Revision types among 214 revised PRs: bug fix 47.7%, docs 29.0%, refactor 27.1%, style 23.4%, chores 21.0%, tests 16.4%, features 15.4%, build 14.0%, CI 7.0%.
- 2601.15195 Where Do AI Coding Agents Fail?
  - 71.48% merged; each extra failed CI check lowers merge odds ~15%.
  - Rejection codes on 562 PRs: abandoned 38%, duplicate 23%, CI/test failure 17%, incorrect 3%, incomplete 2%, unwanted 4%.
- 2606.13468 Rejected fixes: 46.41% of 3,225 fix PRs rejected. Sample of 306: inactivity 17.3%, agent failure 7.5%, CI failure 6.9%, incorrect 5.6%, breaking change 0.3%.
- PatchDiff (2503.15223), github ZJU-CTAG/PatchDiff
  - SWE-bench only runs the PR's modified test files; the full suite shows 7.8% of plausible patches fail (CodeStory 8.4%, LearnByInteract 7.6%, OpenHands 7.2%).
  - 29.6% of plausible patches diverge in behavior. Causes: divergent implementation 46.8%, supplementary changes 27.3%, absent changes 5.2%. Of the divergent ones, 28.6% are certainly incorrect.
- 2506.08311: overfitting (incl. breaking PASS_TO_PASS) 17–26% for SWE-agent, 4–5% for Agentless. Snippet only, UNVERIFIED in full.
- METR 2026-03-10: 296 PRs; grader is 24.2pp above maintainer merge decisions. Categories include "Breaks other code: touches unrelated code and causes breakages". Per-category % only in a figure.

## Key gap
No study measures whether follow-up fixes land in OTHER files or callers. A cross-file ground truth must be built (a research contribution).

## Proposed eval A
AIDev merged agent PRs → fix/revert PRs within 30 days (no file-overlap requirement) → classify the relation (same lines / same file / caller-dependent / tests-config-docs) → LLM judge calibrated with the Who Finishes labels → did the gate flag it at merge time?

## Obstacles
- Needs the GitHub API or clones; fixes are rare (3–5%).
- Multi-language (TS > Py).
- Squash merges break commit-level attribution.
