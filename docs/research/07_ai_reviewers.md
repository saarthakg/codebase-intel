# Track 7: AI code reviewers vs in-loop gate (subagent, 2026-09-27)

## Market changes
- Sweep pivoted to a JetBrains assistant.
- Ellipsis now does managed agents.
- Graphite joined Cursor (Dec 2025).

## Tools
- CodeRabbit:
  - PR/IDE/CLI (--agent JSON)/Claude Code plugin;
  - learnings from chat;
  - 50+ linters, no tests;
  - can block via pre-merge checks in error mode;
  - $24–72/user.
- Greptile: static graph (no git history); reactions-based learning; CLI committed-only; "Fix with your Agent"; $30/seat.
- Graphite Agent: PR; learns from feedback.
- Qodo:
  - Context Engine incl. PR history and "historical implementation patterns";
  - "blast radius" labels (method unspecified);
  - Rule Miner mines accepted past review comments into rules;
  - Agentic Toolbox (Claude Code/Codex plugins).
- Copilot review: full project context; thumbs only; doesn't count toward approvals; CodeQL integration "soon".
- Cursor Bugbot:
  - learned rules from downvotes/replies/human comments;
  - Autofix cloud agents test in VMs;
  - can block (fail-on-unresolved-issues).
- Amp: checks in .agents/checks, fed back to the agent.
- Bito: knowledge graph; rule created after 3 negatives.
- Claude Code Review (managed):
  - multi-agent + verification step;
  - NEVER blocks (neutral conclusion);
  - ~$15–25/review, ~20 min.
- Claude /code-review, /security-review (doesn't execute code).
- Codex review: "runs your code and tests"; flags only P0/P1.
- Saguaro (mesa-dot-dev, Apache-2.0):
  - Claude Code STOP HOOK;
  - tree-sitter/SWC import-graph blast radius;
  - rules; "Violations block Claude and ask it to fix before completing".
  - A DIRECT COMPETITOR for the gate concept.
- CodeScene MCP: pre-commit safeguard.

## Evidence
- Beko (ICSE 2025, Qodo PR-Agent): 73.8% of comments resolved, but PR closure time 5h52→8h20.
- Mozilla/Ubisoft RevMate: 8.1% / 7.2% accepted; functional ~5%, refactoring ~18%.
- Atlassian RovoDev (1,900 repos): 38.7% of comments led to changes; cycle time −30.8%.
- Google: 40–50% of previewed edits applied.
- Meta MetaMateCR: 19.7% applied; showing patches to reviewers made them 5% slower, authors only = no slowdown.
- Microsoft: 600K PRs/mo, 10–20% faster completion.
- MSR 2026 (2604.03196): bot-only-reviewed agent PRs merged 45.2% vs human-reviewed 68.37%; 12/13 bots <60% signal; Copilot 19.79%.
- Martian Code Review Bench: no tool >63% of known issues; volatile leaderboard.
- Vendor claims:
  - Greptile benchmark: Greptile 82%, Bugbot 58%, Copilot 54%, CodeRabbit 44%;
  - Bugbot 78% resolution rate;
  - CodeRabbit: AI PRs 1.7× issues.

## Complaints
- Noise and nitpicks; false positives → abandoned; chatty and drowns human signal.
- Misses the important things; no learning.
- Runs too late ("The agent that wrote the code never sees the critique").
- Hallucinated issues (GitHub docs admit it).

## Gaps (what nobody documents)
1. Co-change/fallout mining from git history (Qodo mines review comments; CodeScene coupling not used in gates).
2. Impact-targeted test selection + execution as gate evidence.
3. Deterministic, reproducible verdicts (so blocking is tolerable).
4. A recorded fix-or-justify protocol.
5. Running while the agent still has context (Saguaro exists).

## Where we'd be worse / overlap
- Semantic bug finding belongs to LLM reviewers.
- "Impact + in-loop gate" alone isn't differentiated (Saguaro, Qodo, CodeScene).
- Convention learning is mature elsewhere.
- History mining is noisy in young repos / monorepo mega-commits / squash merges.
- Latency.

## Positioning
- Complement: "the evidence layer AI reviewers lack".
- Ingest AI reviewer findings as fix-or-justify inputs and down-rank ones the tests disprove.
- Export evidence to PR reviewers (REVIEW.md, knowledge bases).
- Author/agent-facing output; measure resolution rate + human override rate.
- Real competitive set: Saguaro, Qodo toolbox, CodeScene MCP, repowise.
- Differentiate on history + execution + determinism + speed.
