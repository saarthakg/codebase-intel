# Track 5: prior art, hook practice, LSP coverage (verified by subagent, 2026-09-27)

## Claude Code hook contracts
- Stop input: stop_hook_active, last_assistant_message, background_tasks, session_crons.
- Block cap: 8 consecutive continuations, then overridden (CLAUDE_CODE_STOP_HOOK_BLOCK_CAP).
- stop_hook_active only suppresses the immediate re-run; the next user message resets it, so a gate can nag every turn (iris #122).
- decision:block + reason (required); exit 2 = stderr as reason; additionalContext = non-error feedback.
- PostToolUse can't block (the tool already ran).
- Strings capped at 10,000 chars; beyond that, a file path plus a 2k preview, and Claude isn't asked to read it. Default timeout 600s.
- ConfigChange hook can block settings edits. Settings edits are live (file watcher).
- type:"agent" hooks run a verifier subagent.
- /goal is a built-in prompt Stop hook.
- Cursor stop: followup_message, loop_limit 5.

## Competitors
- repowise (7,067★, AGPL-3.0, created Mar 2026, active):
  - tree-sitter across 26 languages + git co-change/hotspots + 51 health detectors;
  - MCP get_risk / get_change_risk; PR mode reports may_break, missing_cochanges, missing_tests; 0–10 Kamei risk score;
  - hooks NEVER block ("never crashes or blocks your agent"); no Stop gate, no fix-or-justify;
  - self-reported ~15% false-positive call edges; Django index 366.8s.
- impact-rs (5★): tree-sitter with Exact/Probable/Heuristic edge tiers; PreToolUse reminder; advisory; "thin blast radius is not proof".
- code-impact-mcp (1★): TS import graph, gate_check with risk = affected/total; pre-commit.
- LAIN (9★): tree-sitter + LSP + co-change + traces; "insufficient_evidence"; no gate.
- CodeScene MCP (65★): maintainability only, not callers; exposes rule-setting tools an agent could use to relax its own gate.
- Pharaoh (SaaS, Neo4j blast radius); Riftmap (infra, cross-repo); Amp checks (LLM); BlockWatch (29★, annotation-based co-change lint).
- Also code-graph-mcp (81★), ckb (109★).

## Hook practice
- Stop hooks run tests; PostToolUse runs tsc/eslint (bartolli, 179★); TDD Guard (2.4k★); protect-tests (blocks deleting/skipping tests).
- bartolli: "Only shows errors for the edited file (not dependencies)", so per-edit hooks drop broken callers.
- Problems:
  - infinite loops (#55754 burned ~50 min; #94041 /goal fired 21×; many hooks omit stop_hook_active);
  - stale scope (iris #122 blocked on a change from 2 days ago);
  - slowness (tsc 2–4s per edit → accumulate paths, run once at Stop: ~100 runs → 1–3);
  - gaming ("a third of approved sessions had a real finding waved through because magic words were present");
  - exit 1 is non-blocking (checks silently ignored);
  - agents can edit hook scripts (#11226); disableAllHooks bypass (#26637).
- ImpossibleBench: Claude cheats >79% via test modification; feedback loops raised cheating 33%→38%; an abort option cut GPT-5 cheating 54%→9% but helped Opus 4.1 much less.

## LSP / type checkers
- Claude Code official LSP plugins (13 languages): diagnostics after edits + LSP tool (find references, call hierarchy); not in cloud; UNVERIFIED whether caller-file diagnostics surface for unopened files.
- Serena (29.8k★): get_diagnostics_for_symbol(check_symbol_references=True) = broken-caller check for typed code; safe_delete_symbol; agent-invoked, not enforced.
- Typed-code caller breakage is increasingly covered by LSP/Serena/full type check.

## Coverage map
- Covered: typed arity/type (full tsc/pyright in a Stop hook), failing tests (Stop test runners).
- Partial: dynamic callers (repowise etc., with confidence), test selection, co-change (repowise, advisory).
- Not covered:
  - tests edited to pass (beyond protect-tests);
  - co-change enforced in-session;
  - unannotated leftover sites (strings/config/docs/SQL);
  - cross-repo code consumers in-session;
  - same-signature behavior changes.

## Google (Sadowski et al., CACM 2018)
- Blocking (compiler) checks must have ~0 effective false positives; review checks up to 10%.
- Tricorder disables an analyzer if "Not useful" / "Please fix" > 10%.
- FindBugs: 16% of warnings fixed, integration dropped.
- 74% of compile-time issues rated real vs 21% in checked-in code (timing matters).
- 57% happy with suggested fixes.

## Design lessons
1. Two tiers: block only near-zero-FP findings (exact callers with mismatch, tests that pass at baseline and fail now); the rest is advice; demote rules above 10% dismissal.
2. Baseline: report only what the change caused.
3. Loop protocol: finding fingerprints, never re-block an unchanged set, allow while background_tasks exist, stay under the cap.
4. Verifiable structured justifications (id + category + evidence), not magic words; a flag-for-human escape.
5. Anti-tamper: treat test edits as findings; PreToolUse deny on gate files + ConfigChange hook.
6. Latency: accumulate in PostToolUse, heavy work at Stop.
7. Whole project, not just the edited file.
8. Output ≤10k chars (~2k ideal); exit 2 / JSON block, never exit 1; silent when green.
9. Differentiation:
   - dynamic-language callers with confidence;
   - leftover sites in strings/config/docs;
   - history co-change enforcement;
   - baseline-relative test failures;
   - all combined in an in-session resolve-or-justify gate.
