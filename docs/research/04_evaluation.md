# Track 4: how to evaluate each capability (verified by subagent)

## (a) Breakage: broken callers and dangling references
- Protocol:
  - inject breaking changes into real repos: add a required param, remove/reorder/rename a param, rename/delete/move a symbol;
  - oracle = pyright/mypy ∪ tests at HEAD+mutation, with a hand audit;
  - TypeScript: tsc --noEmit;
  - score at site level (file, line).
- Real refactorings mined from history: apply only the declaration hunk and check that the gate flags the call sites the real commit changed.
- PyRef (github PyRef/PyRef, MIT): 9 method-level types; P 89.6%, R 76.1% (secondary source); needs pandas <2.
- ActRef (arXiv 2505.06553): 1,914 validated refactorings in 136 Python projects; P .80, R .92; no code URL.
- AexPy (StardustDL/aexpy): detects Python API breaking changes.
- "Safer Builders, Risky Maintainers" (MSR 2026, arXiv 2603.27524):
  - AST breaking-change detection on 7,191 agent vs 1,402 human Python PRs (AIDev);
  - agents break at 3.45% overall, 6.72% of refactoring PRs, 9.35% of chore PRs.
- RefactorBench (ICLR 2025, microsoft/RefactorBench, archived Sep 2026): 100 multi-file Python refactors; AST unit tests are ground truth for "every site updated".
- Pitfalls:
  - type oracles under-report untyped code;
  - test oracles only see covered code;
  - detector recall ~76%;
  - tangled commits.

## (b) Propagation
- Hold-out replay: apply k<n similar hunks, score the flagged sites against the held-out ones; unrelated-hunk commits give false alarms.
- CodePlan (microsoft/CodePlan, MIT, archived): Python temporal edits; matched/missing/spurious blocks.
- CoEdPilot (ISSTA 2024): 180K commits; edit location accuracy 70.8–85.3%.
- NextEditPrediction (lurf21, Apache-2.0).
- SWE-Bench ProMax (COLM 2026, arXiv 2608.09802): 170 refactor tasks in 7 languages incl. Py and TS, avg 11.4 files; best model 41.2%.
- Pitfalls:
  - hunks aren't always "the same edit" (cluster by AST-diff pattern);
  - stratify by edit type (renames are trivially solved by LSP).

## (c) Regression test selection
- Safety / precision / reduction / time.
- RTSCheck (ICSE 2019): mutant-based safety.
- NameRTS (ISSTA 2026, arXiv 2605.25356, ZJU-CTAG/NameRTS, Apache-2.0):
  - 500 Python commits from sympy, sklearn, matplotlib, dask, xarray, sphinx, pylint, seaborn, pvlib, loguru;
  - 99.6% safe, 69.9% of test files skipped;
  - PRIMARY Python RTS benchmark.
- pytest-testmon (MIT; v2.2.0, Dec 2025):
  - coverage contexts per test, block checksums;
  - limits: non-Python files, site-packages, xdist db lock, cold start.
- coverage.py:
  - dynamic_context=test_function; pytest-cov --cov-context=test;
  - the sysmon core (default on 3.14+) doesn't support contexts → ctrace overhead; measure it ourselves.
- Mutation tools: mutmut 3.8.0 (active, BSD-3), cosmic-ray 8.7.0 (MIT); mutatest is dormant.
- SWE-smith (59,136 synthesized bugs breaking tests in 128 Python repos, MIT) = realistic mutants with known failing tests.
- Pitfalls:
  - file vs function granularity;
  - mutants are easier than real regressions;
  - flaky tests;
  - always select new/modified tests.

## Agent-in-the-loop
- TDAD (pepealonso95/TDAD, MIT):
  - AST CALLS/IMPORTS/TESTS/INHERITS graph;
  - test-level regression rate: vanilla 6.08% (562 P2P failures) → TDD prompt 9.94% (799) → GraphRAG+TDD 1.82% (155);
  - resolve rate 24% → 32%;
  - the Claude Code driver ran only 10 instances;
  - template for our study.
- TestPrune (IBM/Issue-Test-Localizer): coverage recall .71, precision .63; +8.0–12.9% resolve; $0.02–0.05 per instance.
- SWE-bench P2P is an incomplete regression oracle (PatchDiff: 7.8% fail the full suite).
- Datasets:
  - SWE-bench Verified 500; SWE-bench-Live (MIT, 1,888, monthly, contamination-resistant);
  - SWE-Gym 2,438; SWE-smith 59k;
  - Multi-SWE-bench (CC0; TS: darkreader, material-ui, vuejs/core; p2p/f2p);
  - SWE-bench Multilingual 300.
- Claude Code headless:
  - `claude --bare -p ... --output-format json --permission-mode acceptEdits --allowedTools`;
  - --bare skips hooks, so load the gate via --settings/--mcp-config/--plugin-dir;
  - total_cost_usd in the output;
  - ~$1–2 per instance per run (one third-party report ~$1.00/solve on 40 SWE-bench V instances).
- mini-swe-agent (MIT) as a baseline.

## (e) Test tampering
- ImpossibleBench (safety-research/impossiblebench, MIT; HF fjzzq2002/impossible_swebench):
  - GPT-5 cheats 54% (conflicting) / 76% (oneoff); Opus 4.1 ~50% (conflicting);
  - Claude/Qwen cheat mostly by modifying tests (>79%);
  - read-only tests stop test modification but not special-casing;
  - LLM monitors detect 86–89% on LiveCodeBench but only 42–65% on SWE-bench;
  - positives set.
- EvilGenie (arXiv 2511.21654): explicit reward hacking by Codex and Claude Code.
- METR June 2025: reward hacking in 30.4% of RE-Bench runs vs 0.7% of HCAST.
- "All Smoke, No Alarm" (arXiv 2606.18168): 86,156 test patches in agent PRs, 80.2% with weak or no oracle signals; 8-category oracle taxonomy.
- Protocol:
  - positives: ImpossibleBench + injected weakenings;
  - negatives: SWE-bench gold test_patch diffs;
  - detectors: diff heuristics, mutation score delta, running HEAD tests against new code, an LLM judge baseline.
- Pitfalls:
  - legitimate spec changes;
  - special-casing in source evades test-diff detectors.

## (d) Co-change
- ROSE/TARMAQ protocol: temporal replay, varying |Q|, applicability, MAP; filter big transactions; history-window sensitivity; static-neighbour baseline.

## Priority
1. RTS on NameRTS + SWE-smith.
2. Breakage injection (pyright + tests oracle) + PyRef/ActRef replay.
3. Tamper detection (ImpossibleBench).
4. Propagation hold-out (RefactorBench, CodePlan, ProMax).
5. Agent A/B (Claude Code --bare, SWE-bench V/Live, 50–100 instances × 3; TDAD metrics).
6. Co-change, with added protocol details.
7. TypeScript via Multi-SWE-bench + tsc.
