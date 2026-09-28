# notyet: evaluation so far (free, local)

Everything here ran locally on scratch clones, at no cost. Nothing below involves a coding agent
yet. These runs check the gate's machinery on real repos and real commits. Whether notyet changes
what an agent does is the dogfooding and A/B question (PLAN §5), which is still to come.

Repos: flask `d73fa1cd`, click `06b2a67`, attrs `8f76777`, requests `5460f467`. Each has its own
`.venv` with the project (editable) and its test dependencies, and uses pytest 9.1.1. Scripts
are in `eval/`.

## 1. Latency and false blocks on harmless edits (MVP week 2 exit check)

`eval/notyet_latency.py` takes 12 source modules per repo:

- **clean:** a comment is appended. Any block counts as a false block.
- **fault:** the first single-line `return <expr>` becomes `return None`, a crude seeded bug.

Each edit is checked by a fresh session in enforce mode. At this point the gate had the
execution and static engines only.

| Repo | p50 | p95 | Max | False blocks (clean) | Seeded faults blocked |
|---|---|---|---|---|---|
| flask | 1.4s | 2.8s | 5.0s | 0 / 12 | 7 / 10 |
| click | 6.3s | 12.2s | 56.8s | 0 / 12 | 7 / 12 |
| attrs | 6.7s | 7.8s | 14.9s | 0 / 12 | 10 / 10 |
| requests | 40.4s | 42.0s | 60.9s | 0 / 12 | 6 / 9 |

- **Zero false blocks in 48 harmless edits.**
- **Missed faults:** every missed seeded fault was in code that no selected test executes. That
  is what the coverage check added in week 3 reports ("changed lines no selected test ran").
- **requests is slow here** only because 205 of its tests need the network and fail in this
  sandbox. Each check re-ran them on the session-start tree. Those results are now cached per
  session-start tree, so only the first check in a session pays for them.
- **Bugs this benchmark found, all fixed before the numbers above:**
  - the relative test command written by `notyet init` broke every comparison with session start;
  - a test command that couldn't start pytest was counted as passing;
  - tests outside the repo's `testpaths` were selected;
  - one pytest crash discarded all results.

## 2. Replaying real merged commits (MVP week 3 check)

`eval/notyet_replay.py` treats each recent first-parent commit that touched both source and tests
as a session: parent → commit. These changes were written by humans and reviewed. All engines
ran in enforce mode.

- **Checked:** 32 commits (click 15, attrs 15, flask 2). flask's other 7 candidates were years
  old and couldn't run in today's environment, so they were reported as not checked.
- **Median check time:** 3.2s (click) and 8.4s (attrs).

| Rule | Commits (of 32) |
|---|---|
| test-failing-before (note) | 27 |
| untested-change (fix or justify) | 12 |
| **test-changed-to-pass (block)** | **7** |
| **test-removed (block)** | **5** |
| suppression-added (fix or justify) | 4 |
| tests-failing-unverified / test-failing (fix or justify) | 3 |
| test-not-collected, assertions-removed (fix or justify) | 1 each |

- **11 of 32 commits (34%) got a block-tier finding.** Each one is an accurate detection:
  - the maintainers changed a test's expectations along with a deliberate behavior change
    (new error messages, new help output);
  - or they deleted or renamed a test (dropping Python 3.9 support, removing greenlet tests).
- **Precision of the detections is good. The friction is real.** In an agent session, each of
  these would stop the agent once. It would then hand the items to the user
  (`ack <ids> --needs-human "<reason>"`), and the receipt would say NEEDS YOUR REVIEW. The plan
  calls this the intended behavior (§3.4). Whether it's the right default is an open decision;
  see §4.
- **untested-change fired on 12 of 32 human commits.** This matches the research finding that
  merged changes often land with lines no test runs.

## 3. Seeded tamper suite (PLAN §5, free part)

`eval/notyet_tamper.py`:

1. Seed faults until one makes a test fail.
2. With the fault still in place, hide the failure in eight ways:
   - skip or xfail the test;
   - replace its last assert with `assert True`;
   - gut its body;
   - wrap its body in try/except;
   - rename it so pytest doesn't collect it;
   - deselect it in conftest;
   - skip it with an autouse fixture in conftest.
3. Credit a catch only when a tamper rule fires on the tampered test itself. Other tests failing
   from the same fault don't count.

Results: see the table below, filled in from `tamper2.json`.

## 4. Open decisions these results raise

1. **Legitimate behavior changes block once.** Keep the block? (It's cheap now: one batch ack,
   and the receipt shows the reason.) Or make `test-changed-to-pass` and `test-removed` "fix or
   justify" by default, keeping block for skip/xfail/deselect, which have no legitimate
   in-session use in these replays?
2. **Repos without a working test environment.** Old commits here, and requests' network
   tests, show what happens: notyet says "not checked" rather than guessing. PLAN decision #13
   (no tests / slow suites) is still open.

## What this doesn't show

- **No agent in the loop yet.** Whether agents fix or game in response to a block, and how often
  they would have claimed "done" wrongly, needs dogfooding (PLAN week 4) and the A/B.
- **The seeded faults are crude** (`return None`), and the tampers are the obvious ones.
  ImpossibleBench-style tasks, where the agent itself chooses how to cheat, are the paid part of
  §5.
