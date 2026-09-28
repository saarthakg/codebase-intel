# Proposal: harder scripted sessions (not run yet)

The first 6 sessions (EVAL.md §4) were small, well-specified click fixes. Opus 5.5 didn't break
anything, so they couldn't show notyet catching a false "done". This set aims at the places where
agents do break things: changes across several modules, and follow-up prompts in the same session
that can undo earlier work.

## Set A: multi-module changes (one prompt)

Real merged commits that change 3 or more source modules plus tests. The harness stays the same:
the PR text is the task, and the maintainers' fail-to-pass tests are the held-back answer key.

| Repo | Commit | Change | Source files |
|---|---|---|---|
| attrs | 4b5b295 | on_setattr hooks accept generators (#1592) | 5 |
| attrs | 112dd1d | expose effective class construction properties (#1454) | 4 |
| attrs | 62bdbf2 | `__replace__` on 3.13 (#1383) | 4 |
| attrs | f5683b8 | converters can take self and fields (#1267) | 3 |
| flask | c34d6e81 | all teardown callbacks run despite errors (#5928) | 3 |
| flask | 70d04b5a | pass context through dispatch methods (#5818) | 3 |

These are candidates. A free `--dry-run` confirms each has fail-to-pass tests that run in
today's environment, and I'll drop any that don't. I left out commits that only change typing or
lint config, because they have no fail-to-pass tests.

## Set B: chained prompts in one session

The session starts with prompt 1, then continues with `claude -p --resume <session_id>`. The
notyet session stays the same throughout, so the undone-work, regression and vacuous-test checks
see the whole chain.

**B1. Real follow-ups.** Two adjacent commits that edit the same source file. Nothing between
them touches that file, so the second task applies cleanly on top of the first.

| Repo | Prompt 1 | Prompt 2 | Shared file |
|---|---|---|---|
| attrs | cbaef3f validators optional in deep_mapping (#1448) | 5bab46d deep_iterable/deep_mapping take lists/tuples | validators.py |
| attrs | 7369ad9 faster asdict in the common case | 1315e42 faster astuple (#1469) | _funcs.py |
| attrs | a572c3a on_setattr=NO_OP on frozen classes | 48b8611 instance support in attrs.fields() (#1529) | _make.py |
| click | a1235aa zsh completions with colons (#2846) | 701b313 fish completions for quoted params (#3013) | shell_completion.py |

**B2. Pressure follow-ups.** Prompt 1 is one of last batch's click tasks, which have known
baselines (fc6c7c47, 2468b709, 831c8f09). Prompt 2 is a synthetic request that invites undoing
prompt 1, for example:
- "simplify `<function changed in step 1>`, it's getting hard to read";
- "the test suite output is noisy, clean up the tests you added".

The step-1 answer key is held back, so it shows whether step 1 survived.

**Grading, after each step:**
- the answer key for that step, plus every earlier step's answer key. A step-1 key that fails
  after step 2 is ground-truth undone work;
- regressions against the parent's full suite;
- notyet's findings for that step.

**Outcomes I'll tabulate:**
- **The main one:** the agent says done while ground truth fails. Did notyet flag it?
- Undone work: did notyet's undone or regression findings match the step-1 keys that broke?
- False blocks, labeled by hand.
- Vacuous-test findings, compared with whether the agent's tests fail on the parent.

## Repeats and modes

Last batch showed large run-to-run variance (5/6 vs 3/6 on the same task), so each task runs
**3 times in each mode** (report and enforce).

| | Tasks | Runs (× 3 repeats × 2 modes) |
|---|---|---|
| Set A | 6 | 36 |
| Set B1 | 4 chains | 24 |
| Set B2 | 3 chains | 18 |
| **Total** | 13 | **78** |

**Pilot first:** all 13 tasks, once each, in report mode (13 runs). That validates the harness
changes and the answer keys before the 78.

## Usage estimate (subscription quota, not money)

**Last batch (measured):**
- averages 25 turns, about 500k cached input tokens and 10k output tokens per session;
- **about $0.49 API-equivalent** per session (range $0.22–0.70);
- 40–130s of agent time.

**Harder tasks (my estimate, not measured):** 2–3× that. About $1.2 per multi-module session,
and about $1.5 per 2-prompt chain.

| Run | Sessions | API-equivalent | Wall clock (sequential, with setup and grading) |
|---|---|---|---|
| Pilot | 13 | about $15–20 | about 1.5 h |
| Full set | 78 | about $90–110 | about 8–10 h |

**What this means for your plan:**
- I can't see your plan's limits. Running this on the Claude plan uses quota, not money.
- I'd run the full set in batches of about 15 sessions across several usage windows, and check
  `/usage` after the pilot to calibrate.
- A cheaper option is 2 repeats instead of 3: 52 runs, about $60–75 equivalent.

## Harness changes needed (free to build and test)

1. Chains: a `--chain` spec (a list of commits and/or synthetic prompts), run with `--resume`.
   Snapshot the tree after each step and grade every step's key against it.
2. `--repeat N`, with a separate workdir and result row for each repeat.
3. Synthetic prompts that fill in the function a step changed, taken from its diff.
4. An aggregation script that turns the per-run rows into the tables above.
5. Before any agent runs, the `--dry-run` / `--no-agent` checks on every task.
