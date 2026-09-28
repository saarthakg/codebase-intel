# notyet

**When your coding agent says it's done, notyet checks whether it is.**

notyet is a completion gate for AI coding agents (Claude Code first). When the agent tries to
stop, notyet runs the tests that exercise the change, and compares them with the tree as it was
when the session started. If something that passed then fails now, a test the agent wrote fails,
or a test disappeared, the agent is told to keep working. It gets a short list of what to fix.
You get a receipt: what was checked, what passed, and what wasn't checked.

> **Status: early MVP, not released.** Python repos with pytest only. It is being built and
> measured in the open; see [docs/PLAN.md](docs/PLAN.md) for the plan and
> [docs/RESEARCH.md](docs/RESEARCH.md) for the evidence behind it.

## Why

The most reported failure of coding agents is the false "done": the agent stops when the work
*looks* finished, but the tests never ran, ran the wrong way, or were quietly changed to pass.
The engineer finds out later and spends the next prompts cleaning up. Instructions don't fix it,
because agents skip checks that are sitting in their context. The check has to run in the
harness, and it has to be based on execution, not on the agent's account of what happened.

## What it checks

Everything is relative to **the session's starting point**: a snapshot of your working tree
(including uncommitted and untracked files) taken when the session starts. Your own
pre-existing edits and already-failing tests are never blamed on the agent.

| Check | Severity |
|---|---|
| A test that passed at session start fails now | **block** |
| A test added this session fails | **block** |
| A test that existed at session start is gone (removed, or its file deleted) | **block** |
| pytest can't run test files it could run at session start (a broken conftest, say) | **block** |
| A new test that pytest never collects | fix or justify |
| New ruff / pyright errors, compared with session start (pyright also checks callers) | fix or justify |
| New `# noqa`, `# type: ignore`, `# pragma: no cover` and similar | fix or justify |
| Flaky failures; failures that already happened at session start | note |

- **block:** only fixing the problem, or handing it to you with a reason, clears it.
- **fix or justify:** the agent may acknowledge the item with a reason, and you see the reason
  on the receipt.
  - An acknowledged untested-lines finding stays acknowledged through small edits to those lines:
    up to 2 new or changed lines, or a tenth of what was acknowledged, counted against the original
    acknowledgment.
  - The receipt says how many lines changed since. More untested code than that raises the finding
    again.
- Anything notyet couldn't check is listed as not checked, never reported as passed.
  - If the tests themselves didn't run, for example because the time budget ran out, the verdict is
    marked "(incomplete)" and the one-line summary says so first.
  - When the budget cuts a run short, results are kept for every test file that finished.

It stays out of the way:
- it only runs when the tree changed;
- it never blocks while background work is running;
- it releases the same findings after two blocks (reporting them as unresolved);
- it reads its own config as it was at session start, so the agent can't loosen it mid-session.

## Try it

```sh
pip install git+https://github.com/saarthakg/notyet
cd your-repo
notyet init                 # writes .notyet.toml; detects your pytest command; asks first
notyet install claude       # adds the hooks to .claude/settings.local.json; shows the diff, asks first
```

Start a Claude Code session and work as usual. The gate begins in `report` mode: it never blocks
and leaves a receipt after each change. Set `mode = "enforce"` in `.notyet.toml` to let it
block.

```sh
notyet status               # the current session and its last check
notyet receipt              # the latest receipt
notyet check                # run the checks now against HEAD, without an agent
```

## Configuration

`.notyet.toml` at the repo root:

```toml
[test]
command = ".venv/bin/python -m pytest"   # notyet adds test selection and junit output
budget_seconds = 60                      # time for tests at each check

[static]                                 # optional
ruff = ".venv/bin/ruff"
pyright = ".venv/bin/pyright"

[gate]
mode = "report"                          # or "enforce"
```

## How it works

- **Snapshots:** the working tree is snapshotted as a git tree through a temporary index. Your
  index and files are never touched.
- **Test selection:** tests the change added or modified, tests importing a changed module
  (through the reverse import graph), tests named after it, and tests under a changed conftest.
  The repo's own `testpaths` are respected.
- **Comparison with session start:** failures are re-run once (a pass means flaky), then on an
  isolated checkout of the session-start tree. A canary test first proves that checkout imports
  its own code; an editable install can otherwise silently test the new code twice.
- **State** lives in `.git/notyet/`: sessions, check results, acknowledgments, receipts.

## Development

```sh
pip install -e ".[dev]"
pytest -q
```

`eval/notyet_latency.py` measures check latency and false blocks on real repos. Point it only
at throwaway clones, because it edits them.
