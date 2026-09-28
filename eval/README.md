# notyet evaluation scripts

Results and how to read them: [docs/EVAL.md](../docs/EVAL.md). Raw outputs are in `results/<date>/`.

## Benchmark clones

The scripts edit repos in place, so they run on throwaway clones, never on a checkout you care
about. Each clone gets its own `.venv` with the project (editable) and its test dependencies,
on Python 3.12. The pins are the commits the 2026-09-28 results used:

| Repo | Upstream | Commit | Extra test dependencies |
|---|---|---|---|
| flask | github.com/pallets/flask | d73fa1cd | `asgiref` |
| click | github.com/pallets/click | 06b2a678 | — |
| attrs | github.com/python-attrs/attrs | 8f767776 | `hypothesis pympler cloudpickle "pytest>9"` |
| requests | github.com/psf/requests | 5460f467 | `-r requirements-dev.txt` |
| httpx (held out) | github.com/encode/httpx | b5addb64 | `-r requirements.txt` plus `-e ".[brotli,cli,http2,socks,zstd]"` |
| rich (held out) | github.com/Textualize/rich | 9d8f9a37 | `attrs` |

```sh
W=~/notyet-bench        # anywhere outside this repo
mkdir -p $W && cd $W
git clone https://github.com/pallets/click && cd click && git checkout -q 06b2a678
uv venv -q -p 3.12 .venv && uv pip install -q -p .venv/bin/python -e . pytest
```

Full history is needed: the replay and sessions scripts walk it. Some tests fail in any sandbox
without network access (requests needs the network, and 205 of its tests fail offline). notyet
reports those as failing before the session, which is the realistic case.

## Scripts

| Script | What it measures | Uses an agent? |
|---|---|---|
| `notyet_latency.py OUT REPO…` | check latency; false blocks on harmless edits; seeded faults blocked | no |
| `notyet_replay.py OUT REPO…` | how often each rule fires on real merged commits (parent → commit) | no |
| `notyet_tamper.py OUT REPO…` | 8 ways of hiding a failing test on a real seeded bug; caught on the tampered test? | no |
| `notyet_sessions.py OUT CLONE --commits A,B --mode report\|enforce` | headless Claude Code on real commits, graded by the maintainers' held-back tests and a full-suite regression check | yes: `claude -p` on the subscription |
| `sessions_aggregate.py RESULTS… [--labels L]` | the SESSIONS_NEXT.md tables from `notyet_sessions.py` rows | no |

Run them from this repo with `PYTHONPATH=. .venv/bin/python eval/<script>.py …`.

About `notyet_sessions.py`:

- It reinstalls the uv-tool copy of notyet, which the hooks run.
- It gives each task a fresh clone whose history ends at the parent commit.
- It strips `PYTHONPATH` from everything it starts.
- `--no-agent` grades the untouched parent. It's a free check of the harness.
- Tasks come from `sessions_tasks.json` (`--tasks eval/sessions_tasks.json [--only ID,…]`). A task is a
  chain of steps, either commits or pressure prompts, run in one session with `claude -p --resume`. After
  each step, the tree is snapshotted, graded on every answer key so far, and restored before the next step.
- `--repeat N` gives each repeat its own workdir and row. Rows already in OUT are skipped, so a batch can
  resume in a later usage window.
- `--dry-run` measures the answer keys along the chain and prints the prompts. It's free.
- `claude -p` saves each session's transcript under `~/.claude/projects/`. That's useful for
  seeing how the agent reacted to a block.

Notes on the 2026-09-28 raw results:

- `tamper.json.gz` covers flask, click, attrs and httpx. rich's entry there is empty because of
  the rootdir bug. `tamper_rich.json.gz` is the rerun after the fix.
- `replay_vacuous.json` is the replay with the vacuous-test rule and the integrity fixes (EVAL.md §5);
  `replay.json` is from before them. `tamper_after_fixes.json.gz` is the tamper suite re-run after them.
- If the clones live under an iCloud-synced folder (like `~/Documents`), iCloud hides `.venv/*.pth` and editable
  installs stop importing. The gate doesn't depend on them, but scripts that call `.venv/bin/python`
  directly (sessions grading) do. Keep the clones outside synced folders.
- `sessions_*.json` are the 6 sessions in docs/EVAL.md §4. The 2 smoke sessions before those ran
  with a PYTHONPATH leak and aren't counted.
