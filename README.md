# codebase-intel

**Before you merge, find what your change forgot.** codebase-intel reads a git repo's history
and structure and tells you, for the change you're making:

- which other files probably need to change too, with a calibrated probability and the evidence;
- who calls the functions you modified;
- which tests to run, most relevant first.

It runs locally, needs no API keys or models, and works as a CLI (for you, a pre-commit hook or
CI) and as an MCP server (for Claude Code, Cursor and other coding agents).

```
$ codebase-intel check
Change vs HEAD, uncommitted changes: 1 file(s)
  src/flask/helpers.py  (get_flashed_messages)

Nothing else is likely to need changing.
Less likely, worth a look: src/flask/app.py (0.18), CHANGES.rst (0.18)

Callers of changed code outside this change:
  get_flashed_messages: src/flask/__init__.py, src/flask/app.py, tests/test_basic.py

Tests to run:
  tests/test_helpers.py
  tests/test_basic.py
  tests/test_testing.py
  ...
  (+34 more linked to this change; --json lists all)
```

## Why this, when agents can read code?

A coding agent can search and read a codebase well. What it can't see from the code is how the
repo *changes*: that `adapters.py` edits come with a `HISTORY.rst` entry, that this module's
tests live in a file named nothing like it, that two config files are always bumped together.
That coupling lives in git history, and mining it takes a pass over thousands of commits, not a
grep. Incomplete changes (the forgotten changelog entry, test, doc page or sibling module) are a
common reason for follow-up fixes, and code review misses them because nothing in the diff points
at them.

codebase-intel precomputes that history, joins it with the import graph and type-resolved call
sites, and scores every candidate file with a model trained on thousands of real changes, so a
warning comes with a probability you can trust (see [How well it works](#how-well-it-works)).

## Install

```bash
pip install git+https://github.com/saarthakg/codebase-intel   # Python 3.10-3.12
```

Indexes are kept in `~/.cache/codebase-intel` (or `$CODEBASE_INTEL_HOME`); nothing is written to
your repo.

## Use

```bash
codebase-intel check                        # uncommitted changes (staged, unstaged, new files)
codebase-intel check --base origin/main     # the whole branch: everything since its merge base
codebase-intel check --staged               # only what's staged, as a pre-commit hook sees it
codebase-intel check --json                 # machine-readable
codebase-intel impact src/pkg/adapters.py   # before editing: what a change here usually touches
codebase-intel impact HTTPAdapter.send      #   ...or a change to one function
```

`check` reports files at probability ≥ 0.2 as likely missing (change it with `--min-confidence`)
and up to five more at ≥ 0.1 as worth a look. The first run in a repo indexes it (well under a
second for most projects, 22 s for Django's 5,700 files); after that the index is rebuilt only
when HEAD moves, so repeated checks while you edit take a fraction of a second.

**As a pre-commit hook** (`.git/hooks/pre-commit`):
```bash
#!/bin/sh
codebase-intel check --staged --fail-above 0.5
```
`--fail-above P` exits 1 when a file at least that likely to be needed is missing from the change.

**In CI**, on a pull request (needs full history: `fetch-depth: 0`):
```yaml
- run: pip install git+https://github.com/saarthakg/codebase-intel
- run: codebase-intel check --base origin/${{ github.base_ref }}
```

**For coding agents (MCP).** From the project you want analyzed:
```bash
claude mcp add codebase-intel -- codebase-intel mcp
```
The agent gets two tools. `check_change` checks its current edits (or a branch, with `base`)
before it finishes, returning likely missing files with their evidence, callers of modified code
and tests to run. `impact` asks about one file or symbol before editing it. Both default to the
directory the client started the server in.

## How it works

1. **Index the commit, not the working tree.** HEAD is extracted with `git archive` into a
   temporary directory and indexed there, so the index matches the committed code exactly and
   uncommitted edits never force a rebuild. Code (Python, TypeScript, JavaScript) is parsed with
   tree-sitter for definitions, references, imports and, in Python, the inferred type of every
   method call's receiver. Every other tracked text file (docs, changelogs, config, lockfiles) is
   indexed by path, since what matters about it is its history.
2. **Mine history per merged change.** Co-change counts come from the last 5,000 changes on the
   main line, one per merged PR (a squash or merge commit), and separately from the last 200.
   Counting each commit inside a PR was much worse on repos with many small commits.
3. **Map the change to symbols.** Uncommitted edits are matched against the index through the
   diff's old side (which is HEAD); a branch's commits through the new side. Only lines actually
   removed or added count, not the diff's context lines.
4. **Gather evidence** for every candidate file: how often it changed with each changed file (in
   both directions, overall and recently), whether it imports or is imported by one, whether it
   calls a changed function (method calls resolved by receiver type and inheritance), whether
   it's a test named after a changed file, path similarity, how often it changes at all, and what
   kind of file it is.
5. **Score** with a logistic regression (`codebase_intel/core/scoring.py`), trained on real
   changes from six repos, and list the evidence behind each suggestion strongest first.

## How well it works

The question the tool answers is "given most of a real change, can you name the file it left
out?" `eval/build_replay.py` replays every merged change on a repo's main line with one file
hidden, and asks the tool about the rest, using only history from before that change. It also
replays each change complete, to count false alarms on changes that needed nothing more. The
model was trained on six repos; **each row below is scored by a model trained on the other five**,
so none of these numbers comes from a repo the model has seen.

| Repo (hidden-file replays) | Hidden file first: before → now | In top 5: before → now | False alarms per complete change at p ≥ 0.2: before → now |
|---|---|---|---|
| psf/requests (1,569) | 0.26 → **0.34** | 0.58 → **0.65** | 34.7 → **0.31** |
| pallets/flask (2,226) | 0.14 → **0.28** | 0.36 → **0.51** | 83.3 → **0.31** |
| encode/httpx (1,606) | 0.28 → **0.33** | 0.51 → **0.57** | 51.6 → **0.39** |
| pallets/click (1,213) | 0.28 → **0.33** | 0.55 → **0.56** | 72.6 → **0.26** |
| Textualize/rich (2,498) | 0.21 → **0.28** | 0.39 → **0.53** | 154.1 → **1.32** |
| python-attrs/attrs (1,690) | 0.30 → **0.32** | 0.51 → **0.56** | 42.3 → **0.56** |

"Before" is the previous version's hand-weighted scoring. Its warnings were mostly noise: dozens
per change that needed nothing more. Ranking by co-change history alone sits in between (e.g.
requests 0.30 first, 0.61 in top 5); the model improves on it everywhere.

**The probabilities are calibrated.** Pooled over all six repos, each scored by a model not
trained on it:

| Probability shown | Share that were the missing file |
|---|---|
| 0.10–0.20 | 14% |
| 0.20–0.30 | 22% |
| 0.30–0.50 | 31% |
| ≥ 0.50 | 47% |

At the default threshold (0.2), 31% of flagged files are the missing one and a complete change
gets 0.57 false warnings on average; 23% of missing files are flagged, and 51-65% land in the
ranked top 5. Lower the threshold to catch more (at 0.05: 44% flagged, 16% of flags right, 2.4
false alarms per change). This is a "did you check X?" hint that's right about a third of the
time, not an oracle.

**By kind of missing file** (in top 5 / flagged at p ≥ 0.2): changelog entries 0.89 / 0.58, source
files 0.59 / 0.25, config and other files 0.62 / 0.31, tests 0.50 / 0.16, prose docs 0.34 / 0.05.
Forgotten changelog entries are caught best; which doc page a change needs is mostly not
predictable from history.

**Tests to run.** When a real change edited a test file, that test was in the list 70-99% of the
time (by repo) and among the first ten shown 61-99%. Lists are long for central modules (median
7-55 tests), because every linked test is included; the order is what makes the top useful.

**Callers.** Method callers are found by inferred receiver type. Checked against the calls each
repo's own test suite makes at runtime (`eval/call_tracer.py`), callers of methods are found with
95-97% recall and at least 76-78% precision (a lower bound: a caller the tests don't exercise
still counts against it).

Caveats. A replay hides exactly one file, but real changes can miss none or several. "Missing"
means "changed in the same merged change", and not every such file was strictly necessary. The
import graph and call sites come from today's code, not the code at each replayed change.

## Limitations

- **History is the main signal.** A new repo, or a new file, gets only structural evidence
  (imports, callers, test names), which is much weaker.
- Type-aware callers are Python-only; TypeScript/JavaScript get imports and name-matched usages.
- The model was trained on six Python open-source libraries. Other kinds of codebases (large
  monorepos, apps, other languages) may be less well calibrated. `eval/train_model.py` retrains
  it on your own replays.
- Signature changes aren't treated specially yet: a changed parameter list should make every
  caller more likely to need an update.
- tree-sitter-languages has no wheels for Python 3.13 yet.

## Development

```bash
pip install -e ".[dev]"
pytest -q                                   # 117 tests
```

Evals (`eval/`), all local:

| Script | Checks |
|---|---|
| `run_eval.py` | the static analysis under impact: definitions, references, import-graph edges vs hand-verified labels (`requests_bench.yaml`, `flask_bench.yaml`) |
| `run_usage_eval.py` | method callers vs calls recorded at runtime by `call_tracer.py` |
| `build_replay.py` + `score_replay.py` | the missing-file replay above, for any scorer |
| `train_model.py` | leave-one-repo-out results, and `--write` regenerates `core/model_weights.py` |

CI (`.github/workflows/ci.yml`) runs the tests, the static and usage evals, and the replay of
requests and Flask on every push, and fails if a metric leaves its bounds in `eval/thresholds.yaml`.

### History of the project

codebase-intel started as a broader "codebase intelligence" tool: semantic code search, question
answering over code with an LLM, and go-to-definition, alongside impact analysis. An honest look
at what it added over the coding agents people already use showed that search, Q&A and
definitions were redundant (agents and IDEs do them as well or better), while predicting what a
change leaves out was not. Those features were removed (see the git history), taking the
embedding model, vector index, LLM backends and web API with them; the install went from ~1 GB
to under 200 MB.
