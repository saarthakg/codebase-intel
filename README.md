# codebase-intel

[![CI](https://github.com/saarthakg/codebase-intel/actions/workflows/ci.yml/badge.svg)](https://github.com/saarthakg/codebase-intel/actions/workflows/ci.yml)

Local code intelligence for Python and TypeScript repos: find code by meaning or by name,
jump to any definition and every usage, see what a change will affect (down to the tests to
run), and ask questions with checked citations. It runs on your machine at no cost, keeps
itself up to date in under a second, and plugs into Claude Code and Cursor as an MCP server.

Every capability is scored against a labeled benchmark on
[`psf/requests`](https://github.com/psf/requests), and CI fails if any score drops.

---

## What it can do

**Find code, however you ask.** Hybrid search combines embeddings (for questions like
"where are redirects followed?"), BM25 keyword matching, and a direct lookup of any
identifier you type (`get_netrc_auth`, `HTTPAdapter.send`). On 25 held-out questions never
used for tuning, the exact function that answers the question is in the top 5 results 92% of
the time, up from 72% with plain embedding search.

**Jump to a definition and every usage.** `Class.method` or bare names, with other matching
definitions listed, source preferred over tests. Usages come from the parse tree, not text
search, so comments and changelogs don't count. On the benchmark: 24/24 definitions and
12/12 reference sets exactly right.

**Know what a change will break, and which tests to run.** Impact analysis ranks every
affected file with a reason ("direct import", "test named for this file", "changed together
in 5 of 12 commits"). It combines five signals: an exact import graph, symbol usages, git
co-change history, test-file naming and code similarity. Give it a `git diff` and it works
out which functions changed and who calls them. Replayed against real `requests` commits,
half the files a commit actually touched are in the top 5.

**Ask questions and get answers you can check.** Answers cite files and line ranges. Every
citation is resolved, and names the answer mentions that don't exist in the repo are
flagged. Answers run free and fully local with Ollama (or Gemini/Claude), stream as they're
written, and are cached, so repeating a question on unchanged code costs nothing.

**Use it where you work.** CLI, a small web UI, a REST API, or an **MCP server** that gives
Claude Code, Cursor or any MCP client these tools directly. The client's own model does the
reasoning, so no extra API cost.

**Cheap to keep current.** Re-indexing re-embeds only code that changed: 0.6 seconds for an
unchanged `requests` versus 12 for a first index. It respects `.gitignore` and skips
lockfiles, minified bundles and oversized files.

Nothing leaves your machine unless you choose a hosted LLM for `/ask`.

---

## Quickstart

### Install

```bash
git clone https://github.com/saarthakg/codebase-intel
cd codebase-intel
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Index a repo

```bash
python scripts/ingest_repo.py --repo /path/to/your/repo --repo-id my-project
# Found 47 files (.gitignore applied)
# Indexed 47 files, 399 chunks, 807 symbols, 2591 references, 107 graph edges.
# Embedded 399 chunks, reused 0 unchanged.
```

The first run downloads the local embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB); after
that everything runs offline. Re-run the same command whenever the code changes: it's
incremental and each run fully replaces the previous index.

### Ask it things

```bash
# Search: natural language or identifiers
python scripts/demo_query.py --repo-id my-project "where is authentication handled?"

# Definition and every usage
python scripts/demo_query.py --repo-id my-project --mode definition --symbol HTTPAdapter.send

# What would a change to this file or symbol affect? Includes tests to run.
python scripts/demo_query.py --repo-id my-project --mode impact --target src/requests/adapters.py

# Impact of your actual uncommitted changes, function by function
git -C /path/to/your/repo diff HEAD | python scripts/demo_query.py --repo-id my-project --mode impact-diff --diff -

# Grounded Q&A, streamed (needs an LLM backend; see below)
python scripts/demo_query.py --repo-id my-project --mode ask --stream "how does redirect handling work?"
```

### Use it from Claude Code, Cursor or any MCP client

`app/mcp_server.py` is an MCP server over stdio. Register it with Claude Code (use your
absolute paths):

```bash
claude mcp add codebase-intel -- /path/to/codebase-intel/.venv/bin/python /path/to/codebase-intel/app/mcp_server.py
```

For Cursor, add to `~/.cursor/mcp.json`:

```json
{"mcpServers": {"codebase-intel": {
  "command": "/path/to/codebase-intel/.venv/bin/python",
  "args": ["/path/to/codebase-intel/app/mcp_server.py"]}}}
```

| Tool | What it returns |
|---|---|
| `search_code` | The best-matching code chunks, with file and line ranges |
| `find_definition` | Where a symbol is defined, every file and line using it, other matches |
| `impact` | Files a change to a file or symbol is likely to affect, with reasons, plus tests to run |
| `impact_of_diff` | For a `git diff`: the changed functions, who uses them, ranked impact, tests to run |
| `list_repos` | Indexed repositories |
| `ingest_repo` | Index or re-index a repo (the only tool that writes) |

`repo_id` can be omitted when one repo is indexed (or set `CODEBASE_INTEL_REPO_ID`). Errors
are written for the agent to act on, e.g. "No definition of 'X'. Try search_code instead."

### Start the API server and web UI

```bash
uvicorn app.main:app --reload
# Web UI:           http://localhost:8000/
# Interactive docs: http://localhost:8000/docs
```

The web UI has a repo picker and tabs for search, definition, impact and streaming ask.

### Configure an LLM (only for asking questions)

Search, definitions and impact never call an LLM. Only `/ask` does, and it can run free and
local. Set the backend in `.env`:

**Ollama (free, local, no key).** Install [Ollama](https://ollama.com), then:
```
ollama pull qwen2.5-coder:7b     # ~4.7 GB download
```
```
LLM_BACKEND=ollama
```
On a 16 GB M2 Pro, `qwen2.5-coder:7b` runs entirely on the GPU and answers in about 15–30
seconds. Pick a model that fits in memory with room to spare: a 13 GB model (`gpt-oss`) on
the same machine spilled onto the CPU and didn't finish an answer in 10 minutes (`ollama ps`
shows the split). Tiny models (0.5B) are fast but give vague, uncited answers, which the
answer checks flag.

**Gemini (free tier, no credit card).** Key from [aistudio.google.com](https://aistudio.google.com):
```
LLM_BACKEND=gemini
GEMINI_API_KEY=your-key-here
```

**Anthropic.**
```
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Hosted backends default to `claude-sonnet-5` / `gemini-flash-latest`; override with
`ANTHROPIC_MODEL` / `GEMINI_MODEL` (or `OLLAMA_MODEL`).

---

## Example output

Real output against `psf/requests` (47 files, ~12K lines of Python). More in
[`examples/demo_questions.md`](examples/demo_questions.md).

**Grounded Q&A, fully local (`qwen2.5-coder:7b` via Ollama): "Where is SSL certificate
verification handled?"**
```
A: SSL certificate verification is handled in the `cert_verify` method of the `HTTPAdapter`
class, which is defined in `src/requests/adapters.py`. This method checks if the URL starts
with "https" and if SSL verification is enabled (`verify=True`). If so, it sets up the
connection to use a CA bundle for certificate verification. [...quotes the method's code...]

Citations:
  src/requests/adapters.py  lines 307–348  (Referenced by file path in the answer)
  src/requests/adapters.py  lines 428–453  (Referenced by file path in the answer)

[ollama/qwen2.5-coder:7b; 8 excerpts, 10173 chars]
```
The code it quoted is verbatim from `adapters.py` (lines 321–342, inside `cert_verify`).

**Impact of changing `adapters.py`:**
```
HIGH CONFIDENCE:
  [0.97] tests/test_adapters.py    — test named for this file
  [0.95] src/requests/models.py    — direct import
  [0.95] src/requests/sessions.py  — direct import
  [0.95] tests/test_requests.py    — direct import
  [0.75] src/requests/cookies.py   — transitive import (2 hops)
  ... 7 more at 2 hops

MEDIUM CONFIDENCE:
  [0.54] pyproject.toml            — changed together in 2 of 4 commits
  [0.54] src/requests/compat.py    — changed together in 2 of 4 commits
  ...

TESTS TO RUN:
  tests/test_adapters.py, tests/test_requests.py, tests/test_utils.py, ...
```

**Impact of a real diff** (`git diff` over two files):
```
CHANGED SYMBOLS:
  src/requests/_types.py::has_read  → used in src/requests/models.py
  src/requests/models.py::RequestEncodingMixin._encode_files
  src/requests/models.py::PreparedRequest.prepare_body
  ...
```

**Search: "what happens after Session.send() is called?"** The query names
`Session.send`, so its definition ranks first:
```
[1] src/requests/sessions.py        lines 752–793
[2] src/requests/api.py             lines 67–99
[3] tests/test_requests.py          lines 2608–2646
...
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Ingest pipeline                       │
│                                                              │
│  scan_repo (git ls-files) → load_file → analyze_file         │
│       │          (tree-sitter: symbols, imports, usages)     │
│       │                                        │             │
│  chunk_file (definition-aligned)      embed_texts (local     │
│       │                               sentence-transformers  │
│       │                               or OpenAI; cached by   │
│       ▼                               content hash)          │
│  SQLite + FTS5: chunks, symbols,           │                 │
│    usages, co-change, caches          vector index (exact    │
│  Import graph (NetworkX → JSON)       cosine, numpy)         │
│  Git history → co-change stats             │                 │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                         Query side                           │
│                                                              │
│  /search         vectors + BM25 + exact symbol → rank fusion │
│  /definition     symbol table + usage index                  │
│  /impact         imports + usages + co-change + named tests  │
│                  + code similarity → ranked, with reasons    │
│  /impact/batch   /impact merged across a change set          │
│  /impact/diff    diff → changed symbols → their users        │
│  /ask(/stream)   search → budgeted context → cache? → LLM    │
│                  → citations + checks                        │
│                                                              │
│  Front ends: CLI · web UI · REST · MCP server (stdio)        │
└──────────────────────────────────────────────────────────────┘
```

---

## API reference

### `POST /ingest`

```json
{"repo_path": "/path/to/repo", "repo_id": "my-project"}
```

```json
{"repo_id": "my-project", "files_indexed": 47, "chunks_indexed": 399,
 "symbols_extracted": 807, "edges_in_graph": 107,
 "files_skipped": {"lockfile": 1, "minified": 2}, "chunks_embedded": 12, "chunks_reused": 387}
```

**What gets indexed:** inside a git repo, the files `git ls-files` reports (tracked and
untracked), so anything your `.gitignore` excludes is skipped; outside git, a directory walk
that skips `node_modules`, `dist`, virtualenvs and similar. Python, TS/JS, Markdown and config
files are included. Lockfiles, minified files and files over 1 MB (`INGEST_MAX_FILE_BYTES`)
are skipped and counted in `files_skipped`.

**Re-ingest:** everything is re-parsed (a fraction of a second), and chunks whose text hasn't
changed reuse their stored embeddings (`chunks_reused`). The DB rebuild is one transaction,
committed only after embedding succeeds, so a failed run leaves the previous index intact.

`repo_id` may only contain letters, digits, `_` and `-`, since it's used as a filesystem path
component. The embedding backend and model are recorded per repo and reused for every later
query, even if `.env` changes.

### `POST /search`

```json
{"repo_id": "my-project", "query": "SSL certificate verification", "top_k": 10, "mode": "hybrid"}
```

Three ranked lists merged with reciprocal rank fusion: semantic (embedding similarity),
keyword (BM25 over paths, symbol names and code, with camelCase split so "adapter" matches
`HTTPAdapter`), and chunks that define any identifier in the query. `mode` is `hybrid`
(default), `semantic` or `keyword`. `score` is the fused rank score in hybrid mode, cosine in
semantic mode and BM25 in keyword mode, so compare scores only within one mode.

### `GET /definition?repo_id=X&symbol=Y`

```json
{"symbol": "HTTPAdapter", "qualified_name": "HTTPAdapter", "kind": "class",
 "defining_file": "src/requests/adapters.py", "start_line": 158, "end_line": 748,
 "references": ["src/requests/models.py", "src/requests/sessions.py",
                "tests/test_adapters.py", "tests/test_requests.py"],
 "reference_locations": [{"file_path": "src/requests/models.py", "line": 90}, ...],
 "other_definitions": []}
```

`symbol` can be bare (`send`) or qualified (`HTTPAdapter.send`). For an ambiguous bare name,
real definitions beat anything else, source beats tests, top-level beats nested, and the rest
are listed in `other_definitions`. Usages are matched by identifier name: `Session.send`
returns every `.send` usage, not only calls on a `Session`.

### `POST /impact`

```json
{"repo_id": "my-project", "target": "src/requests/adapters.py", "depth": 3}
```

`target` is a file path or a symbol. Five signals, each reported as the reason:

| Signal | Confidence | Reason shown |
|---|---|---|
| A test named after the target (`test_adapters.py`, `foo.test.ts`) | 0.97 | `test named for this file` |
| Import graph, direct / 2 hops / 3 hops | 0.95 / 0.75 / 0.50 | `direct import` … |
| Files using the target symbol (symbol targets) | 0.70 | `references symbol` |
| Git history: changed together in N of the target's M commits, p = N / (M + 3) | 0.4 + 0.5·p, max 0.9 | `changed together in N of M commits` |
| Nearest chunks to the target's own code | 0.35 | `semantically related` |

Results come back as `high_confidence` (≥ 0.7), `medium_confidence` (≥ 0.4) and `related`.
Ties go to stronger co-change, then to files that change more often overall, then fewer hops,
then path. `tests` lists the test files among the results, best-first. Co-change comes from
the last 5,000 commits of the repo's git history.

### `POST /impact/batch`

```json
{"repo_id": "my-project", "targets": ["src/requests/adapters.py", "src/requests/certs.py"], "depth": 3}
```

`/impact` across a change set (e.g. `git diff --name-only`), merged into one ranking. Each
file reports `triggered_by`, the targets that surfaced it.

### `POST /impact/diff`

```json
{"repo_id": "my-project", "diff": "<output of git diff HEAD>", "depth": 3}
```

Changed lines are mapped to the innermost symbols they touch (a method rather than its whole
class) and returned as `changed_symbols`, each with the files that use it. Those files rank
at 0.96, above other importers; everything else is as in `/impact/batch`. Usages count only in
files that import the changed file (within `depth` hops), so an unrelated class's `send()`
doesn't match. The diff's new side should be the indexed code: ingest the working tree, then
send `git diff HEAD`. Files not in the index are listed in `unindexed_files`.

### `POST /ask`

```json
{"repo_id": "my-project", "question": "How does redirect handling work?", "top_k": 8, "use_cache": true}
```

```json
{
  "answer": "...follows redirects in SessionRedirectMixin.resolve_redirects [1]...",
  "citations": [{"file_path": "src/requests/sessions.py", "start_line": 186, "end_line": 307,
                 "relevance": "Cited as [1] in the answer"}],
  "uncertainty": null, "unverified_mentions": [],
  "backend": "ollama", "model": "qwen2.5-coder:7b", "cached": false,
  "excerpts_used": 7, "excerpts_omitted": 1, "context_chars": 11420
}
```

- **Context:** hybrid search; overlapping or adjacent chunks from one file are merged, and
  excerpts are added best-first up to 12,000 characters (`excerpts_omitted` counts the rest).
- **Citations:** `[N]` references and mentions of an excerpt's file path (with line numbers
  when given) are both resolved to files and lines. Small local models often cite by path.
- **Checks:** `unverified_mentions` lists code names or paths found neither in the excerpts nor
  in the repo's index, usually an invented name. `uncertainty` is set when the answer cites
  nothing, cites excerpts that don't exist, names unverified things, or says the evidence is
  insufficient.
- **Cache:** keyed by a hash of the full prompt (question + excerpt text) and the model, so a
  hit means the same question on unchanged code. `cached: true` means no LLM call was made;
  `"use_cache": false` forces a fresh answer.

Errors: `503` when the backend isn't usable (missing key, Ollama not running, model not
pulled; the message says what to do), `502` when the provider fails or refuses.

### `POST /ask/stream`

Same request as `/ask`; the response is newline-delimited JSON, so answers appear as they're
written:

```
{"type": "context", "backend": "ollama", "model": "...", "excerpts": [{"n": 1, "file_path": "...", "start_line": 1, "end_line": 40}], "excerpts_omitted": 0}
{"type": "delta", "text": "Redirect handling is "}
{"type": "delta", "text": "implemented in sessions.py [1]..."}
{"type": "answer", "response": { ...the full /ask response, after checks... }}
```

A missing API key is a plain `503`; problems only discoverable mid-stream arrive as
`{"type": "error", "status": 502|503, "detail": "..."}`.

### `GET /repos` and `DELETE /repos/{repo_id}`

List indexed repos with their last ingest stats, or delete one's index, database, graph and
metadata (`404` if it was never ingested).

---

## How well it works

Everything below is measured on `psf/requests`, and CI
(`.github/workflows/ci.yml`) re-runs it on every push against a fresh clone, failing if any
metric drops below its floor in `eval/thresholds.yaml`.

| Capability | Before this work | Now |
|---|---|---|
| Search, held-out questions: answer's lines in top 5 / span MRR | 0.72 / 0.52 | **0.92 / 0.88** |
| Search, identifier queries: in top 5 / span MRR | 0.79 / 0.66 | **0.93 / 0.93** |
| Search, main questions: in top 5 / span MRR | 0.88 / 0.71 | **0.95 / 0.79** |
| Definition accuracy (file + line) | 0.71 | **1.00** |
| References recall / precision | 0.00 / 0.00 | **1.00 / 1.00** |
| Import graph edge recall / precision | 0.68 / 0.90 | **1.00 / 1.00** |
| Impact: true direct importers ranked high-confidence | 0.61 | **1.00** |
| Impact on real commits, held-out: recall@5 / MRR | 0.39 / 0.45 (no history) | **0.52 / 0.64** |
| Re-index of unchanged code | ~12 s | **0.6 s** |

### The benchmark

`eval/` holds a labeled benchmark and runners that score the live API through FastAPI's
`TestClient`. No API key is needed.

```bash
python eval/run_eval.py --repo-id requests --ingest ../requests-demo -v
python eval/run_history_eval.py --git <full clone of psf/requests> --repo-id requests
```

- **42 search questions**, each labeled with the function or class that answers it, scored
  by file (`file_hit@k`, MRR) and by exact lines (`span_hit@k`, `span_mrr`: a result overlaps
  the answering symbol's lines), with average result size alongside, since bigger chunks
  overlap more for free.
- **25 held-out questions**, committed before any search tuning and never used to pick
  parameters, and **14 identifier queries** typed the way developers type them.
- **24 definitions, 12 reference sets, 10 impact targets**, and **the true import graph**
  (107 edges).
- **Real commits** for impact: for each commit after 2018 that changed 2–15 Python files, each
  changed source file is the query and the commit's other changed files are the answer.
  Co-change is learned only from commits up to 2018; weights were chosen on 2019–2022
  (72 queries) and checked once on 2023+ (73 queries).

Labels are hand-written in `eval/build_requests_bench.py`. Line spans, the import graph and
reference sets come from Python's own `ast` module, independent of this tool's tree-sitter
extraction, so the benchmark can't inherit the tool's bugs.

### What moved search

| Change | Main span MRR | Held-out span MRR |
|---|---|---|
| Baseline: fixed 1600-char windows, MiniLM, embeddings only | 0.71 | 0.52 |
| Chunks aligned to function/class boundaries | 0.77 | — |
| + hybrid search (BM25 + exact symbol, rank-fused) | 0.80 | 0.76 |
| + `bge-small-en-v1.5` with a context header per chunk | 0.77 | 0.83 |
| + at most 2 keyword hits per file | **0.79** | **0.88** |

The held-out set was added after chunking, so it has no chunking-only number. Tried and not
shipped: other chunk sizes (1600 chars was best), headers with MiniLM (it truncates at 256
tokens), `bge-base-en-v1.5` (no better, ~4× slower ingest), and cross-encoder rerankers
(clearly worse on code, e.g. main span MRR 0.77 → 0.65, plus 0.25–1 s per query). A tuned
idea that looked better on the main set but worse on held-out (down-weighting keyword matches
for plain-English queries) was reverted.

### What moved impact (real commits)

| Impact ranking | Dev: recall@5 / @10 / MRR | Held-out: recall@5 / @10 / MRR |
|---|---|---|
| Without git history (imports, usages, named tests, similarity) | 0.35 / 0.49 / 0.42 | 0.39 / 0.58 / 0.45 |
| `/impact`, with git co-change | 0.46 / 0.67 / **0.69** | 0.52 / **0.70** / **0.64** |
| `/impact/diff` (uses each commit's diff) | **0.52 / 0.74** / 0.67 | **0.53** / **0.70** / 0.63 |

Recall@k is the share of the commit's other changed files in the top k. Co-change is the
biggest single gain. Diff-level impact adds recall on dev but is roughly even with `/impact`
on held-out, where many commits are typing passes touching most symbols in a file. The graph
numbers are slightly optimistic, since today's import graph is used for past commits.

With 14–73 queries per set, one query moves a metric by roughly 0.01–0.07, so small
differences are noise. That's why every tuning decision was checked against a held-out set.

**Correction.** An earlier version of these impact numbers (e.g. `/impact/diff` at dev
0.51 / 0.73 / 0.69, held-out 0.50 / 0.70 / 0.62) partly depended on the order in which
equal-confidence files came out of the graph. Ties are now broken deterministically and the
numbers above are re-measured. Co-change rates are also shrunk toward zero when backed by few
commits (`n / (commits + 3)`), which changes nothing on full history but stops a shallow
clone's "3 of 3 commits" from scoring as near-certain.

### Bugs the benchmark caught

- **Import resolution.** The graph originally found 81 of 107 real edges, 8 of them wrong:
  `from . import certs` resolved to the package `__init__.py` instead of `certs.py`, and the
  `src/` layout meant no test file had any import edges, so tests never showed up as
  impacted. It's now exact.
- **"References" returned only the defining file.** It queried the definitions table instead
  of usages, so references recall was 0.00. It's now 1.00, from a real usage index.
- **An import cycle reported a file as impacting itself.** `adapters.py` and `models.py` import
  each other in `requests`; the traversal now excludes its starting file (covered by
  `tests/test_graph.py::test_import_cycle_excludes_start_from_its_own_results`).
- **A process crash.** `faiss-cpu` and `torch` bundle conflicting OpenMP runtimes on macOS, so a
  server whose first request was `/impact` aborted on the next `/search`. The vector index is
  now plain numpy (same exact results), with a test that faiss is never imported.

### Tests

```bash
pytest tests/
# 187 passed
```

The suite covers:
- symbol and import extraction (Python, TypeScript, TSX, regex fallback) and import
  resolution (`src/` layouts, submodules, tsconfig aliases, ESM specifiers);
- chunking, hybrid search and rank fusion;
- references and definition ranking;
- every impact signal, deterministic ranking, git history with renames, diff parsing;
- incremental and transactional ingest, `.gitignore`-aware scanning, legacy-index migration;
- `/ask` context assembly, caching, citation checks and streaming for all three LLM backends;
- the MCP tools and the web UI;
- the full FastAPI surface through `TestClient`.

---

## Design decisions

**Why git history is an impact signal.** Import edges miss coupling that isn't an import: a
module and its tests, a schema and the code that serializes it. Version control records that
coupling directly. If a file changed in most of the commits that changed yours, it will
probably need to change again. On real commits it was the single biggest improvement.

**Why impact is scored against real commits.** "Which files import this?" has an exact answer,
and the graph now gets it right. "Which files will this change need to touch?" doesn't, but
history records what happened, so the eval replays commits, learning co-change only from
commits before the ones scored.

**Why diff-level impact matches methods by name.** A diff that only changes `cert_verify`
shouldn't rank every importer of `adapters.py` equally. Callers are found by name among files
that import the changed module. That over-matches generic names like `read`, but also
requiring the class name gave up diff-level's recall gain on real commits (dev recall@5
0.52 → 0.46), because methods are mostly called on instances obtained elsewhere.

**Why chunks follow definitions.** A fixed 1,600-character window cuts functions in half, so the
matching chunk often holds the end of one function and the start of the next. Chunks now
come from the parse tree: a function or small class is one chunk, an oversized class is split
into its methods, decorators and comments stay with their definition, and only a single
function too big to fit falls back to windows.

**Why hybrid search, fused by rank.** Embeddings handle paraphrase ("where are redirects
followed?") and miss exact identifiers; BM25 is the reverse. Reciprocal rank fusion merges
the lists by position, which avoids comparing cosine and BM25 scores and needs no trained
weights.

**Why `bge-small-en-v1.5`.** Same size and speed as `all-MiniLM-L6-v2`, but it reads 512
tokens instead of 256, which lets the per-chunk context header (file, enclosing class,
defined symbols) help instead of crowding out code. Each repo stays pinned to the model it
was indexed with, so changing the default never breaks an existing index.

**Why no reranker.** Off-the-shelf cross-encoders are trained on web passages. Both tried made
code search clearly worse and slower.

**Why answers are checked after generation.** "Only use the excerpts" is just a request in a
prompt. The checks make grounding observable: every citation must resolve and every code name
must exist in the excerpts or the repo. Small local models in particular produce fluent,
uncited answers, and those come back flagged instead of looking as trustworthy as a cited
one. It's a name check, not a fact check.

**Why the answer cache is keyed by prompt content.** The key hashes the exact prompt (which
contains the code excerpts) plus backend, model and prompt version, so a cached answer is
reused only while the code it came from is unchanged. There's no invalidation logic to get
wrong.

**Why the Anthropic call uses `effort: "low"` and no `temperature`.** Current Claude models
think adaptively, and thinking counts against `max_tokens`; the old 1,000-token limit could
leave nothing for the answer. Low effort keeps spend down for what is mostly lookup over
supplied excerpts, and current models reject `temperature`.

**Why incremental ingest caches embeddings instead of diffing files.** Parsing and linking the
whole repo takes a fraction of a second; embedding is ~95% of the cost. So every ingest
re-parses everything (nothing is ever stale) and only embedding is incremental, keyed by a
hash of the exact text. If a chunk's text is unchanged, its vector is too.

**Why a numpy vector index, not FAISS or a hosted vector DB.** No infrastructure and no
network. At repo scale an exact cosine search over a normalized matrix is as fast as FAISS's
flat index and gives the same results. FAISS was dropped after its OpenMP runtime clashed with
torch's and crashed the server. The storage class keeps the name `FAISSStore` for
compatibility.

**Why NetworkX, saved as JSON.** No server or schema migrations for a file-level graph. It's
stored as JSON rather than a pickle, which would run arbitrary code if the file were ever
tampered with.

**Why the graph traversal excludes its starting file.** Real code has import cycles, so a file
is reachable from itself; reporting it as its own dependent is never useful.

**Why the embedding backend is pinned per repo.** If `.env` changes after indexing, embedding
queries with the new model would silently return garbage (or crash on a dimension mismatch).
The backend and model are stored with the index and always reused.

**Why route handlers are sync `def`.** Every endpoint does CPU-bound work with no `await`;
FastAPI runs sync handlers in a worker thread, whereas `async def` would block the event loop
(and `/health`) for the duration.

**Why Python and TypeScript only.** Depth over breadth: real ASTs and proper import resolution
for two languages beat shallow support for six.

---

## Limitations

- File-level dependency graph, not a call graph; method usages are matched by name, so
  `Session.send` also matches every other `.send(...)` call
- Import resolution covers Python (relative, absolute, `src/` layouts) and TS/JS (relative,
  `index` files, tsconfig `baseUrl`/`paths`); imports set up by runtime `sys.path` changes
  aren't detected, and external packages are excluded
- Co-change needs git history: a shallow clone (like `requests-demo`, 64 commits) gives it
  little to work with, and a repo without `.git` gets none
- `/impact/diff` needs the diff's new side to match the indexed code
- No live sync: re-run ingest after changes (incremental, but it re-parses the whole repo)
- Answer quality depends on retrieval, and `unverified_mentions` checks names, not claims
- Broad questions can still surface changelog entries (e.g. `HISTORY.md`); a per-file cap
  limits how many
- The default embedding model is general-purpose, not code-specific (`EMBEDDING_MODEL`
  switches it; `EMBEDDING_BACKEND=openai` uses OpenAI)
- The answer cache has no size limit; it's removed with the repo
- The benchmark covers one Python repo; TypeScript is tested but not scored
- The in-memory repo cache is per process, fine for local use but not for multiple
  `uvicorn` workers

## Future work

- A tree-sitter call graph, so method callers are resolved by type rather than name
- Multi-repo support with cross-repo symbol resolution
- Background ingest jobs with progress, so `/ingest` on a large repo doesn't hold the HTTP
  connection open
