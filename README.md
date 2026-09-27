# codebase-intel

AI-powered codebase intelligence: semantic search, symbol lookup, dependency-aware impact analysis, and grounded repository Q&A.

---

## What it does

Ask developer questions about any Python or TypeScript codebase:

- **"Where is `submit_order()` defined, and who calls it?"** → Symbol definition (bare or `Class.method`) with file + line range, plus every usage
- **"What files import `auth.py`?"** → Dependency graph traversal
- **"What would break if I change `adapters.py`?"** → Multi-signal impact analysis
- **"How does data flow from the API layer to the database?"** → Grounded LLM answer over retrieved code chunks
- **"What does this pull request touch?"** → Batch impact analysis across every changed file, merged into one ranked result

Answers are always grounded in retrieved code — no hallucinated function names or invented behavior.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Ingest Pipeline                      │
│                                                             │
│  walk_repo → load_file → detect_language → chunk_file       │
│       │                                        │            │
│  extract_symbols/imports              embed_texts (local    │
│       │                               sentence-transformers │
│       ▼                               or OpenAI)            │
│  MetadataStore (SQLite)                    │                │
│  DependencyGraph (NetworkX)           FAISSStore            │
│       │                               (IndexFlatIP)         │
│       ▼                                    │                │
│  data/metadata/{repo_id}.db          data/indexes/          │
│  data/metadata/{repo_id}.graph.pkl   {repo_id}.index        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       Query Pipeline                        │
│                                                             │
│  POST /search   → embed query → FAISS search → ranked chunks│
│  GET  /definition → SQLite symbol lookup + usage index      │
│  POST /impact   → graph BFS + symbol refs + FAISS (3 signals│
│  POST /impact/batch → merge /impact across many changed files│
│  POST /ask      → search → LLM (grounded answer + cites)    │
│  GET  /repos    → list ingested repos + last-ingest stats   │
│  DELETE /repos/{repo_id} → remove a repo's on-disk artifacts│
└─────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### Install

```bash
git clone <this-repo>
cd codebase-intel
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Configure your LLM key

The search, definition, and impact features work with no API key. Only `/ask` requires one.

Edit `.env` and set your preferred backend:

**Option A — Gemini (free tier, no credit card):**
Get a free API key at [aistudio.google.com](https://aistudio.google.com), then:
```
LLM_BACKEND=gemini
GEMINI_API_KEY=your-key-here
```

**Option B — Anthropic:**
```
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Both backends default to a current model (`claude-sonnet-5` / `gemini-flash-latest`); override with `ANTHROPIC_MODEL=...` or `GEMINI_MODEL=...` in `.env` if you want a different one.

### Ingest a repo

```bash
python scripts/ingest_repo.py --repo /path/to/your/repo --repo-id my-project
# Indexed 47 files, 405 chunks, 807 symbols, 2591 references, 107 graph edges.
```

`repo_id` may only contain letters, digits, `_`, and `-` (it's used as a filesystem path component, so this is enforced everywhere, not just the CLI). Re-running ingest on the same `repo_id` is safe and idempotent — it fully replaces the previous index rather than accumulating stale chunks/symbols/edges alongside it, and the embedding backend used at ingest time is recorded and reused automatically for every later query against that `repo_id`, even if `EMBEDDING_BACKEND` in `.env` changes afterward.

### Query

```bash
# Semantic search (no API key needed)
python scripts/demo_query.py --repo-id my-project "where is authentication handled?"

# Symbol definition (no API key needed)
python scripts/demo_query.py --repo-id my-project --mode definition --symbol HTTPAdapter

# Impact analysis (no API key needed)
python scripts/demo_query.py --repo-id my-project --mode impact --target src/requests/adapters.py

# Grounded Q&A (requires LLM API key)
python scripts/demo_query.py --repo-id my-project --mode ask "how does redirect handling work?"
```

### Start the API server

```bash
uvicorn app.main:app --reload
# Interactive docs at http://localhost:8000/docs
```

---

## API Reference

### `POST /ingest`

Ingest a repository and build all indexes.

```json
{"repo_path": "/path/to/repo", "repo_id": "my-project"}
```

```json
{"repo_id": "my-project", "files_indexed": 47, "chunks_indexed": 405,
 "symbols_extracted": 807, "edges_in_graph": 107}
```

### `POST /search`

Semantic search over embedded code chunks.

```json
{"repo_id": "my-project", "query": "SSL certificate verification", "top_k": 10}
```

### `GET /definition?repo_id=X&symbol=Y`

Symbol definition lookup plus every usage. `symbol` can be bare (`send`) or qualified
(`HTTPAdapter.send`). When a bare name matches several definitions, source files win over
tests and the rest are listed in `other_definitions`.

```json
{"symbol": "HTTPAdapter", "qualified_name": "HTTPAdapter", "kind": "class",
 "defining_file": "src/requests/adapters.py", "start_line": 158, "end_line": 748,
 "references": ["src/requests/models.py", "src/requests/sessions.py",
                "tests/test_adapters.py", "tests/test_requests.py"],
 "reference_locations": [{"file_path": "src/requests/models.py", "line": 90}, ...],
 "other_definitions": []}
```

References come from an index of identifier usages built from the tree-sitter parse, so
comments and docstrings don't count. They're matched by name: a query for `Session.send`
returns every `.send` usage, not only calls on a `Session`.

### `POST /impact`

Multi-signal impact analysis: which files are likely affected by changing a target?

```json
{"repo_id": "my-project", "target": "src/requests/adapters.py", "depth": 3}
```

Returns `high_confidence` (graph traversal), `medium_confidence` (symbol refs), and `related` (semantic similarity) buckets.

### `POST /impact/batch`

Diff-aware impact analysis: the same three signals, run across every file in a change set and merged into one ranked result. Point `targets` at the output of `git diff --name-only` to see everything a whole PR is likely to affect, not just one file at a time.

```json
{"repo_id": "my-project", "targets": ["src/requests/adapters.py", "src/requests/certs.py"], "depth": 3}
```

Each impacted file additionally reports `triggered_by`: which of the requested targets caused it to show up, and keeps the highest confidence when a file is impacted by more than one target.

### `GET /repos`

List every repo_id that has been ingested, with its most recent ingest stats (file/chunk/symbol/edge counts, embedding backend, timestamp).

### `DELETE /repos/{repo_id}`

Remove a repo's index, database, graph, and metadata from disk, and evict it from the in-memory cache. Returns `404` if the repo_id was never ingested.

### `POST /ask`

Grounded LLM Q&A with citations. Requires a Gemini or Anthropic API key in `.env`.

```json
{"repo_id": "my-project", "question": "How does redirect handling work?", "top_k": 8}
```

```json
{
  "answer": "Redirect handling is implemented in sessions.py [1][2]. After Session.send() receives a response, it checks response.status_code against redirect codes (301, 302, 303, 307, 308)...",
  "citations": [
    {"file_path": "src/requests/sessions.py", "start_line": 340, "end_line": 398,
     "relevance": "Cited as [1] in the answer"}
  ],
  "uncertainty": null
}
```

---

## Example Output

Demoed against `psf/requests` (47 files, ~12K lines of Python) — see
[`examples/demo_questions.md`](examples/demo_questions.md) for the full set of real,
regenerated output.

**Impact analysis of `adapters.py`:**
```
HIGH CONFIDENCE (direct/transitive imports):
  [0.95] tests/test_requests.py    — direct import
  [0.95] src/requests/models.py    — direct import
  [0.95] src/requests/sessions.py  — direct import
  [0.95] tests/test_adapters.py    — direct import
  [0.75] src/requests/cookies.py   — transitive import (2 hops)
  [0.75] src/requests/utils.py     — transitive import (2 hops)
  [0.75] src/requests/__init__.py  — transitive import (2 hops)
  ... 5 more at 2 hops

MEDIUM CONFIDENCE:
  [0.50] tests/test_utils.py       — transitive import (3 hops)
  ... 5 more at 3 hops

RELATED:
  [0.35] pyproject.toml            — semantically related
  [0.35] README.md                 — semantically related
```

**Search: "where is SSL certificate verification handled?":**
```
[1] src/requests/adapters.py  lines 296–337  score=0.457
[2] tests/certs/README.md     lines 1–11     score=0.442
[3] src/requests/adapters.py  lines 431–464  score=0.423
[4] src/requests/certs.py     lines 1–19     score=0.415
[5] src/requests/sessions.py  lines 470–503  score=0.385
```

**Grounded Q&A: "Where is SSL certificate verification handled?"**
```
A: Based on the provided excerpts, SSL/TLS certificate verification is handled in
the following places:
* HTTPAdapter.cert_verify method in src/requests/adapters.py: Specifically
  dedicated to verifying an SSL certificate (exposed for subclassing
  HTTPAdapter) [1]. Adapter methods also accept and process verify (a boolean
  to control TLS certificate verification or a path string to a CA bundle)
  and client cert parameters [3, 10].
* Session in src/requests/sessions.py: Manages the self.verify configuration
  (defaulting to True) and the verify parameter on session requests [6, 7].
* src/requests/certs.py: Provides the preferred default CA certificate bundle
  used for verification via certifi.where() [5].
* Test Server in tests/testserver/server.py: Configures server-side
  verification using ssl.SSLContext for mutual TLS tests [9].

Citations:
  src/requests/adapters.py    lines 296–337   (cited as [1])
  src/requests/certs.py       lines 1–19      (cited as [5])
  tests/testserver/server.py  lines 153–177   (cited as [9])
```

**Batch impact analysis (`POST /impact/batch`) — "what does this PR touch?":**
```
targets: ["src/requests/adapters.py", "src/requests/certs.py"]

HIGH CONFIDENCE:
  [0.95] src/requests/models.py    — direct import    (via src/requests/adapters.py)
  [0.95] src/requests/sessions.py  — direct import    (via src/requests/adapters.py, src/requests/certs.py)
  ...
RELATED:
  [0.35] README.md  — semantically related  (via src/requests/adapters.py, src/requests/certs.py)
```
`sessions.py` and `README.md` are each triggered by *both* changed files — one call
instead of running `/impact` twice and manually merging the results. Full output in
[`examples/demo_questions.md`](examples/demo_questions.md#q8-what-does-a-pr-touching-adapterspy-and-certspy-affect-batch-impact).

---

## Results

Validated end-to-end against `psf/requests` (47 files, ~12K lines of Python) and dogfooded
against its own source. Everything below is measured, not estimated.

**Ingestion is fast and fully local.** 47 files → 405 chunks, 807 symbols, 2,591 symbol
usages and 107 dependency edges in ~8 seconds on a laptop CPU — walking, chunking, tree-sitter parsing, dependency
resolution, and embedding with the local `all-MiniLM-L6-v2` model, no API key and no network
calls required.

**Re-ingestion is provably idempotent.** Ingesting the same `repo_id` twice leaves the exact
same row counts, not double — verified both with a synthetic fixture
(`tests/test_pipeline.py::test_reingest_replaces_not_accumulates`) and by re-ingesting the real
`requests` repo mid-development and diffing chunk counts before/after. A symbol removed from
source (e.g. a rename) is confirmed gone from the DB after re-ingest, not left as an orphaned row.

**Retrieval finds the right code with pure semantic similarity — no keyword matching.**
Querying "where is SSL certificate verification handled?" against all 405 chunks returns
`adapters.py`'s TLS/cert-verification logic and `certs.py`'s CA-bundle resolution in the top 5
results, purely from embedding similarity — neither file name nor the query share much
vocabulary. See [Example Output](#example-output) above for the exact ranked results.

**The impact engine caught a real bug in a mature, heavily-tested library — on itself.**
Running `/impact` against `adapters.py` in the actual `requests` codebase surfaced that
`adapters.py` and `models.py` import each other (a genuine circular import, not a test
fixture). The graph BFS didn't originally exclude the traversal's own starting file from its
results, so it reported "changing `adapters.py` might be impacted by `adapters.py`." That's
exactly the kind of false result a hand-rolled impact tool ships silently until it's run
against real code with a real import cycle. Fixed and covered by a regression test
(`tests/test_graph.py::test_import_cycle_excludes_start_from_its_own_results`) that encodes
the cycle directly rather than relying on the one real repo that happens to have one.

**The dependency graph is exact on `requests`.** Scored against the import graph Python's own
`ast` module derives (see [Evaluation](#evaluation)), all 107 edges are found with no false
ones. An earlier version found 81, 8 of them wrong: `from . import certs` resolved to the
package `__init__.py` instead of `certs.py`, and the `src/` layout meant no test file had any
edges, so tests never showed up as impacted.

**Test suite: 100/100 passing in about a second**, covering ingestion and language
detection, chunking with overlap/line-boundary handling, tree-sitter *and* regex-fallback
symbol extraction for Python, TypeScript and TSX (qualified method names, same-named symbols,
imported names), Python and TypeScript import resolution (`src/` layouts, relative and
submodule imports, dotted filenames, ESM `.js` specifiers, tsconfig `paths` aliases), the
dependency graph and its import-cycle handling, find-references, transactional re-ingest and
legacy-DB migration, `/ask` error handling, all three impact-analysis
signals plus the batch/merge logic, FAISS storage/normalization/dimension-mismatch safety, and
the full FastAPI surface (ingest → search → impact → repos → delete) driven through
`TestClient` rather than mocked at the unit level.

**Fixes were verified against production behavior, not just unit tests.** Re-ingest safety,
`repo_id` path-traversal rejection, and embedding-backend/dimension consistency were each
exercised through the live FastAPI app (`tests/test_repos.py`) in addition to targeted unit
tests, and manually re-run against the real `requests` and `requests-demo` repos while making
the change — the numbers and outputs throughout this README come from those actual runs.

---

## Design Decisions

**Why FAISS over a hosted vector DB:** No infrastructure to run, no network calls, sufficient
performance for repo-scale (~thousands of chunks). A flat `IndexFlatIP` with L2-normalized
vectors gives exact cosine similarity.

**Why NetworkX over Neo4j:** Same reasoning — no server, no schema migrations, sufficient
for file-level dependency DAGs. The graph serializes to a single pickle.

**Why answers are grounded:** LLM hallucination about code is uniquely harmful — invented
function names, wrong file paths, and fabricated behavior are worse than "I don't know."
Every answer cites the exact file and line range it was derived from.

**Why three impact signals:** Import edges alone miss semantic coupling. Semantic similarity
alone produces false positives. Combining graph traversal + symbol references + semantic
similarity gives calibrated confidence scores that are actually useful.

**Why the dependency-graph traversal explicitly excludes its own starting file:** Real
codebases have import cycles (`requests`' own `adapters.py` and `models.py` import each
other). Graph-theoretically, a file is reachable from itself through a cycle, so a naive BFS
will report a file as one of its own transitive dependents. That's never actionable
information for "what breaks if I change this file" — it's excluded unconditionally, cycle or
not, rather than trying to special-case cycle detection.

**Why Python + TypeScript only:** Depth over breadth. Two languages done well (real ASTs
via tree-sitter, proper import resolution) beats six languages done poorly.

**Why route handlers are sync `def`, not `async def`:** Every endpoint does CPU-bound work
(embedding, FAISS search, tree-sitter parsing) with no `await` in the body. FastAPI runs sync
path operations in a worker thread automatically; leaving them `async def` would run that
CPU-bound work directly on the single asyncio event loop and block every other in-flight
request — including `/health` — for the duration.

**Why the embedding backend is pinned per repo, not read from the environment at query
time:** `EMBEDDING_BACKEND` can change between when a repo was ingested and when it's later
queried (e.g. switching `.env` to try OpenAI embeddings on a new repo). Reading the *current*
env var at query time would silently mis-embed the query, either producing meaningless
results or crashing on a dimension mismatch. The backend and model used at ingest time are
recorded alongside the FAISS index and reused for every query against that `repo_id`.

---

## Running Tests

```bash
pytest tests/ -v
# 100 passed
```

---

## Evaluation

`eval/` holds a labeled benchmark against `psf/requests` and a runner that scores the live
API (`/search`, `/definition`, `/impact`) through FastAPI's `TestClient`. No API key needed.

```bash
python eval/run_eval.py --repo-id requests --ingest ../requests-demo -v
python eval/run_eval.py --repo-id requests --out eval/results/<name>.json   # save for comparison
```

- **42 search questions**, each labeled with the function/class that answers it. Scored
  file-level (`file_hit@k`, MRR) and line-level (`span_hit@k`: a result chunk overlaps the
  answering symbol's exact line span).
- **24 definition lookups** (bare and `Class.method` names), **12 reference sets**
  (every file that uses a symbol), **10 impact targets** (their true direct importers).
- **The true import graph** (107 edges), to score the dependency graph directly.

Labels are hand-written in `eval/build_requests_bench.py`. Line spans, the import graph and
reference sets are derived from the `requests` source with Python's own `ast` module, which
is deliberately independent of codebase-intel's tree-sitter/regex extraction, so the
benchmark can't inherit the tool's bugs. Saved runs live in `eval/results/`.

| Metric | Baseline | Now |
|---|---|---|
| Search `file_hit@5` / `span_hit@5` / MRR | 0.95 / 0.88 / 0.78 | 0.95 / 0.88 / 0.78 |
| Definition accuracy (file + line) | 0.71 | **1.00** |
| References recall / precision | 0.00 / 0.00 | **1.00 / 1.00** |
| Impact: true direct importers in `high_confidence` | 0.61 | **1.00** |
| Impact: `high_confidence` precision | 0.70 | **1.00** |
| Import graph edge recall / precision | 0.68 / 0.90 | **1.00 / 1.00** |

Search is unchanged so far: these fixes were to symbols, references and the import graph.
Retrieval quality (AST-aware chunking, hybrid keyword + semantic search) is next.

---

## Limitations

- File-level dependency graph, not call-level (no intra-function call edges)
- No live repo sync — re-run `ingest` after changes (safe to do repeatedly; see Quickstart)
- Import resolution covers Python (relative, absolute, `src/` layouts) and TS/JS (relative,
  `index` files, tsconfig `baseUrl`/`paths`); external packages are excluded. Python imports
  resolved via runtime `sys.path` tweaks aren't detected
- References are matched by identifier name, not by type: usages of `Session.send` include
  every `.send(...)` call
- Answer quality depends on whether the relevant code was retrieved in the top-k chunks
- `all-MiniLM-L6-v2` is 384-dimensional and fast but not state-of-the-art; swap to
  `text-embedding-3-small` via `EMBEDDING_BACKEND=openai` for better retrieval
- `/impact/batch` is file-level diff-awareness (which files does a change set touch), not
  line-level (which *symbols* within a file a specific hunk affects)
- Single-process, in-memory `_loaded_repos` cache — fine for local/single-worker use, not
  designed for multiple concurrent `uvicorn` workers

## Future Work

- Line/hunk-level diff-aware impact analysis: changed lines → affected symbols, not just files
- Tree-sitter call graph for intra-file function-call edges
- Multi-repo support with cross-repo symbol resolution
- Streaming responses for `/ask`
- Background ingest jobs with progress polling, so `/ingest` on a large repo doesn't hold the
  HTTP connection open for the whole run
