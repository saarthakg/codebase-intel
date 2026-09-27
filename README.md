# codebase-intel

AI-powered codebase intelligence: semantic search, symbol lookup, dependency-aware impact analysis, and grounded repository Q&A.

---

## What it does

Ask developer questions about any Python or TypeScript codebase:

- **"Where is `submit_order()` defined, and who calls it?"** → Symbol definition (bare or `Class.method`) with file + line range, plus every usage
- **"What files import `auth.py`?"** → Dependency graph traversal
- **"What would break if I change `adapters.py`?"** → Multi-signal impact analysis
- **"How does data flow from the API layer to the database?"** → Grounded LLM answer over retrieved code chunks
- **"What does this pull request touch, and which tests should I run?"** → Diff-aware impact:
  the functions a diff changed, the files that use them, and the tests to run

Answers are always grounded in retrieved code — no hallucinated function names or invented behavior.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Ingest Pipeline                      │
│                                                             │
│  walk_repo → load_file → detect_language → analyze_file     │
│       │          (tree-sitter: symbols, imports, usages)    │
│       │                                        │            │
│  chunk_file (definition-aligned)      embed_texts (local    │
│       │                               sentence-transformers │
│       ▼                               or OpenAI)            │
│  MetadataStore (SQLite + FTS5)             │                │
│  DependencyGraph (NetworkX)           FAISSStore            │
│       │                               (IndexFlatIP)         │
│       ▼                                    │                │
│  data/metadata/{repo_id}.db          data/indexes/          │
│  data/metadata/{repo_id}.graph.pkl   {repo_id}.index        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       Query Pipeline                        │
│                                                             │
│  POST /search   → FAISS + BM25 + exact symbol → rank fusion │
│  GET  /definition → SQLite symbol lookup + usage index      │
│  POST /impact   → imports + usages + git co-change + named  │
│                   tests + semantic neighbours, ranked       │
│  POST /impact/batch → merge /impact across many changed files│
│  POST /impact/diff  → diff → changed symbols → their users  │
│  POST /ask      → search → cache? → LLM → cite + check      │
│  POST /ask/stream → same, streamed as NDJSON events         │
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

### Configure an LLM (only for `/ask`)

Search, definition and impact never call an LLM. Only `/ask` does, and it can run entirely
free and local.

Edit `.env` and set your preferred backend:

**Option A: Ollama (free, local, no key):**
Install [Ollama](https://ollama.com), then:
```
ollama pull qwen2.5-coder:7b     # ~4.7 GB download
```
```
LLM_BACKEND=ollama
OLLAMA_MODEL=qwen2.5-coder:7b
```
Pick a model that fits in your GPU/unified memory with room to spare. A 13 GB model
(`gpt-oss`) on a 16 GB M2 Pro was partly offloaded to the CPU and didn't finish an answer
within 10 minutes; `ollama ps` shows the CPU/GPU split. Very small models (0.5B) run fast but give vague, often uncited answers,
which the answer checks flag.

**Option B: Gemini (free tier, no credit card):**
Get a free API key at [aistudio.google.com](https://aistudio.google.com), then:
```
LLM_BACKEND=gemini
GEMINI_API_KEY=your-key-here
```

**Option C: Anthropic:**
```
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Hosted backends default to a current model (`claude-sonnet-5` / `gemini-flash-latest`);
override with `ANTHROPIC_MODEL=...` or `GEMINI_MODEL=...`. Whatever the backend, repeated
questions on unchanged code are answered from a local cache without calling the LLM again.

### Ingest a repo

The first ingest downloads the local embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) from
Hugging Face; after that everything runs offline.

```bash
python scripts/ingest_repo.py --repo /path/to/your/repo --repo-id my-project
# Indexed 47 files, 399 chunks, 807 symbols, 2591 references, 107 graph edges.
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
{"repo_id": "my-project", "files_indexed": 47, "chunks_indexed": 399,
 "symbols_extracted": 807, "edges_in_graph": 107}
```

### `POST /search`

Hybrid search over code chunks. Three ranked lists are merged with reciprocal rank fusion:
semantic (embedding similarity), keyword (BM25 over paths, symbol names and code, with
camelCase split so "adapter" matches `HTTPAdapter`), and chunks that define any identifier
typed in the query (`get_netrc_auth`, `HTTPAdapter.send`).

```json
{"repo_id": "my-project", "query": "SSL certificate verification", "top_k": 10, "mode": "hybrid"}
```

`mode` is `hybrid` (default), `semantic` or `keyword`. `score` is the fused rank score in
hybrid mode, cosine similarity in semantic mode and BM25 in keyword mode, so compare scores
only within one mode.

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

Ranks files by how likely they are to need changes too, from five signals, each with its
reason in the output:

| Signal | Confidence | Reason shown |
|---|---|---|
| A test named after the target (`test_adapters.py`, `foo.test.ts`) | 0.97 | `test named for this file` |
| Import graph, direct / 2 hops / 3 hops | 0.95 / 0.75 / 0.50 | `direct import` … |
| Files using the target symbol (symbol targets) | 0.70 | `references symbol` |
| Git history: changed together in a fraction p of the target's commits | 0.4 + 0.5·p, max 0.9 | `changed together in N of M commits` |
| Nearest chunks to the target's own code | 0.35 | `semantically related` |

Results are bucketed into `high_confidence` (≥ 0.7), `medium_confidence` (≥ 0.4) and
`related`, with co-change strength breaking ties within a confidence level. `tests` lists the
test files among them, best-first: the tests to run. Co-change comes from the last 5,000
commits of the ingested repo's git history (none if it isn't a git repo).

### `POST /impact/batch`

File-level impact across a change set, merged into one ranked result. Point `targets` at the
output of `git diff --name-only` to see everything a whole PR is likely to affect.

```json
{"repo_id": "my-project", "targets": ["src/requests/adapters.py", "src/requests/certs.py"], "depth": 3}
```

Each impacted file additionally reports `triggered_by`: which of the requested targets caused it to show up, and keeps the highest confidence when a file is impacted by more than one target.

### `POST /impact/diff`

Impact of an actual diff, function by function:

```json
{"repo_id": "my-project", "diff": "<output of git diff HEAD>", "depth": 3}
```

Changed lines are mapped to the innermost symbols they touch (a method rather than its
whole class), returned as `changed_symbols` with the files that use each one. Those files
rank at 0.96, above other importers of the changed file; everything else is as in
`/impact/batch`. Usages count only in files that import the changed file (within `depth`
hops), so an unrelated class's `send()` doesn't match. Line numbers are matched against the
index, so the diff's new side should be the ingested code: ingest the working tree, then send
`git diff HEAD`. Files in the diff that aren't indexed are listed in `unindexed_files`.
CLI: `git diff HEAD | python scripts/demo_query.py --repo-id X --mode impact-diff --diff -`.

### `GET /repos`

List every repo_id that has been ingested, with its most recent ingest stats (file/chunk/symbol/edge counts, embedding backend, timestamp).

### `DELETE /repos/{repo_id}`

Remove a repo's index, database, graph, and metadata from disk, and evict it from the in-memory cache. Returns `404` if the repo_id was never ingested.

### `POST /ask`

Grounded LLM Q&A with citations. Needs an LLM backend (see Quickstart; Ollama is free).

```json
{"repo_id": "my-project", "question": "How does redirect handling work?", "top_k": 8, "use_cache": true}
```

```json
{
  "answer": "Redirect handling is implemented in sessions.py [1][2]. After Session.send() receives a response, it checks response.status_code against redirect codes (301, 302, 303, 307, 308)...",
  "citations": [
    {"file_path": "src/requests/sessions.py", "start_line": 340, "end_line": 398,
     "relevance": "Cited as [1] in the answer"}
  ],
  "uncertainty": null,
  "unverified_mentions": [],
  "backend": "ollama", "model": "qwen2.5-coder:7b", "cached": false,
  "excerpts_used": 7, "excerpts_omitted": 1, "context_chars": 11420
}
```

- **Retrieval and context:** the question goes through hybrid search; overlapping or
  adjacent chunks from one file are merged, and excerpts are added best-first up to a
  12,000-character budget. `excerpts_omitted` counts retrieved excerpts that didn't fit.
- **Cache:** answers are cached by a hash of the full prompt (question + excerpt text) and the
  model, so a hit means the same question on unchanged code. `cached: true` means no LLM call
  was made; send `"use_cache": false` to force a fresh answer.
- **Checks:** `[N]` citations are mapped back to files and lines. `unverified_mentions` lists
  code names or file paths in the answer that appear neither in the excerpts nor anywhere in
  the repo's index, which usually means an invented name. `uncertainty` is set when the answer
  cites nothing, cites excerpts that don't exist, names unverified things, or says the
  evidence is insufficient.

Errors: `503` when the backend isn't usable (missing key, Ollama not running, model not
pulled; the message says what to do), `502` when the provider fails or refuses.

### `POST /ask/stream`

Same request as `/ask`; the response is newline-delimited JSON, so answers from slow local
models appear as they're written:

```
{"type": "context", "backend": "ollama", "model": "...", "excerpts": [{"n": 1, "file_path": "...", "start_line": 1, "end_line": 40}], "excerpts_omitted": 0}
{"type": "delta", "text": "Redirect handling is "}
{"type": "delta", "text": "implemented in sessions.py [1]..."}
{"type": "answer", "response": { ...the full /ask response, after checks... }}
```

A missing API key is still a plain `503`. Problems only discoverable mid-stream arrive as
`{"type": "error", "status": 502|503, "detail": "..."}`. CLI:
`python scripts/demo_query.py --repo-id X --mode ask --stream "question"`.

---

## Example Output

Demoed against `psf/requests` (47 files, ~12K lines of Python) — see
[`examples/demo_questions.md`](examples/demo_questions.md) for the full set of real,
regenerated output.

**Impact analysis of `adapters.py`:**
```
HIGH CONFIDENCE:
  [0.97] tests/test_adapters.py    — test named for this file
  [0.95] tests/test_requests.py    — direct import
  [0.95] src/requests/models.py    — direct import
  [0.95] src/requests/sessions.py  — direct import
  [0.75] src/requests/cookies.py   — transitive import (2 hops)
  ... 7 more at 2 hops

MEDIUM CONFIDENCE:
  [0.65] pyproject.toml            — changed together in 2 of 4 commits
  [0.65] src/requests/compat.py    — changed together in 2 of 4 commits
  ... 

TESTS TO RUN:
  tests/test_adapters.py, tests/test_requests.py, tests/test_utils.py, ...
```

**Search: "what happens after Session.send() is called?":**
```
[1] src/requests/sessions.py        lines 752–793    score=0.046   (Session.send itself)
[2] src/requests/api.py             lines 67–99      score=0.029
[3] tests/test_requests.py          lines 2608–2646  score=0.028
[4] src/requests/sessions.py        lines 108–132    score=0.028
[5] HISTORY.md                      lines 1653–1704  score=0.028
```

**Grounded Q&A: "Where is SSL certificate verification handled?"** (generated before the
hybrid-search and full-chunk changes; not re-run since, as it uses a paid API)
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

**Ingestion is fast and fully local.** 47 files → 399 chunks, 807 symbols, 2,591 symbol
usages and 107 dependency edges in ~12 seconds on a laptop CPU — walking, tree-sitter parsing,
chunking, dependency resolution, keyword indexing and embedding with the local
`bge-small-en-v1.5` model. No API key, and no network calls after the one-time model download.

**Re-ingestion is provably idempotent.** Ingesting the same `repo_id` twice leaves the exact
same row counts, not double — verified both with a synthetic fixture
(`tests/test_pipeline.py::test_reingest_replaces_not_accumulates`) and by re-ingesting the real
`requests` repo mid-development and diffing chunk counts before/after. A symbol removed from
source (e.g. a rename) is confirmed gone from the DB after re-ingest, not left as an orphaned row.

**Search gains held up on questions that weren't used for tuning.** Every search change was
scored on 42 questions, and parameters were chosen on those only. A further 25 questions were
written and committed *before* any tuning and used only to check that gains transfer. On them,
the answering function's exact lines are in the top 5 for 92% of questions, up from 72%,
and span MRR rose from 0.52 to 0.88. A tuned idea that didn't transfer (down-weighting keyword
matches for plain-English queries) was reverted. See [Evaluation](#evaluation).

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

**Test suite: 165/165 passing in about 3 seconds**, covering ingestion and language
detection, chunking with overlap/line-boundary handling, tree-sitter *and* regex-fallback
symbol extraction for Python, TypeScript and TSX (qualified method names, same-named symbols,
imported names), Python and TypeScript import resolution (`src/` layouts, relative and
submodule imports, dotted filenames, ESM `.js` specifiers, tsconfig `paths` aliases), the
dependency graph and its import-cycle handling, find-references, transactional re-ingest and
legacy-DB migration, structure-aware chunking, keyword indexing, rank fusion, per-repo
embedding-model pinning, `/ask` context assembly, caching, answer checks, streaming and
error handling for all three LLM backends, git history with renames, co-change, test-file
matching and diff-level impact, all three impact-analysis
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

**Why several impact signals, and why git history is one of them:** Import edges miss
coupling that isn't an import: a module and its tests, a schema and the code that serializes
it. Version control records that coupling directly. If a file changed in most of the commits
that changed yours, it will probably need to change again. On real `requests` commits, adding
co-change was the single biggest improvement to impact ranking (see Evaluation).

**Why impact is scored against real commits:** "Which files import this?" has an exact answer,
and the import graph now gets it right. "Which files will this change actually need to touch?"
doesn't, but history records what happened. The history eval replays commits and checks
whether the files they changed are ranked near the top. Co-change is learned only from commits
*before* the ones scored, and weights were chosen on 2019–2022 commits, then checked once
on 2023+.

**Why diff-level impact matches methods by name:** A diff that only changes `cert_verify`
shouldn't rank every importer of `adapters.py` equally. Callers are found by name among files
that import the changed module. That over-matches generic names like `read`, but requiring
the class name too was tested and threw away every gain on real commits, because methods are
mostly called on instances obtained elsewhere (`r.connection.send(...)`).

**Why the dependency-graph traversal explicitly excludes its own starting file:** Real
codebases have import cycles (`requests`' own `adapters.py` and `models.py` import each
other). Graph-theoretically, a file is reachable from itself through a cycle, so a naive BFS
will report a file as one of its own transitive dependents. That's never actionable
information for "what breaks if I change this file" — it's excluded unconditionally, cycle or
not, rather than trying to special-case cycle detection.

**Why chunks follow definitions instead of fixed windows:** A fixed 1,600-character window cuts
functions in half, so the chunk that matches a query often holds the end of one function and
the start of the next. Chunks now come from the parse tree: a function or small class is one
chunk, an oversized class is split into its methods, decorators and comments stay with the
definition they describe, and only a single function too big to fit falls back to windows.

**Why hybrid search, fused by rank:** Embeddings are good at paraphrase ("where are redirects
followed?") and bad at exact identifiers; BM25 is the reverse. Reciprocal rank fusion merges
the lists by rank position, which sidesteps the fact that cosine and BM25 scores aren't on
comparable scales, and needs no trained weights. Queries that contain an identifier also pull
in the chunk that defines it, straight from the symbol table.

**Why `bge-small-en-v1.5` by default:** Same size and speed as `all-MiniLM-L6-v2`, but it reads
512 tokens instead of 256, which is what lets the per-chunk context header (file, enclosing
class, defined symbols) help instead of crowding out code. Each repo stays pinned to the model
it was ingested with, so changing the default never breaks an existing index.

**Why no reranker:** Off-the-shelf cross-encoders are trained on web search passages. Both
ones tried made results clearly worse on code while adding latency to every query.

**Why `/ask` caches by prompt content, not by question:** The cache key hashes the exact
prompt, which contains the excerpt text, plus the backend, model and a prompt-format version.
So a cached answer is only reused while the code it was answered from is unchanged; editing
that code and re-ingesting naturally misses the cache, with no invalidation logic to get wrong.

**Why answers are checked after generation:** "Grounded" is only a request in the system
prompt. The checks make it observable: every cited `[N]` must exist, and every code name or
path the answer mentions must appear in the excerpts or the repo index. Small local models
in particular produce fluent answers with no citations, and those now come back flagged
instead of looking as trustworthy as a cited one.

**Why the Anthropic call uses `effort: "low"` and no `temperature`:** Current Claude models
think adaptively, and thinking tokens count against `max_tokens`; the old `max_tokens=1000`
could leave nothing for the answer. Low effort keeps spend down for what is mostly lookup
over supplied excerpts, and `temperature` is rejected by current models.

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
# 165 passed
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
  file-level (`file_hit@k`, MRR) and line-level (`span_hit@k`, `span_mrr`: a result chunk
  overlaps the answering symbol's exact line span), with the average result size alongside,
  since bigger chunks overlap more spans for free.
- **25 held-out search questions**, committed before any search tuning and never used to pick
  parameters, and **14 identifier queries** typed the way developers do (`get_netrc_auth`,
  `HTTPAdapter.send timeout handling`).
- **24 definition lookups** (bare and `Class.method` names), **12 reference sets**
  (every file that uses a symbol), **10 impact targets** (their true direct importers).
- **The true import graph** (107 edges), to score the dependency graph directly.

Labels are hand-written in `eval/build_requests_bench.py`. Line spans, the import graph and
reference sets are derived from the `requests` source with Python's own `ast` module, which
is deliberately independent of codebase-intel's tree-sitter/regex extraction, so the
benchmark can't inherit the tool's bugs. Saved runs live in `eval/results/`.

| Metric | Baseline | Now |
|---|---|---|
| Search, main 42: `span_hit@5` / span MRR | 0.88 / 0.71 | **0.95 / 0.79** |
| Search, held-out 25: `span_hit@5` / span MRR | 0.72 / 0.52 | **0.92 / 0.88** |
| Search, identifier 14: `span_hit@5` / span MRR | 0.79 / 0.66 | **0.93 / 0.93** |
| Search: avg lines per top-5 result | 44 | 39 |
| Definition accuracy (file + line) | 0.71 | **1.00** |
| References recall / precision | 0.00 / 0.00 | **1.00 / 1.00** |
| Impact: true direct importers in `high_confidence` | 0.61 | **1.00** |
| Impact: `high_confidence` precision | 0.70 | **1.00** |
| Import graph edge recall / precision | 0.68 / 0.90 | **1.00 / 1.00** |

What moved search, in order (span MRR on main / held-out; the held-out set was added after
chunking was done, so it has no chunking-only number):

| Change | Main | Held-out |
|---|---|---|
| Baseline: fixed 1600-char windows, MiniLM, semantic only | 0.71 | 0.52 |
| Definition-aligned chunks | 0.77 | — |
| + hybrid search (BM25 + exact symbol, rank-fused) | 0.80 | 0.76 |
| + `bge-small-en-v1.5` with a context header per chunk | 0.77 | 0.83 |
| + at most 2 keyword hits per file | **0.79** | **0.88** |

Tried and not shipped: chunk budgets of 1000/1200/2000/2400 chars (1600 was best), context
headers with MiniLM (it truncates at 256 tokens), `bge-base-en-v1.5` (no better, ~4× slower
ingest), cross-encoder rerankers `ms-marco-MiniLM-L-6-v2` and `bge-reranker-base` (both clearly
worse on code, e.g. main span MRR 0.77 → 0.65, and +0.25–1s per query).

With 25–42 questions per set, one question moves a metric by 0.02–0.04, so small differences
are noise. That's why decisions were checked against the held-out set.

### Impact against real commits

`eval/run_history_eval.py` replays `requests` commits: for every commit after 2018 that changed
2–15 existing Python files, each changed source file is the query and the commit's other
changed files are what impact analysis should find. Co-change is built only from commits up
to 2018. Weights were chosen on 2019–2022 commits (72 queries); 2023+ (73 queries) is
held out. It needs a full clone at the indexed commit (see the script's docstring).

| Impact ranking | Dev: recall@5 / @10 / MRR | Held-out: recall@5 / @10 / MRR |
|---|---|---|
| Import graph + semantic (before) | 0.35 / 0.47 / 0.40 | 0.43 / 0.66 / 0.46 |
| + git co-change | 0.45 / 0.65 / 0.65 | 0.50 / 0.68 / 0.59 |
| + semantic query from the file's own code | 0.46 / 0.66 / 0.65 | 0.50 / 0.70 / 0.59 |
| + named tests ranked first | 0.46 / 0.66 / 0.68 | 0.50 / 0.70 / 0.60 |
| `/impact/diff` (uses each commit's diff) | **0.51 / 0.73 / 0.69** | 0.50 / 0.70 / **0.62** |

Recall@k is the share of the commit's other changed files in the top k. The graph numbers
are slightly optimistic, since today's import graph is used for past commits. Diff-level
impact helps less on held-out because many 2023+ commits are typing passes that touch most
symbols in a file.

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
- `unverified_mentions` is a name check, not a fact check: an answer can use only real names
  and still describe them wrongly
- The answer cache has no size limit or expiry; it lives in each repo's DB and is removed
  with `DELETE /repos/{repo_id}`
- Local models trade quality for cost. The `/ask` examples in this README were generated
  with hosted models
- The default local model (`bge-small-en-v1.5`) is general-purpose, not code-specific. Set
  `EMBEDDING_MODEL` to any sentence-transformers model, or `EMBEDDING_BACKEND=openai`
- Keyword search still surfaces changelog/prose entries for broad questions (e.g. `HISTORY.md`
  for "where is SSL certificate verification handled?"); a per-file cap limits how many, but
  one can still rank first
- The search benchmark covers one Python repo; TypeScript retrieval is tested but not scored
- Co-change needs git history: a shallow clone (like `requests-demo`, 64 commits) gives it
  little to work with, and a repo without `.git` gets none
- `/impact/diff` needs the diff's new side to match the ingested code, since it maps line
  numbers to symbols in the index
- Single-process, in-memory `_loaded_repos` cache — fine for local/single-worker use, not
  designed for multiple concurrent `uvicorn` workers

## Future Work

- Tree-sitter call graph for intra-file function-call edges
- Multi-repo support with cross-repo symbol resolution
- Background ingest jobs with progress polling, so `/ingest` on a large repo doesn't hold the
  HTTP connection open for the whole run
