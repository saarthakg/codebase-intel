# codebase-intel

AI-powered codebase intelligence: semantic search, symbol lookup, dependency-aware impact analysis, and grounded repository Q&A.

---

## What it does

Ask developer questions about any Python or TypeScript codebase:

- **"Where is `submit_order()` defined?"** → Symbol definition lookup with file + line
- **"What files import `auth.py`?"** → Dependency graph traversal
- **"What would break if I change `adapters.py`?"** → Multi-signal impact analysis
- **"How does data flow from the API layer to the database?"** → Grounded LLM answer over retrieved code chunks

Answers are always grounded in retrieved code — no hallucinated function names or invented behavior.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Ingest Pipeline                      │
│                                                             │
│  walk_repo → load_file → detect_language → chunk_file      │
│       │                                        │            │
│  extract_symbols/imports              embed_texts (local    │
│       │                               sentence-transformers │
│       ▼                               or OpenAI)            │
│  MetadataStore (SQLite)                    │                │
│  DependencyGraph (NetworkX)           FAISSStore            │
│       │                               (IndexFlatIP)         │
│       ▼                                    │                │
│  data/metadata/{repo_id}.db          data/indexes/         │
│  data/metadata/{repo_id}.graph.pkl   {repo_id}.index       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       Query Pipeline                        │
│                                                             │
│  POST /search   → embed query → FAISS search → ranked chunks│
│  GET  /definition → SQLite symbol lookup → graph references │
│  POST /impact   → graph BFS + symbol refs + FAISS (3 signals│
│  POST /ask      → search → LLM (grounded answer + cites)   │
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

### Ingest a repo

```bash
python scripts/ingest_repo.py --repo /path/to/your/repo --repo-id my-project
# Indexed 46 files, 378 chunks, 757 symbols, 62 graph edges.
```

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
{"repo_id": "my-project", "files_indexed": 46, "chunks_indexed": 378,
 "symbols_extracted": 757, "edges_in_graph": 62}
```

### `POST /search`

Semantic search over embedded code chunks.

```json
{"repo_id": "my-project", "query": "SSL certificate verification", "top_k": 10}
```

### `GET /definition?repo_id=X&symbol=Y`

Symbol definition lookup with cross-reference list.

```json
{"symbol": "HTTPAdapter", "defining_file": "src/requests/adapters.py",
 "start_line": 144, "references": ["src/requests/sessions.py"]}
```

### `POST /impact`

Multi-signal impact analysis: which files are likely affected by changing a target?

```json
{"repo_id": "my-project", "target": "src/requests/adapters.py", "depth": 3}
```

Returns `high_confidence` (graph traversal), `medium_confidence` (symbol refs), and `related` (semantic similarity) buckets.

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

Demoed against `psf/requests` — see [`examples/demo_questions.md`](examples/demo_questions.md) for full real output.

**Impact analysis of `adapters.py`:**
```
HIGH CONFIDENCE:
  [0.95] src/requests/sessions.py  — direct import
  [0.75] src/requests/__init__.py  — transitive import (2 hops)

MEDIUM CONFIDENCE:
  [0.50] src/requests/utils.py  — transitive import (3 hops)
  [0.50] src/requests/api.py    — transitive import (3 hops)

RELATED:
  [0.35] tests/test_adapters.py  — semantically related
```

**Search: "where is SSL verification handled?":**
```
[1] src/requests/certs.py        lines 1–19    score=0.415
[2] src/requests/adapters.py     lines 289–334 score=0.381
[3] src/requests/adapters.py     lines 395–425 score=0.410
```

**Grounded Q&A: "Where is SSL certificate verification handled?"**
```
A: SSL certificate verification is handled primarily in src/requests/adapters.py
through the processing of the verify parameter:

- Logic for CA Bundles: The HTTPAdapter determines the CA bundle location.
  If verify=True, it defaults to DEFAULT_CA_BUNDLE_PATH; if verify is a
  string, it uses that as the bundle path [5].
- Default CA Source: The default CA bundle is provided by certifi,
  defined in src/requests/certs.py [3].
- Connection Configuration: For HTTPS, if verification is enabled,
  the adapter sets conn.cert_reqs = "CERT_REQUIRED" [5].

Citations:
  src/requests/certs.py       lines 1–19
  src/requests/adapters.py    lines 289–334
  src/requests/adapters.py    lines 395–425
```

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

**Why Python + TypeScript only:** Depth over breadth. Two languages done well (real ASTs
via tree-sitter, proper import resolution) beats six languages done poorly.

---

## Running Tests

```bash
pytest tests/ -v
# 52 passed
```

---

## Limitations

- File-level dependency graph, not call-level (no intra-function call edges)
- No live repo sync — re-run `ingest` after changes
- Import resolution is best-effort for relative paths; external packages are excluded
- Answer quality depends on whether the relevant code was retrieved in the top-k chunks
- `all-MiniLM-L6-v2` is 384-dimensional and fast but not state-of-the-art; swap to
  `text-embedding-3-small` via `EMBEDDING_BACKEND=openai` for better retrieval

## Future Work

- Diff-aware impact analysis: changed lines → affected symbols
- Tree-sitter call graph for intra-file function-call edges
- Multi-repo support with cross-repo symbol resolution
- Streaming responses for `/ask`
