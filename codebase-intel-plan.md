# codebase-intel — Complete Build Plan

> An AI-powered codebase intelligence tool combining semantic search, symbol extraction,
> dependency graphs, and impact analysis for developer Q&A over any repository.
> This document is the canonical reference for Claude Code to follow throughout the project.

---

## Project Goal

Build a tool that ingests a code repository and lets a developer ask:

- *Where is `submit_order()` defined?*
- *What files import `auth.py`?*
- *What would break if I change `portfolio.py`?*
- *How does data flow from the API layer to the database?*

The key distinction: this is **developer infrastructure**, not "chat with your codebase."
The answers must be grounded in retrieved evidence — no LLM hallucinations about code structure.

---

## Final Product Scope

**Must ship:**
- Repo ingestion pipeline with file filtering and language detection
- Code chunking with overlap and per-chunk metadata
- Embedding pipeline + FAISS vector index
- Symbol extraction (functions, classes, imports) for Python and TypeScript
- File-level dependency graph (NetworkX)
- Impact analysis engine (graph traversal + semantic similarity)
- FastAPI backend with 5 endpoints
- CLI scripts for ingestion and querying
- README with demo, architecture diagram, and example output

**Do not build:**
- Full IDE extension
- Live GitHub syncing
- Multi-user auth
- Perfect call-graph accuracy (this requires full compiler instrumentation — out of scope)
- Neo4j, Qdrant, or other heavy infrastructure (FAISS + NetworkX is correct for V1)
- Frontend (CLI-first; tiny React UI is optional only if core is done)

**Language support:** Python + TypeScript only. Do not attempt Go, Java, Rust, etc. in V1.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI | async, clean schema, Pydantic |
| Embeddings | `text-embedding-3-small` (OpenAI) or `sentence-transformers/all-MiniLM-L6-v2` (local) | both work; local avoids API cost |
| Vector store | FAISS (flat index) | simple, no server, fast enough for repo-scale |
| Graph | NetworkX | sufficient for file-level dependency DAG |
| Parsing | `tree-sitter` (primary), regex fallback | tree-sitter gives real ASTs for Python + TS |
| LLM | Claude via Anthropic SDK (`claude-sonnet-4-20250514`) | grounded Q&A |
| Metadata | SQLite via `sqlite3` stdlib | no ORM needed |
| Storage | Local filesystem | `data/indexes/`, `data/metadata/` |
| Python version | 3.11+ | |

---

## Repository Structure

```
codebase-intel/
├── README.md
├── requirements.txt
├── .env.example              # ANTHROPIC_API_KEY, OPENAI_API_KEY (if using OpenAI embeddings)
├── .gitignore
├── app/
│   ├── main.py               # FastAPI app entrypoint
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_ingest.py  # POST /ingest
│   │   ├── routes_search.py  # POST /search, GET /definition
│   │   ├── routes_impact.py  # POST /impact
│   │   └── routes_ask.py     # POST /ask
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ingest.py         # repo walking, file filtering
│   │   ├── chunking.py       # code chunking with metadata
│   │   ├── embeddings.py     # embed chunks, load/save model
│   │   ├── search.py         # FAISS query + result formatting
│   │   ├── symbols.py        # tree-sitter symbol extraction
│   │   ├── graph.py          # NetworkX dependency graph
│   │   ├── impact.py         # impact analysis engine
│   │   └── answer.py         # grounded LLM answer generation
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        # Pydantic request/response models
│   └── storage/
│       ├── __init__.py
│       ├── faiss_store.py    # FAISS index wrapper
│       └── metadata_store.py # SQLite metadata wrapper
├── scripts/
│   ├── ingest_repo.py        # CLI: python scripts/ingest_repo.py --repo <path>
│   └── demo_query.py         # CLI: python scripts/demo_query.py "where is auth handled?"
├── data/
│   ├── indexes/              # FAISS index files (.index)
│   └── metadata/             # SQLite .db files, graph pickles
├── examples/
│   └── demo_questions.md     # curated demo Q&A for README
└── tests/
    ├── test_ingest.py
    ├── test_symbols.py
    ├── test_graph.py
    ├── test_impact.py
    └── test_search.py
```

---

## Data Models (`app/models/schemas.py`)

Define all Pydantic models here. Every API request and response uses these.

```python
from pydantic import BaseModel
from typing import Optional

# --- Shared ---

class ChunkMetadata(BaseModel):
    chunk_id: str           # uuid
    file_path: str          # relative to repo root
    language: str           # "python" | "typescript" | "unknown"
    start_line: int
    end_line: int
    symbols: list[str]      # function/class names found in this chunk
    imports: list[str]      # import targets found in this chunk
    content: str            # raw source text of chunk

# --- Ingest ---

class IngestRequest(BaseModel):
    repo_path: str
    repo_id: str            # user-chosen name, e.g. "lob-sim-cpp"

class IngestResponse(BaseModel):
    repo_id: str
    files_indexed: int
    chunks_indexed: int
    symbols_extracted: int
    edges_in_graph: int

# --- Search ---

class SearchRequest(BaseModel):
    repo_id: str
    query: str
    top_k: int = 10

class SearchResult(BaseModel):
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    score: float            # cosine similarity
    snippet: str            # first 300 chars of chunk

class SearchResponse(BaseModel):
    results: list[SearchResult]

# --- Definition ---

class DefinitionResponse(BaseModel):
    symbol: str
    defining_file: str
    start_line: Optional[int]
    references: list[str]   # files that reference this symbol

# --- Impact ---

class ImpactRequest(BaseModel):
    repo_id: str
    target: str             # file path or symbol name
    depth: int = 3          # graph traversal depth

class ImpactedFile(BaseModel):
    file_path: str
    reason: str             # "direct import" | "symbol reference" | "semantic similarity"
    confidence: float       # 0.0–1.0
    depth: int              # hops from target in graph

class ImpactResponse(BaseModel):
    target: str
    high_confidence: list[ImpactedFile]
    medium_confidence: list[ImpactedFile]
    related: list[ImpactedFile]

# --- Ask ---

class AskRequest(BaseModel):
    repo_id: str
    question: str
    top_k: int = 8

class Citation(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    relevance: str          # one sentence explaining why this chunk was used

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    uncertainty: Optional[str]  # null if confident; else a caveat
```

---

## Core Module Specs

### `app/core/ingest.py`

**Responsibility:** Walk a repo directory, filter to relevant files, return a list of file paths.

```python
SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", "__pycache__",
    ".venv", "venv", ".env", "coverage", ".next", ".nuxt",
    "target", "out", "bin", "obj", ".idea", ".vscode"
}

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf",
    ".zip", ".tar", ".gz", ".lock", ".sum", ".whl", ".egg"
}

INCLUDE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".md", ".txt", ".yaml", ".yml", ".toml", ".json"
}
```

Key function signatures:
```python
def walk_repo(repo_path: str) -> list[str]:
    """Return list of absolute file paths to index."""

def detect_language(file_path: str) -> str:
    """Return 'python', 'typescript', 'javascript', 'markdown', or 'unknown'."""

def load_file(file_path: str) -> Optional[str]:
    """Read file contents. Return None if binary or unreadable."""
```

---

### `app/core/chunking.py`

**Responsibility:** Split a file into overlapping chunks with metadata attached.

**Chunking strategy:**
- Target chunk size: **400 tokens** (approximate by character count: ~1600 chars)
- Overlap: **50 tokens** (~200 chars) between adjacent chunks
- Never split mid-line: find the nearest newline within the target range
- Each chunk preserves its `file_path`, `start_line`, `end_line`, detected `symbols`, and detected `imports`

```python
def chunk_file(
    content: str,
    file_path: str,
    language: str,
    chunk_size_chars: int = 1600,
    overlap_chars: int = 200,
) -> list[ChunkMetadata]:
    """Split file content into overlapping chunks with metadata."""
```

**Symbol pre-scan:** Before chunking, run a fast regex pass to collect all symbol and import names
present in the file. Attach these to every chunk from that file. The tree-sitter extraction
(in `symbols.py`) will refine this per-chunk later during indexing.

---

### `app/core/embeddings.py`

**Responsibility:** Embed chunks, manage the embedding model, load/save index.

**Model selection (in order of preference):**
1. `sentence-transformers/all-MiniLM-L6-v2` — local, no API key, 384-dim, fast
2. `text-embedding-3-small` (OpenAI) — better quality, costs API credits

Use an environment variable `EMBEDDING_BACKEND=local|openai` to switch.

```python
def embed_texts(texts: list[str]) -> np.ndarray:
    """Return (N, D) float32 numpy array of embeddings."""

def embed_query(query: str) -> np.ndarray:
    """Return (1, D) float32 numpy array."""
```

Batch size: 64 chunks per embedding call. Show a progress bar via `tqdm`.

---

### `app/storage/faiss_store.py`

**Responsibility:** FAISS flat L2 index wrapper with save/load.

```python
class FAISSStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine if normalized
        self.id_map: list[str] = []          # position → chunk_id

    def add(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        """Normalize embeddings, add to index, record chunk_ids."""

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) pairs."""

    def save(self, path: str) -> None:
        """Save index + id_map to disk."""

    def load(self, path: str) -> None:
        """Load index + id_map from disk."""
```

**Important:** Normalize all embeddings before adding to `IndexFlatIP` so inner product = cosine similarity.
Do this consistently for both indexed vectors and query vectors.

---

### `app/storage/metadata_store.py`

**Responsibility:** SQLite store for chunk metadata, symbols, and file-to-file edges.

**Schema (create on first use):**

```sql
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    repo_id     TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    language    TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    symbols     TEXT NOT NULL,   -- JSON array of strings
    imports     TEXT NOT NULL,   -- JSON array of strings
    content     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS symbols (
    symbol_name TEXT NOT NULL,
    repo_id     TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    start_line  INTEGER,
    kind        TEXT,            -- "function" | "class" | "method" | "variable"
    PRIMARY KEY (symbol_name, repo_id, file_path)
);

CREATE TABLE IF NOT EXISTS edges (
    repo_id     TEXT NOT NULL,
    source_file TEXT NOT NULL,
    target_file TEXT NOT NULL,
    edge_type   TEXT NOT NULL,   -- "import" | "reference"
    PRIMARY KEY (repo_id, source_file, target_file, edge_type)
);
```

Key methods:
```python
class MetadataStore:
    def upsert_chunk(self, chunk: ChunkMetadata, repo_id: str) -> None
    def get_chunk(self, chunk_id: str) -> Optional[ChunkMetadata]
    def get_chunks_by_file(self, repo_id: str, file_path: str) -> list[ChunkMetadata]
    def upsert_symbol(self, name: str, repo_id: str, file_path: str, line: int, kind: str) -> None
    def find_symbol(self, repo_id: str, name: str) -> list[dict]
    def upsert_edge(self, repo_id: str, source: str, target: str, edge_type: str) -> None
    def get_edges_from(self, repo_id: str, file_path: str) -> list[dict]
    def get_edges_to(self, repo_id: str, file_path: str) -> list[dict]
```

---

### `app/core/symbols.py`

**Responsibility:** Extract symbols and imports from source files using tree-sitter.

**Setup:**
```python
# pip install tree-sitter tree-sitter-python tree-sitter-typescript
from tree_sitter_languages import get_language, get_parser
```

Use `tree-sitter-languages` package which bundles pre-compiled grammars — no manual compilation needed.

**For Python files, extract:**
- Function definitions: `function_definition` nodes → name + start line
- Class definitions: `class_definition` nodes → name + start line
- Import statements: `import_statement`, `import_from_statement` → module names

**For TypeScript/JavaScript files, extract:**
- Function declarations, arrow functions assigned to variables
- Class declarations
- Import declarations → module specifiers

**Output per file:**
```python
@dataclass
class SymbolInfo:
    name: str
    kind: str        # "function" | "class" | "method"
    start_line: int
    file_path: str

@dataclass
class ImportInfo:
    source_file: str
    imported_module: str     # e.g. "auth", "./utils", "fastapi"
    is_relative: bool

def extract_symbols(content: str, file_path: str, language: str) -> list[SymbolInfo]
def extract_imports(content: str, file_path: str, language: str) -> list[ImportInfo]
```

**Fallback:** If tree-sitter fails on a file, fall back to regex patterns:
- Python functions: `r'^def\s+(\w+)\s*\('`
- Python classes: `r'^class\s+(\w+)'`
- Python imports: `r'^(?:import|from)\s+([\w.]+)'`
- TS/JS functions: `r'(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:async\s*)?\(|[<(])'`

---

### `app/core/graph.py`

**Responsibility:** Build and query the file-level dependency graph.

```python
import networkx as nx

class DependencyGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    def add_file(self, file_path: str) -> None:
        """Add a node for this file."""

    def add_import_edge(self, source_file: str, target_file: str) -> None:
        """source_file imports target_file. Edge direction: source → target."""

    def dependents_of(self, file_path: str, depth: int = 3) -> list[dict]:
        """Return files that DEPEND ON file_path (reverse edges), up to `depth` hops.
        Returns: [{"file": str, "depth": int}]"""

    def dependencies_of(self, file_path: str, depth: int = 3) -> list[dict]:
        """Return files that file_path DEPENDS ON (forward edges), up to `depth` hops."""

    def files_referencing_symbol(self, symbol: str, metadata_store: MetadataStore, repo_id: str) -> list[str]:
        """Return file paths that contain this symbol name in their chunks."""

    def save(self, path: str) -> None:
        """Pickle the graph to disk."""

    def load(self, path: str) -> None:
        """Load graph from disk."""
```

**Import resolution:** When a Python file has `from auth import validate_token`,
resolve `auth` to an actual file path in the repo using:
1. Check if `auth.py` or `auth/__init__.py` exists relative to the source file
2. Check repo root
3. If not found in repo, mark as external and skip the edge

For TypeScript, resolve relative imports (`./utils`, `../services/db`) similarly.
Skip node_modules imports (no edge created).

---

### `app/core/impact.py`

**Responsibility:** The core differentiator. Rank files by likelihood of being affected by a change.

```python
def analyze_impact(
    target: str,            # file path or symbol name
    repo_id: str,
    graph: DependencyGraph,
    faiss_store: FAISSStore,
    metadata_store: MetadataStore,
    embeddings_module,
    depth: int = 3,
) -> ImpactResponse:
```

**Algorithm (combine three signals):**

**Signal 1 — Graph traversal (highest confidence):**
- If target is a file: get all files that directly or transitively import it (reverse BFS up to `depth`)
- If target is a symbol: get the defining file, then apply file-level graph traversal
- Assign confidence: depth=1 → 0.95, depth=2 → 0.75, depth=3 → 0.50
- Label reason: `"direct import"` or `"transitive import (N hops)"`

**Signal 2 — Symbol reference search (medium confidence):**
- If target is a symbol name: query `metadata_store.find_symbol()` for all files referencing it
- Exclude the defining file
- Confidence: 0.70
- Label reason: `"references symbol"`

**Signal 3 — Semantic similarity (lower confidence):**
- Embed the target's name or file path as a query string
- Run FAISS search, top 5 results
- Exclude files already in signals 1 and 2
- Confidence: 0.35
- Label reason: `"semantically related"`

**Merge and bucket:**
```
high_confidence:   confidence >= 0.7
medium_confidence: confidence >= 0.4
related:           confidence < 0.4
```

Within each bucket, sort by confidence descending. Deduplicate by file path, keeping highest confidence.

---

### `app/core/answer.py`

**Responsibility:** Grounded LLM answer generation over retrieved chunks.

```python
def generate_answer(
    question: str,
    retrieved_chunks: list[ChunkMetadata],
    repo_id: str,
) -> AskResponse:
```

**Prompt construction:**

```python
SYSTEM_PROMPT = """You are a codebase assistant. You answer questions about source code
using ONLY the provided code excerpts. You must cite the specific files and line ranges
that support your answer. If the evidence is insufficient, say so explicitly.
Never invent code, function names, or behavior not present in the excerpts."""

def build_prompt(question: str, chunks: list[ChunkMetadata]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        block = f"""
[{i+1}] File: {chunk.file_path} (lines {chunk.start_line}–{chunk.end_line})
```
{chunk.content[:800]}
```"""
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)
    return f"""Code excerpts from the repository:\n{context}\n\nQuestion: {question}\n\nAnswer based strictly on the excerpts above. Cite by [N] number."""
```

**Response parsing:**
The LLM returns a natural language answer with `[N]` citations. Parse these to build the
`citations` list in `AskResponse`. Map `[N]` → `chunks[N-1]`.

**Model:** Use `claude-sonnet-4-20250514` with `max_tokens=1000`.

---

## API Endpoints (`app/main.py` + route files)

### `POST /ingest`

Request: `IngestRequest`

Steps:
1. Walk repo → list of file paths
2. For each file: load content, detect language, chunk it
3. Extract symbols and imports per file → store in metadata DB
4. Build dependency graph → add edges → save graph
5. Embed all chunks in batches → add to FAISS index
6. Store all chunk metadata in SQLite

Response: `IngestResponse` with counts.

Save artifacts:
- FAISS index: `data/indexes/{repo_id}.index`
- FAISS id_map: `data/indexes/{repo_id}.idmap.json`
- Graph: `data/metadata/{repo_id}.graph.pkl`
- SQLite DB: `data/metadata/{repo_id}.db`

---

### `POST /search`

Request: `SearchRequest`

Steps:
1. Load FAISS index for `repo_id`
2. Embed query
3. Search top_k chunks
4. Load full chunk metadata from SQLite for each result
5. Return ranked results

Response: `SearchResponse`

---

### `GET /definition?repo_id=X&symbol=Y`

Steps:
1. Query `metadata_store.find_symbol(repo_id, symbol)`
2. Find the defining entry (kind = "function" or "class")
3. Query edges to find all files that reference this symbol name in their chunk text
4. Return `DefinitionResponse`

---

### `POST /impact`

Request: `ImpactRequest`

Steps:
1. Load graph and metadata store for `repo_id`
2. Load FAISS store
3. Run `analyze_impact()`
4. Return `ImpactResponse`

---

### `POST /ask`

Request: `AskRequest`

Steps:
1. Run semantic search for top_k chunks
2. Optionally augment: if question mentions a symbol name, also fetch chunks from its defining file
3. Call `generate_answer()`
4. Return `AskResponse`

---

## CLI Scripts

### `scripts/ingest_repo.py`

```bash
python scripts/ingest_repo.py --repo /path/to/repo --repo-id my-project
```

Calls the ingest logic directly (not via HTTP). Prints progress. At the end, prints:
```
Indexed 47 files, 312 chunks, 284 symbols, 89 graph edges.
Saved to data/indexes/my-project.index
```

### `scripts/demo_query.py`

```bash
python scripts/demo_query.py --repo-id my-project "where is authentication handled?"
python scripts/demo_query.py --repo-id my-project --mode definition --symbol submit_order
python scripts/demo_query.py --repo-id my-project --mode impact --target auth.py
```

Modes: `search` (default), `definition`, `impact`, `ask`

---

## State Management (Important)

Each repo has its own isolated state identified by `repo_id`. All artifacts are namespaced:

```
data/indexes/{repo_id}.index
data/indexes/{repo_id}.idmap.json
data/metadata/{repo_id}.db
data/metadata/{repo_id}.graph.pkl
```

The FastAPI app loads these lazily on first request for a given `repo_id` and caches them in memory.
Use a module-level dict: `_loaded_repos: dict[str, RepoState]` where `RepoState` holds the
FAISSStore, MetadataStore, and DependencyGraph for that repo.

---

## Unit Tests

### `tests/test_ingest.py`
- Walk a temp directory, verify skip dirs work
- Language detection for `.py`, `.ts`, `.js`, `.md`
- File load returns None for binary files

### `tests/test_symbols.py`
- Extract functions from a sample Python string (inline fixture)
- Extract classes from a sample Python string
- Extract imports from a sample Python string
- Extract function from a sample TypeScript string
- Regex fallback produces correct results when tree-sitter unavailable

### `tests/test_graph.py`
- Add files and edges, verify `dependents_of()` returns correct results
- Verify depth limiting works (depth=1 only returns direct dependents)
- Verify deduplication across multiple paths

### `tests/test_impact.py`
- Mock FAISSStore and MetadataStore
- Verify high/medium/related bucketing
- Verify symbol-name target correctly resolves through defining file

### `tests/test_search.py`
- Add known embeddings to FAISSStore, search returns correct chunk_id
- Verify normalization (inner product = 1.0 for self-search)

---

## `requirements.txt`

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
numpy>=1.26.0
faiss-cpu>=1.8.0
networkx>=3.3
sentence-transformers>=2.7.0   # for local embeddings
tree-sitter-languages>=1.10.2  # pre-built grammars
anthropic>=0.25.0
openai>=1.0.0                  # optional, for OpenAI embeddings
python-dotenv>=1.0.0
tqdm>=4.66.0
pytest>=8.0.0
httpx>=0.27.0                  # for testing FastAPI
```

---

## `.env.example`

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...           # optional
EMBEDDING_BACKEND=local         # "local" or "openai"
```

---

## Demo Repo Choice

Pick **one** well-known open-source Python repo to demo against. Best choices:

- **FastAPI itself** (`tiangolo/fastapi`) — meta and impressive
- **Your own lob-sim-cpp** — but it's C++, which you don't support in V1
- **Requests library** (`psf/requests`) — small, clean, everyone knows it
- **Flask** — similar reasons

Recommended: `requests`. It's small enough to index in seconds, well-structured, and everyone knows what it does — so demo answers are immediately verifiable.

Curate 5–6 demo Q&A pairs in `examples/demo_questions.md`:

```markdown
## Demo Questions

**Q: Where is SSL certificate verification handled?**
A: [actual answer from your tool with citations]

**Q: What files would be affected if I changed `adapters/http.py`?**
A: [actual impact analysis output]

**Q: What happens after `Session.send()` is called?**
A: [actual answer]
```

Run these against your tool and paste the real output. Do not fake this.

---

## README Structure

```markdown
# codebase-intel

AI-powered codebase intelligence: semantic search, symbol lookup, dependency-aware impact analysis,
and grounded repository Q&A.

## What it does
## Architecture
[diagram: Ingest → Chunk → Embed → FAISS / Symbol Extract → NetworkX / Impact → FastAPI → Answer]

## Quickstart
### Install
### Ingest a repo
### Query

## API Reference
### POST /ingest
### POST /search
### GET /definition
### POST /impact
### POST /ask

## Example Output
[paste real terminal output from demo questions]

## Design Decisions
- Why FAISS over a hosted vector DB: simplicity, no infra, sufficient for repo scale
- Why NetworkX over Neo4j: same reason
- Why answers are grounded: LLM hallucination about code is worse than "I don't know"
- Why Python + TypeScript only: depth over breadth

## Limitations
- L2 data only (file-level graph, not call-level)
- No live repo sync
- Import resolution is best-effort for relative paths
- Answer quality depends on chunk retrieval quality

## Future Work
- Diff-aware impact analysis (changed lines → affected symbols)
- Tree-sitter call graph for intra-file resolution
- Multi-repo support
```

---

## Build Order for Claude Code

Implement in this exact order — each step is independently testable:

1. `app/models/schemas.py` — all Pydantic models
2. `app/storage/metadata_store.py` + `tests/test_ingest.py` (partial)
3. `app/core/ingest.py` — walk + filter + detect language
4. `app/core/chunking.py` — chunker with metadata
5. `app/core/symbols.py` + `tests/test_symbols.py` — tree-sitter extraction
6. `app/core/graph.py` + `tests/test_graph.py` — NetworkX dependency graph
7. `app/core/embeddings.py` — embedding model wrapper (local first)
8. `app/storage/faiss_store.py` + `tests/test_search.py` — FAISS wrapper
9. `scripts/ingest_repo.py` — wire everything together, test on a real repo
10. `app/core/search.py` — semantic search query pipeline
11. `scripts/demo_query.py --mode search` — verify search works end-to-end
12. `app/core/impact.py` + `tests/test_impact.py` — impact analysis
13. `scripts/demo_query.py --mode impact` — verify impact works
14. `app/core/answer.py` — grounded LLM answer generation
15. `app/main.py` + all route files — FastAPI app
16. `examples/demo_questions.md` — run real demos, paste real output
17. `README.md` — write after you have real demo output

---

## Interview Pitch

> I built a codebase intelligence tool that combines semantic search over embedded code chunks
> with tree-sitter symbol extraction and a NetworkX dependency graph to support impact analysis,
> definition lookup, and grounded repository Q&A. The key design choice was to treat it as
> developer infrastructure rather than a chatbot — answers always cite the specific files and
> line ranges they're based on, and the impact engine ranks files by confidence using graph
> traversal, symbol references, and semantic similarity as independent signals.
