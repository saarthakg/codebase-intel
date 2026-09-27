from pydantic import BaseModel, field_validator
from typing import Literal, Optional

from app.core.validation import validate_repo_id


class _RepoScoped(BaseModel):
    """Base for any request keyed by repo_id — enforces the filesystem-safe format."""
    repo_id: str

    @field_validator("repo_id")
    @classmethod
    def _check_repo_id(cls, v: str) -> str:
        return validate_repo_id(v)


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

class IngestRequest(_RepoScoped):
    """repo_id: user-chosen name, e.g. 'my-project' (inherited from _RepoScoped)."""
    repo_path: str


class IngestResponse(BaseModel):
    repo_id: str
    files_indexed: int
    chunks_indexed: int
    symbols_extracted: int
    edges_in_graph: int
    files_skipped: dict[str, int] = {}   # reason ("too_large", "minified", "lockfile", ...) → count
    chunks_embedded: int = 0             # newly embedded this run
    chunks_reused: int = 0               # unchanged since the last ingest; vectors reused


# --- Search ---

class SearchRequest(_RepoScoped):
    query: str
    top_k: int = 10
    # "hybrid" (default): semantic + keyword + exact-symbol, rank-fused.
    # "semantic": embeddings only. "keyword": BM25 only.
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"


class SearchResult(BaseModel):
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    score: float            # cosine (semantic), BM25 (keyword) or fused RRF score (hybrid)
    snippet: str            # first 300 chars of chunk


class SearchResponse(BaseModel):
    results: list[SearchResult]


# --- Definition ---

class SymbolLocation(BaseModel):
    qualified_name: str     # e.g. "HTTPAdapter.send"
    kind: Optional[str]
    file_path: str
    start_line: Optional[int]


class ReferenceLocation(BaseModel):
    file_path: str
    line: int


class DefinitionResponse(BaseModel):
    symbol: str
    qualified_name: Optional[str] = None
    kind: Optional[str] = None
    defining_file: str
    start_line: Optional[int]
    end_line: Optional[int] = None
    references: list[str]   # files that use this symbol (matched by identifier name)
    reference_locations: list[ReferenceLocation] = []
    # Other symbols matching the query, e.g. every other class's `send` method
    other_definitions: list[SymbolLocation] = []


# --- Impact ---

class ImpactRequest(_RepoScoped):
    target: str             # file path or symbol name
    depth: int = 3          # graph traversal depth


class ImpactedFile(BaseModel):
    file_path: str
    reason: str             # e.g. "direct import", "test named for this file", "changed together in 5 of 12 commits"
    confidence: float       # 0.0–1.0
    depth: int              # hops from target in graph


class ImpactResponse(BaseModel):
    target: str
    high_confidence: list[ImpactedFile]
    medium_confidence: list[ImpactedFile]
    related: list[ImpactedFile]
    # The test files among the above, best-first: what to run for this change.
    tests: list[ImpactedFile] = []


class ImpactBatchRequest(_RepoScoped):
    targets: list[str]      # e.g. changed files from `git diff --name-only`
    depth: int = 3


class BatchImpactedFile(BaseModel):
    file_path: str
    reason: str
    confidence: float
    depth: int
    triggered_by: list[str]  # which of the requested targets caused this impact


class ImpactBatchResponse(BaseModel):
    targets: list[str]
    high_confidence: list[BatchImpactedFile]
    medium_confidence: list[BatchImpactedFile]
    related: list[BatchImpactedFile]
    tests: list[BatchImpactedFile] = []


class ImpactDiffRequest(_RepoScoped):
    diff: str               # unified diff, e.g. the output of `git diff`
    depth: int = 3


class ChangedSymbol(BaseModel):
    file_path: str
    qualified_name: str     # innermost symbol the diff touched, e.g. "HTTPAdapter.cert_verify"
    used_in: list[str]      # files (depending on file_path) that use this symbol


class DiffImpactResponse(BaseModel):
    targets: list[str] = []                     # changed, indexed files
    changed_symbols: list[ChangedSymbol] = []
    high_confidence: list[BatchImpactedFile] = []
    medium_confidence: list[BatchImpactedFile] = []
    related: list[BatchImpactedFile] = []
    tests: list[BatchImpactedFile] = []
    unindexed_files: list[str] = []             # in the diff but not in the index (new, or not ingested)


# --- Ask ---

class AskRequest(_RepoScoped):
    question: str
    top_k: int = 8
    # Serve a cached answer when the same question hits the same code with the
    # same model. Set false to force a fresh LLM call.
    use_cache: bool = True


class Citation(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    relevance: str          # one sentence explaining why this chunk was used


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    uncertainty: Optional[str] = None  # null if confident; else a caveat
    # Code names / file paths in the answer found neither in the excerpts nor in
    # the repo index — likely invented.
    unverified_mentions: list[str] = []
    backend: Optional[str] = None
    model: Optional[str] = None
    cached: bool = False               # true when served from the answer cache (no LLM call)
    excerpts_used: int = 0
    excerpts_omitted: int = 0          # retrieved but dropped to stay within the context budget
    context_chars: int = 0


# --- Repos ---

class RepoInfo(BaseModel):
    repo_id: str
    files_indexed: int
    chunks_indexed: int
    symbols_extracted: int
    edges_in_graph: int
    embedding_backend: Optional[str] = None
    ingested_at: Optional[str] = None  # ISO-8601 UTC timestamp


class RepoListResponse(BaseModel):
    repos: list[RepoInfo]


class DeleteRepoResponse(BaseModel):
    repo_id: str
    deleted: bool
