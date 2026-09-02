from pydantic import BaseModel, field_validator
from typing import Optional

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


# --- Search ---

class SearchRequest(_RepoScoped):
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

class ImpactRequest(_RepoScoped):
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


# --- Ask ---

class AskRequest(_RepoScoped):
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
    uncertainty: Optional[str] = None  # null if confident; else a caveat


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
