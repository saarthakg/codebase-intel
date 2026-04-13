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
    repo_id: str            # user-chosen name, e.g. "my-project"


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
    uncertainty: Optional[str] = None  # null if confident; else a caveat
