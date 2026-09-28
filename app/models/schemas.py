"""Result types for impact analysis and symbol lookup."""
from typing import Optional

from pydantic import BaseModel


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
