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
