"""Symbol definition lookup, shared by GET /definition, /impact and the CLI."""
from pathlib import PurePosixPath
from typing import Optional

from app.models.schemas import DefinitionResponse, ReferenceLocation, SymbolLocation
from app.storage.metadata_store import MetadataStore

DEFINITION_KINDS = ("class", "function", "method", "interface", "type", "enum")

_TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs"}


def is_test_path(file_path: str) -> bool:
    path = PurePosixPath(file_path)
    name = path.name
    return (
        any(part in _TEST_DIRS for part in path.parts[:-1])
        or name.startswith("test_")
        or name == "conftest.py"
        or name.endswith(("_test.py", ".test.ts", ".test.tsx", ".test.js", ".spec.ts", ".spec.tsx", ".spec.js"))
    )


def rank_definitions(symbol: str, rows: list[dict]) -> list[dict]:
    """Order candidate definitions best-first.

    1. real definitions (class/function/...) before anything else
    2. source files before tests: a top-level `prepare_url` fixture in
       conftest.py shouldn't win over PreparedRequest.prepare_url
    3. an exact qualified-name match ("send" → a top-level `send`) before
       nested ones (`HTTPAdapter.send`)
    4. then file path and line, for determinism
    """
    return sorted(
        rows,
        key=lambda r: (
            r.get("kind") not in DEFINITION_KINDS,
            is_test_path(r["file_path"]),
            r.get("qualified_name", r.get("symbol_name")) != symbol,
            r["file_path"],
            r.get("start_line") or 0,
        ),
    )


def best_definition(symbol: str, metadata_store: MetadataStore, repo_id: str) -> Optional[dict]:
    ranked = rank_definitions(symbol, metadata_store.find_symbol(repo_id, symbol))
    return ranked[0] if ranked else None


def lookup_definition(
    symbol: str, metadata_store: MetadataStore, repo_id: str
) -> Optional[DefinitionResponse]:
    """Resolve `symbol` (bare "send" or qualified "HTTPAdapter.send") to its
    best definition, the other candidates, and every usage in the repo."""
    ranked = rank_definitions(symbol, metadata_store.find_symbol(repo_id, symbol))
    if not ranked:
        return None
    best = ranked[0]

    # Usages are recorded by identifier, so a dotted query matches on its last
    # component (`Session.send` → every `.send` usage). The definition site
    # itself is never a usage; drop any hit on the definition's own line.
    refs = [
        r for r in metadata_store.find_references(repo_id, symbol)
        if not (r["file_path"] == best["file_path"] and r["line"] == best.get("start_line"))
    ]
    return DefinitionResponse(
        symbol=symbol,
        qualified_name=best.get("qualified_name") or best.get("symbol_name") or symbol,
        kind=best.get("kind"),
        defining_file=best["file_path"],
        start_line=best.get("start_line"),
        end_line=best.get("end_line"),
        references=sorted({r["file_path"] for r in refs}),
        reference_locations=[ReferenceLocation(file_path=r["file_path"], line=r["line"]) for r in refs],
        other_definitions=[
            SymbolLocation(
                qualified_name=r.get("qualified_name") or r.get("symbol_name") or symbol,
                kind=r.get("kind"),
                file_path=r["file_path"],
                start_line=r.get("start_line"),
            )
            for r in ranked[1:]
        ],
    )
