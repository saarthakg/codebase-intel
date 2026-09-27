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


_TEST_NAME_RE = __import__("re").compile(
    r"^(?P<prefix>test_)?(?P<stem>.+?)(?P<suffix>_test|_tests|\.test|\.spec|Test|Tests)?$"
)


def tested_module_stem(test_path: str) -> Optional[str]:
    """For a test file, the module name it's named after: 'tests/test_utils.py' →
    'utils', 'src/foo.spec.ts' → 'foo', 'FooTest.java' → 'Foo'. None otherwise."""
    if not is_test_path(test_path):
        return None
    name = PurePosixPath(test_path).name
    for ext in (".py", ".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    m = _TEST_NAME_RE.match(name)
    if not (m.group("prefix") or m.group("suffix")):
        return None  # a helper that lives in tests/ (e.g. tests/testserver/server.py)
    stem = m.group("stem")
    return stem if stem not in ("conftest", "test", "tests", "__init__") else None


def tests_named_for(file_path: str, candidates) -> list[str]:
    """Test files among `candidates` named after `file_path`'s module
    (adapters.py → tests/test_adapters.py, foo.ts → foo.test.ts)."""
    if is_test_path(file_path):
        return []
    stem = PurePosixPath(file_path).stem
    if stem == "__init__":
        stem = PurePosixPath(file_path).parent.name
    return sorted(c for c in candidates if c != file_path and tested_module_stem(c) == stem)


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
    symbol: str, metadata_store: MetadataStore, repo_id: str, graph=None
) -> Optional[DefinitionResponse]:
    """Resolve `symbol` (bare "send" or qualified "HTTPAdapter.send") to its
    best definition, the other candidates, and every usage in the repo.

    With `graph`, a method's usages are narrowed to files whose references can
    reach that method by inferred receiver type (see app/core/usages.py), so
    `Session.send` no longer lists every `.send(...)` in the repo."""
    ranked = rank_definitions(symbol, metadata_store.find_symbol(repo_id, symbol))
    if not ranked:
        return None
    best = ranked[0]

    # Usages are recorded by identifier, so a dotted query matches on its last
    # component. The definition site itself is never a usage; drop any hit on
    # the definition's own line.
    refs = [
        r for r in metadata_store.find_references(repo_id, symbol)
        if not (r["file_path"] == best["file_path"] and r["line"] == best.get("start_line"))
    ]
    qualified = best.get("qualified_name") or ""
    if graph is not None and best.get("kind") == "method" and "." in qualified:
        from app.core.usages import symbol_users
        users = set(symbol_users(repo_id, qualified, best["file_path"], graph, metadata_store))
        users.add(best["file_path"])  # calls within the defining file stay listed
        refs = [r for r in refs if r["file_path"] in users]
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
