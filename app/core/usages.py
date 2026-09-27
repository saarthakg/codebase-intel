"""Which files use a given symbol: the core of diff-level impact."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.graph import DependencyGraph
    from app.storage.metadata_store import MetadataStore


def symbol_users(
    repo_id: str,
    qualified_name: str,
    defining_file: str,
    graph: "DependencyGraph",
    metadata_store: "MetadataStore",
    depth: int = 3,
) -> list[str]:
    """Files that use `qualified_name` (defined in `defining_file`).

    Matched by the symbol's bare name, counted only in files that import the
    defining file within `depth` hops, so an unrelated module's `send` doesn't
    match. Generic method names (`read`, `get`) still over-match: requiring
    the class name in the user file as well lost the recall gain on the
    history eval, since methods are mostly called on instances obtained
    elsewhere (`r.connection.send(...)`).
    """
    dependents = {d["file"] for d in graph.dependents_of(defining_file, depth=depth)}
    bare = qualified_name.rsplit(".", 1)[-1]
    return sorted({
        r["file_path"] for r in metadata_store.find_references(repo_id, bare)
        if r["file_path"] in dependents
    })
