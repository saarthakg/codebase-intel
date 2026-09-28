"""Which files use a given symbol: the core of diff-level impact."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codebase_intel.core.graph import DependencyGraph
    from codebase_intel.storage.metadata_store import MetadataStore

from codebase_intel.core.typeinfer import UNKNOWN


def symbol_users(
    repo_id: str,
    qualified_name: str,
    defining_file: str,
    graph: "DependencyGraph",
    metadata_store: "MetadataStore",
    depth: int = 3,
) -> list[str]:
    """Files that use `qualified_name` (defined in `defining_file`).

    Functions and classes are matched by name among files that import the
    defining file (within `depth` hops); their names are usually distinctive.

    Methods use the receiver types inferred at ingest (app/core/typeinfer.py):
    a reference to `m` counts for `X.m` when its receiver is X, a subclass
    that inherits m from X, or a base class whose `m` can dispatch to X's
    override. References whose receiver couldn't be inferred fall back to
    name matching among importers; references known to be on another type
    (another repo class, or an external type like a file object) don't count.
    """
    dependents = {d["file"] for d in graph.dependents_of(defining_file, depth=depth)}
    rows = metadata_store.find_symbol(repo_id, qualified_name)
    row = next((r for r in rows if r["file_path"] == defining_file), rows[0] if rows else None)
    is_method = row is not None and row.get("kind") == "method" and "." in (row.get("qualified_name") or "")

    if not is_method or not metadata_store.has_method_refs(repo_id):
        bare = qualified_name.rsplit(".", 1)[-1]
        return sorted({
            r["file_path"] for r in metadata_store.find_references(repo_id, bare)
            if r["file_path"] in dependents
        })

    owner, method = row["qualified_name"].rsplit(".", 1)
    owner = owner.rsplit(".", 1)[-1]  # innermost class of a nested qualified name
    bases = metadata_store.class_bases_map(repo_id)
    definers = {
        r["qualified_name"].rsplit(".", 1)[0].rsplit(".", 1)[-1]
        for r in metadata_store.find_symbol(repo_id, method)
        if r.get("kind") == "method" and "." in (r.get("qualified_name") or "")
    }
    receivers = _dispatching_receivers(owner, method, bases, definers)

    typed = metadata_store.method_ref_files(repo_id, method, sorted(receivers))
    if method.startswith("__") and method.endswith("__"):
        # Constructors and protocol methods (with, for, calling an instance) are
        # recorded only with a known receiver type; no call site spells them
        # out, so there's no name to fall back on.
        return sorted(typed)
    if "Protocol" in _mro(owner, bases):
        # A typing.Protocol method never runs: calls land on the concrete object
        # (a file, BytesIO, ...). Only code typed against the protocol uses it.
        return sorted(typed)
    fallback = metadata_store.method_ref_files(repo_id, method, [UNKNOWN]) & dependents
    return sorted(typed | fallback)


def _mro(cls: str, bases: dict[str, list[str]]) -> list[str]:
    seen, order, queue = set(), [], [cls]
    while queue:
        c = queue.pop(0)
        if c in seen:
            continue
        seen.add(c)
        order.append(c)
        queue.extend(bases.get(c, []))
    return order


def _strict_ancestors(cls: str, bases: dict[str, list[str]]) -> set[str]:
    return set(_mro(cls, bases)) - {cls}


def _dispatching_receivers(owner: str, method: str, bases: dict[str, list[str]], definers: set[str]) -> set[str]:
    """Receiver classes through which calling `method` can run `owner.method`."""
    receivers = {owner}
    # Subclasses that inherit owner's implementation (first definer in their MRO is owner).
    for cls in bases:
        if cls != owner and owner in _mro(cls, bases):
            first = next((c for c in _mro(cls, bases) if c in definers), None)
            if first == owner:
                receivers.add(cls)
    # Base classes that have the method: a call through them can dispatch to owner's override.
    for anc in _strict_ancestors(owner, bases):
        if any(c in definers for c in _mro(anc, bases)):
            receivers.add(anc)
    return receivers
