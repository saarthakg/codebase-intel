from typing import TYPE_CHECKING, Optional

from app.core.definitions import best_definition
from app.core.graph import DependencyGraph
from app.models.schemas import BatchImpactedFile, ImpactBatchResponse, ImpactedFile, ImpactResponse
from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore

if TYPE_CHECKING:
    from app.core.history import CoChange

# Confidence scores by depth
_GRAPH_CONFIDENCE = {1: 0.95, 2: 0.75, 3: 0.50}
_SYMBOL_CONFIDENCE = 0.70
_SEMANTIC_CONFIDENCE = 0.35
# Co-change: a file that changed together with the target in a fraction p of
# the target's commits gets confidence 0.4 + 0.5·p (capped at 0.9), and p also
# breaks ties among files with equal confidence from other signals (e.g. the
# many "2 hops" files). Chosen on the history eval's 2019–2022 dev commits
# (signal-only and tiebreak-only were both worse), checked once on 2023+.
_COCHANGE_BASE = 0.4
_COCHANGE_SCALE = 0.5
_COCHANGE_CAP = 0.9


def analyze_impact(
    target: str,
    repo_id: str,
    graph: DependencyGraph,
    faiss_store: FAISSStore,
    metadata_store: MetadataStore,
    embeddings_module,
    depth: int = 3,
    cochange: Optional["CoChange"] = None,
) -> ImpactResponse:
    """Rank files by likelihood of being affected by a change to `target`.

    `target` may be a file path (e.g. 'requests/adapters.py') or a symbol name
    (e.g. 'HTTPAdapter').
    """
    # Accumulate: file_path → best (confidence, reason, depth)
    results: dict[str, tuple[float, str, int]] = {}

    def _add(file_path: str, confidence: float, reason: str, hop: int = 0) -> None:
        existing = results.get(file_path)
        if existing is None or confidence > existing[0]:
            results[file_path] = (confidence, reason, hop)

    # ── Signal 1: Graph traversal ──────────────────────────────────────────────
    # Determine the root file to traverse from
    root_file: str | None = None
    defining_file: str | None = None  # the file that *defines* the symbol

    # Check if target looks like a file path (exists as a node in graph)
    if target in graph.G.nodes:
        root_file = target
    else:
        # Try treating as symbol → find defining file
        defining_entry = best_definition(target, metadata_store, repo_id)
        if defining_entry and defining_entry.get("kind") in ("function", "class", "method"):
            defining_file = defining_entry["file_path"]
            root_file = defining_file

    if root_file:
        dependents = graph.dependents_of(root_file, depth=depth)
        for entry in dependents:
            d = entry["depth"]
            conf = _GRAPH_CONFIDENCE.get(d, 0.30)
            reason = "direct import" if d == 1 else f"transitive import ({d} hops)"
            _add(entry["file"], conf, reason, d)

    # ── Signal 2: Symbol reference search ─────────────────────────────────────
    # Only apply if target looks like a symbol name (not a file path). Uses the
    # identifier-usage index — previously this queried the *definitions* table,
    # so it only ever found other files defining a same-named symbol.
    if target not in graph.G.nodes and defining_file is not None:
        for fp in {r["file_path"] for r in metadata_store.find_references(repo_id, target)}:
            if fp == defining_file:
                continue  # skip the defining file itself
            _add(fp, _SYMBOL_CONFIDENCE, "references symbol", 0)

    # ── Signal 4: Co-change history ───────────────────────────────────────────
    cochange_p: dict[str, float] = {}
    if cochange is not None and root_file:
        total = cochange.file_commits.get(root_file, 0)
        for other, p, n in cochange.related(root_file):
            if other == root_file:
                continue
            cochange_p[other] = p
            _add(other, min(_COCHANGE_CAP, _COCHANGE_BASE + _COCHANGE_SCALE * p),
                 f"changed together in {n} of {total} commits", 0)

    # ── Signal 3: Semantic similarity ─────────────────────────────────────────
    try:
        backend, model = embeddings_module.index_embedding_settings(faiss_store)
        query_emb = embeddings_module.embed_query(target, backend=backend, model=model)
        hits = faiss_store.search(query_emb, top_k=5)
        for chunk_id, _score in hits:
            chunk = metadata_store.get_chunk(chunk_id)
            if chunk is None:
                continue
            fp = chunk.file_path
            if fp in results:
                continue  # already covered by higher-signal
            if fp == root_file:
                continue
            _add(fp, _SEMANTIC_CONFIDENCE, "semantically related", 0)
    except Exception:
        pass  # FAISS/embedding failure is non-fatal

    # ── Bucket and sort ───────────────────────────────────────────────────────
    high_confidence: list[ImpactedFile] = []
    medium_confidence: list[ImpactedFile] = []
    related: list[ImpactedFile] = []

    for file_path, (confidence, reason, hop) in sorted(
        results.items(), key=lambda x: (-x[1][0], -cochange_p.get(x[0], 0.0))
    ):
        item = ImpactedFile(
            file_path=file_path,
            reason=reason,
            confidence=confidence,
            depth=hop,
        )
        if confidence >= 0.7:
            high_confidence.append(item)
        elif confidence >= 0.4:
            medium_confidence.append(item)
        else:
            related.append(item)

    return ImpactResponse(
        target=target,
        high_confidence=high_confidence,
        medium_confidence=medium_confidence,
        related=related,
    )


def analyze_impact_batch(
    targets: list[str],
    repo_id: str,
    graph: DependencyGraph,
    faiss_store: FAISSStore,
    metadata_store: MetadataStore,
    embeddings_module,
    depth: int = 3,
    cochange: Optional["CoChange"] = None,
) -> ImpactBatchResponse:
    """Diff-aware impact analysis: merge impact across several changed targets.

    Intended for a PR/diff workflow — pass the list of files changed in a
    commit (e.g. `git diff --name-only`) and get back the union of everything
    those changes are likely to affect, each impacted file annotated with
    which of the changed targets triggered it. A file is excluded from its
    own results (a target can't be "impacted by itself").
    """
    # file_path → (confidence, reason, depth, {triggering targets})
    merged: dict[str, tuple[float, str, int, set[str]]] = {}

    for target in targets:
        single = analyze_impact(
            target, repo_id, graph, faiss_store, metadata_store, embeddings_module, depth=depth,
            cochange=cochange,
        )
        for item in single.high_confidence + single.medium_confidence + single.related:
            if item.file_path == target:
                continue  # a target can't be impacted by itself
            existing = merged.get(item.file_path)
            if existing is None:
                merged[item.file_path] = (item.confidence, item.reason, item.depth, {target})
            else:
                conf, reason, hop, triggers = existing
                triggers = triggers | {target}
                if item.confidence > conf:
                    merged[item.file_path] = (item.confidence, item.reason, item.depth, triggers)
                else:
                    merged[item.file_path] = (conf, reason, hop, triggers)

    high_confidence: list[BatchImpactedFile] = []
    medium_confidence: list[BatchImpactedFile] = []
    related: list[BatchImpactedFile] = []

    for file_path, (confidence, reason, hop, triggers) in sorted(
        merged.items(), key=lambda x: -x[1][0]
    ):
        item = BatchImpactedFile(
            file_path=file_path,
            reason=reason,
            confidence=confidence,
            depth=hop,
            triggered_by=sorted(triggers),
        )
        if confidence >= 0.7:
            high_confidence.append(item)
        elif confidence >= 0.4:
            medium_confidence.append(item)
        else:
            related.append(item)

    return ImpactBatchResponse(
        targets=targets,
        high_confidence=high_confidence,
        medium_confidence=medium_confidence,
        related=related,
    )
