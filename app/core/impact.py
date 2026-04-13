from app.core.graph import DependencyGraph
from app.models.schemas import ImpactedFile, ImpactResponse
from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore

# Confidence scores by depth
_GRAPH_CONFIDENCE = {1: 0.95, 2: 0.75, 3: 0.50}
_SYMBOL_CONFIDENCE = 0.70
_SEMANTIC_CONFIDENCE = 0.35


def analyze_impact(
    target: str,
    repo_id: str,
    graph: DependencyGraph,
    faiss_store: FAISSStore,
    metadata_store: MetadataStore,
    embeddings_module,
    depth: int = 3,
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
        symbol_hits = metadata_store.find_symbol(repo_id, target)
        if symbol_hits:
            defining_entry = next(
                (h for h in symbol_hits if h["kind"] in ("function", "class", "method")),
                None,
            )
            if defining_entry:
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
    # Only apply if target looks like a symbol name (not a file path)
    if target not in graph.G.nodes:
        symbol_refs = metadata_store.find_symbol(repo_id, target)
        for ref in symbol_refs:
            fp = ref["file_path"]
            if fp == defining_file:
                continue  # skip the defining file itself
            _add(fp, _SYMBOL_CONFIDENCE, "references symbol", 0)

    # ── Signal 3: Semantic similarity ─────────────────────────────────────────
    try:
        query_emb = embeddings_module.embed_query(target)
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
        results.items(), key=lambda x: -x[1][0]
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
