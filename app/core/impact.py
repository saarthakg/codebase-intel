from typing import TYPE_CHECKING, Optional

from app.core.definitions import best_definition, is_test_path, tests_named_for
from app.core.graph import DependencyGraph
from app.models.schemas import BatchImpactedFile, ImpactBatchResponse, ImpactedFile, ImpactResponse
from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore

if TYPE_CHECKING:
    from app.core.history import CoChange

# Each signal gives a file a confidence, and a file found by several signals
# gets their noisy-OR, 1 − ∏(1 − cᵢ): the chance at least one is right if they
# were independent. Scores are tuned for "will this file change in the same
# commit?", on the history eval's 2019–2022 dev commits of requests and Flask,
# and checked once on 2023+. There an import alone is weak evidence and co-change
# history strong: taking the max with direct imports at 0.95 ranked every
# importer above every history-backed file (requests held-out recall@5 0.52,
# Flask 0.33; now 0.58 and 0.43). An importer with no other evidence is medium.
# Hop count made no difference once imports were weak, so every hop counts the
# same and only breaks ties.
_GRAPH_CONFIDENCE = 0.40
_SYMBOL_CONFIDENCE = 0.70
_SEMANTIC_CONFIDENCE = 0.35
# A test named after the target (adapters.py → tests/test_adapters.py) is the
# file most likely to change with it.
_NAMED_TEST_CONFIDENCE = 0.97
# Semantic neighbours: read this many chunks, keep up to this many new files.
_SEMANTIC_POOL = 20
_SEMANTIC_FILES = 5
# Co-change: a file that changed together with the target in a fraction p of
# the target's commits gets confidence 0.4 + 0.5·p (capped at 0.9), and p also
# breaks ties among files with equal confidence from other signals.
_COCHANGE_BASE = 0.4
_COCHANGE_SCALE = 0.5
_COCHANGE_CAP = 0.9


def combine_evidence(evidence: list[tuple[float, str, int]]) -> tuple[float, str, int]:
    """One (confidence, reason, hop) per file from every (confidence, reason,
    hop) found for it: noisy-OR confidence, reasons strongest first, the
    shortest import path."""
    miss = 1.0
    for confidence, _, _ in evidence:
        miss *= 1.0 - confidence
    reasons = [r for _, r, _ in sorted(evidence, key=lambda e: -e[0])]
    hop = min((h for _, _, h in evidence if h), default=0)
    return 1.0 - miss, "; ".join(dict.fromkeys(reasons)), hop


def _rank_key(file_path, confidence, hop, cochange=None, cochange_p=None):
    """Sort key for impacted files, best first.

    1. confidence
    2. co-change strength with the target (how often they changed together)
    3. base rate: how often the file changes at all. Among files with equal
       evidence, one touched by most commits is likelier to change again.
    4. fewest hops, then path, so ties never depend on graph storage order.
    """
    p = (cochange_p or {}).get(file_path, 0.0)
    base = cochange.file_commits.get(file_path, 0) if cochange is not None else 0
    return (-confidence, -p, -base, hop, file_path)


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
    # file_path → every (confidence, reason, hop) found for it
    results: dict[str, list[tuple[float, str, int]]] = {}

    def _add(file_path: str, confidence: float, reason: str, hop: int = 0) -> None:
        results.setdefault(file_path, []).append((confidence, reason, hop))

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
            reason = "direct import" if d == 1 else f"transitive import ({d} hops)"
            _add(entry["file"], _GRAPH_CONFIDENCE, reason, d)

    # ── Signal 2: Symbol reference search ─────────────────────────────────────
    # Only apply if target looks like a symbol name (not a file path). Uses the
    # identifier-usage index — previously this queried the *definitions* table,
    # so it only ever found other files defining a same-named symbol.
    if target not in graph.G.nodes and defining_file is not None:
        if defining_entry.get("kind") == "method" and "." in (defining_entry.get("qualified_name") or ""):
            # Methods: callers by inferred receiver type (app/core/usages.py)
            from app.core.usages import symbol_users
            users = set(symbol_users(repo_id, defining_entry["qualified_name"], defining_file, graph, metadata_store))
        else:
            users = {r["file_path"] for r in metadata_store.find_references(repo_id, target)}
        for fp in users:
            if fp == defining_file:
                continue  # skip the defining file itself
            _add(fp, _SYMBOL_CONFIDENCE, "references symbol", 0)

    # ── Signal 5: Tests named after the target file ───────────────────────────
    if root_file:
        for test_file in tests_named_for(root_file, graph.G.nodes):
            _add(test_file, _NAMED_TEST_CONFIDENCE, "test named for this file", 0)

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
    # Query with the target's own code: the mean of its chunk vectors (the
    # defining chunk for a symbol). Embedding the *text* of a file path, as
    # before, says almost nothing about what the file does.
    try:
        query_emb = None
        if root_file and root_file == target:
            source_ids = [c.chunk_id for c in metadata_store.get_chunks_by_file(repo_id, root_file)]
        elif root_file:
            source_ids = metadata_store.chunks_defining(repo_id, [target])
        else:
            source_ids = []
        vectors = faiss_store.vectors_for(source_ids) if source_ids else None
        if vectors is not None and len(vectors):
            query_emb = vectors.mean(axis=0, keepdims=True)
        if query_emb is None:  # target not indexed: fall back to embedding its text
            backend, model = embeddings_module.index_embedding_settings(faiss_store)
            query_emb = embeddings_module.embed_query(target, backend=backend, model=model)
        hits = faiss_store.search(query_emb, top_k=_SEMANTIC_POOL)
        added = 0
        for chunk_id, _score in hits:
            chunk = metadata_store.get_chunk(chunk_id)
            if chunk is None:
                continue
            fp = chunk.file_path
            if fp in results or fp == root_file:
                continue  # already covered by a stronger signal, or the target itself
            _add(fp, _SEMANTIC_CONFIDENCE, "semantically related", 0)
            added += 1
            if added >= _SEMANTIC_FILES:
                break
    except Exception:
        pass  # FAISS/embedding failure is non-fatal

    # ── Bucket and sort ───────────────────────────────────────────────────────
    high_confidence: list[ImpactedFile] = []
    medium_confidence: list[ImpactedFile] = []
    related: list[ImpactedFile] = []

    combined = {fp: combine_evidence(ev) for fp, ev in results.items()}
    for file_path, (confidence, reason, hop) in sorted(
        combined.items(), key=lambda x: _rank_key(x[0], x[1][0], x[1][2], cochange, cochange_p)
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
        tests=[f for f in high_confidence + medium_confidence + related if is_test_path(f.file_path)],
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
        merged.items(), key=lambda x: _rank_key(x[0], x[1][0], x[1][2], cochange)
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
        tests=[f for f in high_confidence + medium_confidence + related if is_test_path(f.file_path)],
    )
