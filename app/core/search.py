from typing import Optional

from app.core.embeddings import embed_query, index_embedding_settings
from app.core.text import keyword_query_terms, query_identifiers
from app.models.schemas import SearchResult
from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore

SEARCH_MODES = ("hybrid", "semantic", "keyword")

# Reciprocal rank fusion constant (Cormack et al. 2009). Larger k flattens the
# advantage of being ranked first in any single list.
RRF_K = 60
# How deep to read each ranked list before fusing.
CANDIDATE_POOL = 50


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Merge ranked id lists: score(id) = Σ 1 / (k + rank). Best-first.

    Unweighted on purpose: down-weighting the keyword list for prose queries
    looked better on the main eval set but was worse on the held-out set.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def search_chunks(
    query: str,
    repo_id: str,
    top_k: int,
    faiss_store: FAISSStore,
    metadata_store: MetadataStore,
    embedding_backend: Optional[str] = None,
    mode: str = "hybrid",
) -> list[SearchResult]:
    """Rank chunks for `query`.

    mode="semantic": embedding cosine similarity (FAISS). `score` is the cosine.
    mode="keyword":  BM25 over path, symbol names and code (SQLite FTS5).
    mode="hybrid":   both, plus chunks that *define* any code-looking identifier
                     in the query (e.g. "get_netrc_auth"), merged with
                     reciprocal rank fusion. `score` is the fused RRF score.

    Semantic search alone misses exact identifiers and rare terms; keyword
    search alone misses paraphrases. Fusing ranks (not raw scores) sidesteps
    the fact that cosine and BM25 scores aren't on comparable scales.

    `embedding_backend` should be the backend the repo was actually ingested
    with (faiss_store.embedding_backend) so the query is embedded consistently
    with the index, regardless of the current EMBEDDING_BACKEND env var.
    """
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {SEARCH_MODES}, got {mode!r}")
    pool = max(top_k, CANDIDATE_POOL)

    semantic: list[tuple[str, float]] = []
    if mode in ("semantic", "hybrid"):
        index_backend, index_model = index_embedding_settings(faiss_store)
        query_vec = embed_query(query, backend=embedding_backend or index_backend, model=index_model)
        semantic = faiss_store.search(query_vec, pool)

    keyword: list[tuple[str, float]] = []
    if mode in ("keyword", "hybrid"):
        keyword = metadata_store.keyword_search(repo_id, keyword_query_terms(query), pool)

    if mode == "semantic":
        ranked = semantic
    elif mode == "keyword":
        ranked = keyword
    else:
        lists = [[cid for cid, _ in semantic], [cid for cid, _ in keyword]]
        definers = metadata_store.chunks_defining(repo_id, query_identifiers(query))
        if definers:
            lists.append(definers[:pool])
        ranked = reciprocal_rank_fusion(lists)

    results: list[SearchResult] = []
    for chunk_id, score in ranked:
        if len(results) >= top_k:
            break
        chunk = metadata_store.get_chunk(chunk_id)
        if chunk is None:
            continue
        results.append(
            SearchResult(
                chunk_id=chunk.chunk_id,
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                score=score,
                snippet=chunk.content[:300],
            )
        )
    return results
