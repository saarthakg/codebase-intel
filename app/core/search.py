from app.core.embeddings import embed_query
from app.models.schemas import SearchResult
from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore


def search_chunks(
    query: str,
    repo_id: str,
    top_k: int,
    faiss_store: FAISSStore,
    metadata_store: MetadataStore,
) -> list[SearchResult]:
    """Embed query, search FAISS, return ranked SearchResult list."""
    query_emb = embed_query(query)
    hits = faiss_store.search(query_emb, top_k)
    results: list[SearchResult] = []
    for chunk_id, score in hits:
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
