import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.chunking import chunk_file
from app.core.embeddings import embed_texts, get_embedding_dim
from app.core.graph import DependencyGraph, resolve_python_import, resolve_ts_import
from app.core.ingest import detect_language, load_file, walk_repo
from app.core.symbols import extract_imports, extract_symbols
from app.models.schemas import IngestRequest, IngestResponse
from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore

router = APIRouter()

DATA_INDEXES = Path(__file__).parent.parent.parent / "data" / "indexes"
DATA_METADATA = Path(__file__).parent.parent.parent / "data" / "metadata"


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    repo_path = str(Path(request.repo_path).resolve())
    if not Path(repo_path).exists():
        raise HTTPException(status_code=400, detail=f"repo_path does not exist: {repo_path}")

    repo_id = request.repo_id
    DATA_INDEXES.mkdir(parents=True, exist_ok=True)
    DATA_METADATA.mkdir(parents=True, exist_ok=True)

    db_path = str(DATA_METADATA / f"{repo_id}.db")
    index_path = str(DATA_INDEXES / f"{repo_id}.index")
    graph_path = str(DATA_METADATA / f"{repo_id}.graph.pkl")

    metadata_store = MetadataStore(db_path)
    graph = DependencyGraph()

    file_paths = walk_repo(repo_path)
    all_chunks = []
    total_symbols = 0

    for file_path in file_paths:
        content = load_file(file_path)
        if content is None:
            continue
        rel_path = os.path.relpath(file_path, repo_path)
        language = detect_language(file_path)
        graph.add_file(rel_path)

        symbols = extract_symbols(content, rel_path, language)
        imports = extract_imports(content, rel_path, language)

        for sym in symbols:
            metadata_store.upsert_symbol(
                name=sym.name, repo_id=repo_id, file_path=rel_path,
                line=sym.start_line, kind=sym.kind,
            )
            total_symbols += 1

        chunks = chunk_file(content, rel_path, language)
        for chunk in chunks:
            metadata_store.upsert_chunk(chunk, repo_id)
        all_chunks.extend(chunks)

        for imp in imports:
            resolved = None
            if language == "python":
                resolved = resolve_python_import(imp.imported_module, file_path, repo_path)
            elif language in ("typescript", "javascript"):
                resolved = resolve_ts_import(imp.imported_module, file_path, repo_path)
            if resolved:
                rel_target = os.path.relpath(resolved, repo_path)
                graph.add_import_edge(rel_path, rel_target)
                metadata_store.upsert_edge(repo_id, rel_path, rel_target, "import")

    graph.save(graph_path)
    edge_count = graph.edge_count

    if all_chunks:
        texts = [c.content for c in all_chunks]
        chunk_ids = [c.chunk_id for c in all_chunks]
        embeddings = embed_texts(texts)
        dim = embeddings.shape[1]
        faiss_store = FAISSStore(dim=dim)
        faiss_store.add(embeddings, chunk_ids)
        faiss_store.save(index_path)

    # Invalidate cached state for this repo_id
    from app.main import _loaded_repos
    _loaded_repos.pop(repo_id, None)

    return IngestResponse(
        repo_id=repo_id,
        files_indexed=len(file_paths),
        chunks_indexed=len(all_chunks),
        symbols_extracted=total_symbols,
        edges_in_graph=edge_count,
    )
