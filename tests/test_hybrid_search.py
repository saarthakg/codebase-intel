import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from app.core.search import reciprocal_rank_fusion, search_chunks
from app.core.text import camel_parts, keyword_query_terms, query_identifiers
from app.models.schemas import ChunkMetadata
from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore


def _chunk(cid: str, path: str, content: str, symbols=(), start=1) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=cid, file_path=path, language="python", start_line=start,
        end_line=start + content.count("\n"), symbols=list(symbols), imports=[], content=content,
    )


# ── text helpers ──────────────────────────────────────────────────────────────

def test_camel_parts():
    assert camel_parts("HTTPAdapter") == ["HTTP", "Adapter"]
    assert camel_parts("RequestsCookieJar") == ["Requests", "Cookie", "Jar"]
    assert camel_parts("get_netrc_auth") == ["get", "netrc", "auth"]


def test_keyword_terms_drop_question_filler_and_expand_camel_case():
    assert keyword_query_terms("where is the HTTPAdapter handled?") == ["httpadapter", "handled", "http", "adapter"]


def test_query_identifiers_only_picks_code_looking_tokens():
    assert query_identifiers("how does get_netrc_auth work") == ["get_netrc_auth"]
    assert query_identifiers("HTTPAdapter.send timeout handling") == ["HTTPAdapter.send"]
    assert query_identifiers("getUserName and plain words") == ["getUserName"]
    assert query_identifiers("where is authentication handled") == []


# ── keyword index ─────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    s = MetadataStore(str(tmp_path / "m.db"))
    s.add_chunks([
        _chunk("c1", "src/adapters.py", "class HTTPAdapter:\n    def send(self): ...\n", ["HTTPAdapter"]),
        _chunk("c2", "src/cookies.py", "class RequestsCookieJar:\n    pass\n", ["RequestsCookieJar"]),
        _chunk("c3", "docs/redirects.md", "Redirects are followed automatically.\n"),
    ], "r1")
    s.add_chunks([_chunk("other", "src/adapters.py", "class HTTPAdapter: pass\n")], "r2")
    s.commit()
    return s


def test_keyword_search_matches_camel_case_parts(store):
    ids = [cid for cid, _ in store.keyword_search("r1", ["adapter"], 10)]
    assert ids == ["c1"]
    ids = [cid for cid, _ in store.keyword_search("r1", ["cookie", "jar"], 10)]
    assert ids == ["c2"]


def test_keyword_search_stems_and_is_scoped_to_repo(store):
    assert [cid for cid, _ in store.keyword_search("r1", ["redirect"], 10)] == ["c3"]
    assert [cid for cid, _ in store.keyword_search("r2", ["adapter"], 10)] == ["other"]


def test_keyword_search_handles_quotes_and_empty_terms(store):
    assert store.keyword_search("r1", [], 10) == []
    assert store.keyword_search("r1", ['say "hi"'], 10) == []  # no crash on FTS syntax


def test_clear_repo_clears_keyword_index(store):
    store.clear_repo("r1")
    assert store.keyword_search("r1", ["adapter"], 10) == []
    assert store.keyword_search("r2", ["adapter"], 10)


def test_keyword_index_is_backfilled_for_existing_dbs(tmp_path):
    """DBs ingested before keyword search existed get their chunks indexed on open."""
    db = tmp_path / "old.db"
    MetadataStore(str(db)).close()
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE chunks_fts")
    conn.execute(
        "INSERT INTO chunks VALUES ('c1','r','a.py','python',1,2,'[]','[]','def get_netrc_auth(): pass')"
    )
    conn.commit()
    conn.close()
    assert [cid for cid, _ in MetadataStore(str(db)).keyword_search("r", ["netrc"], 10)] == ["c1"]


def test_chunks_defining_prefers_source_over_tests(tmp_path):
    s = MetadataStore(str(tmp_path / "m.db"))
    s.add_chunks([
        _chunk("t", "tests/test_x.py", "def helper(): pass\n"),
        _chunk("s", "src/x.py", "def helper(): pass\n"),
    ], "r")
    s.upsert_symbol("helper", "r", "tests/test_x.py", 1, "function")
    s.upsert_symbol("helper", "r", "src/x.py", 1, "function")
    assert s.chunks_defining("r", ["helper"]) == ["s", "t"]


# ── fusion and modes ──────────────────────────────────────────────────────────

def test_rrf_rewards_agreement_between_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "x"]])
    order = [cid for cid, _ in fused]
    assert order[0] == "b" or order[0] == "c"
    assert order.index("b") < order.index("a")  # b is 2nd in both; a is only in one list


def _semantic_store(order: list[str]) -> FAISSStore:
    """FAISS store whose search always returns `order` (query vector = e0)."""
    dim = len(order)
    faiss_store = FAISSStore(dim=dim)
    for rank, cid in enumerate(order):
        v = np.zeros((1, dim), dtype=np.float32)
        v[0, 0] = 1.0 - 0.1 * rank
        v[0, rank] += 0.5 if rank else 0.0
        faiss_store.add(v, [cid])
    return faiss_store


def test_search_modes(store):
    faiss_store = _semantic_store(["c3", "c2", "c1"])
    query_vec = lambda q, backend=None, **kw: np.eye(1, 3, dtype=np.float32)
    with patch("app.core.search.embed_query", side_effect=query_vec):
        semantic = search_chunks("adapter", "r1", 3, faiss_store, store, mode="semantic")
        keyword = search_chunks("adapter", "r1", 3, faiss_store, store, mode="keyword")
        hybrid = search_chunks("adapter", "r1", 3, faiss_store, store, mode="hybrid")
    assert semantic[0].chunk_id == "c3"
    assert [r.chunk_id for r in keyword] == ["c1"]
    # c1 is last semantically but the only keyword hit → fused to the top
    assert hybrid[0].chunk_id == "c1"
    with pytest.raises(ValueError):
        search_chunks("x", "r1", 3, faiss_store, store, mode="fuzzy")


def test_search_endpoint_accepts_mode_and_rejects_unknown(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.core import paths
    from app.main import _loaded_repos, app

    monkeypatch.setattr(paths, "DATA_INDEXES", tmp_path / "indexes")
    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path / "metadata")
    _loaded_repos.clear()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "netrc.py").write_text("def get_netrc_auth(url):\n    return None\n")
    (repo / "other.py").write_text("def unrelated():\n    return 1\n")

    fake = lambda texts, backend=None, **kw: np.ones((len(texts), 8), dtype=np.float32)
    client = TestClient(app)
    with patch("app.core.pipeline.embed_texts", side_effect=fake), \
         patch("app.core.search.embed_query", side_effect=lambda q, backend=None, **kw: fake([q])):
        assert client.post("/ingest", json={"repo_path": str(repo), "repo_id": "hy"}).status_code == 200
        r = client.post("/search", json={"repo_id": "hy", "query": "get_netrc_auth", "mode": "hybrid"})
        bad = client.post("/search", json={"repo_id": "hy", "query": "x", "mode": "fuzzy"})
    _loaded_repos.clear()
    assert r.status_code == 200
    assert r.json()["results"][0]["file_path"] == "netrc.py"
    assert bad.status_code == 422
