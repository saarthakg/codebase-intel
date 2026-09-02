from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core import paths
from app.main import _loaded_repos, app


def _fake_embed_texts(texts, backend=None):
    return np.random.rand(len(texts), 8).astype(np.float32)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_INDEXES", tmp_path / "indexes")
    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path / "metadata")
    _loaded_repos.clear()
    yield TestClient(app)
    _loaded_repos.clear()


@pytest.fixture
def ingested_repo(tmp_path, client):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def foo():\n    return 1\n")
    with patch("app.core.pipeline.embed_texts", side_effect=_fake_embed_texts):
        r = client.post("/ingest", json={"repo_path": str(repo), "repo_id": "testrepo"})
    assert r.status_code == 200
    return "testrepo"


def test_list_repos_empty(client):
    r = client.get("/repos")
    assert r.status_code == 200
    assert r.json() == {"repos": []}


def test_list_repos_after_ingest(client, ingested_repo):
    r = client.get("/repos")
    assert r.status_code == 200
    repos = r.json()["repos"]
    assert len(repos) == 1
    assert repos[0]["repo_id"] == "testrepo"
    assert repos[0]["files_indexed"] == 1
    assert repos[0]["embedding_backend"]
    assert repos[0]["ingested_at"]


def test_delete_repo_removes_artifacts(client, ingested_repo):
    assert paths.index_path("testrepo").exists()
    r = client.delete("/repos/testrepo")
    assert r.status_code == 200
    assert r.json() == {"repo_id": "testrepo", "deleted": True}
    assert not paths.index_path("testrepo").exists()
    assert not paths.db_path("testrepo").exists()

    # Search should now 404 instead of silently using stale cached state.
    r = client.post("/search", json={"repo_id": "testrepo", "query": "foo"})
    assert r.status_code == 404


def test_delete_nonexistent_repo_404(client):
    r = client.delete("/repos/ghost")
    assert r.status_code == 404


def test_ingest_rejects_path_traversal_repo_id(client, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    r = client.post("/ingest", json={"repo_path": str(repo), "repo_id": "../escape"})
    assert r.status_code == 422


def test_reingest_via_api_replaces_state(client, ingested_repo, tmp_path):
    """Ingesting the same repo_id again should be reflected immediately —
    the in-memory cache from the first ingest must not shadow the new data."""
    repo_dir = next(tmp_path.glob("repo"))
    (repo_dir / "b.py").write_text("def bar():\n    return 2\n")
    with patch("app.core.pipeline.embed_texts", side_effect=_fake_embed_texts):
        r = client.post("/ingest", json={"repo_path": str(repo_dir), "repo_id": "testrepo"})
    assert r.status_code == 200
    assert r.json()["files_indexed"] == 2

    r = client.get("/definition", params={"repo_id": "testrepo", "symbol": "bar"})
    assert r.status_code == 200
    assert r.json()["defining_file"] == "b.py"
