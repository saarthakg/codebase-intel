import os
import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.history import CoChange, Commit, cochange_for_repo, read_history

_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t"}


def _git(repo, *args, date="2020-01-01T00:00:00"):
    env = {**_ENV, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _commit(repo, files: dict[str, str], date: str, msg: str = "c"):
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg, date=date)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _commit(r, {"pkg/core.py": "a=1\n", "pkg/util.py": "b=1\n", "tests/test_core.py": "c=1\n"}, "2020-01-01T00:00:00")
    _commit(r, {"pkg/core.py": "a=2\n", "tests/test_core.py": "c=2\n"}, "2020-02-01T00:00:00")
    _commit(r, {"pkg/core.py": "a=3\n", "tests/test_core.py": "c=3\n"}, "2020-03-01T00:00:00")
    _commit(r, {"pkg/core.py": "a=4\n", "pkg/util.py": "b=2\n"}, "2020-04-01T00:00:00")
    # move pkg/ → src/pkg/ (a rename commit)
    _git(r, "mv", "pkg", "src")
    _git(r, "commit", "-q", "-m", "move", date="2021-01-01T00:00:00")
    _commit(r, {"src/core.py": "a=5\n", "tests/test_core.py": "c=5\n"}, "2021-06-01T00:00:00")
    return r


def test_history_follows_renames_to_current_paths(repo):
    commits = read_history(str(repo))
    assert commits[0].date == "2021-06-01"
    first = commits[-1]
    assert set(first.files) == {"src/core.py", "src/util.py", "tests/test_core.py"}
    assert all("pkg/" not in f for c in commits for f in c.files)


def test_date_filter_keeps_renames(repo):
    """--until passed to git would hide the 2021 rename and leave old paths."""
    old = read_history(str(repo), until="2020-12-31")
    assert len(old) == 4
    assert {f for c in old for f in c.files} == {"src/core.py", "src/util.py", "tests/test_core.py"}


def test_cochange_confidence_and_min_support(repo):
    cc = CoChange.from_commits(read_history(str(repo)))
    related = {other: (round(p, 2), n) for other, p, n in cc.related("src/core.py")}
    # core.py appears in all 6 commits (the pkg/ → src/ move counts: renamed
    # files changed together); the test changed with it in 4, util.py in 3.
    # Rates are shrunk by 3 prior commits: 4/(6+3), 3/(6+3).
    assert cc.file_commits["src/core.py"] == 6
    assert related == {"tests/test_core.py": (0.44, 4), "src/util.py": (0.33, 3)}
    assert cc.related("missing.py") == []


def test_large_commits_are_ignored():
    big = Commit("x", "2020-01-01", [f"f{i}.py" for i in range(40)])
    small = Commit("y", "2020-01-02", ["a.py", "b.py"])
    cc = CoChange.from_commits([big, small, small])
    assert cc.commits_used == 2
    assert cc.related("a.py", min_support=2) == [("b.py", 2 / (2 + 3), 2)]
    assert "f1.py" not in cc.file_commits


def test_rows_round_trip():
    cc = CoChange.from_commits([Commit("y", "d", ["a.py", "b.py"])] * 3)
    files, pairs = cc.to_rows()
    again = CoChange.from_rows(files, pairs)
    assert again.related("a.py") == cc.related("a.py")


def test_subdirectory_of_a_git_repo_uses_relative_paths(repo):
    commits = read_history(str(repo / "src"))
    assert {f for c in commits for f in c.files} == {"core.py", "util.py"}


def test_not_a_git_repo_gives_no_history(tmp_path):
    assert read_history(str(tmp_path)) == []
    assert cochange_for_repo(str(tmp_path), keep={"a.py"}).commits_used == 0


# ── Integration: ingest stores it, impact uses it ────────────────────────────

def _fake_embed(texts, backend=None, **kw):
    return np.random.rand(len(texts), 8).astype(np.float32)


def test_ingest_stores_cochange_and_impact_uses_it(repo, tmp_path, monkeypatch):
    from app.core import paths
    from app.core.graph import DependencyGraph
    from app.core.impact import analyze_impact
    from app.core.pipeline import run_ingestion
    from app.storage.faiss_store import FAISSStore
    from app.storage.metadata_store import MetadataStore

    monkeypatch.setattr(paths, "DATA_INDEXES", tmp_path / "indexes")
    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path / "metadata")
    with patch("app.core.pipeline.embed_texts", side_effect=_fake_embed):
        summary = run_ingestion(str(repo), "hist")
    assert summary["files_with_history"] == 3

    store = MetadataStore(str(paths.db_path("hist")))
    cochange = store.load_cochange("hist")
    graph = DependencyGraph()
    graph.load(str(paths.graph_path("hist")))
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    class NoEmbed:
        def index_embedding_settings(self, s): return None, None
        def embed_query(self, *a, **k): return np.zeros((1, 8), dtype=np.float32)

    resp = analyze_impact("src/core.py", "hist", graph, faiss, store, NoEmbed(), cochange=cochange)
    hits = {f.file_path: f for f in resp.high_confidence + resp.medium_confidence + resp.related}
    # No import links between these files; only history ties util.py to core.py.
    # (tests/test_core.py is found by its name, at higher confidence.)
    assert hits["src/util.py"].reason == "changed together in 3 of 6 commits"
    assert hits["src/util.py"].confidence == pytest.approx(0.4 + 0.5 * 3 / (6 + 3))
    assert hits["tests/test_core.py"].reason == "test named for this file; changed together in 4 of 6 commits"

    without = analyze_impact("src/core.py", "hist", graph, faiss, store, NoEmbed())
    assert "src/util.py" not in {f.file_path for f in without.high_confidence + without.medium_confidence + without.related}


def test_commits_record_path_at_the_time(repo):
    commits = read_history(str(repo))
    oldest = commits[-1]
    assert oldest.paths_then["src/core.py"] == "pkg/core.py"
    assert commits[0].paths_then["src/core.py"] == "src/core.py"



def test_sparse_history_is_not_high_confidence():
    """3-of-3 commits (a shallow clone) must not look like near-certain coupling."""
    from app.core.history import CONFIDENCE_PRIOR_COMMITS
    cc = CoChange.from_commits([Commit(str(i), "d", ["a.py", "b.py"]) for i in range(3)])
    [(other, p, n)] = cc.related("a.py")
    assert (other, n) == ("b.py", 3) and p == 3 / (3 + CONFIDENCE_PRIOR_COMMITS) == 0.5
    many = CoChange.from_commits([Commit(str(i), "d", ["a.py", "b.py"]) for i in range(60)])
    assert many.related("a.py")[0][1] > 0.95  # plenty of evidence: barely shrunk
