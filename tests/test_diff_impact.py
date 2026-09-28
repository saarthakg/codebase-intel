from unittest.mock import MagicMock, patch

import numpy as np

from app.core.diff_impact import parse_unified_diff, symbols_touched

DIFF = """\
diff --git a/src/pkg/adapters.py b/src/pkg/adapters.py
index 111..222 100644
--- a/src/pkg/adapters.py
+++ b/src/pkg/adapters.py
@@ -10,2 +10,3 @@ class HTTPAdapter:
-        old = 1
-        old = 2
+        new = 1
+        new = 2
+        new = 3
@@ -40,3 +41,0 @@ def helper():
-    gone = 1
-    gone = 2
-    gone = 3
diff --git a/src/pkg/new.py b/src/pkg/new.py
new file mode 100644
--- /dev/null
+++ b/src/pkg/new.py
@@ -0,0 +1,2 @@
+def fresh():
+    pass
diff --git a/src/pkg/dead.py b/src/pkg/dead.py
deleted file mode 100644
--- a/src/pkg/dead.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def old():
-    pass
"""


def test_parse_unified_diff():
    files = {f.path: f for f in parse_unified_diff(DIFF)}
    adapters = files["src/pkg/adapters.py"]
    assert adapters.new_ranges == [(10, 12), (41, 41)]  # pure deletion → the line it points at
    assert adapters.old_ranges == [(10, 11), (40, 42)]
    assert files["src/pkg/new.py"].new_ranges == [(1, 2)] and files["src/pkg/new.py"].old_ranges == []
    assert files["src/pkg/dead.py"].deleted and files["src/pkg/dead.py"].new_ranges == []


def test_symbols_touched_reports_innermost():
    rows = [
        {"qualified_name": "HTTPAdapter", "start_line": 1, "end_line": 50},
        {"qualified_name": "HTTPAdapter.send", "start_line": 8, "end_line": 20},
        {"qualified_name": "HTTPAdapter.close", "start_line": 22, "end_line": 30},
        {"qualified_name": "helper", "start_line": 60, "end_line": 70},
    ]
    names = lambda ranges: [s["qualified_name"] for s in symbols_touched(rows, ranges)]
    assert names([(10, 12)]) == ["HTTPAdapter.send"]            # not also the class
    assert names([(3, 4)]) == ["HTTPAdapter"]                    # class-level lines
    assert names([(18, 24)]) == ["HTTPAdapter.send", "HTTPAdapter.close"]
    assert names([(55, 56)]) == []                               # module-level code


def _ingest(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.core import paths
    from app.main import _loaded_repos, app

    monkeypatch.setattr(paths, "DATA_INDEXES", tmp_path / "indexes")
    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path / "metadata")
    _loaded_repos.clear()
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "adapters.py").write_text(
        "class Adapter:\n"            # 1
        "    def send(self):\n"       # 2
        "        return 1\n"          # 3
        "\n"                          # 4
        "    def close(self):\n"      # 5
        "        return 2\n"          # 6
    )
    (repo / "pkg" / "sender.py").write_text("from pkg.adapters import Adapter\n\ndef go():\n    return Adapter().send()\n")
    (repo / "pkg" / "closer.py").write_text("from pkg.adapters import Adapter\n\ndef stop():\n    return Adapter().close()\n")
    # Unrelated module with its own `send`: must not count as a user of Adapter.send
    (repo / "other.py").write_text("class Mailer:\n    def send(self):\n        return self.send()\n")
    fake = lambda texts, backend=None, **kw: np.random.rand(len(texts), 8).astype(np.float32)
    monkeypatch.setattr("app.core.pipeline.embed_texts", fake)
    client = TestClient(app)
    assert client.post("/ingest", json={"repo_path": str(repo), "repo_id": "dif"}).status_code == 200
    return client


def test_impact_diff_ranks_users_of_changed_symbol_first(tmp_path, monkeypatch):
    client = _ingest(tmp_path, monkeypatch)
    diff = (
        "--- a/pkg/adapters.py\n+++ b/pkg/adapters.py\n"
        "@@ -3 +3 @@\n-        return 1\n+        return 10\n"
        "--- a/brand_new.py\n+++ b/brand_new.py\n@@ -0,0 +1 @@\n+x = 1\n"
    )
    body = client.post("/impact/diff", json={"repo_id": "dif", "diff": diff}).json()
    assert body["changed_symbols"] == [
        {"file_path": "pkg/adapters.py", "qualified_name": "Adapter.send", "used_in": ["pkg/sender.py"]}
    ]
    assert body["high_confidence"][0]["file_path"] == "pkg/sender.py"
    assert body["high_confidence"][0]["reason"] == "uses changed Adapter.send; direct import"
    ranked = [f["file_path"] for f in body["high_confidence"] + body["medium_confidence"] + body["related"]]
    assert ranked.index("pkg/sender.py") < ranked.index("pkg/closer.py")  # both import it; only one calls send
    users = {f["file_path"] for f in body["high_confidence"] + body["medium_confidence"] + body["related"]
             if f["reason"].startswith("uses changed")}
    assert "other.py" not in users  # its own class's send() doesn't count
    assert body["unindexed_files"] == ["brand_new.py"]


def test_impact_diff_rejects_empty_diff(tmp_path, monkeypatch):
    client = _ingest(tmp_path, monkeypatch)
    assert client.post("/impact/diff", json={"repo_id": "dif", "diff": "  "}).status_code == 400


def test_method_users_follow_types_and_dispatch(tmp_path, monkeypatch):
    """Callers of Adapter.send: a file calling it via the base type counts
    (dispatch), a file calling a *different* class's send doesn't, and an
    untyped receiver in an importing file falls back to name matching."""
    from fastapi.testclient import TestClient
    from app.core import paths
    from app.core.usages import symbol_users
    from app.main import _loaded_repos, app, get_repo_state

    monkeypatch.setattr(paths, "DATA_INDEXES", tmp_path / "indexes")
    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path / "metadata")
    _loaded_repos.clear()
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "adapters.py").write_text(
        "class Base:\n    def send(self, r): ...\n\n"
        "class Adapter(Base):\n    def send(self, r): ...\n\n"
        "def get_adapter() -> Base: ...\n"
    )
    (repo / "pkg" / "mail.py").write_text("class Mailer:\n    def send(self, m): ...\n")
    (repo / "pkg" / "proto.py").write_text(
        "from typing import Protocol\n\nclass Reader(Protocol):\n    def read(self): ...\n\n"
        "def consume(r: Reader):\n    return r.read()\n")
    (repo / "pkg" / "files.py").write_text(
        "from pkg import proto\n\ndef load(fp):\n    return fp.read()\n")
    (repo / "pkg" / "via_base.py").write_text(
        "from pkg.adapters import get_adapter\n\ndef go():\n    get_adapter().send(1)\n")
    (repo / "pkg" / "other.py").write_text(
        "from pkg.adapters import Adapter\nfrom pkg.mail import Mailer\n\n"
        "def go(m: Mailer):\n    m.send(1)\n")
    (repo / "pkg" / "untyped.py").write_text(
        "from pkg import adapters\n\ndef go(x):\n    x.send(1)\n")
    fake = lambda texts, backend=None, **kw: np.random.rand(len(texts), 8).astype(np.float32)
    monkeypatch.setattr("app.core.pipeline.embed_texts", fake)
    assert TestClient(app).post("/ingest", json={"repo_path": str(repo), "repo_id": "typed"}).status_code == 200

    state = get_repo_state("typed")
    users = symbol_users("typed", "Adapter.send", "pkg/adapters.py", state.graph, state.metadata_store)
    assert "pkg/via_base.py" in users       # Base.send can dispatch to Adapter.send
    assert "pkg/untyped.py" in users        # unknown receiver, importer: name fallback
    assert "pkg/other.py" not in users      # m is a Mailer: known to be another class
    mail_users = symbol_users("typed", "Mailer.send", "pkg/mail.py", state.graph, state.metadata_store)
    assert mail_users == ["pkg/other.py"]
    # Protocol method: only code typed against the protocol, never untyped .read() calls
    reader_users = symbol_users("typed", "Reader.read", "pkg/proto.py", state.graph, state.metadata_store)
    assert reader_users == ["pkg/proto.py"]
    _loaded_repos.clear()
