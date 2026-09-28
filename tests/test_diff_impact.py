from codebase_intel.core.diff_impact import parse_unified_diff, symbols_touched

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


def _index(tmp_path, monkeypatch, files: dict[str, str], repo_id: str):
    from codebase_intel.core import paths
    from codebase_intel.core.pipeline import run_ingestion
    from codebase_intel.state import _loaded_repos, get_repo_state

    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path / "data")
    _loaded_repos.clear()
    repo = tmp_path / "repo"
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    run_ingestion(str(repo), repo_id)
    return get_repo_state(repo_id)


ADAPTER_REPO = {
    "pkg/__init__.py": "",
    "pkg/adapters.py": (
        "class Adapter:\n"            # 1
        "    def send(self):\n"       # 2
        "        return 1\n"          # 3
        "\n"                          # 4
        "    def close(self):\n"      # 5
        "        return 2\n"          # 6
    ),
    "pkg/sender.py": "from pkg.adapters import Adapter\n\ndef go():\n    return Adapter().send()\n",
    "pkg/closer.py": "from pkg.adapters import Adapter\n\ndef stop():\n    return Adapter().close()\n",
    # Unrelated module with its own `send`: must not count as a user of Adapter.send
    "other.py": "class Mailer:\n    def send(self):\n        return self.send()\n",
}


def _listed(resp):
    return {f.file_path: f for f in resp.high_confidence + resp.medium_confidence + resp.related}


def test_diff_impact_ranks_users_of_changed_symbol_first(tmp_path, monkeypatch):
    from codebase_intel.core.diff_impact import analyze_diff
    state = _index(tmp_path, monkeypatch, ADAPTER_REPO, "dif")
    diff = (
        "--- a/pkg/adapters.py\n+++ b/pkg/adapters.py\n"
        "@@ -3 +3 @@\n-        return 1\n+        return 10\n"
        "--- a/brand_new.py\n+++ b/brand_new.py\n@@ -0,0 +1 @@\n+x = 1\n"
    )
    resp = analyze_diff(diff, "dif", state.graph, state.metadata_store, cochange=state.cochange)
    assert [(c.file_path, c.qualified_name, c.used_in) for c in resp.changed_symbols] == [
        ("pkg/adapters.py", "Adapter.send", ["pkg/sender.py"])
    ]
    assert resp.high_confidence[0].file_path == "pkg/sender.py"
    assert resp.high_confidence[0].reason == "uses changed Adapter.send; direct import"
    ranked = list(_listed(resp))
    assert ranked.index("pkg/sender.py") < ranked.index("pkg/closer.py")  # both import it; only one calls send
    users = {f for f, item in _listed(resp).items() if item.reason.startswith("uses changed")}
    assert "other.py" not in users  # its own class's send() doesn't count
    assert resp.unindexed_files == ["brand_new.py"]


def test_method_users_follow_types_and_dispatch(tmp_path, monkeypatch):
    """Callers of Adapter.send: a file calling it via the base type counts
    (dispatch), a file calling a *different* class's send doesn't, and an
    untyped receiver in an importing file falls back to name matching."""
    from codebase_intel.core.usages import symbol_users
    state = _index(tmp_path, monkeypatch, {
        "pkg/__init__.py": "",
        "pkg/adapters.py": (
            "class Base:\n    def send(self, r): ...\n\n"
            "class Adapter(Base):\n    def send(self, r): ...\n\n"
            "def get_adapter() -> Base: ...\n"
        ),
        "pkg/mail.py": "class Mailer:\n    def send(self, m): ...\n",
        "pkg/proto.py": (
            "from typing import Protocol\n\nclass Reader(Protocol):\n    def read(self): ...\n\n"
            "def consume(r: Reader):\n    return r.read()\n"),
        "pkg/files.py": "from pkg import proto\n\ndef load(fp):\n    return fp.read()\n",
        "pkg/via_base.py": "from pkg.adapters import get_adapter\n\ndef go():\n    get_adapter().send(1)\n",
        "pkg/other.py": (
            "from pkg.adapters import Adapter\nfrom pkg.mail import Mailer\n\n"
            "def go(m: Mailer):\n    m.send(1)\n"),
        "pkg/untyped.py": "from pkg import adapters\n\ndef go(x):\n    x.send(1)\n",
    }, "typed")
    users = symbol_users("typed", "Adapter.send", "pkg/adapters.py", state.graph, state.metadata_store)
    assert "pkg/via_base.py" in users       # Base.send can dispatch to Adapter.send
    assert "pkg/untyped.py" in users        # unknown receiver, importer: name fallback
    assert "pkg/other.py" not in users      # m is a Mailer: known to be another class
    mail_users = symbol_users("typed", "Mailer.send", "pkg/mail.py", state.graph, state.metadata_store)
    assert mail_users == ["pkg/other.py"]
    # Protocol method: only code typed against the protocol, never untyped .read() calls
    reader_users = symbol_users("typed", "Reader.read", "pkg/proto.py", state.graph, state.metadata_store)
    assert reader_users == ["pkg/proto.py"]


def test_file_symbol_and_batch_impact_end_to_end(tmp_path, monkeypatch):
    from codebase_intel.core.impact import analyze_impact, analyze_impact_batch
    state = _index(tmp_path, monkeypatch, ADAPTER_REPO, "dif")
    args = ("dif", state.graph, state.metadata_store)

    by_file = _listed(analyze_impact("pkg/adapters.py", *args, cochange=state.cochange))
    assert by_file["pkg/sender.py"].reason == "direct import"
    assert by_file["pkg/closer.py"].reason == "direct import"
    assert "pkg/adapters.py" not in by_file

    by_symbol = _listed(analyze_impact("Adapter.send", *args, cochange=state.cochange))
    ranked = list(by_symbol)
    assert ranked.index("pkg/sender.py") < ranked.index("pkg/closer.py")  # only sender calls send
    assert by_symbol["pkg/sender.py"].reason.startswith("references symbol")

    batch = _listed(analyze_impact_batch(["pkg/sender.py", "pkg/adapters.py"], *args, cochange=state.cochange))
    assert batch["pkg/closer.py"].triggered_by == ["pkg/adapters.py"]
