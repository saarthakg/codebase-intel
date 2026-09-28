import pytest

from codebase_intel.core import scoring
from codebase_intel.core.features import Candidate, gather
from codebase_intel.core.graph import DependencyGraph
from codebase_intel.core.history import CoChange, Commit
from codebase_intel.core.predict import predict, target_change
from tests.test_diffs import ADAPTER_REPO, _index


def _graph(*edges, files=()):
    g = DependencyGraph()
    for f in files:
        g.add_file(f)
    for a, b in edges:
        g.add_file(a)
        g.add_file(b)
        g.add_import_edge(a, b)
    return g


class _NoSymbols:
    def find_symbol(self, *a):
        return []

    def symbols_in_file(self, *a):
        return []


def test_gather_collects_each_kind_of_evidence():
    g = _graph(("A.py", "B.py"), ("B.py", "C.py"), files=("tests/test_C.py", "docs/c.md"))
    history = CoChange.from_commits([Commit(str(i), "d", ["C.py", "docs/c.md"]) for i in range(3)])
    cands = gather({"C.py": []}, "r", g, _NoSymbols(), history)
    assert "C.py" not in cands                                   # the change itself is never a candidate
    assert cands["B.py"].import_hops == 1 and cands["A.py"].import_hops == 2
    assert cands["tests/test_C.py"].named_test == "C.py" and cands["tests/test_C.py"].kind == "test"
    doc = cands["docs/c.md"]
    assert (doc.cc_n, doc.cc_with, doc.cc_total, doc.kind) == (3, "C.py", 3, "docs")
    assert doc.cc_p == pytest.approx(3 / (3 + 3))
    assert cands["B.py"].because_of == {"C.py"}


def test_a_changed_test_points_at_the_module_it_tests():
    g = _graph(files=("pkg/utils.py", "tests/test_utils.py"))
    cands = gather({"tests/test_utils.py": []}, "r", g, _NoSymbols(), None)
    assert cands["pkg/utils.py"].tested_by_change == "tests/test_utils.py"


def test_probability_rises_with_history_and_is_readable():
    weak = Candidate(file="x.py", import_hops=2, imports="a.py")
    strong = Candidate(file="x.py", import_hops=2, imports="a.py", cc_p=0.6, cc_rev_p=0.6, cc_n=6,
                       cc_queries=1, cc_with="a.py", cc_total=9, commits=10)
    assert scoring.probability(strong) > 5 * scoring.probability(weak)
    assert scoring.reasons(strong)[0] == "changed together with a.py in 6 of its 9 changes"
    assert "imports a.py indirectly (2 hops)" in scoring.reasons(strong)
    # The dict form (replay datasets) scores the same as the object
    from dataclasses import asdict
    assert scoring.probability(asdict(strong)) == pytest.approx(scoring.probability(strong))


def test_predict_end_to_end(tmp_path, monkeypatch):
    state = _index(tmp_path, monkeypatch, ADAPTER_REPO, "dif")

    by_file = {p.file: p for p in predict({"pkg/adapters.py": []}, "dif", state)}
    assert {"pkg/sender.py", "pkg/closer.py"} <= set(by_file)   # every importer is a candidate
    assert "imports pkg/adapters.py" in by_file["pkg/sender.py"].reasons

    ranked = predict({"pkg/adapters.py": ["Adapter.send"]}, "dif", state)
    files = [p.file for p in ranked]
    assert files.index("pkg/sender.py") < files.index("pkg/closer.py")  # only sender calls send
    sender = ranked[files.index("pkg/sender.py")]
    assert sender.uses == [("pkg/adapters.py", "Adapter.send")]
    assert any(r.startswith("uses changed Adapter.send") for r in sender.reasons)
    assert "other.py" not in {p.file for p in ranked if p.uses}   # its own class's send() doesn't count


def test_target_change_resolves_files_and_symbols(tmp_path, monkeypatch):
    state = _index(tmp_path, monkeypatch, ADAPTER_REPO, "dif")
    assert target_change("pkg/adapters.py", "dif", state) == {"pkg/adapters.py": []}
    assert target_change("Adapter.send", "dif", state) == {"pkg/adapters.py": ["Adapter.send"]}
    assert target_change("nope", "dif", state) == {}
