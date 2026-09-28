"""Advisory co-change lines from git history."""
from notyet import config, snapshot, store
from notyet.engines import history as advice
from notyet.findings import Context
from tests.conftest import _git, _write


def check(repo):
    root = str(repo)
    base = snapshot.head_tree(root)
    session = store.Session(session_id="h", started=0, baseline_tree=base, baseline_head=base,
                            baseline_source="session-start")
    current = snapshot.snapshot(root)
    ctx = Context(root=root, session=session, config=config.load(root, base), baseline_tree=base,
                  current_tree=current, deltas=snapshot.diff_trees(root, base, current))
    return advice.run(ctx)


def test_a_usual_companion_left_untouched_is_mentioned(repo):
    _write(repo, "CHANGES.md", "")
    for i in range(4):   # calc.py and CHANGES.md change together 4 times; strings.py once
        _write(repo, "pkg/calc.py", (repo / "pkg/calc.py").read_text() + f"\n# v{i}\n")
        _write(repo, "CHANGES.md", (repo / "CHANGES.md").read_text() + f"- change {i}\n")
        if i == 0:
            _write(repo, "pkg/strings.py", (repo / "pkg/strings.py").read_text() + "\n# once\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", f"change {i}")
    last = _git(repo, "rev-parse", "HEAD").strip()
    _write(repo, "pkg/calc.py", (repo / "pkg/calc.py").read_text() + "\n# now\n")
    result = check(repo)
    assert result.advice == [f"pkg/calc.py usually changes with CHANGES.md, which this session didn't touch: "
                             f"in 4 of 5 changes, e.g. {last[:8]}"]
    assert result.findings == []

    _write(repo, "CHANGES.md", (repo / "CHANGES.md").read_text() + "- now\n")   # touched: nothing to say
    assert check(repo).advice == []


def test_no_history_no_advice(repo):
    _write(repo, "pkg/calc.py", (repo / "pkg/calc.py").read_text() + "\n# x\n")
    assert check(repo).advice == []
