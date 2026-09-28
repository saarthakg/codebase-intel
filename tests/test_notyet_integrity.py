"""The integrity engine: tests weakened instead of code fixed."""
from notyet.engines import integrity
from tests.conftest import _write


def check(repo):
    from notyet import config, snapshot, store
    from notyet.findings import Context
    root = str(repo)
    base = snapshot.head_tree(root)
    session = store.Session(session_id="t", started=0, baseline_tree=base, baseline_head=base,
                            baseline_source="session-start")
    current = snapshot.snapshot(root)
    ctx = Context(root=root, session=session, config=config.load(root, base), baseline_tree=base,
                  current_tree=current, deltas=snapshot.diff_trees(root, base, current))
    return integrity.run(ctx)


def rules(result):
    return sorted((f.rule, f.severity, f.location) for f in result.findings)


BROKEN_ADD = "def add(a, b):\n    return a + b + 1\n\n\ndef sub(a, b):\n    return a - b\n"


def test_skipping_a_failing_test_blocks(repo):
    _write(repo, "pkg/calc.py", BROKEN_ADD)
    _write(repo, "tests/test_calc.py", "import pytest\nfrom pkg.calc import add, sub\n\n\n@pytest.mark.skip(reason='flaky')\n"
                                       "def test_add():\n    assert add(1, 2) == 3\n\n\ndef test_sub():\n    assert sub(3, 1) == 2\n")
    assert rules(check(repo)) == [("test-disabled", "block", "tests/test_calc.py::test_add")]


def test_skip_call_in_body_and_module_pytestmark(repo):
    _write(repo, "tests/test_calc.py", "import pytest\nfrom pkg.calc import add, sub\n\n\ndef test_add():\n"
                                       "    pytest.skip('later')\n    assert add(1, 2) == 3\n\n\n"
                                       "def test_sub():\n    assert sub(3, 1) == 2\n")
    _write(repo, "tests/test_strings.py", "import pytest\nfrom pkg.strings import shout\n\npytestmark = pytest.mark.xfail\n\n\n"
                                          "def test_shout():\n    assert shout('a') == 'A'\n")
    found = [r[2] for r in rules(check(repo)) if r[0] == "test-disabled"]
    assert found == ["tests/test_calc.py::test_add", "tests/test_strings.py::test_shout"]


def test_editing_the_expected_value_to_match_a_bug_blocks(repo):
    _write(repo, "pkg/calc.py", BROKEN_ADD)
    _write(repo, "tests/test_calc.py", "from pkg.calc import add, sub\n\n\ndef test_add():\n    assert add(1, 2) == 4\n\n\n"
                                       "def test_sub():\n    assert sub(3, 1) == 2\n")
    result = check(repo)
    assert rules(result) == [("test-changed-to-pass", "block", "tests/test_calc.py::test_add")]
    assert any("assert" in e for e in result.findings[0].evidence)


def test_removing_assertions_is_fix_or_justify(repo):
    _write(repo, "tests/test_calc.py", "from pkg.calc import add, sub\n\n\ndef test_add():\n    add(1, 2)\n\n\n"
                                       "def test_sub():\n    assert sub(3, 1) == 2\n")
    assert rules(check(repo)) == [("assertions-removed", "resolve", "tests/test_calc.py::test_add")]


def test_strengthening_or_refactoring_a_test_is_fine(repo):
    _write(repo, "tests/test_calc.py", "from pkg.calc import add, sub\n\n\ndef test_add():\n    total = add(1, 2)\n"
                                       "    assert total == 3\n    assert add(0, 0) == 0\n\n\n"
                                       "def test_sub():\n    assert sub(3, 1) == 2\n")
    result = check(repo)
    assert result.findings == [] and "ran the session-start version of 1 edited test(s)" in result.checks[0]


def test_a_requested_behavior_change_blocks_until_handed_to_the_user(repo):
    """Changing a test's expectation is legitimate when the user asked for the
    new behavior; the agent then hands it over with needs-human, and the gate
    lets it stop."""
    from notyet import gate, snapshot, store
    root = str(repo)
    base = snapshot.head_tree(root)
    session = store.Session(session_id="req", started=0, baseline_tree=base, baseline_head=base,
                            baseline_source="session-start")
    _write(repo, "pkg/calc.py", "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return b - a\n")
    _write(repo, "tests/test_calc.py", "from pkg.calc import add, sub\n\n\ndef test_add():\n    assert add(1, 2) == 3\n\n\n"
                                       "def test_sub():\n    assert sub(3, 1) == -2\n")
    _write(repo, ".notyet.toml", (repo / ".notyet.toml").read_text())
    first = gate.check(root, session)
    assert first.verdict == "blocked" and [f.rule for f in first.findings] == ["test-changed-to-pass"]
    fid = first.findings[0].id
    session.acks[fid] = {"reason": "user asked for sub to be reversed", "category": "needs-human"}
    assert gate.check(root, session).verdict == "needs-review"      # may stop; not a pass


def _commit(repo, rel, text):
    from tests.conftest import _git
    _write(repo, rel, text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", f"set up {rel}")


PARAM_TEST = ("import pytest\nfrom pkg.calc import add\n\n\n"
              "@pytest.mark.parametrize('a,b,out', [(1, 2, {})])\ndef test_add(a, b, out):\n    assert add(a, b) == out\n")


def test_changing_a_parametrize_expectation_to_match_a_bug_blocks(repo):
    _commit(repo, "tests/test_calc.py", PARAM_TEST.format(3))
    _write(repo, "pkg/calc.py", BROKEN_ADD)
    _write(repo, "tests/test_calc.py", PARAM_TEST.format(4))        # the body is untouched
    assert rules(check(repo)) == [("test-changed-to-pass", "block", "tests/test_calc.py::test_add")]


CONST_TEST = "from pkg.calc import add\n\nEXPECTED = {}\n\n\ndef test_add():\n    assert add(1, 2) == EXPECTED\n"


def test_changing_a_module_level_expectation_to_match_a_bug_blocks(repo):
    _commit(repo, "tests/test_calc.py", CONST_TEST.format(3))
    _write(repo, "pkg/calc.py", BROKEN_ADD)
    _write(repo, "tests/test_calc.py", CONST_TEST.format(4))
    assert rules(check(repo)) == [("test-changed-to-pass", "block", "tests/test_calc.py::test_add")]


def test_a_test_file_that_cannot_import_does_not_hide_a_changed_expectation(repo):
    """The session renames shapes.perimeter (and its test's import); the old
    test_shapes.py can't import the new code, and pytest then drops every
    other targeted result unless run file by file."""
    _write(repo, "pkg/calc.py", BROKEN_ADD)
    _write(repo, "pkg/shapes.py", "from pkg.calc import add\n\n\ndef outline(w, h):\n    return 2 * (w + h)\n")
    _write(repo, "tests/test_shapes.py", "from pkg.shapes import outline\n\n\ndef test_perimeter():\n"
                                         "    assert outline(2, 3) == 10\n")
    _write(repo, "tests/test_calc.py", (repo / "tests/test_calc.py").read_text().replace("== 3", "== 4"))
    found = rules(check(repo))
    assert ("test-changed-to-pass", "block", "tests/test_calc.py::test_add") in found
    assert ("test-changed-to-pass", "block", "tests/test_shapes.py::test_perimeter") in found   # renamed API


def test_changed_functions_reads_decorators_and_module_values():
    before = "X = 1\n\n\ndef helper():\n    return 1\n\n\ndef test_a():\n    assert X\n\n\ndef test_b():\n    assert helper()\n\n\ndef test_c():\n    pass\n"
    after = before.replace("X = 1", "X = 2") + "\n\ndef test_d():\n    pass\n"
    assert integrity.changed_functions(before, after) == {"test_a": "edited", "test_d": "new"}
    after = before.replace("return 1", "return 2")
    assert integrity.changed_functions(before, after) == {"test_b": "edited"}
