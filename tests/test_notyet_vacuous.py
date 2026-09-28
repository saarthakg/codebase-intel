"""Vacuous new tests: new or edited tests run against the session-start code."""
from notyet import gate, snapshot, store
from notyet.engines import execution
from tests.conftest import _git, _write
from tests.test_notyet_execution import check, rules

BUGGY_SUB = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return b - a\n"
FIXED_SUB = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
CALC_TESTS = "from pkg.calc import add, sub\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def with_buggy_sub(repo):
    """sub is broken at session start, and its old test is gone (so the bug went unnoticed)."""
    _write(repo, "pkg/calc.py", BUGGY_SUB)
    _write(repo, "tests/test_calc.py", CALC_TESTS)
    _git(repo, "commit", "-qam", "sub is broken")


def vacuous(result):
    return [f for f in result.findings if f.rule == "test-vacuous"]


def test_a_test_that_reproduces_the_bug_is_not_vacuous(repo):
    with_buggy_sub(repo)
    _write(repo, "pkg/calc.py", FIXED_SUB)
    _write(repo, "tests/test_calc.py", CALC_TESTS + "\n\ndef test_sub():\n    assert sub(3, 1) == 2\n")
    result = check(repo)
    assert rules(result) == []
    assert any("1 of 1 fail there" in c for c in result.checks)


def test_a_new_test_that_passes_before_the_fix_is_vacuous(repo):
    with_buggy_sub(repo)
    _write(repo, "pkg/calc.py", FIXED_SUB)
    # sub(2, 2) is 0 either way: this test can't tell the fix from the bug
    _write(repo, "tests/test_calc.py", CALC_TESTS + "\n\ndef test_sub():\n    assert sub(2, 2) == 0\n")
    result = check(repo)
    assert rules(result) == [("test-vacuous", "resolve", "tests/test_calc.py::test_sub")]
    assert vacuous(result)[0].evidence == ["tests/test_calc.py::test_sub"]


def test_regression_tests_next_to_a_real_one_are_fine(repo):
    """Passing on both sides is expected for tests of behavior that already
    worked; one test that tells old from new is enough."""
    with_buggy_sub(repo)
    _write(repo, "pkg/calc.py", FIXED_SUB)
    _write(repo, "tests/test_calc.py", CALC_TESTS + "\n\ndef test_sub():\n    assert sub(3, 1) == 2\n"
                                                    "\n\ndef test_sub_same():\n    assert sub(2, 2) == 0\n"
                                                    "\n\ndef test_add_zero():\n    assert add(0, 0) == 0\n")
    result = check(repo)
    assert vacuous(result) == []
    assert any("1 of 3 fail there" in c and "2 pass on both sides" in c for c in result.checks)


def test_parametrized_test_tells_the_change_if_any_case_does(repo):
    with_buggy_sub(repo)
    _write(repo, "pkg/calc.py", FIXED_SUB)
    _write(repo, "tests/test_calc.py", "import pytest\n\n" + CALC_TESTS +
           "\n\n@pytest.mark.parametrize('a,b,out', [(2, 2, 0), (3, 1, 2)])\n"
           "def test_sub(a, b, out):\n    assert sub(a, b) == out\n")
    assert vacuous(check(repo)) == []


def test_edited_test_counts_too(repo):
    _write(repo, "pkg/calc.py", "def add(a, b):\n    return b + a\n\n\ndef sub(a, b):\n    return a - b\n")
    _write(repo, "tests/test_calc.py", (repo / "tests/test_calc.py").read_text().replace("add(1, 2)", "add(2, 1)"))
    result = check(repo)
    assert [f.location for f in vacuous(result)] == ["tests/test_calc.py::test_add"]


def test_a_test_of_new_code_fails_at_session_start_by_importing_it(repo):
    _write(repo, "pkg/calc.py", FIXED_SUB + "\n\ndef mul(a, b):\n    return a * b\n")
    _write(repo, "tests/test_mul.py", "from pkg.calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n")
    result = check(repo)
    assert vacuous(result) == [] and any("1 of 1 fail there" in c for c in result.checks)


def test_sessions_that_only_add_tests_are_not_compared(repo):
    _write(repo, "tests/test_calc.py", (repo / "tests/test_calc.py").read_text()
           + "\n\ndef test_add_zero():\n    assert add(0, 0) == 0\n")
    result = check(repo)
    assert vacuous(result) == [] and not any("session-start code" in c for c in result.checks)


def test_new_tests_that_fail_now_are_left_to_new_test_failing(repo):
    _write(repo, "pkg/calc.py", FIXED_SUB + "\n# touched\n")
    _write(repo, "tests/test_calc.py", (repo / "tests/test_calc.py").read_text()
           + "\n\ndef test_wrong():\n    assert add(1, 1) == 3\n")
    assert rules(check(repo)) == [("new-test-failing", "block", "tests/test_calc.py::test_wrong")]


def test_new_conftest_fixture_is_copied_to_the_session_start_side(repo):
    with_buggy_sub(repo)
    _write(repo, "pkg/calc.py", FIXED_SUB)
    _write(repo, "tests/conftest.py", "import pytest\n\n\n@pytest.fixture\ndef pair():\n    return (3, 1)\n")
    _write(repo, "tests/test_calc.py", CALC_TESTS + "\n\ndef test_sub(pair):\n    assert sub(*pair) == 2\n")
    result = check(repo)
    assert vacuous(result) == []          # it fails on the bug, not on a missing fixture...
    assert any("1 of 1 fail there" in c for c in result.checks)


def test_a_refactor_with_characterization_tests_is_acknowledged_not_blocked(repo):
    """The legitimate both-sides case, through the gate in enforce mode: a
    fix-or-justify item, cleared by an acknowledgment with the reason."""
    _write(repo, "pkg/calc.py", "def add(a, b):\n    return sum((a, b))\n\n\ndef sub(a, b):\n    return a - b\n")
    _write(repo, "tests/test_calc.py", (repo / "tests/test_calc.py").read_text()
           + "\n\ndef test_add_negative():\n    assert add(-1, -2) == -3\n")
    root = str(repo)
    session = store.Session(session_id="r", started=0, baseline_tree=snapshot.head_tree(root),
                            baseline_head=snapshot.head_tree(root), baseline_source="session-start")
    first = gate.check(root, session)
    assert first.verdict == "blocked" and [f.rule for f in first.findings] == ["test-vacuous"]
    session.acks[first.findings[0].id] = {"category": "acknowledge", "reason": "refactor; behavior unchanged"}
    assert gate.check(root, session).verdict == "passed"


def test_the_session_start_run_is_cached(repo, monkeypatch):
    with_buggy_sub(repo)
    _write(repo, "pkg/calc.py", FIXED_SUB)
    _write(repo, "tests/test_calc.py", CALC_TESTS + "\n\ndef test_sub():\n    assert sub(2, 2) == 0\n")
    first = check(repo)
    calls = []
    real = snapshot.materialize
    monkeypatch.setattr(snapshot, "materialize", lambda *a: calls.append(a) or real(*a))
    _write(repo, "pkg/calc.py", FIXED_SUB + "\n# same fix\n")
    assert rules(check(repo)) == rules(first) and calls == []


def test_an_untrusted_session_start_checkout_is_not_checked(repo, monkeypatch):
    with_buggy_sub(repo)
    _write(repo, "pkg/calc.py", FIXED_SUB)
    _write(repo, "tests/test_calc.py", CALC_TESTS + "\n\ndef test_sub():\n    assert sub(2, 2) == 0\n")
    monkeypatch.setattr(execution, "_canary", lambda *a: (False, "canary failed"))
    from notyet.engines import vacuous as vacuous_mod
    monkeypatch.setattr(vacuous_mod, "_canary", lambda *a: (False, "canary failed"))
    result = check(repo)
    assert vacuous(result) == []
    assert any("new tests against the session-start code: canary failed" in n for n in result.not_checked)
