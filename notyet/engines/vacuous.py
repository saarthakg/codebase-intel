"""Vacuous new tests: does any test the session wrote actually test the change?

Each new or edited test function (see integrity.changed_functions: a new
`parametrize` case or a changed module-level value it uses counts) that
passes now is run against the session-start code: an isolated checkout of the session-start tree with the
session's test-side files (test modules, conftest.py, anything under a test
directory) copied in. A test that fails there tells the old code from the new
one. A test that passes there too doesn't exercise the change.

That alone isn't wrong: regression tests for behavior that already worked,
and characterization tests around a refactor, pass on both sides by design.
So the rule is per session, not per test:
  - the session changed source code, and none of its new or edited tests
    fails at session start: fix or justify ("test-vacuous"). A refactor or a
    regression test is a fine justification;
  - at least one fails at session start: the change is tested; the ones that
    pass on both sides are only counted on the receipt;
  - only tests changed: nothing to compare, not reported.
The same import canary as the other session-start runs guards the checkout.
"""
import hashlib
import os
import tempfile
from pathlib import PurePosixPath

from notyet import snapshot, testrun
from notyet.engines.execution import (
    _BaselineCache,
    _canary,
    in_test_dir,
    is_test_module,
)
from notyet.engines.integrity import changed_functions
from notyet.findings import Context, EngineResult, Finding
from notyet.pyresolve import find_python_source_roots

MAX_TESTS = 60
MAX_EVIDENCE = 5


def is_test_side(path: str) -> bool:
    return is_test_module(path) or PurePosixPath(path).name == "conftest.py" or in_test_dir(path)


def written_tests(ctx: Context) -> list[str]:
    """Node ids ("file::func", "file::Class::func") of test functions the session added or edited."""
    out = []
    for d in ctx.deltas:
        if d.status == "D" or not is_test_module(d.path):
            continue
        before = (snapshot.show(ctx.root, ctx.baseline_tree, d.old_path or d.path) or "") if d.status != "A" else ""
        after = snapshot.show(ctx.root, ctx.current_tree, d.path) or ""
        out += [f"{d.path}::{name}" for name in changed_functions(before, after)]
    return sorted(out)


def check(ctx: Context, command: str, current: dict[str, testrun.TestResult], result: EngineResult) -> None:
    source_changed = any(d.path.endswith(".py") and not is_test_side(d.path) for d in ctx.deltas)
    if not source_changed:
        return
    passing_now = [n for n in written_tests(ctx) if testrun.outcome_of(n, current) == "passed"][:MAX_TESTS]
    if not passing_now:
        return
    before, why = _run_at_session_start(ctx, command, passing_now)
    if why:
        result.not_checked.append(f"new tests against the session-start code: {why}")
        return
    outcomes = {n: testrun.outcome_of(n, before) for n in passing_now}
    tells = sorted(n for n, o in outcomes.items() if o == "failed")
    same = sorted(n for n, o in outcomes.items() if o == "passed")
    unknown = len(outcomes) - len(tells) - len(same)
    tail = f"; {unknown} couldn't be compared (skipped or not collected there)" if unknown else ""
    if tells:
        result.checks.append(
            f"new tests against the session-start code: {len(tells)} of {len(outcomes)} fail there, so they test "
            f"the change; {len(same)} pass on both sides (regression or characterization tests){tail}")
    elif same:
        result.findings.append(Finding(
            rule="test-vacuous", severity="resolve", location=same[0], key="|".join(same),
            title=f"none of the {len(same)} new or edited test(s) fails on the session-start code, "
                  f"so none of them tests what this session changed",
            evidence=same[:MAX_EVIDENCE] + ([f"… and {len(same) - MAX_EVIDENCE} more"] if len(same) > MAX_EVIDENCE else []),
            action="Add a test that fails without your change (for a bug fix: one that reproduces the bug). "
                   "If passing on both sides is right, e.g. a refactor that keeps behavior or a regression test "
                   "for behavior that already worked, acknowledge it with that reason."))
    else:
        result.not_checked.append(f"new tests against the session-start code: none of {len(outcomes)} could be "
                                  f"compared (skipped or not collected there)")


def _run_at_session_start(ctx: Context, command: str, node_ids: list[str]) -> tuple[dict[str, testrun.TestResult], str]:
    """Run `node_ids` on the session-start tree plus the session's test-side
    files. Cached per session-start tree and test-side content."""
    overlay = sorted(d.path for d in ctx.deltas if d.status != "D" and is_test_side(d.path))
    listing = snapshot.git(ctx.root, "ls-tree", "-r", ctx.current_tree, "--", *overlay) if overlay else ""
    key = hashlib.sha1(("|".join(sorted(node_ids)) + "\n" + listing).encode()).hexdigest()[:16]
    cache = _BaselineCache(ctx.root, ctx.baseline_tree, command)
    if key in cache.vacuous:
        return {n: testrun.TestResult(n, *r) for n, r in cache.vacuous[key].items()}, ""

    files = sorted({n.split("::")[0] for n in node_ids})
    dirs = sorted({os.path.dirname(p) for p in files})
    with tempfile.TemporaryDirectory(prefix="notyet-vacuous-") as tmp:
        tmp = os.path.realpath(tmp)
        snapshot.materialize(ctx.root, ctx.baseline_tree, tmp)
        if overlay:
            snapshot.materialize(ctx.root, ctx.current_tree, tmp, paths=overlay)
        env = {"PYTHONPATH": os.pathsep.join(str(r) for r in find_python_source_roots(tmp))}
        # a copied-in conftest.py could change how imports resolve: prove it again
        if any(PurePosixPath(p).name == "conftest.py" for p in overlay) or not cache.canary_ok(dirs):
            ok, why = _canary(ctx, tmp, files, env)
            if not ok:
                return {}, why
            if not any(PurePosixPath(p).name == "conftest.py" for p in overlay):
                cache.canary_passed(dirs)
        timeout = max(30.0, min(ctx.config.budget_seconds, ctx.time_left() - 5))
        run = testrun.run_tests(command, tmp, node_ids, env=env, timeout=timeout)
    if run.crashed:
        return {}, f"the test command failed on the session-start tree ({run.crashed})"
    if run.timed_out:
        return {}, "ran out of time"
    cache.vacuous = {key: {n: [r.outcome, r.message[:300]] for n, r in run.results.items()}}
    cache.save()
    return run.results, ""
