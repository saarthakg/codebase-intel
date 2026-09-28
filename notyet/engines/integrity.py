"""Test integrity: did the session weaken the tests instead of fixing the code?

For every test function that existed at session start and was edited:
  - a skip or xfail added (decorator, `pytest.skip()` in the body, a module
    `pytestmark`): the test no longer checks anything. Block.
  - the session-start version of the test, run against the new code, fails:
    the test was changed to accept new behavior. Block: only a fix, or
    handing it to the user (a requested behavior change is a fine reason).
  - fewer assertions than before: fix or justify.

The session-start version runs in a checkout of the current tree with just
that test file swapped back, so it tests exactly the code the agent wrote.
A failure there only blocks if the same test passed on the session-start
tree (checked in an isolated checkout, behind the import canary).
"""
import ast
import os
import tempfile
from dataclasses import dataclass

from notyet import snapshot, testrun
from notyet.engines.execution import _canary, is_test_module
from notyet.findings import Context, EngineResult, Finding
from notyet.pyresolve import find_python_source_roots

DISABLING_DECORATORS = ("mark.skip", "mark.xfail", "unittest.skip", "skipIf", "skipUnless", "expectedFailure")
DISABLING_CALLS = ("pytest.skip", "pytest.xfail", "self.skipTest", "pytest.importorskip")
MAX_OLD_TEST_RUNS = 40


@dataclass
class TestFunc:
    name: str                 # "test_x" or "TestC::test_x"
    node: ast.AST
    disablers: frozenset[str]
    assertions: int
    body: str                 # normalized source of the body, to tell edits apart


def run(ctx: Context) -> EngineResult:
    result = EngineResult()
    edited_files = [(d.old_path or d.path, d.path) for d in ctx.deltas
                    if d.status in ("M", "R") and is_test_module(d.path)]
    if not edited_files:
        return result

    to_rerun: dict[str, list[str]] = {}          # current path → edited test functions
    old_paths: dict[str, str] = {}
    for old_path, path in edited_files:
        before = _functions(snapshot.show(ctx.root, ctx.baseline_tree, old_path) or "")
        after = _functions(snapshot.show(ctx.root, ctx.current_tree, path) or "")
        for name in sorted(before.keys() & after.keys()):
            b, a = before[name], after[name]
            nid = f"{path}::{name}"
            added = sorted(a.disablers - b.disablers)
            if added:
                result.findings.append(Finding(
                    rule="test-disabled", severity="block", location=nid,
                    title=f"{nid} was switched off this session ({', '.join(added)})",
                    evidence=added[:3],
                    action="Remove the skip/xfail and fix the code. If the test should really be off, "
                           "hand it to the user: `ack <id> --needs-human \"why\"`."))
                continue
            if a.assertions < b.assertions:
                result.findings.append(Finding(
                    rule="assertions-removed", severity="resolve", location=nid,
                    title=f"{nid} checks less than it did: {b.assertions} assertion(s) at session start, "
                          f"{a.assertions} now",
                    action="Restore the assertions, or acknowledge why they no longer apply."))
            if a.body != b.body:
                to_rerun.setdefault(path, []).append(name)
                old_paths[path] = old_path

    if to_rerun:
        _run_old_versions(ctx, to_rerun, old_paths, result)
    return result


def _run_old_versions(ctx: Context, to_rerun: dict[str, list[str]], old_paths: dict[str, str],
                      result: EngineResult) -> None:
    """Run the session-start version of each edited test against the new code."""
    cfg = ctx.config
    if not cfg.test_command:
        return
    command = testrun.anchored(cfg.test_command, ctx.root)
    node_ids = [f"{p}::{n}" for p, names in to_rerun.items() for n in names][:MAX_OLD_TEST_RUNS]
    with tempfile.TemporaryDirectory(prefix="notyet-oldtests-") as tmp:
        tmp = os.path.realpath(tmp)
        snapshot.materialize(ctx.root, ctx.current_tree, tmp)
        for path in to_rerun:
            with open(os.path.join(tmp, path), "w") as f:
                f.write(snapshot.show(ctx.root, ctx.baseline_tree, old_paths[path]) or "")
        env = {"PYTHONPATH": os.pathsep.join(str(r) for r in find_python_source_roots(tmp))}
        old_on_new = testrun.run_pytest(command, tmp, node_ids, env=env, timeout=max(30, cfg.budget_seconds))
    if old_on_new.crashed:
        result.not_checked.append(f"session-start versions of edited tests: {old_on_new.crashed}")
        return
    failing = [r for r in old_on_new.results.values() if r.bad]
    result.checks.append(f"ran the session-start version of {len(node_ids)} edited test(s) against the new code "
                         f"({len(failing)} failed)")
    if not failing:
        return

    # Did those tests pass on the session-start tree? Otherwise they prove nothing.
    passed_before, why = _passed_at_session_start(ctx, command, [r.nodeid for r in failing])
    if why:
        result.not_checked.append(f"session-start versions of edited tests: {why}")
    for r in failing:
        evidence = [line for line in r.message.splitlines() if line.strip()][:3]
        if r.nodeid in passed_before:
            result.findings.append(Finding(
                rule="test-changed-to-pass", severity="block", location=r.nodeid,
                title=f"{r.nodeid} as it was at session start fails on the new code; the test was edited "
                      f"instead", evidence=evidence,
                action="Fix the code so the original test passes. If the behavior change was asked for, "
                       "hand it to the user: `ack <id> --needs-human \"the change that was requested\"`."))
        elif why:
            result.findings.append(Finding(
                rule="test-changed-to-pass", severity="resolve", location=r.nodeid,
                title=f"{r.nodeid} as it was at session start fails on the new code (couldn't confirm it "
                      f"passed at session start)", evidence=evidence,
                action="Fix the code, or acknowledge the behavior change with the reason."))


def _passed_at_session_start(ctx: Context, command: str, node_ids: list[str]) -> tuple[set[str], str]:
    with tempfile.TemporaryDirectory(prefix="notyet-baseline-") as tmp:
        tmp = os.path.realpath(tmp)
        snapshot.materialize(ctx.root, ctx.baseline_tree, tmp)
        env = {"PYTHONPATH": os.pathsep.join(str(r) for r in find_python_source_roots(tmp))}
        files = sorted({n.split("::")[0] for n in node_ids if os.path.exists(os.path.join(tmp, n.split("::")[0]))})
        if not files:
            return set(), "the edited test files didn't exist at session start (renamed?)"
        ok, why = _canary(ctx, tmp, files, env)
        if not ok:
            return set(), why
        run = testrun.run_pytest(command, tmp, node_ids, env=env, timeout=max(30, ctx.config.budget_seconds))
    if run.crashed:
        return set(), f"the test command failed on the session-start tree ({run.crashed})"
    return {n for n, r in run.results.items() if r.outcome == "passed"}, ""


# ── reading test functions ─────────────────────────────────────────────────────

def _functions(source: str) -> dict[str, TestFunc]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    module_marks = _module_disablers(tree)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            out[node.name] = _read(node.name, node, module_marks)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            class_marks = module_marks | _decorator_disablers(node)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                    out[f"{node.name}::{item.name}"] = _read(f"{node.name}::{item.name}", item, class_marks)
    return out


def _read(name: str, node, inherited: frozenset[str]) -> TestFunc:
    disablers = set(inherited) | _decorator_disablers(node)
    assertions = 0
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert):
            assertions += 1
        elif isinstance(sub, ast.Call):
            func = ast.unparse(sub.func)
            if func.split(".")[-1].startswith("assert") or func in ("pytest.raises", "pytest.warns"):
                assertions += 1
            if func in DISABLING_CALLS:
                disablers.add(f"{func}()")
    body = "\n".join(ast.unparse(stmt) for stmt in node.body)
    return TestFunc(name, node, frozenset(disablers), assertions, body)


def _decorator_disablers(node) -> frozenset[str]:
    found = set()
    for dec in getattr(node, "decorator_list", []):
        text = ast.unparse(dec)
        if any(k in text for k in DISABLING_DECORATORS):
            found.add(f"@{text}")
    return frozenset(found)


def _module_disablers(tree: ast.Module) -> frozenset[str]:
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            text = ast.unparse(node.value)
            if any(k in text for k in DISABLING_DECORATORS):
                found.add(f"pytestmark = {text}")
    return frozenset(found)
