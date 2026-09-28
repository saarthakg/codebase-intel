"""Execution evidence: run the tests that exercise the change, and compare
failures with the tree as it was at session start.

Selection, most relevant first:
  1. test files the change added or modified;
  2. test files importing a changed module, directly or through other modules;
  3. test files named after a changed module (test_<name>.py);
  4. for a changed conftest.py or test helper, every test file under its directory.

Every failure is re-run once on the current tree (a pass means flaky, never a
block), then run on an isolated checkout of the session-start tree. A canary
test proves that checkout imports its own code rather than the working tree
(editable installs can silently redirect imports); without that proof a
failure can't be called a regression, and it's reported as fix-or-justify.
"""
import ast
import configparser
import os
import tempfile
import time
import tomllib
from pathlib import PurePosixPath

from notyet.pyresolve import find_python_source_roots
from notyet import pyimports, snapshot, testrun
from notyet.findings import Context, EngineResult, Finding

DOC_SUFFIXES = (".md", ".rst", ".txt", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico")
BATCH_FILES = 12
MAX_UNVERIFIED_ITEMS = 5
CANARY = "test_notyet_baseline_canary.py"


def is_test_module(path: str) -> bool:
    name = PurePosixPath(path).name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def in_test_dir(path: str) -> bool:
    return any(p in ("tests", "test", "testing") for p in PurePosixPath(path).parts[:-1])


def pytest_testpaths(root: str) -> list[str]:
    """`testpaths` from the repo's pytest configuration, in pytest's order of precedence."""
    for name in ("pytest.ini", ".pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        if name == "pyproject.toml":
            try:
                with open(path, "rb") as f:
                    data = tomllib.load(f)
            except (OSError, tomllib.TOMLDecodeError):
                continue
            tool = data.get("tool", {}).get("pytest", {})
            opts = tool.get("ini_options", tool) if tool else None
            if opts is None:
                continue
            paths = opts.get("testpaths", [])
            return [paths] if isinstance(paths, str) else [str(p) for p in paths]
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(path)
        except configparser.Error:
            continue
        section = "tool:pytest" if name == "setup.cfg" else "pytest"
        if parser.has_section(section):
            return parser.get(section, "testpaths", fallback="").split()
    return []


def select_tests(ctx: Context, graph: pyimports.ImportGraph) -> tuple[list[tuple[str, int, str]], list[str]]:
    """[(test file, priority, why)] and notes about the selection."""
    notes = []
    current_tests = {p for p in graph.blobs if is_test_module(p)}
    testpaths = pytest_testpaths(ctx.root)
    if testpaths:   # the repo's own pytest setup only runs these; neither do we
        current_tests = {p for p in current_tests
                         if any(p == t or p.startswith(t.rstrip("/") + "/") for t in testpaths)}
    picked: dict[str, tuple[int, str]] = {}

    def pick(path: str, priority: int, why: str) -> None:
        if path in current_tests and (path not in picked or picked[path][0] > priority):
            picked[path] = (priority, why)

    changed = [d for d in ctx.deltas if d.status != "D"]
    deleted = [d.path for d in ctx.deltas if d.status == "D"] + [d.old_path for d in ctx.deltas if d.old_path]
    changed_src = [d.path for d in changed if d.path.endswith(".py") and not is_test_module(d.path)]

    for d in changed:
        if is_test_module(d.path):
            pick(d.path, 0, "changed test")
    for path, depth in graph.importers(changed_src).items():
        pick(path, depth, "imports a changed module" if depth == 1 else f"imports a changed module ({depth} steps)")
    stems = {PurePosixPath(p).stem for p in changed_src} - {"__init__"}
    for t in current_tests:
        stem = PurePosixPath(t).stem
        if stem.removeprefix("test_").removesuffix("_test") in stems:
            pick(t, 1, "named after a changed module")
    for d in changed:
        name = PurePosixPath(d.path).name
        if d.path.endswith(".py") and (name == "conftest.py" or (in_test_dir(d.path) and not is_test_module(d.path))):
            folder = str(PurePosixPath(d.path).parent)
            for t in current_tests:
                if folder in (".", "") or t.startswith(folder + "/"):
                    pick(t, 2, f"under changed {name}")
    if deleted and any(p.endswith(".py") for p in deleted):
        notes.append("deleted Python modules: importers can't be traced from the current tree; "
                     "their tests are only run if selected another way")
    other = [d.path for d in changed if not d.path.endswith(".py") and not d.path.endswith(DOC_SUFFIXES)]
    if other:
        for t in current_tests:
            try:
                text = open(os.path.join(ctx.root, t), encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for o in other:
                if PurePosixPath(o).name in text:
                    pick(t, 2, f"mentions {PurePosixPath(o).name}")
        notes.append(f"non-Python changes ({len(other)} file(s)): tests chosen by mentioning the file name")
    ordered = sorted(((p, pr, why) for p, (pr, why) in picked.items()), key=lambda x: (x[1], x[0]))
    return ordered, notes


def _batches(tests: list[tuple[str, int, str]]) -> list[list[str]]:
    """The most relevant files first, in one batch; the rest in chunks, so a
    time budget cut still leaves complete results for what ran."""
    first = [t for t, pr, _ in tests if pr <= 1]
    rest = [t for t, pr, _ in tests if pr > 1]
    batches = [first] if first else []
    batches += [rest[i:i + BATCH_FILES] for i in range(0, len(rest), BATCH_FILES)]
    return batches


def _test_functions(source: str) -> set[str]:
    """Node-id suffixes of the test functions a test module defines."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            out.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            out |= {f"{node.name}::{n.name}" for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")}
    return out


def _strip_params(nodeid: str) -> str:
    return nodeid.split("[", 1)[0]


def run(ctx: Context) -> EngineResult:
    result = EngineResult()
    cfg = ctx.config
    if not cfg.test_command:
        result.not_checked.append("tests: no test command configured (run `notyet init`)")
        return result
    if all(d.path.endswith(DOC_SUFFIXES) for d in ctx.deltas):
        result.checks.append("tests: not needed; only documentation changed")
        return result

    command = testrun.anchored(cfg.test_command, ctx.root)
    started = time.monotonic()
    graph = pyimports.ImportGraph(ctx.root, ctx.current_tree)
    tests, notes = select_tests(ctx, graph)
    result.not_checked += notes
    if not tests:
        result.not_checked.append("tests: no tests exercise the changed files (none import them or are named for them)")
        return result

    # ── run the selection on the current tree ────────────────────────────────
    current = testrun.Run()
    skipped_files: list[str] = []
    crashed: list[tuple[list[str], str]] = []
    for batch in _batches(tests):
        remaining = cfg.budget_seconds - (time.monotonic() - started)
        if remaining <= 1:
            skipped_files += batch
            continue
        run_ = testrun.run_pytest(command, ctx.root, batch, timeout=remaining)
        if run_.crashed:
            crashed.append((batch, run_.crashed))
            continue
        if run_.timed_out:
            skipped_files += [f for f in batch if not any(n.startswith(f + "::") for n in run_.results)]
        current.results.update(run_.results)
    for batch, why in crashed:
        _report_crash(ctx, command, batch, why, result)
    if crashed and not current.results:
        return result
    ran_files = sorted({n.split("::")[0] for n in current.results})
    if skipped_files:
        result.not_checked.append(f"tests: the {cfg.budget_seconds}s budget ran out before {len(skipped_files)} "
                                  f"selected test file(s) ran: " + ", ".join(skipped_files[:5])
                                  + (" …" if len(skipped_files) > 5 else ""))

    failing = current.failures
    flaky: list[str] = []
    if failing:  # re-run failures once: a pass means flaky, not broken
        rerun = testrun.run_pytest(command, ctx.root, [f.nodeid for f in failing],
                                   timeout=max(30, cfg.budget_seconds / 2))
        for f in list(failing):
            again = rerun.results.get(f.nodeid)
            if again is not None and not again.bad:
                flaky.append(f.nodeid)
        failing = [f for f in failing if f.nodeid not in flaky]

    # ── compare with session start ───────────────────────────────────────────
    changed_tests = [d.path for d in ctx.deltas if d.status != "D" and is_test_module(d.path)]
    need_baseline = bool(failing) or bool(changed_tests)
    baseline: dict[str, testrun.TestResult] = {}
    baseline_collected: dict[str, set[str]] = {}
    baseline_ok, baseline_note = False, ""
    if need_baseline:
        baseline_ok, baseline_note, baseline, baseline_collected = _baseline(ctx, failing, changed_tests)
        if not baseline_ok:
            result.not_checked.append(f"comparison with session start: {baseline_note}")

    unverified: list[testrun.TestResult] = []
    for f in failing:
        test_file = f.nodeid.split("::")[0]
        existed = baseline_ok and (_strip_params(f.nodeid) in {
            _strip_params(n) for n in baseline_collected.get(test_file, set())} or f.nodeid in baseline)
        before = baseline.get(f.nodeid)
        evidence = [line for line in f.message.splitlines() if line.strip()][:3]
        if baseline_ok and before is not None and before.bad:
            result.findings.append(Finding(
                rule="test-failing-before", severity="note", location=f.nodeid,
                title=f"{f.nodeid} fails, but it already failed at session start", evidence=evidence))
        elif baseline_ok and before is not None:
            result.findings.append(Finding(
                rule="test-regression", severity="block", location=f.nodeid,
                title=f"{f.nodeid} passed at session start and fails now", evidence=evidence,
                action="Fix the code so it passes again. Don't change the test to match new behavior "
                       "unless that behavior change was asked for."))
        elif baseline_ok and not existed:
            result.findings.append(Finding(
                rule="new-test-failing", severity="block", location=f.nodeid,
                title=f"{f.nodeid} is new in this session and fails", evidence=evidence,
                action="Make the code pass the test (or fix the test if the test itself is wrong)."))
        else:
            unverified.append(f)
    if len(unverified) > MAX_UNVERIFIED_ITEMS:
        # one item, not hundreds: without a session-start comparison these are
        # as likely to be environment failures as anything this session did
        ids = sorted(f.nodeid for f in unverified)
        result.findings.append(Finding(
            rule="tests-failing-unverified", severity="resolve", location=ids[0],
            title=f"{len(ids)} selected tests fail, and they couldn't be compared with session start",
            evidence=[f"{f.nodeid}: {(f.message.splitlines() or [''])[0][:120]}" for f in unverified[:5]],
            action="Check whether this session caused them (run them, or `git stash` and re-run); "
                   "fix what it caused, then acknowledge the rest with what you found.",
            key="|".join(ids)))
    else:
        for f in unverified:
            result.findings.append(Finding(
                rule="test-failing", severity="resolve", location=f.nodeid,
                title=f"{f.nodeid} fails (couldn't confirm whether it passed at session start)",
                evidence=[line for line in f.message.splitlines() if line.strip()][:3],
                action="Fix it, or explain why it's expected to fail."))
    for nid in flaky:
        result.findings.append(Finding(rule="test-flaky", severity="note", location=nid,
                                       title=f"{nid} failed once and passed on re-run (flaky)"))

    # ── were the changed tests actually collected, and did any disappear? ─────
    if changed_tests:
        now_collected = testrun.collect(command, ctx.root, changed_tests, timeout=60) or set()
        for path in changed_tests:
            source = snapshot.show(ctx.root, ctx.current_tree, path) or ""
            before_source = snapshot.show(ctx.root, ctx.baseline_tree, path) or ""
            new_funcs = _test_functions(source) - _test_functions(before_source)
            collected_here = {_strip_params(n).split("::", 1)[1] for n in now_collected if n.startswith(path + "::")}
            for func in sorted(new_funcs - collected_here):
                result.findings.append(Finding(
                    rule="test-not-collected", severity="resolve", location=f"{path}::{func}",
                    title=f"{path}::{func} was added but pytest doesn't collect it, so it never runs",
                    action="Check the file/function naming and pytest configuration."))
            if baseline_ok:
                gone = {_strip_params(n) for n in baseline_collected.get(path, set())} - \
                       {_strip_params(n) for n in now_collected}
                for nid in sorted(gone):
                    result.findings.append(Finding(
                        rule="test-removed", severity="block", location=nid,
                        title=f"{nid} existed at session start and is gone now",
                        action="Restore it, or if removing it is right, hand it to the user: "
                               "`ack <id> --needs-human \"why\"`."))

    # ── deleted or renamed test files: compare their test functions statically ─
    # (a test that reappears in a test file added this session was moved, not removed;
    # git reports a heavily edited rename as a delete plus an add)
    added_elsewhere = {f for d in ctx.deltas if d.status == "A" and is_test_module(d.path)
                       for f in _test_functions(snapshot.show(ctx.root, ctx.current_tree, d.path) or "")}
    for d in ctx.deltas:
        old = d.path if d.status == "D" else d.old_path
        if not old or not is_test_module(old):
            continue
        before_funcs = _test_functions(snapshot.show(ctx.root, ctx.baseline_tree, old) or "")
        after_funcs = set() if d.status == "D" else _test_functions(snapshot.show(ctx.root, ctx.current_tree, d.path) or "")
        for func in sorted(before_funcs - after_funcs - added_elsewhere):
            nid = f"{old}::{func}"
            result.findings.append(Finding(
                rule="test-removed", severity="block", location=nid,
                title=f"{nid} existed at session start and is gone now"
                      + (" (its file was deleted)" if d.status == "D" else f" (file renamed to {d.path})"),
                action="Restore it, or if removing it is right, hand it to the user: "
                       "`ack <id> --needs-human \"why\"`."))

    passed = sum(1 for r in current.results.values() if r.outcome == "passed")
    bad = sum(1 for r in current.results.values() if r.bad)
    skipped = sum(1 for r in current.results.values() if r.outcome == "skipped")
    why = sorted({w for _, _, w in tests})
    result.checks.append(f"tests: ran {len(current.results)} test(s) in {len(ran_files)} file(s) "
                         f"({passed} passed, {bad} failed, {skipped} skipped) in {time.monotonic() - started:.0f}s; "
                         f"selected because: {', '.join(why)}")
    if baseline_ok and need_baseline:
        result.checks.append("failures and changed tests compared with an isolated checkout of the session-start tree")
    return result


def _report_crash(ctx: Context, command: str, batch: list[str], why: str, result: EngineResult) -> None:
    """pytest couldn't run these files now. If they collected at session start,
    this session broke them (a conftest, an import at module level)."""
    with tempfile.TemporaryDirectory(prefix="notyet-baseline-") as tmp:
        tmp = os.path.realpath(tmp)
        snapshot.materialize(ctx.root, ctx.baseline_tree, tmp)
        present = [p for p in batch if os.path.exists(os.path.join(tmp, p))]
        env = {"PYTHONPATH": os.pathsep.join(str(r) for r in find_python_source_roots(tmp))}
        before = testrun.collect(command, tmp, present, env=env, timeout=60) if present else None
    if before:
        result.findings.append(Finding(
            rule="test-run-broken", severity="block", location=batch[0],
            title=f"pytest can't run {len(batch)} test file(s) that it could run at session start",
            evidence=[why[:300]], key="|".join(sorted(batch)),
            action="Fix the import or configuration error so these tests run again."))
    else:
        result.not_checked.append(f"tests: couldn't run {len(batch)} test file(s), and they didn't run at "
                                  f"session start either: {why[:300]}")


def _baseline(ctx: Context, failing: list[testrun.TestResult], changed_tests: list[str]):
    """Collect the relevant test files on the session-start tree, then re-run
    there the failing tests that existed then."""
    cfg = ctx.config
    results: dict[str, testrun.TestResult] = {}
    command = testrun.anchored(cfg.test_command, ctx.root)
    collected: dict[str, set[str]] = {}
    with tempfile.TemporaryDirectory(prefix="notyet-baseline-") as tmp:
        # Resolve symlinks (macOS temp dirs live behind /var -> /private/var):
        # source roots come back resolved, and every path check compares with them.
        tmp = os.path.realpath(tmp)
        snapshot.materialize(ctx.root, ctx.baseline_tree, tmp)
        roots = find_python_source_roots(tmp)
        env = {"PYTHONPATH": os.pathsep.join(str(r) for r in roots)}
        files = sorted({f.nodeid.split("::")[0] for f in failing} | set(changed_tests))
        present = [p for p in files if os.path.exists(os.path.join(tmp, p))]
        if not present:
            return True, "", results, collected       # nothing existed then: everything is new
        ok, why = _canary(ctx, tmp, present, env)
        if not ok:
            return False, why, results, collected
        got = testrun.collect(command, tmp, present, env=env, timeout=60)
        if got is None:
            return False, "couldn't collect the relevant test files at session start", results, collected
        for p in present:
            collected[p] = {n for n in got if n.startswith(p + "::")}
        existed = [f.nodeid for f in failing if f.nodeid in got]
        if existed:
            run = testrun.run_pytest(command, tmp, existed, env=env, timeout=max(30, cfg.budget_seconds))
            if run.crashed:
                return False, f"the test command failed on the session-start tree ({run.crashed})", results, collected
            results = run.results
    return True, "", results, collected


def _canary(ctx: Context, tmp: str, test_files: list[str], env) -> tuple[bool, str]:
    """Prove the session-start checkout imports its own copy of the changed
    modules. The canary sits next to each test file being compared, so pytest
    sets up imports exactly as it will for those tests."""
    command = testrun.anchored(ctx.config.test_command, ctx.root)
    roots = find_python_source_roots(tmp)
    modules = []
    for d in ctx.deltas:
        if not d.path.endswith(".py") or is_test_module(d.path) or d.status == "A":
            continue
        full = os.path.join(tmp, d.path)
        for r in sorted(roots, key=lambda r: -len(str(r))):
            if full.startswith(str(r) + os.sep):
                modules.append(os.path.relpath(full, r)[:-3].replace(os.sep, ".").removesuffix(".__init__"))
                break
    modules = [m for m in dict.fromkeys(modules) if m and all(p.isidentifier() for p in m.split("."))][:5]
    if not modules:
        return True, ""   # only new or non-importable files changed: nothing that could be shadowed
    body = "import importlib, os\n\n" + f"ROOT = os.path.realpath({tmp!r})\n\n" + "".join(
        f"def test_canary_{i}():\n    mod = importlib.import_module({m!r})\n"
        f"    assert os.path.realpath(mod.__file__).startswith(ROOT + os.sep), mod.__file__\n\n"
        for i, m in enumerate(modules))
    canaries = []
    for folder in sorted({os.path.dirname(os.path.join(tmp, p)) for p in test_files}):
        path = os.path.join(folder, CANARY)
        with open(path, "w") as f:
            f.write(body)
        canaries.append(os.path.relpath(path, tmp))
    run = testrun.run_pytest(command, tmp, canaries, env=env, timeout=60)
    if run.crashed or not run.results:
        return False, "couldn't verify the session-start checkout imports its own code"
    if any(r.bad for r in run.results.values()):
        return False, ("the session-start checkout imports the working tree's code instead of its own "
                       "(an editable install?), so failures can't be compared with session start")
    return True, ""
