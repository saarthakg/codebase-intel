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

The session's new and edited tests are also run against the session-start
code, to see whether any of them tests the change (engines/vacuous.py).
"""
import ast
import configparser
import hashlib
import json
import os
import tempfile
import time
import tomllib
from pathlib import PurePosixPath

from notyet.pyresolve import find_python_source_roots
from notyet import coverage, pyimports, snapshot, store, testrun
from notyet.findings import Context, EngineResult, Finding

DOC_SUFFIXES = (".md", ".rst", ".txt", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico")
BATCH_FILES = 12
MAX_UNVERIFIED_ITEMS = 5
MAX_INFRA_ITEMS = 10
INFRA_FILES = {"conftest.py", "pytest.ini", ".pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"}
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


def source_path_env(root: str, env: dict | None = None) -> dict:
    """`env` with PYTHONPATH led by the checkout's own source roots. The
    session-start checkout is always run this way; running the working tree
    the same way means both sides import alike, and a venv whose install
    broke (or points elsewhere) mid-session isn't blamed on the session."""
    env = dict(env or {})
    rest = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([str(r) for r in find_python_source_roots(root)] + ([rest] if rest else []))
    return env


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
    changed_src = [d.path for d in ctx.deltas if d.status != "D" and d.path.endswith(".py")
                   and not is_test_module(d.path) and not in_test_dir(d.path)]
    if not tests:
        result.not_checked.append("tests: no tests exercise the changed files (none import them or are named for them)")
        result.findings += _coverage_findings(ctx, changed_src, {})
        return result

    # ── run the selection on the current tree, recording which changed lines run ─
    current = testrun.Run()
    skipped_files: list[str] = []
    crashed: list[tuple[list[str], str]] = []
    with tempfile.TemporaryDirectory(prefix="notyet-lines-") as trace_dir:
        env, extra, trace_out = None, None, None
        if changed_src:
            env, trace_out = coverage.setup(trace_dir, ctx.root, changed_src)
            extra = ["-p", coverage.PLUGIN]
        env = source_path_env(ctx.root, env)
        for batch in _batches(tests):
            remaining = cfg.budget_seconds - (time.monotonic() - started)
            if remaining <= 1:
                skipped_files += batch
                continue
            run_ = testrun.run_pytest(command, ctx.root, batch, timeout=remaining, env=env, extra=extra)
            if run_.crashed:
                crashed.append((batch, run_.crashed))
                continue
            if run_.timed_out:
                skipped_files += [f for f in batch if not any(n.startswith(f + "::") for n in run_.results)]
            current.results.update(run_.results)
        hits = coverage.read(trace_out, ctx.root) if trace_out else {}
    coverage_check = None
    if changed_src:
        if hits is None:
            result.not_checked.append("coverage: couldn't record which changed lines ran")
        elif skipped_files or crashed:
            result.not_checked.append("coverage: not all selected tests ran, so changed lines that didn't run "
                                      "aren't reported")
        else:
            result.findings += _coverage_findings(ctx, changed_src, hits)
            coverage_check = f"coverage: recorded which changed lines in {len(changed_src)} file(s) ran"
    for batch, why in crashed:
        _report_crash(ctx, command, batch, why, result)
    if crashed and not current.results:
        return result
    ran_files = sorted({n.split("::")[0] for n in current.results})
    if skipped_files:
        selected = len({f for batch in _batches(tests) for f in batch})
        result.gaps.append(f"tests NOT run: the {cfg.budget_seconds}s budget ran out before any of the {selected} "
                           f"selected test file(s) finished" if not current.results else
                           f"tests incomplete: the {cfg.budget_seconds}s budget ran out; {len(skipped_files)} of the "
                           f"{selected} selected test file(s) didn't run")
        result.not_checked.append(f"tests: the {cfg.budget_seconds}s budget ran out before {len(skipped_files)} "
                                  f"selected test file(s) ran: " + ", ".join(skipped_files[:5])
                                  + (" …" if len(skipped_files) > 5 else ""))

    failing = current.failures
    flaky: list[str] = []
    if failing:  # re-run failures once: a pass means flaky, not broken
        rerun = testrun.run_pytest(command, ctx.root, [f.nodeid for f in failing], env=source_path_env(ctx.root),
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
    baseline_errors: set[str] = set()
    baseline_ok, baseline_note = False, ""
    if need_baseline:
        baseline_ok, baseline_note, baseline, baseline_collected, baseline_errors = _baseline(ctx, failing, changed_tests)
        if not baseline_ok:
            result.not_checked.append(f"comparison with session start: {baseline_note}")

    unverified: list[testrun.TestResult] = []
    for f in [f for f in failing if "::" not in f.nodeid]:     # a test file that can't be imported/collected
        evidence = [line for line in f.message.splitlines() if line.strip()][:3]
        if baseline_ok and f.nodeid in baseline_errors:
            result.findings.append(Finding(
                rule="test-failing-before", severity="note", location=f.nodeid, evidence=evidence,
                title=f"{f.nodeid} can't be collected, but it couldn't at session start either"))
        elif baseline_ok and baseline_collected.get(f.nodeid):
            result.findings.append(Finding(
                rule="test-regression", severity="block", location=f.nodeid, evidence=evidence,
                title=f"{f.nodeid} can't be collected now, so none of its "
                      f"{len(baseline_collected[f.nodeid])} test(s) run; it could at session start",
                action="Fix the import or collection error."))
        else:
            unverified.append(f)
    failing = [f for f in failing if "::" in f.nodeid]
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
        now_collected = testrun.collect(command, ctx.root, changed_tests, env=source_path_env(ctx.root),
                                        timeout=60) or set()
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

    # ── do the session's new tests fail on the session-start code? ───────────
    if changed_tests and not skipped_files and ctx.time_left() > 10:
        from notyet.engines import vacuous
        vacuous.check(ctx, command, current.results, result)

    # ── test infrastructure changed: did any test stop being collected or start skipping? ─
    infra = sorted(d.path for d in ctx.deltas if PurePosixPath(d.path).name in INFRA_FILES)
    if infra:
        _infra_check(ctx, command, current, infra, result)

    passed = sum(1 for r in current.results.values() if r.outcome == "passed")
    bad = sum(1 for r in current.results.values() if r.bad)
    skipped = sum(1 for r in current.results.values() if r.outcome == "skipped")
    why = sorted({w for _, _, w in tests})
    result.checks.append(f"tests: ran {len(current.results)} test(s) in {len(ran_files)} file(s) "
                         f"({passed} passed, {bad} failed, {skipped} skipped) in {time.monotonic() - started:.0f}s; "
                         f"selected because: {', '.join(why)}")
    if baseline_ok and need_baseline:
        result.checks.append("failures and changed tests compared with an isolated checkout of the session-start tree")
    if coverage_check:
        result.checks.append(coverage_check)
    return result


def _coverage_findings(ctx: Context, changed_src: list[str], hits: dict[str, set[int]]) -> list[Finding]:
    deltas = [d for d in ctx.deltas if d.path in set(changed_src)]
    added = snapshot.added_lines(ctx.root, ctx.baseline_tree, ctx.current_tree, deltas)
    sources = {p: snapshot.show(ctx.root, ctx.current_tree, p) or "" for p in changed_src}
    return coverage.findings(sources, added, hits)


def _infra_check(ctx: Context, command: str, current: testrun.Run, infra: list[str], result: EngineResult) -> None:
    """A conftest hook, an addopts `--deselect`/`-k`, or an autouse skip can
    switch tests off without touching a test file. Compare the whole suite's
    collection with session start, and re-run newly skipped tests there."""
    now = testrun.collect(command, ctx.root, [], env=source_path_env(ctx.root), timeout=90)
    with tempfile.TemporaryDirectory(prefix="notyet-baseline-") as tmp:
        tmp = os.path.realpath(tmp)
        snapshot.materialize(ctx.root, ctx.baseline_tree, tmp)
        env = {"PYTHONPATH": os.pathsep.join(str(r) for r in find_python_source_roots(tmp))}
        before = testrun.collect(command, tmp, [], env=env, timeout=90)
        skipped_now = [n for n, r in current.results.items() if r.outcome == "skipped"][:100]
        ran_before = (testrun.run_pytest(command, tmp, [n for n in skipped_now if before and n in before], env=env,
                                         timeout=max(30, ctx.config.budget_seconds))
                      if skipped_now and before else testrun.Run())
    if now is None or before is None:
        result.not_checked.append(f"test collection with {', '.join(infra)} changed: couldn't collect the suite "
                                  f"{'now' if now is None else 'at session start'}")
        return
    gone_files = {d.path for d in ctx.deltas if d.status == "D"} | {d.old_path for d in ctx.deltas if d.old_path}
    gone = sorted(n for n in before - now if n.split("::")[0] not in gone_files)   # deleted files: reported already
    reported = {f.location for f in result.findings}
    for nid in [n for n in gone if n not in reported][:MAX_INFRA_ITEMS]:
        result.findings.append(Finding(
            rule="test-removed", severity="block", location=nid,
            title=f"{nid} was collected at session start and isn't now ({', '.join(infra)} changed)",
            action="Undo whatever stops it being collected, or hand it to the user: "
                   "`ack <id> --needs-human \"why\"`."))
    for nid, r in ran_before.results.items():
        if r.outcome == "passed" and nid not in reported:
            result.findings.append(Finding(
                rule="test-disabled", severity="block", location=nid,
                title=f"{nid} passed at session start and is skipped now ({', '.join(infra)} changed)",
                evidence=[(current.results[nid].message.splitlines() or [""])[0][:160]] if nid in current.results else [],
                action="Remove whatever skips it and fix the code, or hand it to the user: "
                       "`ack <id> --needs-human \"why\"`."))
    result.checks.append(f"test collection compared with session start because {', '.join(infra)} changed "
                         f"({len(before)} then, {len(now)} now)")


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
    there the failing tests that existed then. The session-start tree never
    changes within a session, so everything learned about it is cached (per
    tree and test command) and later checks only run what's new."""
    cfg = ctx.config
    command = testrun.anchored(cfg.test_command, ctx.root)
    cache = _BaselineCache(ctx.root, ctx.baseline_tree, command)
    in_baseline = set(snapshot.git(ctx.root, "ls-tree", "-r", "--name-only", ctx.baseline_tree).splitlines())
    files = sorted({f.nodeid.split("::")[0] for f in failing} | set(changed_tests))  # file-level ids are files
    present = [p for p in files if p in in_baseline]
    if not present:
        return True, "", {}, {}, set()                  # nothing existed then: everything is new
    dirs = sorted({os.path.dirname(p) for p in present})
    need_collect = [p for p in present if p not in cache.collected]
    wanted = [f.nodeid for f in failing]

    def ids_known() -> list[str]:
        return [n for n in wanted if any(n in cache.collected.get(p, ()) for p in present)
                and n not in cache.results]

    if need_collect or ids_known():
        with tempfile.TemporaryDirectory(prefix="notyet-baseline-") as tmp:
            # Resolve symlinks (macOS temp dirs live behind /var -> /private/var):
            # source roots come back resolved, and every path check compares with them.
            tmp = os.path.realpath(tmp)
            snapshot.materialize(ctx.root, ctx.baseline_tree, tmp)
            env = {"PYTHONPATH": os.pathsep.join(str(r) for r in find_python_source_roots(tmp))}
            if not cache.canary_ok(dirs):
                ok, why = _canary(ctx, tmp, present, env)
                if not ok:
                    return False, why, {}, {}, set()
                cache.canary_passed(dirs)
            if need_collect:
                got = testrun.collect(command, tmp, need_collect, env=env, timeout=60)
                if got is None:
                    return False, "couldn't collect the relevant test files at session start", {}, {}, set()
                for p in need_collect:
                    cache.collected[p] = sorted(n for n in got if n.startswith(p + "::"))
                    if p in got.errors:
                        cache.errors[p] = True
            to_run = ids_known()
            if to_run:
                run = testrun.run_pytest(command, tmp, to_run, env=env, timeout=max(30, cfg.budget_seconds))
                if run.crashed:
                    return False, f"the test command failed on the session-start tree ({run.crashed})", {}, {}, set()
                for n in to_run:
                    r = run.results.get(n)
                    cache.results[n] = [r.outcome, r.message] if r else ["missing", ""]
        cache.save()
    collected = {p: set(cache.collected.get(p, ())) for p in present}
    errors = {p for p in present if cache.errors.get(p)}
    results = {n: testrun.TestResult(n, *cache.results[n]) for n in wanted
               if n in cache.results and cache.results[n][0] != "missing"}
    return True, "", results, collected, errors


def _code_fingerprint() -> str:
    """notyet's own code that produces cached results: a new version of it
    must not reuse results the old one computed."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha1()
    for rel in ("testrun.py", "snapshot.py", "engines/execution.py", "engines/vacuous.py"):
        try:
            with open(os.path.join(here, rel), "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:12]


class _BaselineCache:
    KEEP = 8

    def __init__(self, root: str, tree: str, command: str):
        key = hashlib.sha1(f"{tree}|{command}|{_code_fingerprint()}".encode()).hexdigest()[:16]
        self.path = store.state_dir(root) / "cache" / f"baseline-{key}.json"
        self.path.parent.mkdir(exist_ok=True)
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            data = {}
        self.collected: dict[str, list[str]] = data.get("collected", {})
        self.errors: dict[str, bool] = data.get("errors", {})
        self.results: dict[str, list[str]] = data.get("results", {})
        self.canaries: list[str] = data.get("canaries", [])
        self.vacuous: dict[str, dict[str, list[str]]] = data.get("vacuous", {})   # see engines/vacuous.py

    def canary_ok(self, dirs: list[str]) -> bool:
        return all(d in self.canaries for d in dirs)

    def canary_passed(self, dirs: list[str]) -> None:
        self.canaries = sorted(set(self.canaries) | set(dirs))

    def save(self) -> None:
        self.path.write_text(json.dumps({"collected": self.collected, "errors": self.errors,
                                         "results": self.results, "canaries": self.canaries,
                                         "vacuous": self.vacuous}))
        old = sorted(self.path.parent.glob("baseline-*.json"), key=lambda p: p.stat().st_mtime)
        for p in old[:-self.KEEP]:
            p.unlink(missing_ok=True)


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
