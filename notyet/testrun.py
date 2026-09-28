"""Running pytest and reading what actually happened.

Results come from pytest's junit XML, never from the exit code alone: an
agent (or a background shell) reporting "exit 0" is exactly the signal that
can't be trusted. Every run gets a hard timeout.
"""
import os
import re
import shlex
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestResult:
    nodeid: str
    outcome: str                  # passed | failed | error | skipped
    message: str = ""

    @property
    def bad(self) -> bool:
        return self.outcome in ("failed", "error")


@dataclass
class Run:
    results: dict[str, TestResult] = field(default_factory=dict)   # nodeid → result
    seconds: float = 0.0
    timed_out: bool = False
    crashed: Optional[str] = None     # pytest itself failed to run (usage error, internal error)

    @property
    def failures(self) -> list[TestResult]:
        return [r for r in self.results.values() if r.bad]


def _nodeid(case: ET.Element) -> str:
    """Rebuild pytest's node id from a junit (xunit1) testcase: file
    "tests/test_x.py", classname "tests.test_x.TestC", name "test_y[p]" ->
    "tests/test_x.py::TestC::test_y[p]"."""
    classname, name, path = case.get("classname", ""), case.get("name", ""), case.get("file")
    if not path:
        return f"{classname}::{name}"
    module = path[:-3].replace("/", ".") if path.endswith(".py") else path
    if not classname and name == module:
        return path       # the file itself failed to import or collect
    inner = classname[len(module) + 1:] if classname.startswith(module + ".") else ""
    return "::".join([path, *(inner.split(".") if inner else []), name])


def parse_junit(path: str) -> dict[str, TestResult]:
    results: dict[str, TestResult] = {}
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError):
        return results
    for case in tree.iter("testcase"):
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = case.find(tag)
            if el is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                message = (el.get("message") or el.text or "").strip()
                break
        nid = _nodeid(case)
        if nid in results and results[nid].bad:
            continue   # a teardown error recorded after a failure: keep the first
        results[nid] = TestResult(nid, outcome, message[:2000])
    return results


def anchored(command: str, root: str) -> str:
    """`command` with a repo-relative interpreter or tool (".venv/bin/python -m
    pytest") made absolute, so it also runs from a checkout elsewhere."""
    try:
        argv = shlex.split(command)
    except ValueError:
        return command
    if argv and "/" in argv[0] and not os.path.isabs(argv[0]) and os.path.exists(os.path.join(root, argv[0])):
        argv[0] = os.path.abspath(os.path.join(root, argv[0]))
        return shlex.join(argv)
    return command


def run_pytest(command: str, cwd: str, targets: list[str], timeout: float,
               env: Optional[dict] = None, extra: Optional[list[str]] = None) -> Run:
    """Run `command` (a pytest invocation) on `targets` (files or node ids)."""
    with tempfile.TemporaryDirectory(prefix="notyet-junit-") as tmp:
        junit = os.path.join(tmp, "junit.xml")
        argv = shlex.split(command) + [
            *targets, "-q", "-p", "no:cacheprovider", "--continue-on-collection-errors", f"--junitxml={junit}",
            f"--rootdir={cwd}",   # node ids relative to the repo, even with a nested tests/pytest.ini
            "-o", "junit_family=xunit1", *(extra or []),
        ]
        start = time.monotonic()
        try:
            proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                                  env={**os.environ, **(env or {}), "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired:
            return Run(results=parse_junit(junit), seconds=time.monotonic() - start, timed_out=True)
        except FileNotFoundError as e:
            return Run(crashed=f"couldn't start the test command: {e}")
        run = Run(results=parse_junit(junit), seconds=time.monotonic() - start)
        # pytest exit codes: 0 ok, 1 test failures, 5 no tests; 2-4 are usage/internal errors.
        # No results with any other code means pytest never ran (e.g. "No module named pytest").
        if not run.results and proc.returncode not in (0, 5):
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
            run.crashed = f"the test command exited {proc.returncode} without results: " + " | ".join(tail)
        return run


def run_tests(command: str, cwd: str, node_ids: list[str], timeout: float, env: dict | None = None) -> Run:
    """`run_pytest` on node ids. When one targeted file can't be imported,
    pytest drops the results of every other targeted file too; then the files
    are run whole, where it carries on past the error."""
    run = run_pytest(command, cwd, node_ids, timeout=timeout, env=env)
    collection_error = any("::" not in n.strip(":") and r.bad for n, r in run.results.items())
    if collection_error and any(outcome_of(n, run.results) == "missing" for n in node_ids):
        files = list(dict.fromkeys(n.split("::")[0] for n in node_ids))
        run = run_pytest(command, cwd, files, timeout=timeout, env=env)
    return run


def outcome_of(nodeid: str, results: dict[str, TestResult]) -> str:
    """passed | failed | skipped | missing for a test function, across its
    parameter sets. A file that fails to import fails every test in it."""
    file = nodeid.split("::")[0]
    if file in results and results[file].bad:
        return "failed"
    mine = [r for n, r in results.items() if n == nodeid or n.startswith(nodeid + "[")]
    if not mine:
        return "missing"
    if any(r.bad for r in mine):
        return "failed"
    if all(r.outcome == "passed" for r in mine):
        return "passed"
    return "skipped"


def failure_of(nodeid: str, results: dict[str, TestResult]) -> str:
    """The message of the first failure behind outcome_of(...) == "failed"."""
    file = nodeid.split("::")[0]
    if file in results and results[file].bad:
        return results[file].message
    return next((r.message for n, r in results.items() if (n == nodeid or n.startswith(nodeid + "[")) and r.bad), "")


class Collected(set):
    """Collected node ids, plus the files that failed to import or collect."""
    def __init__(self, ids=(), errors=()):
        super().__init__(ids)
        self.errors: set[str] = set(errors)


_COLLECT_ERROR = re.compile(r"^ERROR (\S+?\.py)\b")


def collect(command: str, cwd: str, targets: list[str], timeout: float,
            env: Optional[dict] = None) -> Optional[Collected]:
    """Node ids pytest collects from `targets` (and files it couldn't), or
    None if pytest itself failed."""
    argv = shlex.split(command) + [*targets, "--collect-only", "-q", "-p", "no:cacheprovider",
                                   "--continue-on-collection-errors", f"--rootdir={cwd}"]
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, **(env or {}), "PYTHONDONTWRITEBYTECODE": "1"})
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    lines = proc.stdout.splitlines()
    errors = {m.group(1) for line in lines if (m := _COLLECT_ERROR.match(line))}
    if proc.returncode not in (0, 5) and not errors:
        return None
    return Collected((line.strip() for line in lines if "::" in line and not line.startswith((" ", "ERROR"))), errors)
