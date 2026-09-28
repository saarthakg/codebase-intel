"""Running pytest and reading what actually happened.

Results come from pytest's junit XML, never from the exit code alone: an
agent (or a background shell) reporting "exit 0" is exactly the signal that
can't be trusted. Every run gets a hard timeout.
"""
import os
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
            *targets, "-q", "-p", "no:cacheprovider", f"--junitxml={junit}",
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


def collect(command: str, cwd: str, targets: list[str], timeout: float,
            env: Optional[dict] = None) -> Optional[set[str]]:
    """Node ids pytest collects from `targets`, or None if collection failed."""
    argv = shlex.split(command) + [*targets, "--collect-only", "-q", "-p", "no:cacheprovider"]
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, **(env or {}), "PYTHONDONTWRITEBYTECODE": "1"})
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode not in (0, 5):
        return None
    return {line.strip() for line in proc.stdout.splitlines() if "::" in line and not line.startswith(" ")}
