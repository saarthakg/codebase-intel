"""pytest plugin: record which files call which functions while a repo's tests run.

Runtime ground truth for "who calls HTTPAdapter.send?": a profile hook sees
every call into the repo's own code, with the callee's qualified name straight
from its code object (co_qualname) and the caller's file from the calling
frame. Decorator wrappers are seen through, so a call via a decorator is
credited to the file that made it. Every recorded caller really made the call; calls on paths the test
suite never exercises are missing, so it gives a trustworthy recall target and
a lower bound on precision.

Run inside an environment with the target repo's test dependencies:
  TRACE_REPO=/path/to/repo TRACE_OUT=calls.json PYTHONPATH=/path/to/codebase-intel/eval \\
    python -m pytest -p call_tracer -q /path/to/repo/tests

Output: {"callee_file::Qualified.name": ["caller_file", ...], ...} with paths
relative to TRACE_REPO.
"""
import json
import os
import sys
import threading
from pathlib import Path

_REPO = os.path.realpath(os.environ.get("TRACE_REPO", "."))
_PREFIX = _REPO + os.sep
_calls: dict[tuple[str, str], set[str]] = {}
_rel_cache: dict[str, str | None] = {}


def _rel(filename: str) -> str | None:
    """Repo-relative path, or None for code outside the repo (stdlib, deps, venvs)."""
    if filename not in _rel_cache:
        path = os.path.realpath(filename)
        ok = path.startswith(_PREFIX) and "site-packages" not in path and f"{os.sep}." not in path[len(_PREFIX):]
        _rel_cache[filename] = path[len(_PREFIX):] if ok else None
    return _rel_cache[filename]


def _is_wrapper_of(frame, callee_code) -> bool:
    if "<locals>" not in frame.f_code.co_qualname or not frame.f_code.co_freevars:
        return False
    for name in frame.f_code.co_freevars:
        value = frame.f_locals.get(name)
        func = getattr(value, "__func__", value)  # bound method → function
        if getattr(func, "__code__", None) is callee_code:
            return True
    return False


def _profile(frame, event, arg):
    if event != "call":
        return
    code = frame.f_code
    callee = _rel(code.co_filename)
    if callee is None or "<" in code.co_qualname:  # skip lambdas, comprehensions, closures
        return
    caller_frame = frame.f_back
    # See through decorator wrappers: a nested function that calls the very
    # function it closed over (functools.wraps-style `wrapper(*a, **kw)`) isn't
    # the real caller; whoever called the wrapper is.
    while caller_frame is not None and _is_wrapper_of(caller_frame, code):
        caller_frame = caller_frame.f_back
    if caller_frame is None:
        return
    caller = _rel(caller_frame.f_code.co_filename)
    if caller is None:
        return
    _calls.setdefault((callee, code.co_qualname), set()).add(caller)


def pytest_configure(config):
    sys.setprofile(_profile)
    threading.setprofile(_profile)


def pytest_unconfigure(config):
    sys.setprofile(None)
    threading.setprofile(None)
    out = os.environ.get("TRACE_OUT", "calls.json")
    data = {f"{f}::{q}": sorted(callers) for (f, q), callers in sorted(_calls.items())}
    Path(out).write_text(json.dumps(data, indent=1))
    print(f"\ncall_tracer: {len(data)} callees traced -> {out}", file=sys.stderr)
