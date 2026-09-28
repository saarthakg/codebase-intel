"""pytest plugin, loaded into the repo's own test run as `-p notyet_lines`:
records which lines of the changed files execute. Standard library only, and
it watches nothing but those files: sys.monitoring (3.12+) disables every
other code location after its first event, sys.settrace (3.11) never traces
other files' frames.

Config through the environment:
  NOTYET_TRACE_FILES  os.pathsep-separated absolute paths to watch
  NOTYET_TRACE_OUT    JSON file to merge {path: [lines]} into at exit
"""
import json
import os
import sys
import threading

_targets: dict[str, str] = {}         # filename as seen in code objects → canonical path
for _p in filter(None, os.environ.get("NOTYET_TRACE_FILES", "").split(os.pathsep)):
    _targets[_p] = _p
    _targets[os.path.realpath(_p)] = _p
_hits: dict[str, set[int]] = {p: set() for p in set(_targets.values())}
_known: dict[str, str | None] = {}


def _canonical(filename: str) -> str | None:
    if filename not in _known:
        _known[filename] = _targets.get(filename) or _targets.get(os.path.realpath(filename))
    return _known[filename]


def _start() -> None:
    if not _hits:
        return
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is not None:
        for tool in (4, 5, 3, 2):
            try:
                monitoring.use_tool_id(tool, "notyet")
            except ValueError:
                continue

            def on_line(code, line, _tool=tool):
                path = _canonical(code.co_filename)
                if path is None:
                    return monitoring.DISABLE
                _hits[path].add(line)
                return monitoring.DISABLE     # one hit per line is all we need

            monitoring.register_callback(tool, monitoring.events.LINE, on_line)
            monitoring.set_events(tool, monitoring.events.LINE)
            return

    def local(frame, event, arg):
        if event == "line":
            _hits[_canonical(frame.f_code.co_filename)].add(frame.f_lineno)
        return local

    def call(frame, event, arg):
        if _canonical(frame.f_code.co_filename) is None:
            return None
        _hits[_canonical(frame.f_code.co_filename)].add(frame.f_lineno)
        return local

    sys.settrace(call)
    threading.settrace(call)


def _save() -> None:
    out = os.environ.get("NOTYET_TRACE_OUT")
    if not out or not _hits:
        return
    try:
        with open(out) as f:
            merged = {k: set(v) for k, v in json.load(f).items()}
    except (OSError, ValueError):
        merged = {}
    for path, lines in _hits.items():
        merged.setdefault(path, set()).update(lines)
    tmp = f"{out}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({k: sorted(v) for k, v in merged.items()}, f)
    os.replace(tmp, out)


_start()


def pytest_unconfigure(config):
    _save()
