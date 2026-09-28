"""Static evidence: what the session added to the repo's own linter and type
checker output, and suppression comments it added.

ruff and pyright (when configured in .notyet.toml) run twice with the same
flags: on a checkout of the session-start tree and on a checkout of the
current tree. Only diagnostics the session introduced are reported, so a
repo with 500 existing warnings reports none of them. Both checkouts use the
session-start linter configuration, so loosening pyproject.toml mid-session
doesn't loosen the check. Pyright also checks the files importing the
changed modules: that's where a changed signature breaks callers.

Nothing here blocks: a new lint or type error is fix-or-justify.
"""
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from notyet import pyimports, snapshot
from notyet.findings import Context, EngineResult, Finding

LINT_CONFIGS = ("pyproject.toml", "ruff.toml", ".ruff.toml", "pyrightconfig.json", "setup.cfg")
SUPPRESSION = re.compile(
    r"#\s*(noqa\b|type:\s*ignore|pyright:\s*ignore|pyright:\s*basic|mypy:\s*ignore-errors"
    r"|ruff:\s*noqa|pragma:\s*no\s*cover|fmt:\s*off)", re.IGNORECASE)
MAX_FINDINGS_PER_TOOL = 15
MAX_PYRIGHT_FILES = 150


@dataclass(frozen=True)
class Diagnostic:
    path: str        # repo-relative
    line: int        # 1-based
    code: str
    message: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.path, self.code, self.message)


def run(ctx: Context) -> EngineResult:
    result = EngineResult()
    py_changed = [d for d in ctx.deltas if d.status != "D" and d.path.endswith(".py")]
    if not py_changed:
        return result
    added_lines = snapshot.added_lines(ctx.root, ctx.baseline_tree, ctx.current_tree, py_changed)
    result.findings += _suppressions(ctx, py_changed, added_lines)

    tools = [t for t in ("ruff", "pyright") if getattr(ctx.config, t)]
    if not tools:
        return result
    started = time.monotonic()
    renamed = {d.path: d.old_path for d in py_changed if d.old_path}
    changed_now = [d.path for d in py_changed]
    targets = {"ruff": changed_now}
    if "pyright" in tools:
        graph = pyimports.ImportGraph(ctx.root, ctx.current_tree)
        depth = graph.importers(changed_now, max_depth=2)
        importers = sorted(depth, key=lambda p: (depth[p], p))
        targets["pyright"] = list(dict.fromkeys(changed_now + importers))[:MAX_PYRIGHT_FILES]

    with tempfile.TemporaryDirectory(prefix="notyet-static-") as tmp:
        tmp = os.path.realpath(tmp)
        before_dir, after_dir = os.path.join(tmp, "before"), os.path.join(tmp, "after")
        snapshot.materialize(ctx.root, ctx.baseline_tree, before_dir)
        snapshot.materialize(ctx.root, ctx.current_tree, after_dir)
        tampered = _use_baseline_configs(ctx, after_dir)
        if tampered:
            result.advice.append(f"linter config changed this session ({', '.join(tampered)}); "
                                 f"static checks used the session-start version")
        for tool in tools:
            remaining = ctx.config.static_budget_seconds - (time.monotonic() - started)
            if remaining <= 1:
                result.not_checked.append(f"{tool}: the {ctx.config.static_budget_seconds}s static budget ran out")
                continue
            now_files = targets[tool]
            then_files = [renamed.get(p, p) for p in now_files]
            then_files = [p for p in then_files if os.path.exists(os.path.join(before_dir, p))]
            with ThreadPoolExecutor(2) as pool:
                after_job = pool.submit(_run_tool, ctx, tool, after_dir, now_files, remaining)
                before_job = pool.submit(_run_tool, ctx, tool, before_dir, then_files, remaining)
                after, after_err = after_job.result()
                before, before_err = before_job.result()
            if after_err or before_err:
                result.not_checked.append(f"{tool}: {after_err or before_err}")
                continue
            back = {old: new for new, old in renamed.items()}
            before = [Diagnostic(back.get(d.path, d.path), d.line, d.code, d.message) for d in before]
            new = _introduced(before, after, added_lines)
            result.findings += _findings(tool, new)
            result.checks.append(f"{tool}: {len(new)} new diagnostic(s) in {len(now_files)} file(s) "
                                 f"compared with session start")
    return result


def _introduced(before: list[Diagnostic], after: list[Diagnostic],
                added_lines: dict[str, set[int]]) -> list[Diagnostic]:
    """Diagnostics in `after` beyond what `before` had, matched by (file, code,
    message) so moved code doesn't count. When a key gained occurrences, the
    ones on lines this session wrote are reported first."""
    had = Counter(d.key for d in before)
    by_key: dict[tuple, list[Diagnostic]] = {}
    for d in after:
        by_key.setdefault(d.key, []).append(d)
    new = []
    for key, ds in by_key.items():
        extra = len(ds) - had.get(key, 0)
        if extra > 0:
            ds.sort(key=lambda d: (d.line not in added_lines.get(d.path, set()), d.line))
            new += ds[:extra]
    return sorted(new, key=lambda d: (d.path, d.line))


def _findings(tool: str, new: list[Diagnostic]) -> list[Finding]:
    rule = "lint-new" if tool == "ruff" else "type-error-new"
    out = []
    occurrence: Counter = Counter()
    for d in new[:MAX_FINDINGS_PER_TOOL]:
        occurrence[d.key] += 1
        first_line = d.message.splitlines()[0] if d.message else ""
        out.append(Finding(
            rule=rule, severity="resolve", location=f"{d.path}:{d.line}",
            title=f"{tool} {d.code}: {first_line}" + (f" ({d.path}:{d.line})" if d.path else ""),
            evidence=[line.strip() for line in d.message.splitlines()[1:3] if line.strip()],
            action="Fix it, or acknowledge it with the reason it's intended.",
            key=f"{tool}|{d.path}|{d.code}|{d.message}|{occurrence[d.key]}"))
    if len(new) > MAX_FINDINGS_PER_TOOL:
        rest = new[MAX_FINDINGS_PER_TOOL:]
        out.append(Finding(
            rule=rule, severity="resolve", location=rest[0].path,
            title=f"{tool}: {len(rest)} more new diagnostic(s)",
            evidence=[f"{d.path}:{d.line} {d.code}" for d in rest[:5]],
            action=f"Run {tool} on the changed files and fix them.",
            key=f"{tool}|more|{len(rest)}"))
    return out


# ── running the tools ──────────────────────────────────────────────────────────

def _run_tool(ctx: Context, tool: str, cwd: str, files: list[str], timeout: float):
    if not files:
        return [], None
    command = getattr(ctx.config, tool)
    argv = shlex.split(command)
    if argv and not os.path.isabs(argv[0]) and os.path.exists(os.path.join(ctx.root, argv[0])):
        argv[0] = os.path.join(ctx.root, argv[0])      # a repo-relative tool, e.g. .venv/bin/ruff
    if tool == "ruff":
        argv += ["check", "--output-format=json", "--no-cache", "--force-exclude", "--exit-zero", *files]
    else:
        argv += ["--outputjson", *_pyright_python(ctx), *files]
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [], f"timed out after {timeout:.0f}s"
    except FileNotFoundError as e:
        return [], f"couldn't run it ({e})"
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return [], f"exited {proc.returncode} without JSON output: " + " | ".join(tail)
    rel = lambda p: os.path.relpath(os.path.realpath(p), cwd) if p else ""   # noqa: E731
    if tool == "ruff":
        return [Diagnostic(rel(d.get("filename")), (d.get("location") or {}).get("row", 0),
                           d.get("code") or "syntax", d.get("message", "")) for d in data], None
    return [Diagnostic(rel(d.get("file")), d["range"]["start"]["line"] + 1, d.get("rule") or d.get("severity", ""),
                       d.get("message", ""))
            for d in data.get("generalDiagnostics", []) if d.get("severity") == "error"], None


def _pyright_python(ctx: Context) -> list[str]:
    """Point pyright at the interpreter the tests use, so both checkouts see
    the same installed packages."""
    try:
        first = shlex.split(ctx.config.test_command or "")[0]
    except (ValueError, IndexError):
        return []
    if "python" not in os.path.basename(first):
        return []
    path = first if os.path.isabs(first) else os.path.join(ctx.root, first)
    return ["--pythonpath", path] if os.path.exists(path) else []


def _use_baseline_configs(ctx: Context, after_dir: str) -> list[str]:
    changed = []
    for name in LINT_CONFIGS:
        then = snapshot.show(ctx.root, ctx.baseline_tree, name)
        target = os.path.join(after_dir, name)
        now = open(target).read() if os.path.exists(target) else None
        if then == now:
            continue
        changed.append(name)
        if then is None:
            os.remove(target)
        else:
            with open(target, "w") as f:
                f.write(then)
    return changed


# ── suppressions ───────────────────────────────────────────────────────────────

def _suppressions(ctx: Context, deltas, added_lines: dict[str, set[int]]) -> list[Finding]:
    """Suppression comments on lines this session wrote, when the file now has
    more of that kind than it had at session start."""
    findings = []
    for d in deltas:
        now = (snapshot.show(ctx.root, ctx.current_tree, d.path) or "").splitlines()
        then = snapshot.show(ctx.root, ctx.baseline_tree, d.old_path or d.path) or ""
        had = Counter(_kind(m) for m in SUPPRESSION.finditer(then))
        gained = Counter(_kind(m) for line in now for m in SUPPRESSION.finditer(line)) - had
        for n in sorted(added_lines.get(d.path, ())):
            if n - 1 >= len(now):
                continue
            for m in SUPPRESSION.finditer(now[n - 1]):
                kind = _kind(m)
                if gained[kind] <= 0:
                    continue   # moved or reformatted, not new
                gained[kind] -= 1
                findings.append(Finding(
                    rule="suppression-added", severity="resolve", location=f"{d.path}:{n}",
                    title=f"added `# {m.group(1)}` at {d.path}:{n}",
                    evidence=[now[n - 1].strip()[:160]],
                    action="Fix what it silences, or acknowledge it with the reason it's needed.",
                    key=f"{d.path}|{kind}|{now[n - 1].strip()}"))
    return findings


def _kind(m: re.Match) -> str:
    return re.sub(r"\s+", " ", m.group(1).lower())
