"""Changed lines that no selected test executed.

The execution engine runs the selected tests with notyet's line plugin
(notyet/lineplugin.py) watching the changed source files; this module sets
that up and turns the result into findings. Only executable lines count
(not comments, blank lines or docstrings), and lines excluded the usual ways
aren't expected to run: `# pragma: no cover`, `if TYPE_CHECKING:` blocks,
`if __name__ == "__main__":` blocks, `raise NotImplementedError`, and
`...`/`pass` bodies.
"""
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

from notyet.findings import Finding

PLUGIN = "notyet_lines"


def setup(tmp: str, root: str, files: list[str]) -> tuple[dict, str]:
    """(env for the test run, path of the JSON it writes)."""
    shutil.copy(Path(__file__).with_name("lineplugin.py"), os.path.join(tmp, f"{PLUGIN}.py"))
    out = os.path.join(tmp, "lines.json")
    existing = os.environ.get("PYTHONPATH")
    env = {
        "PYTHONPATH": tmp + (os.pathsep + existing if existing else ""),
        "NOTYET_TRACE_FILES": os.pathsep.join(os.path.join(root, f) for f in files),
        "NOTYET_TRACE_OUT": out,
    }
    return env, out


def read(out: str, root: str) -> dict[str, set[int]] | None:
    try:
        with open(out) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return {os.path.relpath(k, root): set(v) for k, v in data.items()}


def executable_lines(source: str, filename: str = "<changed>") -> set[int]:
    try:
        tree = ast.parse(source)
        code = compile(tree, filename, "exec")
    except (SyntaxError, ValueError):
        return set()
    lines: set[int] = set()
    stack = [code]
    while stack:
        co = stack.pop()
        lines |= {line for _, _, line in co.co_lines() if line}
        stack += [c for c in co.co_consts if hasattr(c, "co_lines")]
    return lines - _excluded(tree, source)


def _excluded(tree: ast.AST, source: str) -> set[int]:
    pragma = {i for i, text in enumerate(source.splitlines(), 1) if "pragma: no cover" in text}
    out = set(pragma)
    for node in ast.walk(tree):
        if getattr(node, "body", None) and getattr(node, "lineno", None) in pragma:  # the whole block, as coverage.py does
            out |= set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            if test in ("TYPE_CHECKING", "typing.TYPE_CHECKING") or test.replace("'", '"') == '__name__ == "__main__"':
                out |= set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        elif isinstance(node, ast.Raise) and node.exc is not None and "NotImplementedError" in ast.unparse(node.exc):
            out.add(node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            if all(isinstance(s, ast.Pass) for s in body):
                out |= set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
            for s in node.body:  # docstrings aren't code a test should hit
                if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                    out |= set(range(s.lineno, (s.end_lineno or s.lineno) + 1))
    return out


def findings(sources: dict[str, str], added: dict[str, set[int]], hits: dict[str, set[int]]) -> list[Finding]:
    """One finding per file with changed executable lines that never ran."""
    out = []
    for path in sorted(sources):
        lines = sorted((added.get(path, set()) & executable_lines(sources[path], path)) - hits.get(path, set()))
        if not lines:
            continue
        text = sources[path].splitlines()
        evidence = [f"{n}: {text[n - 1].strip()[:120]}" for n in lines[:3] if n - 1 < len(text)]
        digest = hashlib.sha1("\n".join(text[n - 1].strip() for n in lines if n - 1 < len(text)).encode()).hexdigest()[:10]
        out.append(Finding(
            rule="untested-change", severity="resolve", location=f"{path}:{lines[0]}",
            title=f"{len(lines)} changed line(s) in {path} never ran in the selected tests ({_ranges(lines)})",
            evidence=evidence, key=f"{path}|{digest}",
            action="Add or extend a test that exercises them, or acknowledge why they can't be tested."))
    return out


def _ranges(lines: list[int]) -> str:
    parts, start, prev = [], lines[0], lines[0]
    for n in lines[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(parts[:6]) + (" …" if len(parts) > 6 else "")
