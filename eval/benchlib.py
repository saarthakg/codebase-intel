"""Shared machinery for building benchmark YAML files from a Python repo.

A repo's builder script supplies only hand-written labels (search questions
and which symbols/files to probe). Everything mechanical: exact line spans,
the true import graph, which files reference a symbol, is derived here with
Python's own `ast` module, which is deliberately independent of
codebase-intel's tree-sitter/regex extraction so a benchmark can't inherit
the tool's bugs.
"""
import ast
import subprocess
from pathlib import Path

import yaml

Label = tuple[str, list[tuple[str, str]]]  # (query, [(file, qualified symbol), ...])


def symbol_spans(path: Path) -> dict[str, list[int]]:
    """Qualified name → [start_line, end_line] for every def/class in a file.

    @overload stubs share a name; their spans are merged so the entry covers
    every overload plus the implementation.
    """
    out: dict[str, list[int]] = {}

    def visit(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = f"{prefix}{child.name}"
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                if qual in out:
                    out[qual] = [min(out[qual][0], start), max(out[qual][1], child.end_lineno)]
                else:
                    out[qual] = [start, child.end_lineno]
                visit(child, qual + ".")
            else:
                visit(child, prefix)

    visit(ast.parse(path.read_text()), "")
    return out


def _module_name(rel: str) -> list[str]:
    parts = list(Path(rel).with_suffix("").parts)
    if parts[0] == "src":
        parts = parts[1:]
    return parts


def import_graph(source: Path, py_files: list[str], extra_roots: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """file → sorted list of in-repo files it imports (ground truth via ast).

    Absolute imports are resolved like Python's module search: a script's own
    directory first (only for files outside a package, since that's the only
    time Python puts it on sys.path), then the repo root and src/ (module names
    as seen by the package itself), then `extra_roots`: directories a
    benchmark declares importable, e.g. an example project with its own
    pyproject.toml, or a folder the test suite adds to sys.path.
    """
    modules: dict[str, str] = {}
    for rel in py_files:
        parts = _module_name(rel)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules[".".join(parts)] = rel

    def modules_under(root: str) -> dict[str, str]:
        """Dotted names of every file below `root`, relative to it."""
        found = {}
        prefix = root.rstrip("/") + "/"
        for rel in py_files:
            if rel.startswith(prefix):
                parts = list(Path(rel[len(prefix):]).with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                if parts:
                    found[".".join(parts)] = rel
        return found

    extra_maps = [modules_under(r) for r in extra_roots]
    script_dir_maps: dict[str, dict[str, str]] = {}

    def longest_known(mod: str, table: dict[str, str]):
        while mod and mod not in table:
            mod = mod.rpartition(".")[0]
        return table.get(mod) if mod else None

    def resolve_absolute(mod: str, rel: str, want_exact: bool = False):
        tables = []
        rel_dir = str(Path(rel).parent)
        if rel_dir != "." and not (source / rel_dir / "__init__.py").exists():
            if rel_dir not in script_dir_maps:
                script_dir_maps[rel_dir] = modules_under(rel_dir)
            tables.append(script_dir_maps[rel_dir])
        tables += [modules] + extra_maps
        for table in tables:
            hit = table.get(mod) if want_exact else longest_known(mod, table)
            if hit:
                return hit
        return None

    edges: dict[str, list[str]] = {}
    for rel in py_files:
        parts = _module_name(rel)
        package = parts[:-1]  # for __init__.py this is the package itself
        targets: set[str] = set()
        for node in ast.walk(ast.parse((source / rel).read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    hit = resolve_absolute(alias.name, rel)
                    if hit:
                        targets.add(hit)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - (node.level - 1)]
                    module = ".".join(base + ([node.module] if node.module else []))
                    for alias in node.names:
                        sub = f"{module}.{alias.name}"
                        hit = modules.get(sub) or longest_known(module, modules)
                        if hit:
                            targets.add(hit)
                else:
                    module = node.module or ""
                    for alias in node.names:
                        hit = (resolve_absolute(f"{module}.{alias.name}", rel, want_exact=True)
                               or resolve_absolute(module, rel))
                        if hit:
                            targets.add(hit)
        targets.discard(rel)
        if targets:
            edges[rel] = sorted(targets)
    return edges


def referencing_files(source: Path, symbol: str, py_files: list[str]) -> list[str]:
    hits = []
    for rel in py_files:
        for node in ast.walk(ast.parse((source / rel).read_text())):
            if (
                (isinstance(node, ast.Name) and node.id == symbol)
                or (isinstance(node, ast.Attribute) and node.attr == symbol)
                or (isinstance(node, ast.ImportFrom) and any(a.name == symbol for a in node.names))
            ):
                hits.append(rel)
                break
    return sorted(hits)


def build_bench(
    source: Path,
    *,
    repo: str,
    builder: str,
    prefix: str,
    search_sets: dict[str, list[Label]],
    definitions: list[tuple[str, str]],
    ref_symbols: list[str],
    impact_targets: list[str],
    extra_roots: tuple[str, ...] = (),
) -> dict:
    """Assemble a benchmark dict.

    `search_sets` maps a section name ("search", "search_holdout", ...) to
    labeled questions; `definitions` are (symbol, file relative to `prefix`);
    `impact_targets` are files relative to `prefix`.
    """
    source = Path(source).resolve()
    py_files = sorted(
        p.relative_to(source).as_posix() for p in source.rglob("*.py") if ".git" not in p.parts
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or None

    spans: dict[str, dict[str, list[int]]] = {}

    def span(rel: str, qual: str) -> list[int]:
        if rel not in spans:
            spans[rel] = symbol_spans(source / rel)
        if qual in spans[rel]:
            return spans[rel][qual]
        # Bare method name (e.g. "cert_verify") — must match exactly one Class.method.
        matches = [k for k in spans[rel] if k.endswith("." + qual)]
        if len(matches) != 1:
            raise KeyError(f"{qual!r} in {rel}: expected one match, got {matches}")
        return spans[rel][matches[0]]

    edges = import_graph(source, py_files, extra_roots)
    dependents: dict[str, set[str]] = {}
    for src, targets in edges.items():
        for t in targets:
            dependents.setdefault(t, set()).add(src)

    bench = {
        "repo": repo,
        "repo_commit": commit,
        "notes": (
            f"Labels are hand-written in {builder}; line spans, the import "
            "graph and reference sets are derived with Python's ast module, independent of "
            "codebase-intel's own extraction. Regenerate with: "
            f"python {builder} --source <{repo.split('/')[-1]} checkout>"
        ),
    }
    for name, labels in search_sets.items():
        bench[name] = [
            {
                "query": q,
                "expected": [{"file": f, "symbol": s, "lines": span(f, s)} for f, s in expected],
            }
            for q, expected in labels
        ]
    bench["definition"] = [
        {"symbol": s, "file": prefix + f, "line": span(prefix + f, s)[0]} for s, f in definitions
    ]
    bench["references"] = [
        {"symbol": s, "files": referencing_files(source, s, py_files)} for s in ref_symbols
    ]
    bench["impact"] = [
        {"target": prefix + t, "direct_dependents": sorted(dependents.get(prefix + t, []))}
        for t in impact_targets
    ]
    bench["import_graph"] = edges
    return bench


def write_bench(bench: dict, out: str) -> None:
    with open(out, "w") as f:
        yaml.safe_dump(bench, f, sort_keys=False, width=110)
    sections = ", ".join(
        f"{len(v)} {k}" for k, v in bench.items() if isinstance(v, list)
    )
    edges = sum(len(v) for v in bench["import_graph"].values())
    print(f"Wrote {out}: {sections}, {edges} ground-truth import edges")
