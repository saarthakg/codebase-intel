#!/usr/bin/env python3
"""Build eval/requests_bench.yaml from hand-written labels + the requests source.

Labels below are hand-written (query → the function/class that answers it).
Everything mechanical — exact line spans, the true import graph, which files
reference a symbol — is derived with Python's own `ast` module, which is
deliberately independent of codebase-intel's tree-sitter/regex extraction so
the benchmark can't inherit the tool's bugs.

Usage:
  python eval/build_requests_bench.py --source ../requests-demo
"""
import argparse
import ast
import subprocess
from pathlib import Path

import yaml

S = "src/requests/"

# query → [(file, qualified symbol that answers it), ...]  (any one is a hit)
SEARCH = [
    ("where is SSL certificate verification handled?", [(S + "adapters.py", "HTTPAdapter.cert_verify")]),
    ("how are HTTP redirects followed?", [(S + "sessions.py", "SessionRedirectMixin.resolve_redirects")]),
    ("strip the Authorization header when redirecting to a different host",
     [(S + "sessions.py", "SessionRedirectMixin.should_strip_auth"), (S + "sessions.py", "SessionRedirectMixin.rebuild_auth")]),
    ("change POST to GET on a 303 See Other redirect", [(S + "sessions.py", "SessionRedirectMixin.rebuild_method")]),
    ("HTTP digest authentication challenge response",
     [(S + "auth.py", "HTTPDigestAuth.build_digest_header"), (S + "auth.py", "HTTPDigestAuth.handle_401")]),
    ("build the basic auth header from username and password", [(S + "auth.py", "_basic_auth_str")]),
    ("read credentials from the .netrc file", [(S + "utils.py", "get_netrc_auth")]),
    ("which proxy should be used for a given URL", [(S + "utils.py", "select_proxy"), (S + "utils.py", "resolve_proxies")]),
    ("NO_PROXY environment variable bypass logic", [(S + "utils.py", "should_bypass_proxies")]),
    ("merge proxy and verify settings from environment variables into the request",
     [(S + "sessions.py", "Session.merge_environment_settings")]),
    ("connection pooling and pool manager setup",
     [(S + "adapters.py", "HTTPAdapter.init_poolmanager"), (S + "adapters.py", "HTTPAdapter.proxy_manager_for")]),
    ("SOCKS proxy support", [(S + "adapters.py", "HTTPAdapter.proxy_manager_for"), (S + "adapters.py", "SOCKSProxyManager")]),
    ("configure retries for failed connections", [(S + "adapters.py", "HTTPAdapter.__init__")]),
    ("how are connect and read timeouts applied when sending", [(S + "adapters.py", "HTTPAdapter.send")]),
    ("convert a urllib3 response into a Response object", [(S + "adapters.py", "HTTPAdapter.build_response")]),
    ("mount a transport adapter for a URL prefix", [(S + "sessions.py", "Session.mount"), (S + "sessions.py", "Session.get_adapter")]),
    ("encode multipart file uploads", [(S + "models.py", "RequestEncodingMixin._encode_files")]),
    ("encode query string parameters", [(S + "models.py", "RequestEncodingMixin._encode_params")]),
    ("validate and normalize the request URL, IDNA hostnames",
     [(S + "models.py", "PreparedRequest.prepare_url"), (S + "models.py", "PreparedRequest._get_idna_encoded_host")]),
    ("set the Content-Length header for the body",
     [(S + "models.py", "PreparedRequest.prepare_content_length"), (S + "models.py", "PreparedRequest.prepare_body")]),
    ("serialize a JSON request body", [(S + "models.py", "PreparedRequest.prepare_body")]),
    ("stream the response body in chunks", [(S + "models.py", "Response.iter_content")]),
    ("iterate over the response one line at a time", [(S + "models.py", "Response.iter_lines")]),
    ("raise an exception for 4xx and 5xx status codes", [(S + "models.py", "Response.raise_for_status")]),
    ("decode the response body as JSON", [(S + "models.py", "Response.json")]),
    ("guess the text encoding of a response with charset detection",
     [(S + "models.py", "Response.apparent_encoding"), (S + "models.py", "Response.text"),
      (S + "utils.py", "get_encoding_from_headers")]),
    ("parse the Link header into a dict", [(S + "utils.py", "parse_header_links"), (S + "models.py", "Response.links")]),
    ("reject header values containing newlines",
     [(S + "utils.py", "check_header_validity"), (S + "utils.py", "_validate_header_part")]),
    ("rewind a file-like request body before resending", [(S + "utils.py", "rewind_body")]),
    ("compute the length of a file-like object or string body", [(S + "utils.py", "super_len")]),
    ("default User-Agent string", [(S + "utils.py", "default_user_agent"), (S + "utils.py", "default_headers")]),
    ("case-insensitive dictionary for headers", [(S + "structures.py", "CaseInsensitiveDict")]),
    ("cookie jar that behaves like a dict", [(S + "cookies.py", "RequestsCookieJar")]),
    ("extract cookies from a response into the jar", [(S + "cookies.py", "extract_cookies_to_jar")]),
    ("merge session cookies with request cookies",
     [(S + "cookies.py", "merge_cookies"), (S + "sessions.py", "Session.prepare_request")]),
    ("dispatch response hooks", [(S + "hooks.py", "dispatch_hook")]),
    ("merge session-level and request-level settings", [(S + "sessions.py", "merge_setting"), (S + "sessions.py", "merge_hooks")]),
    ("check urllib3 and chardet versions are compatible", [(S + "__init__.py", "check_compatibility")]),
    ("print system and dependency info for bug reports", [(S + "help.py", "info")]),
    ("mapping of HTTP status names to numeric codes", [(S + "status_codes.py", "_init")]),
    ("exception raised when there are too many redirects", [(S + "exceptions.py", "TooManyRedirects")]),
    ("requests.get convenience function", [(S + "api.py", "get"), (S + "api.py", "request")]),
]

# symbol (bare or Class.method) → defining file
DEFINITIONS = [
    ("HTTPAdapter", "adapters.py"), ("Session", "sessions.py"), ("PreparedRequest", "models.py"),
    ("Response", "models.py"), ("CaseInsensitiveDict", "structures.py"), ("RequestsCookieJar", "cookies.py"),
    ("HTTPDigestAuth", "auth.py"), ("dispatch_hook", "hooks.py"), ("get_netrc_auth", "utils.py"),
    ("super_len", "utils.py"), ("to_native_string", "_internal_utils.py"), ("check_compatibility", "__init__.py"),
    ("cert_verify", "adapters.py"), ("resolve_redirects", "sessions.py"), ("prepare_url", "models.py"),
    ("raise_for_status", "models.py"), ("build_digest_header", "auth.py"), ("merge_environment_settings", "sessions.py"),
    # Qualified method names — the bare name is ambiguous (e.g. `send` exists on several classes).
    ("HTTPAdapter.send", "adapters.py"), ("Session.send", "sessions.py"), ("Session.request", "sessions.py"),
    ("PreparedRequest.prepare_body", "models.py"), ("Response.json", "models.py"), ("HTTPBasicAuth.__call__", "auth.py"),
]

# Distinctive names whose usages we can find unambiguously.
REF_SYMBOLS = [
    "HTTPAdapter", "PreparedRequest", "CaseInsensitiveDict", "dispatch_hook", "to_native_string",
    "extract_cookies_to_jar", "super_len", "get_netrc_auth", "cookiejar_from_dict", "TooManyRedirects",
    "merge_cookies", "rewind_body",
]

IMPACT_TARGETS = [
    "adapters.py", "certs.py", "hooks.py", "structures.py", "help.py", "cookies.py",
    "_internal_utils.py", "status_codes.py", "api.py", "exceptions.py",
]


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


def import_graph(source: Path, py_files: list[str]) -> dict[str, list[str]]:
    """file → sorted list of in-repo files it imports (ground truth via ast)."""
    modules: dict[str, str] = {}
    for rel in py_files:
        parts = _module_name(rel)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules[".".join(parts)] = rel

    def longest_known(mod: str):
        while mod and mod not in modules:
            mod = mod.rpartition(".")[0]
        return modules.get(mod) if mod else None

    edges: dict[str, list[str]] = {}
    for rel in py_files:
        parts = _module_name(rel)
        package = parts[:-1]  # for __init__.py this is the package itself
        targets: set[str] = set()
        for node in ast.walk(ast.parse((source / rel).read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    hit = longest_known(alias.name)
                    if hit:
                        targets.add(hit)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - (node.level - 1)]
                    module = ".".join(base + ([node.module] if node.module else []))
                else:
                    module = node.module or ""
                for alias in node.names:
                    sub = f"{module}.{alias.name}"
                    hit = modules.get(sub) or longest_known(module)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to the requests checkout")
    parser.add_argument("--out", default=str(Path(__file__).parent / "requests_bench.yaml"))
    args = parser.parse_args()

    source = Path(args.source).resolve()
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

    edges = import_graph(source, py_files)
    dependents: dict[str, set[str]] = {}
    for src, targets in edges.items():
        for t in targets:
            dependents.setdefault(t, set()).add(src)

    bench = {
        "repo": "psf/requests",
        "repo_commit": commit,
        "notes": (
            "Labels are hand-written in eval/build_requests_bench.py; line spans, the import "
            "graph and reference sets are derived with Python's ast module, independent of "
            "codebase-intel's own extraction. Regenerate with: "
            "python eval/build_requests_bench.py --source <requests checkout>"
        ),
        "search": [
            {
                "query": q,
                "expected": [{"file": f, "symbol": s, "lines": span(f, s)} for f, s in expected],
            }
            for q, expected in SEARCH
        ],
        "definition": [
            {"symbol": s, "file": S + f, "line": span(S + f, s)[0]} for s, f in DEFINITIONS
        ],
        "references": [
            {"symbol": s, "files": referencing_files(source, s, py_files)} for s in REF_SYMBOLS
        ],
        "impact": [
            {"target": S + t, "direct_dependents": sorted(dependents.get(S + t, []))}
            for t in IMPACT_TARGETS
        ],
        "import_graph": edges,
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(bench, f, sort_keys=False, width=110)
    print(
        f"Wrote {args.out}: {len(bench['search'])} search, {len(bench['definition'])} definition, "
        f"{len(bench['references'])} references, {len(bench['impact'])} impact cases, "
        f"{sum(len(v) for v in edges.values())} ground-truth import edges"
    )


if __name__ == "__main__":
    main()
