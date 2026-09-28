"""Evidence that a file belongs in a change: every candidate file for a set
of changed files, with the raw signals for it.

Scoring (turning these into one confidence) is separate, in scoring.py, so
the replay eval and the product use the same evidence and different scorers
can be compared on it.
"""
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Optional

from codebase_intel.core.definitions import is_test_path, tested_module_stem, tests_named_for
from codebase_intel.core.history import CONFIDENCE_PRIOR_COMMITS
from codebase_intel.core.usages import symbol_users

if TYPE_CHECKING:
    from codebase_intel.core.graph import DependencyGraph
    from codebase_intel.core.history import CoChange
    from codebase_intel.storage.metadata_store import MetadataStore

_DOC_RE = re.compile(r"\.(rst|md|txt)$|(^|/)docs?/")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Candidate:
    file: str
    # Co-change with the changed files
    cc_p: float = 0.0            # max over changed q of n(q,c) / (commits(q) + k): P(c changes | q changes)
    cc_rev_p: float = 0.0        # max over q of n(q,c) / (commits(c) + k): P(q changes | c changes)
    cc_n: int = 0                # max over q of commits shared with q
    cc_queries: int = 0          # how many changed files it has co-changed with
    cc_with: str = ""            # the changed file behind cc_p, and its commit count (for reasons)
    cc_total: int = 0
    rc_p: float = 0.0            # cc_p over recent history only
    rc_n: int = 0                # cc_n over recent history only
    rc_window: int = 0           # changes in the recent-history window
    # Structure
    import_hops: int = 0         # fewest hops at which c imports a changed file (0: doesn't)
    imports: str = ""            # ...that changed file
    imported_hops: int = 0       # 1 if a changed file imports c directly
    imported_by: str = ""
    symbol_uses: int = 0         # changed symbols c uses
    named_test: str = ""         # the changed file c is a test named after
    tested_by_change: str = ""   # the changed test named after c
    same_dir: bool = False       # in the same directory as a changed file
    path_sim: float = 0.0        # max token Jaccard between c's path and a changed file's
    # The candidate itself
    commits: int = 0             # how often c changes at all
    kind: str = "source"         # source / test / docs / other
    because_of: set[str] = field(default_factory=set)
    uses: list[tuple[str, str]] = field(default_factory=list)  # (changed file, symbol) it uses


def file_kind(path: str) -> str:
    if is_test_path(path):
        return "test"
    if path.endswith((".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")):
        return "source"
    if _DOC_RE.search(path):
        return "docs"
    return "other"


def _tokens(path: str) -> set[str]:
    return set(_TOKEN_RE.findall(path.lower())) - {"py", "src", "test", "tests"}


def gather(
    changed: dict[str, list[str]],
    repo_id: str,
    graph: "DependencyGraph",
    metadata_store: "MetadataStore",
    cochange: Optional["CoChange"],
    depth: int = 3,
    recent: Optional["CoChange"] = None,
) -> dict[str, Candidate]:
    """Candidates for a change: `changed` maps each changed (indexed) file to
    the qualified names of the symbols it touched. `recent`: co-change over
    the latest changes only."""
    query = set(changed)
    cands: dict[str, Candidate] = {}

    def cand(f: str) -> Optional[Candidate]:
        if f in query:
            return None
        if f not in cands:
            cands[f] = Candidate(file=f, kind=file_kind(f))
        return cands[f]

    k = CONFIDENCE_PRIOR_COMMITS
    for q in query:
        if cochange is not None:
            total_q = cochange.file_commits.get(q, 0)
            for other, _p, n in cochange.related(q, limit=200):
                c = cand(other)
                if c is None:
                    continue
                if n / (total_q + k) > c.cc_p:
                    c.cc_p, c.cc_with, c.cc_total = n / (total_q + k), q, total_q
                c.cc_rev_p = max(c.cc_rev_p, n / (cochange.file_commits.get(other, 0) + k))
                c.cc_n = max(c.cc_n, n)
                c.cc_queries += 1
                c.because_of.add(q)
        if recent is not None:
            total_q = recent.file_commits.get(q, 0)
            for other, _p, n in recent.related(q, limit=200):
                c = cand(other)
                if c is not None:
                    c.rc_window = recent.commits_used
                    c.rc_p = max(c.rc_p, n / (total_q + k))
                    c.rc_n = max(c.rc_n, n)
                    c.because_of.add(q)
        if q in graph.G.nodes:
            for entry in graph.dependents_of(q, depth=depth):
                c = cand(entry["file"])
                if c is not None and (not c.import_hops or entry["depth"] < c.import_hops):
                    c.import_hops, c.imports = entry["depth"], q
                    c.because_of.add(q)
            for entry in graph.dependencies_of(q, depth=1):
                c = cand(entry["file"])
                if c is not None:
                    c.imported_hops, c.imported_by = 1, q
                    c.because_of.add(q)
            for t in tests_named_for(q, graph.G.nodes):
                c = cand(t)
                if c is not None:
                    c.named_test = q
                    c.because_of.add(q)
            stem = tested_module_stem(q)
            if stem:
                for f in graph.G.nodes:
                    if f not in query and not is_test_path(f) and PurePosixPath(f).stem == stem:
                        c = cand(f)
                        c.tested_by_change = q
                        c.because_of.add(q)
        for symbol in changed[q]:
            for user in symbol_users(repo_id, symbol, q, graph, metadata_store, depth):
                c = cand(user)
                if c is not None:
                    c.symbol_uses += 1
                    c.uses.append((q, symbol))
                    c.because_of.add(q)

    q_dirs = {str(PurePosixPath(q).parent) for q in query}
    q_tokens = [_tokens(q) for q in query]
    for c in cands.values():
        c.same_dir = str(PurePosixPath(c.file).parent) in q_dirs
        toks = _tokens(c.file)
        c.path_sim = max((len(toks & t) / len(toks | t) for t in q_tokens if toks | t), default=0.0)
        c.commits = cochange.file_commits.get(c.file, 0) if cochange is not None else 0
    return cands
