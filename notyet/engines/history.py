"""Advice from history: files that usually change along with the changed ones.

"src/app.py usually changes with CHANGES.rst: in 5 of 7 changes, e.g. a1b2c3d."
Counted per merged change on the main line (notyet/history.py), from the
latest 5000. Only strong, repeated pairs are mentioned, at most three lines,
and never as a block: history says what's usual, not what's required.
Cached per HEAD commit in .git/notyet/cache/.
"""
import json

from notyet import history, snapshot, store
from notyet.findings import Context, EngineResult

MIN_TOGETHER = 3
MIN_CONFIDENCE = 0.5
MAX_LINES = 3


def run(ctx: Context) -> EngineResult:
    result = EngineResult()
    changed = {d.path for d in ctx.deltas if d.status != "D"}
    if not changed:
        return result
    stats = _stats(ctx.root)
    if stats is None:
        return result
    files, pairs = stats
    existing = set(snapshot.git(ctx.root, "ls-tree", "-r", "--name-only", ctx.current_tree).splitlines())
    lines = []
    for path in sorted(changed):
        total = files.get(path, 0)
        for other, (together, sha) in pairs.get(path, {}).items():
            confidence = together / (total + history.CONFIDENCE_PRIOR_COMMITS)
            if other in changed or other not in existing or confidence < MIN_CONFIDENCE:
                continue
            lines.append((confidence, together, path, other, total, sha))
    lines.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
    for _, together, path, other, total, sha in lines[:MAX_LINES]:
        result.advice.append(f"{path} usually changes with {other}, which this session didn't touch: "
                             f"in {together} of {total} changes, e.g. {sha[:8]}")
    return result


def _stats(root: str):
    """({file: #changes}, {file: {other: (#together, latest sha)}}) for pairs seen at least MIN_TOGETHER times."""
    try:
        head = snapshot.git(root, "rev-parse", "HEAD").strip()
    except snapshot.GitError:
        return None
    path = store.state_dir(root) / "cache" / "history.json"
    path.parent.mkdir(exist_ok=True)
    try:
        cached = json.loads(path.read_text())
        if cached.get("head") == head:
            return cached["files"], {a: {b: tuple(v) for b, v in o.items()} for a, o in cached["pairs"].items()}
    except (OSError, ValueError, KeyError):
        pass
    commits = history.read_history(root, max_commits=history.MAX_CHANGES, first_parent=True)  # newest first
    stats = history.CoChange.from_commits(commits)
    latest: dict[tuple[str, str], str] = {}
    for c in commits:
        if len(c.files) > history.MAX_FILES_PER_COMMIT:
            continue
        for a in c.files:
            for b in c.files:
                if a != b and (a, b) not in latest:
                    latest[(a, b)] = c.sha
    pairs = {a: {b: (n, latest[(a, b)]) for b, n in others.items() if n >= MIN_TOGETHER}
             for a, others in stats.pairs.items()}
    pairs = {a: o for a, o in pairs.items() if o}
    files = dict(stats.file_commits)
    path.write_text(json.dumps({"head": head, "files": files, "pairs": pairs}))
    return files, pairs
