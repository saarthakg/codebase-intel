"""Git co-change history: files that tend to change in the same commits.

Import edges miss coupling that isn't an import: a module and its tests, a
schema and the code that serializes it, two files implementing both halves of
a protocol. Version history records that coupling directly: if B changed in
most commits that changed A, a change to A is likely to need a change to B.
"""
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

# Commits touching more files than this are refactors, renames, formatting
# passes or dependency bumps; they'd link everything to everything.
MAX_FILES_PER_COMMIT = 30
# Pairs seen fewer times than this are noise, not coupling.
MIN_SUPPORT = 2
# Shrinks co-change rates toward 0 when a file has few commits: with sparse
# history (a shallow clone), "changed together in 3 of 3 commits" would
# otherwise score 1.0. With k=3 that's 0.5. Full-history results on the
# history eval are identical for k in 0..10; it only bites on thin evidence.
CONFIDENCE_PRIOR_COMMITS = 3


@dataclass
class Commit:
    sha: str
    date: str               # ISO date, YYYY-MM-DD
    files: list[str]        # paths as they are named *today* (renames followed)
    # today's path → the path it had in this commit (differs for renamed files)
    paths_then: dict[str, str] = field(default_factory=dict)


def read_history(
    repo_path: str,
    max_commits: Optional[int] = None,
    until: Optional[str] = None,
    since: Optional[str] = None,
) -> list[Commit]:
    """Commits newest-first, with every path translated to its current name.

    Renames are followed: a file moved from `requests/adapters.py` to
    `src/requests/adapters.py` is reported under the new path in older
    commits too, so its history isn't split in two. Returns [] if `repo_path`
    isn't a git work tree (or git isn't installed).

    `until`/`since` (YYYY-MM-DD, inclusive) filter *after* walking from HEAD:
    passing them to git would hide renames made outside the window, leaving
    old paths untranslated.
    """
    # --relative: paths relative to repo_path (and only files under it), so
    # ingesting a subdirectory of a larger git repo lines up with its index.
    cmd = ["git", "-C", str(repo_path), "log", "-M", "--name-status", "--no-merges", "--relative",
           "--format=@@%H %ad", "--date=short"]
    if max_commits and not (until or since):
        cmd.append(f"--max-count={max_commits}")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []

    # Walking newest → oldest: `current[old] = new` once we've seen the rename,
    # so every older mention of `old` is reported under today's name.
    current: dict[str, str] = {}
    commits: list[Commit] = []
    sha = date = None
    files: list[str] = []
    then: dict[str, str] = {}

    def flush():
        if sha is not None:
            commits.append(Commit(sha, date, list(dict.fromkeys(files)), dict(then)))

    for line in out.splitlines():
        if line.startswith("@@"):
            flush()
            sha, date = line[2:].split(" ", 1)
            files = []
            then = {}
        elif line.strip():
            parts = line.split("\t")
            status = parts[0]
            if status.startswith(("R", "C")) and len(parts) == 3:
                old, new = parts[1], parts[2]
                now = current.get(new, new)
                if status.startswith("R"):
                    current[old] = now
                files.append(now)
                then[now] = new
            elif len(parts) >= 2:
                now = current.get(parts[1], parts[1])
                files.append(now)
                then[now] = parts[1]
    flush()
    if until:
        commits = [c for c in commits if c.date <= until]
    if since:
        commits = [c for c in commits if c.date >= since]
    return commits[:max_commits] if max_commits else commits


@dataclass
class CoChange:
    """Pairwise co-change counts over a set of commits."""
    file_commits: Counter = field(default_factory=Counter)                          # file → #commits
    pairs: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))  # a → {b: #together}
    commits_used: int = 0

    @classmethod
    def from_commits(
        cls,
        commits: list[Commit],
        keep: Optional[set[str]] = None,
        max_files: int = MAX_FILES_PER_COMMIT,
    ) -> "CoChange":
        """`keep`: only count files in this set (e.g. files that still exist)."""
        stats = cls()
        for commit in commits:
            files = [f for f in commit.files if keep is None or f in keep]
            if not files or len(commit.files) > max_files:
                continue
            stats.commits_used += 1
            for f in files:
                stats.file_commits[f] += 1
            for a in files:
                for b in files:
                    if a != b:
                        stats.pairs[a][b] += 1
        return stats

    def related(self, file: str, min_support: int = MIN_SUPPORT, limit: int = 50) -> list[tuple[str, float, int]]:
        """Files that co-changed with `file`: (other, confidence, support) best-first.

        confidence ≈ P(other changed | file changed) = together / (commits(file) + k),
        with k = CONFIDENCE_PRIOR_COMMITS shrinking rates backed by few commits.
        """
        total = self.file_commits.get(file, 0)
        if not total:
            return []
        out = [
            (other, n / (total + CONFIDENCE_PRIOR_COMMITS), n)
            for other, n in self.pairs.get(file, {}).items()
            if n >= min_support
        ]
        out.sort(key=lambda t: (-t[1], -t[2], t[0]))
        return out[:limit]

    def to_rows(self, min_support: int = MIN_SUPPORT) -> tuple[list[tuple[str, int]], list[tuple[str, str, int]]]:
        """(file_commit_rows, pair_rows) for storage; pairs below min_support are dropped."""
        files = sorted(self.file_commits.items())
        pairs = [
            (a, b, n)
            for a, others in sorted(self.pairs.items())
            for b, n in sorted(others.items())
            if n >= min_support
        ]
        return files, pairs

    @classmethod
    def from_rows(cls, file_rows, pair_rows, commits_used: int = 0) -> "CoChange":
        stats = cls(commits_used=commits_used)
        for f, n in file_rows:
            stats.file_commits[f] = n
        for a, b, n in pair_rows:
            stats.pairs[a][b] = n
        return stats


def cochange_for_repo(repo_path: str, keep: set[str], max_commits: int = 5000) -> CoChange:
    """Co-change stats for the files in `keep` from the repo's most recent
    `max_commits` commits. Empty if `repo_path` isn't inside a git repo."""
    return CoChange.from_commits(read_history(repo_path, max_commits=max_commits), keep=keep)
