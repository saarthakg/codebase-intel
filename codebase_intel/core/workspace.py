"""A git repository as codebase-intel sees it: where it is, what it's
called, and an index that matches its current commit.

The index is built from a snapshot of HEAD (via `git archive`), never from
the working tree. That keeps it exact for the committed code, so a diff of
uncommitted edits lines up with it (its old side is HEAD), and it only needs
rebuilding when HEAD moves, not after every edit.
"""
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Callable, Optional

from codebase_intel.core import paths
from codebase_intel.core.pipeline import run_ingestion
from codebase_intel.state import RepoState, forget_repo, get_repo_state


class NotAGitRepo(ValueError):
    pass


def git(root: str, *args: str) -> str:
    """stdout of a git command in `root`; raises CalledProcessError on failure."""
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, check=True).stdout


def repo_root(path: str = ".") -> str:
    try:
        return git(str(Path(path).expanduser().resolve()), "rev-parse", "--show-toplevel").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        raise NotAGitRepo(f"{path} is not inside a git repository.") from None


def head_sha(root: str) -> Optional[str]:
    try:
        return git(root, "rev-parse", "--verify", "--quiet", "HEAD").strip() or None
    except subprocess.CalledProcessError:
        return None  # no commits yet


def repo_id_for(root: str) -> str:
    """Stable id for a checkout: its folder name plus a hash of its path, so two
    clones with the same name don't share an index."""
    name = re.sub(r"[^A-Za-z0-9_-]", "-", Path(root).name)[:40] or "repo"
    return f"{name}-{hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:8]}"


def indexed_head(repo_id: str) -> Optional[str]:
    meta = paths.meta_path(repo_id)
    if not meta.exists():
        return None
    try:
        return json.loads(meta.read_text()).get("head")
    except (OSError, ValueError):
        return None


def _snapshot(root: str, rev: str, dest: str) -> None:
    """Extract the tree of `rev` into `dest` (read-only for the repo)."""
    proc = subprocess.Popen(["git", "-C", root, "archive", "--format=tar", rev],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:  # streamed, never held in memory
        tar.extractall(dest, filter="data")
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, "git archive", stderr=proc.stderr.read())


def ensure_index(path: str = ".", progress: Optional[Callable[[str], None]] = None) -> tuple[str, str, RepoState]:
    """(repo root, repo_id, state), indexing HEAD first if the index is missing
    or was built from another commit."""
    root = repo_root(path)
    repo_id = repo_id_for(root)
    head = head_sha(root)
    if head is None:
        raise NotAGitRepo(f"{root} has no commits yet.")
    if indexed_head(repo_id) != head:
        if progress:
            progress(f"Indexing {root} at {head[:10]}...")
        with tempfile.TemporaryDirectory(prefix="codebase-intel-") as tmp:
            _snapshot(root, head, tmp)
            run_ingestion(tmp, repo_id, progress, history_path=root,
                          extra_meta={"repo_path": root, "head": head})
        forget_repo(repo_id)
    return root, repo_id, get_repo_state(repo_id)
