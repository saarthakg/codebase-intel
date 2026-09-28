"""Git plumbing: snapshot the working tree, diff snapshots, materialize one.

A snapshot is a git tree object holding every file in the working tree,
including uncommitted and untracked (non-ignored) ones. It's built with a
temporary copy of the index, so the user's index, HEAD and files are never
touched; the only side effect is new objects in the object store, which git
garbage-collects in time if nothing references them.
"""
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class GitError(RuntimeError):
    pass


def git(root: str, *args: str, env: Optional[dict] = None, input: Optional[str] = None) -> str:
    try:
        return subprocess.run(
            ["git", "-C", root, *args], capture_output=True, text=True, check=True,
            env={**os.environ, **(env or {})}, input=input,
        ).stdout
    except FileNotFoundError:
        raise GitError("git is not installed") from None
    except subprocess.CalledProcessError as e:
        raise GitError(f"git {' '.join(args[:2])} failed: {e.stderr.strip()[:300]}") from None


def repo_root(path: str = ".") -> str:
    return git(str(Path(path).resolve()), "rev-parse", "--show-toplevel").strip()


def git_dir(root: str) -> Path:
    """The repo's .git directory (a worktree's own git dir for linked worktrees)."""
    return Path(git(root, "rev-parse", "--absolute-git-dir").strip())


def head_tree(root: str) -> Optional[str]:
    try:
        return git(root, "rev-parse", "--verify", "--quiet", "HEAD^{tree}").strip() or None
    except GitError:
        return None  # no commits yet


EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def snapshot(root: str) -> str:
    """Tree id of the current working tree (tracked, modified and untracked
    files; .gitignore'd files excluded)."""
    real_index = git_dir(root) / "index"
    with tempfile.TemporaryDirectory(prefix="notyet-index-") as tmp:
        index = Path(tmp) / "index"
        if real_index.exists():
            # Reuse its stat cache so unchanged files aren't rehashed. copy2 keeps the
            # index's mtime, which git's racy-git check relies on: with a fresh mtime,
            # a same-size edit made in the same second as the index was written would
            # look unchanged.
            shutil.copy2(real_index, index)
        env = {"GIT_INDEX_FILE": str(index)}
        git(root, "add", "--all", env=env)
        return git(root, "write-tree", env=env).strip()


@dataclass
class FileDelta:
    status: str             # A, M, D, R (rename), T (type change)
    path: str               # path in the newer tree
    old_path: Optional[str] = None


def diff_trees(root: str, old: str, new: str) -> list[FileDelta]:
    out = git(root, "diff-tree", "-r", "-M", "--name-status", "--no-commit-id", "-z", old, new)
    parts = [p for p in out.split("\0") if p]
    deltas: list[FileDelta] = []
    i = 0
    while i < len(parts):
        status = parts[i]
        if status[0] in ("R", "C"):
            deltas.append(FileDelta("R", parts[i + 2], parts[i + 1]))
            i += 3
        else:
            deltas.append(FileDelta(status[0], parts[i + 1]))
            i += 2
    return deltas


def unified_diff(root: str, old: str, new: str, *paths: str, context: int = 0) -> str:
    return git(root, "diff", f"-U{context}", "-M", "--no-color", "--no-ext-diff", old, new, "--", *paths)


def show(root: str, tree: str, path: str) -> Optional[str]:
    """File contents at `path` in `tree`, or None if absent or binary."""
    try:
        text = git(root, "cat-file", "-p", f"{tree}:{path}")
    except GitError:
        return None
    return None if "\0" in text else text


def materialize(root: str, tree: str, dest: str) -> None:
    """Write the files of `tree` into `dest` (read-only for the repo)."""
    proc = subprocess.Popen(["git", "-C", root, "archive", "--format=tar", tree],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        tar.extractall(dest, filter="data")
    if proc.wait() != 0:
        raise GitError(f"git archive failed: {proc.stderr.read().decode()[:300]}")
