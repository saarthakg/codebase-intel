import os
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", "__pycache__",
    ".venv", "venv", ".env", "coverage", ".next", ".nuxt",
    "target", "out", "bin", "obj", ".idea", ".vscode",
}

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf",
    ".zip", ".tar", ".gz", ".lock", ".sum", ".whl", ".egg",
}

INCLUDE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".md", ".txt", ".yaml", ".yml", ".toml", ".json",
}


# Larger files are almost always generated, vendored or data.
# Override with INGEST_MAX_FILE_BYTES.
DEFAULT_MAX_FILE_BYTES = 1_000_000

# Dependency lockfiles: huge, generated, and never what a code question is about.
LOCKFILE_NAMES = {
    "package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock",
    "poetry.lock", "Pipfile.lock", "composer.lock", "Cargo.lock", "Gemfile.lock",
    "uv.lock", "bun.lockb",
}

# Minified/bundled code: one enormous line. Checked on the first 64 KB.
_MINIFIED_SAMPLE_BYTES = 65_536
_MINIFIED_AVG_LINE = 300
_MINIFIED_EXTS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".json"}


@dataclass
class RepoScan:
    files: list[str]                                   # absolute paths to index
    skipped: Counter = field(default_factory=Counter)  # reason → count
    used_git: bool = False                             # .gitignore rules applied via git
    indexed: int = 0                                   # files actually read and indexed (set by the pipeline)


def _max_file_bytes() -> int:
    return int(os.environ.get("INGEST_MAX_FILE_BYTES", DEFAULT_MAX_FILE_BYTES))


def _git_candidates(repo: Path) -> Optional[list[Path]]:
    """Files git considers part of the work tree (tracked + untracked, minus
    everything .gitignore'd), or None if `repo` isn't inside a git repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    paths = [repo / p for p in out.decode("utf-8", errors="surrogateescape").split("\0") if p]
    return [p for p in paths if p.is_file()]  # tracked-but-deleted files are listed too


def _walk_candidates(repo: Path) -> list[Path]:
    results = []
    for dirpath, dirnames, filenames in os.walk(repo):
        # Prune skip dirs in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        results += [Path(dirpath) / f for f in filenames]
    return results


def _looks_minified(path: Path) -> bool:
    if ".min." in path.name:
        return True
    if path.suffix.lower() not in _MINIFIED_EXTS:
        return False
    try:
        with open(path, "rb") as f:
            sample = f.read(_MINIFIED_SAMPLE_BYTES)
    except OSError:
        return False
    lines = sample.count(b"\n") + 1
    return len(sample) > 2_000 and len(sample) / lines > _MINIFIED_AVG_LINE


def scan_repo(repo_path: str) -> RepoScan:
    """Choose the files to index, and count what was skipped and why.

    Inside a git repo the candidates come from `git ls-files`, so every
    .gitignore (nested ones and global excludes too) is honored; otherwise
    the tree is walked. Either way SKIP_DIRS, dot-directories and the
    extension allow-list apply, and lockfiles, minified files and files over
    INGEST_MAX_FILE_BYTES are skipped.
    """
    repo = Path(repo_path).resolve()
    candidates = _git_candidates(repo)
    used_git = candidates is not None
    if candidates is None:
        candidates = _walk_candidates(repo)

    max_bytes = _max_file_bytes()
    scan = RepoScan(files=[], used_git=used_git)
    for path in sorted(candidates):
        rel_parts = path.relative_to(repo).parts
        if any(p in SKIP_DIRS or p.startswith(".") for p in rel_parts[:-1]):
            continue
        ext = path.suffix.lower()
        if ext in SKIP_EXTENSIONS or ext not in INCLUDE_EXTENSIONS:
            continue
        if path.name in LOCKFILE_NAMES:
            scan.skipped["lockfile"] += 1
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_bytes:
            scan.skipped["too_large"] += 1
            continue
        if _looks_minified(path):
            scan.skipped["minified"] += 1
            continue
        scan.files.append(str(path))
    return scan


def walk_repo(repo_path: str) -> list[str]:
    """Return list of absolute file paths to index (see scan_repo)."""
    return scan_repo(repo_path).files


def detect_language(file_path: str) -> str:
    """Return 'python', 'typescript', 'javascript', 'markdown', or 'unknown'."""
    ext = Path(file_path).suffix.lower()
    mapping = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".md": "markdown",
        ".txt": "unknown",
        ".yaml": "unknown",
        ".yml": "unknown",
        ".toml": "unknown",
        ".json": "unknown",
    }
    return mapping.get(ext, "unknown")


def load_file(file_path: str) -> Optional[str]:
    """Read file contents. Return None if binary or unreadable."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="strict") as f:
            content = f.read()
        # Sanity-check: reject files with null bytes (binary smuggled as text)
        if "\x00" in content:
            return None
        return content
    except (UnicodeDecodeError, OSError):
        return None
