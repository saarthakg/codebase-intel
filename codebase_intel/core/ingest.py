import os
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".env", ".next", ".nuxt",
    ".idea", ".vscode", ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}

# Binary formats: never part of an impact answer worth reading.
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp", ".pdf", ".psd",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".whl", ".egg", ".jar",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".mp3", ".mp4", ".mov", ".wav",
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".o", ".a", ".class", ".bin",
}

# Code is parsed for symbols, references and imports; every other text file
# (docs, changelogs, config, lockfiles) is indexed by path only, since what
# matters for impact is that history can link it to the code it changes with.
CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx"}

# Larger code files are almost always generated or vendored, and aren't
# parsed. Override with INGEST_MAX_FILE_BYTES.
DEFAULT_MAX_FILE_BYTES = 1_000_000

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
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
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
    the tree is walked. Either way SKIP_DIRS and binary extensions are
    skipped, and so are minified code and code over INGEST_MAX_FILE_BYTES.
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
        if any(p in SKIP_DIRS for p in rel_parts[:-1]):
            continue
        ext = path.suffix.lower()
        if ext in SKIP_EXTENSIONS:
            continue
        if ext not in CODE_EXTENSIONS:
            scan.files.append(str(path))  # path-only; binaries are dropped on reading
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


def is_code(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in CODE_EXTENSIONS


def looks_binary(file_path: str) -> bool:
    """Null bytes, or not UTF-8, in the first 8 KB."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(8192)
    except OSError:
        return True
    if b"\x00" in head:
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as e:
        return e.start < len(head) - 3  # a multi-byte character cut at the boundary is fine
    return False


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
