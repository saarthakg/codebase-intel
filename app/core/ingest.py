import os
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


def walk_repo(repo_path: str) -> list[str]:
    """Return list of absolute file paths to index."""
    results = []
    repo = Path(repo_path).resolve()
    for dirpath, dirnames, filenames in os.walk(repo):
        # Prune skip dirs in-place so os.walk doesn't descend into them
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in SKIP_EXTENSIONS:
                continue
            if ext not in INCLUDE_EXTENSIONS:
                continue
            results.append(str(Path(dirpath) / fname))
    return results


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
