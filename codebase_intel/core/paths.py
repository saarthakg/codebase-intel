"""Central definitions of where a repo's on-disk artifacts live.

Every module that needs to read or write a repo's index/db/graph/meta files
should go through these helpers instead of re-deriving the paths, so the
on-disk layout only has one source of truth.
"""
import os
from pathlib import Path

from codebase_intel.core.validation import validate_repo_id


def _default_home() -> Path:
    """$CODEBASE_INTEL_HOME, else the user cache dir. Indexes are derived
    data: deleting the directory only means the next check re-indexes."""
    if os.environ.get("CODEBASE_INTEL_HOME"):
        return Path(os.environ["CODEBASE_INTEL_HOME"]).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "codebase-intel"


DATA_METADATA = _default_home()


def ensure_data_dirs() -> None:
    DATA_METADATA.mkdir(parents=True, exist_ok=True)


def db_path(repo_id: str) -> Path:
    return DATA_METADATA / f"{validate_repo_id(repo_id)}.db"


def graph_path(repo_id: str) -> Path:
    return DATA_METADATA / f"{validate_repo_id(repo_id)}.graph.json"


def legacy_graph_path(repo_id: str) -> Path:
    """Where indexes built before the JSON format kept a pickled graph."""
    return DATA_METADATA / f"{validate_repo_id(repo_id)}.graph.pkl"


def meta_path(repo_id: str) -> Path:
    return DATA_METADATA / f"{validate_repo_id(repo_id)}.meta.json"


def repo_artifact_paths(repo_id: str) -> list[Path]:
    """All files on disk that belong to a given repo_id (for deletion)."""
    return [
        db_path(repo_id),
        graph_path(repo_id),
        legacy_graph_path(repo_id),
        meta_path(repo_id),
    ]


def known_repo_ids() -> list[str]:
    """repo_ids that have been indexed (they have a saved import graph)."""
    if not DATA_METADATA.exists():
        return []
    return sorted(p.name[: -len(".graph.json")] for p in DATA_METADATA.glob("*.graph.json"))
