"""Central definitions of where a repo's on-disk artifacts live.

Every module that needs to read or write a repo's index/db/graph/meta files
should go through these helpers instead of re-deriving the paths, so the
on-disk layout only has one source of truth.
"""
from pathlib import Path

from app.core.validation import validate_repo_id

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_INDEXES = PROJECT_ROOT / "data" / "indexes"
DATA_METADATA = PROJECT_ROOT / "data" / "metadata"


def ensure_data_dirs() -> None:
    DATA_INDEXES.mkdir(parents=True, exist_ok=True)
    DATA_METADATA.mkdir(parents=True, exist_ok=True)


def legacy_index_paths(repo_id: str) -> list[Path]:
    """Vector-index files written before search was removed."""
    rid = validate_repo_id(repo_id)
    return [DATA_INDEXES / f"{rid}.index", DATA_INDEXES / f"{rid}.idmap.json"]


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
        *legacy_index_paths(repo_id),
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
