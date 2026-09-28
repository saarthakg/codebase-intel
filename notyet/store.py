"""Per-repo state under .git/notyet/: sessions, gate runs and receipts.

Kept inside the git directory so nothing appears in the working tree (the
agent's change never includes it) and nothing gets committed.
"""
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from notyet import snapshot

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def state_dir(root: str) -> Path:
    d = snapshot.git_dir(root) / "notyet"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class GateRun:
    tree: str
    at: float
    verdict: str                       # passed | blocked | reported | not-checked
    finding_ids: list[str] = field(default_factory=list)   # standing (unresolved) findings
    receipt: Optional[str] = None      # path of the receipt file
    result: dict = field(default_factory=dict)             # the engines' full result, to re-decide without re-running


@dataclass
class Session:
    session_id: str
    started: float
    baseline_tree: str
    baseline_head: Optional[str]       # HEAD's tree at session start, for "was it already failing?"
    baseline_source: str               # session-start | first-prompt | head (hooks installed mid-session)
    prompts: list[str] = field(default_factory=list)
    runs: list[GateRun] = field(default_factory=list)
    consecutive_blocks: int = 0
    acks: dict[str, dict] = field(default_factory=dict)   # finding id → {category, reason, at}
    turns: list[dict] = field(default_factory=list)      # {"at", "tree"}: the working tree at each prompt
    stops: list[dict] = field(default_factory=list)      # {"at", "busy"}: each Stop, including skipped ones


def _session_path(root: str, session_id: str) -> Path:
    d = state_dir(root) / "sessions"
    d.mkdir(exist_ok=True)
    return d / f"{_SAFE.sub('_', session_id)[:120]}.json"


def load_session(root: str, session_id: str) -> Optional[Session]:
    path = _session_path(root, session_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data["runs"] = [GateRun(**r) for r in data.get("runs", [])]
    return Session(**data)


def save_session(root: str, session: Session) -> None:
    _write_atomic(_session_path(root, session.session_id), json.dumps(asdict(session), indent=1))


def start_session(root: str, session_id: str, source: str) -> Session:
    """Create the session with a baseline snapshot of the tree as it is now."""
    session = Session(
        session_id=session_id, started=time.time(), baseline_tree=snapshot.snapshot(root),
        baseline_head=snapshot.head_tree(root), baseline_source=source,
    )
    save_session(root, session)
    return session


def latest_session(root: str) -> Optional[Session]:
    d = state_dir(root) / "sessions"
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime) if d.exists() else []
    return load_session(root, json.loads(files[-1].read_text())["session_id"]) if files else None


def all_sessions(root: str) -> list[Session]:
    d = state_dir(root) / "sessions"
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime) if d.exists() else []
    return [s for s in (load_session(root, json.loads(f.read_text())["session_id"]) for f in files) if s]


def receipts_dir(root: str) -> Path:
    d = state_dir(root) / "receipts"
    d.mkdir(exist_ok=True)
    return d


def log_error(root: str, message: str) -> None:
    try:
        with open(state_dir(root) / "errors.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)
