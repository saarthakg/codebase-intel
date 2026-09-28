"""Configuration: `.notyet.toml` at the repo root.

During a session the gate reads the file as it was when the session started
(from the baseline snapshot), so an agent editing it mid-session can't
loosen its own gate; a changed config is reported on the receipt instead.
"""
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from notyet import snapshot

CONFIG_FILE = ".notyet.toml"

TEMPLATE = """\
# notyet: a completion gate for AI coding agents. https://github.com/saarthakg/codebase-intel

[test]
# How to run this repo's tests. notyet adds its own flags (test selection,
# junit output), so this must be a pytest invocation for now.
command = "{command}"
# Seconds the gate may spend running tests at each check.
budget_seconds = 60

[gate]
# "report": never block; leave a receipt and a short summary.
# "enforce": block the agent's stop on execution evidence (see docs/PLAN.md).
mode = "report"
"""


@dataclass
class Config:
    test_command: Optional[str] = None
    budget_seconds: int = 60
    mode: str = "report"                      # report | enforce
    source: str = "defaults"                  # where the config was read from
    problems: list[str] = field(default_factory=list)

    @property
    def enforce(self) -> bool:
        return self.mode == "enforce"


def parse(text: Optional[str], source: str) -> Config:
    cfg = Config(source=source if text is not None else "defaults")
    if text is None:
        return cfg
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        cfg.problems.append(f"{CONFIG_FILE} is not valid TOML ({e}); using defaults")
        return cfg
    test = data.get("test", {})
    gate = data.get("gate", {})
    cfg.test_command = (test.get("command") or "").strip() or None
    cfg.budget_seconds = int(test.get("budget_seconds", cfg.budget_seconds))
    mode = gate.get("mode", cfg.mode)
    if mode not in ("report", "enforce"):
        cfg.problems.append(f"gate.mode {mode!r} isn't report or enforce; using report")
        mode = "report"
    cfg.mode = mode
    return cfg


def load(root: str, tree: Optional[str] = None) -> Config:
    """Config from `tree` (a snapshot) if given, else from the working tree."""
    if tree is not None:
        return parse(snapshot.show(root, tree, CONFIG_FILE), f"{CONFIG_FILE} at session start")
    path = Path(root) / CONFIG_FILE
    return parse(path.read_text() if path.exists() else None, CONFIG_FILE)


def detect_test_command(root: str) -> Optional[str]:
    """A pytest command for this repo, if it looks like it uses pytest."""
    r = Path(root)
    uses_pytest = (
        (r / "pytest.ini").exists()
        or (r / "conftest.py").exists()
        or ("[tool.pytest" in _read(r / "pyproject.toml"))
        or ("[tool:pytest]" in _read(r / "setup.cfg"))
        or ("[pytest]" in _read(r / "tox.ini"))
        or any(r.glob("tests/test_*.py")) or any(r.glob("test/test_*.py"))
        or any(r.glob("tests/**/test_*.py"))
    )
    if not uses_pytest:
        return None
    for venv in (".venv", "venv", "env"):
        py = r / venv / "bin" / "python"
        if py.exists():
            return f"{venv}/bin/python -m pytest"
    return "python -m pytest"


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""
