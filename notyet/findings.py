"""Findings, and the context engines work from."""
import hashlib
from dataclasses import dataclass, field

from notyet.config import Config
from notyet.snapshot import FileDelta
from notyet.store import Session

# block:   execution evidence; only a fix or needs-human clears it
# resolve: fix it, or acknowledge it with a reason the human will see
# note:    information for the agent and the receipt
SEVERITIES = ("block", "resolve", "note")


@dataclass
class Finding:
    rule: str                  # e.g. "test-regression", "test-dropped"
    severity: str
    title: str                 # one line, for the agent and the receipt
    location: str = ""         # file[:line] or test id
    evidence: list[str] = field(default_factory=list)   # short lines: test output, commit ids
    action: str = ""           # what to do about it
    key: str = ""              # what makes it the same finding across runs (defaults to location)

    @property
    def id(self) -> str:
        raw = f"{self.rule}|{self.key or self.location}"
        return hashlib.sha1(raw.encode()).hexdigest()[:8]


@dataclass
class Context:
    root: str
    session: Session
    config: Config
    baseline_tree: str
    current_tree: str
    deltas: list[FileDelta]


@dataclass
class EngineResult:
    findings: list[Finding] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)       # what was checked, for the receipt
    not_checked: list[str] = field(default_factory=list)  # negative space
    advice: list[str] = field(default_factory=list)       # advisory lines (history, etc.)

    def extend(self, other: "EngineResult") -> None:
        self.findings += other.findings
        self.checks += other.checks
        self.not_checked += other.not_checked
        self.advice += other.advice
