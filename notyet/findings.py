"""Findings, and the context engines work from."""
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

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
    # the items a finding is about, when it groups several (untested-change: one digest per changed
    # line). An acknowledgment of such a finding carries over to later findings of the same group
    # while the items it didn't cover stay few (see find_ack), even though the id changes.
    covers: list[str] = field(default_factory=list)

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
    deadline: Optional[float] = None      # time.monotonic() by which the whole check must end

    def time_left(self) -> float:
        return float("inf") if self.deadline is None else self.deadline - time.monotonic()


@dataclass
class EngineResult:
    findings: list[Finding] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)       # what was checked, for the receipt
    not_checked: list[str] = field(default_factory=list)  # negative space
    advice: list[str] = field(default_factory=list)       # advisory lines (history, etc.)
    gaps: list[str] = field(default_factory=list)         # a core check that didn't run: the summary leads with it

    def extend(self, other: "EngineResult") -> None:
        self.findings += other.findings
        self.checks += other.checks
        self.not_checked += other.not_checked
        self.advice += other.advice
        self.gaps += other.gaps


def group(f: Finding) -> str:
    """What a finding's acknowledgment can carry over to: same rule, same place."""
    return f"{f.rule}|{(f.key or f.location).split('|')[0]}"


def find_ack(f: Finding, acks: dict[str, dict]) -> Optional[dict]:
    """The acknowledgment that applies to `f`: one for its id, or, for a finding that groups items,
    one for the same group that covered all but a few of its items now: at most CARRY_MIN, or a
    tenth of what was acknowledged. Items are counted against that acknowledgment, not the last
    check, so untested code that keeps growing turn by turn is raised again."""
    ack = acks.get(f.id)
    if ack is not None or not f.covers:
        return ack
    for a in acks.values():
        covered = a.get("covers", [])
        if a.get("group") == group(f) and uncovered(f, a) <= max(CARRY_MIN, len(covered) // 10):
            return a
    return None


def uncovered(f: Finding, ack: dict) -> int:
    """How many of `f`'s items the acknowledgment didn't cover."""
    return len(set(f.covers) - set(ack.get("covers", [])))


CARRY_MIN = 2
