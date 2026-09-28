"""The gate: run the engines on the session's change and decide whether the
agent may stop.

Loop safety (docs/PLAN.md §3.6):
- engines run only when the tree changed since the last check; an unchanged
  tree is re-decided from the saved result and the current acknowledgments;
- a finding set that comes back unchanged is blocked at most twice, then the
  stop is allowed and the receipt leads with "unresolved";
- never more than MAX_BLOCKS consecutive blocks (Claude Code's own cap is 8);
- never blocks while background work is running.
"""
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from notyet import config as config_mod
from notyet import receipt as receipt_mod
from notyet import snapshot, store
from notyet.findings import Context, EngineResult, Finding

MAX_BLOCKS = 6
MAX_SAME_SET_BLOCKS = 2

Engine = Callable[[Context], EngineResult]


def default_engines() -> list[Engine]:
    from notyet.engines import execution, static
    return [execution.run, static.run]


@dataclass
class Decision:
    verdict: str        # passed | blocked | reported | unresolved | not-checked | no-change
    block_reason: Optional[str] = None    # text for the agent when blocking
    summary: str = ""                     # short text for the user
    receipt_path: Optional[str] = None
    findings: list[Finding] = field(default_factory=list)


def outstanding(findings: list[Finding], acks: dict[str, dict]) -> list[Finding]:
    """Findings that still stand: blocks not marked needs-human, resolves not acknowledged."""
    out = []
    for f in findings:
        ack = acks.get(f.id)
        if f.severity == "block" and not (ack and ack.get("category") == "needs-human"):
            out.append(f)
        elif f.severity == "resolve" and not ack:
            out.append(f)
    return out


def run_engines(ctx: Context, engines: Optional[list[Engine]] = None) -> EngineResult:
    result = EngineResult()
    for engine in (engines if engines is not None else default_engines()):
        try:
            result.extend(engine(ctx))
        except Exception as e:  # an engine failure is reported, never fatal
            result.not_checked.append(f"{getattr(engine, '__module__', 'engine')} failed: {e}")
    return result


def check(root: str, session: store.Session, engines: Optional[list[Engine]] = None,
          background_busy: bool = False) -> Decision:
    if background_busy:
        return Decision(verdict="no-change")  # the session isn't done; check when it is
    current = snapshot.snapshot(root)
    if current == session.baseline_tree:
        return Decision(verdict="no-change")
    cfg = config_mod.load(root, session.baseline_tree)
    ctx = Context(root=root, session=session, config=cfg, baseline_tree=session.baseline_tree,
                  current_tree=current, deltas=snapshot.diff_trees(root, session.baseline_tree, current))

    previous = session.runs[-1] if session.runs else None
    if previous is not None and previous.tree == current:
        if previous.verdict != "blocked":
            return Decision(verdict="no-change")      # this exact tree was already checked
        result = _result_from(previous.result)       # re-decide with current acknowledgments
    else:
        result = run_engines(ctx, engines)
        result.not_checked = cfg.problems + result.not_checked
        live = config_mod.load(root)
        if _gate_settings(live) != _gate_settings(cfg):
            result.advice.append(f"{config_mod.CONFIG_FILE} changed during this session; "
                                 f"the gate used the version from session start")

    standing = outstanding(result.findings, session.acks)
    repeats = _same_set_streak(session, [f.id for f in standing])

    if not standing and not result.checks:
        verdict = "not-checked"
    elif not standing:
        verdict = "passed"
    elif not cfg.enforce:
        verdict = "reported"
    elif repeats >= MAX_SAME_SET_BLOCKS or session.consecutive_blocks >= MAX_BLOCKS:
        verdict = "unresolved"
    else:
        verdict = "blocked"

    path = receipt_mod.write(root, session, ctx, result, standing, verdict)
    session.runs.append(store.GateRun(
        tree=current, at=time.time(), verdict=verdict, finding_ids=[f.id for f in standing],
        receipt=path, result=_result_to(result)))
    session.consecutive_blocks = session.consecutive_blocks + 1 if verdict == "blocked" else 0
    store.save_session(root, session)

    decision = Decision(verdict=verdict, receipt_path=path, findings=result.findings,
                        summary=receipt_mod.summary(verdict, standing, result, path))
    if verdict == "blocked":
        decision.block_reason = receipt_mod.agent_message(standing, last_chance=repeats + 1 >= MAX_SAME_SET_BLOCKS)
    return decision


def _gate_settings(cfg: config_mod.Config) -> tuple:
    return (cfg.test_command, cfg.mode, cfg.budget_seconds, cfg.ruff, cfg.pyright, cfg.static_budget_seconds)


def _same_set_streak(session: store.Session, ids: list[str]) -> int:
    """How many of the latest runs in a row were blocked with exactly this finding set."""
    want, n = sorted(ids), 0
    for run in reversed(session.runs):
        if run.verdict == "blocked" and sorted(run.finding_ids) == want:
            n += 1
        else:
            break
    return n


def _result_to(result: EngineResult) -> dict:
    return {"findings": [asdict(f) for f in result.findings], "checks": result.checks,
            "not_checked": result.not_checked, "advice": result.advice}


def _result_from(data: dict) -> EngineResult:
    return EngineResult(findings=[Finding(**f) for f in data.get("findings", [])],
                        checks=data.get("checks", []), not_checked=data.get("not_checked", []),
                        advice=data.get("advice", []))
