"""What the gate says: the receipt for the human, a short summary shown to
the user, and the message the agent gets when it's blocked.

The agent-facing message stays small (Claude Code caps hook text at 10,000
characters, and long messages get skimmed): the top findings, what to do,
and how to acknowledge.
"""
import sys
import time

from notyet import store
from notyet.findings import Context, EngineResult, Finding

MAX_AGENT_FINDINGS = 8
MAX_AGENT_CHARS = 3500

VERDICT_TEXT = {
    "passed": "PASSED",
    "blocked": "NOT DONE YET",
    "unresolved": "UNRESOLVED",
    "reported": "REPORTED (report mode: not enforced)",
    "not-checked": "NOT CHECKED",
}


def invocation() -> str:
    """How the agent should call notyet (the same interpreter the hook runs in)."""
    return f"{sys.executable} -m notyet"


def write(root: str, session: store.Session, ctx: Context, result: EngineResult,
          standing: list[Finding], verdict: str) -> str:
    n = len(session.runs) + 1
    path = store.receipts_dir(root) / f"{_safe(session.session_id)}-{n:03d}.md"
    path.write_text(render(session, ctx, result, standing, verdict))
    (store.receipts_dir(root) / "latest.md").write_text(path.read_text())
    return str(path)


def render(session: store.Session, ctx: Context, result: EngineResult,
           standing: list[Finding], verdict: str) -> str:
    standing_ids = {f.id for f in standing}
    lines = [
        "# notyet receipt",
        "",
        f"**Verdict: {VERDICT_TEXT.get(verdict, verdict)}**",
        "",
        f"Session `{session.session_id}` · checked {time.strftime('%Y-%m-%d %H:%M:%S')} · "
        f"compared with: {_baseline_label(session)}",
        "",
    ]
    needs_human = [f for f in result.findings if session.acks.get(f.id, {}).get("category") == "needs-human"]
    if verdict == "unresolved" or needs_human:
        lines += ["## Needs your attention", ""]
        if verdict == "unresolved":
            lines += [f"- **unresolved** `{f.id}` {f.title}" + (f" ({f.location})" if f.location else "")
                      for f in standing]
        lines += [f"- **needs human** `{f.id}` {f.title}: {session.acks[f.id].get('reason', '')}"
                  for f in needs_human]
        lines.append("")

    lines += [f"## Change since {'session start' if session.baseline_source != 'head' else 'HEAD'}: "
              f"{len(ctx.deltas)} file(s)", ""]
    lines += [f"- {d.status} {d.path}" + (f" (from {d.old_path})" if d.old_path else "") for d in ctx.deltas[:50]]
    if len(ctx.deltas) > 50:
        lines.append(f"- … and {len(ctx.deltas) - 50} more")
    lines.append("")

    if result.findings:
        lines += ["## Findings", ""]
        for f in result.findings:
            state = "open" if f.id in standing_ids else _ack_state(session, f)
            lines.append(f"- [{f.severity}] `{f.id}` {f.title}" + (f" ({f.location})" if f.location else "")
                         + f": {state}")
            lines += [f"  - {e}" for e in f.evidence[:6]]
        lines.append("")

    acked = [(fid, a) for fid, a in session.acks.items() if a.get("category") != "needs-human"]
    if acked:
        lines += ["## Acknowledged by the agent", ""]
        lines += [f"- `{fid}` ({a.get('category', 'acknowledged')}): {a.get('reason', '')}" for fid, a in acked]
        lines.append("")

    lines += ["## Checks run", ""] + ([f"- {c}" for c in result.checks] or ["- none"]) + [""]
    lines += ["## Not checked", ""] + ([f"- {c}" for c in result.not_checked] or ["- nothing left out"]) + [""]
    if session.baseline_source != "session-start":
        lines += [f"Note: {_baseline_note(session)}", ""]
    lines.append("Edits you made yourself during this session count as part of the change.")
    if result.advice:
        lines += ["", "## Advice (not enforced)", ""] + [f"- {a}" for a in result.advice]
    return "\n".join(lines) + "\n"


def summary(verdict: str, standing: list[Finding], result: EngineResult, path: str) -> str:
    head = f"notyet: {VERDICT_TEXT.get(verdict, verdict)}"
    if verdict == "passed":
        detail = "; ".join(result.checks[:2])
    elif verdict in ("blocked", "reported", "unresolved"):
        detail = f"{len(standing)} open finding(s): " + "; ".join(f.title for f in standing[:2])
    else:
        detail = "; ".join(result.not_checked[:2]) or "no checks ran"
    return f"{head}. {detail} (receipt: {path})"


def agent_message(standing: list[Finding], last_chance: bool) -> str:
    cmd = invocation()
    lines = [f"notyet: not done yet. {len(standing)} item(s) to resolve before stopping.", ""]
    for i, f in enumerate(standing[:MAX_AGENT_FINDINGS], 1):
        lines.append(f"{i}. [{f.severity}] {f.title}" + (f" ({f.location})" if f.location else "") + f"  id={f.id}")
        lines += [f"   {e}" for e in f.evidence[:3]]
        if f.action:
            lines.append(f"   -> {f.action}")
    if len(standing) > MAX_AGENT_FINDINGS:
        lines.append(f"... and {len(standing) - MAX_AGENT_FINDINGS} more (see the receipt).")
    lines += [
        "",
        "Fix these, then finish. If an item is intended and you're sure, acknowledge it with a specific "
        f"reason, which the user will see: `{cmd} ack <id> \"<reason>\"`. Items marked [block] can only be "
        f"fixed or handed to the user: `{cmd} ack <id> --needs-human \"<reason>\"`.",
    ]
    if last_chance:
        lines.append("This is the last automatic check for these items: anything still open will be "
                     "reported to the user as unresolved.")
    text = "\n".join(lines)
    return text if len(text) <= MAX_AGENT_CHARS else text[:MAX_AGENT_CHARS] + "\n... (truncated; see the receipt)"


def _ack_state(session: store.Session, f: Finding) -> str:
    ack = session.acks.get(f.id)
    if not ack:
        return "resolved"
    return f"{ack.get('category', 'acknowledged')}: {ack.get('reason', '')}"


def _baseline_label(session: store.Session) -> str:
    return {"session-start": "the tree at session start", "first-prompt": "the tree at the first prompt",
            "head": "HEAD (the session's start wasn't recorded)"}.get(session.baseline_source, session.baseline_source)


def _baseline_note(session: store.Session) -> str:
    if session.baseline_source == "head":
        return ("notyet was installed mid-session, so the change is measured from HEAD and may include "
                "edits made before this session.")
    return "the change is measured from the first prompt notyet saw."


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:60]
