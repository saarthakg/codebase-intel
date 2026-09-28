"""Claude Code integration: hook entry points and installing them.

Hook contract (code.claude.com/docs/en/hooks, checked Sept 2026):
- every hook gets JSON on stdin with session_id, cwd, hook_event_name, ...;
- SessionStart has `source` (startup | resume | clear | compact | fork);
- UserPromptSubmit has `prompt`;
- Stop has `stop_hook_active`, `last_assistant_message` and `background_tasks`,
  and may answer {"decision": "block", "reason": ...}; 8 consecutive blocks
  are the host's cap;
- `systemMessage` shows the user a message; hook text is capped at 10,000 chars.

A notyet hook must never break the agent's session: any error is logged to
.git/notyet/errors.log and the hook exits 0.
"""
import difflib
import json
import shlex
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from notyet import gate, snapshot, store

EVENTS = {"session-start": "SessionStart", "prompt": "UserPromptSubmit", "stop": "Stop"}
MARKER = " -m notyet hook claude "


def handle(event: str, payload: dict) -> Optional[dict]:
    """Run one hook event. Returns the JSON to print, or None."""
    cwd = payload.get("cwd") or "."
    try:
        root = snapshot.repo_root(cwd)
    except snapshot.GitError:
        return None  # not a git repo: nothing to gate
    try:
        return _dispatch(root, event, payload)
    except Exception as e:
        store.log_error(root, f"{event}: {e}\n{traceback.format_exc()}")
        return {"systemMessage": f"notyet: internal error in the {event} hook ({e}); the stop was not "
                                 f"checked. Details: .git/notyet/errors.log"}


def _dispatch(root: str, event: str, payload: dict) -> Optional[dict]:
    session_id = payload.get("session_id") or "unknown"
    session = store.load_session(root, session_id)

    if event == "session-start":
        source = payload.get("source", "startup")
        if session is None or source in ("startup", "clear"):
            store.start_session(root, session_id, "session-start")
        return None

    if event == "prompt":
        if session is None:
            session = store.start_session(root, session_id, "first-prompt")
        prompt = (payload.get("prompt") or "").strip()
        if prompt:
            session.prompts.append(prompt[:2000])
        # what earlier turns (and the user between turns) left: undone work is measured against it
        session.turns.append({"at": time.time(), "tree": snapshot.snapshot(root)})
        store.save_session(root, session)
        return None

    if event == "stop":
        if session is None:
            # Installed mid-session: the start wasn't seen, so measure from HEAD.
            session = store.start_session(root, session_id, "head")
            head = snapshot.head_tree(root)
            session.baseline_tree = head or snapshot.EMPTY_TREE
            store.save_session(root, session)
        busy = any(t.get("status") in (None, "running", "pending") for t in payload.get("background_tasks") or [])
        decision = gate.check(root, session, background_busy=busy)
        if decision.verdict == "blocked":
            return {"decision": "block", "reason": decision.block_reason}
        if decision.summary:
            return {"systemMessage": decision.summary}
        return None

    raise ValueError(f"unknown hook event {event!r}")


def main(event: str) -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        payload = {}
    out = handle(event, payload)
    if out is not None:
        print(json.dumps(out))
    return 0


# ── Installing ────────────────────────────────────────────────────────────────

def settings_path(root: str, scope: str) -> Path:
    if scope == "project":
        return Path(root) / ".claude" / "settings.json"        # shared with the team
    if scope == "local":
        return Path(root) / ".claude" / "settings.local.json"  # just you, not committed
    if scope == "user":
        return Path.home() / ".claude" / "settings.json"       # every project
    raise ValueError(f"scope must be project, local or user, not {scope!r}")


def hook_command(event: str) -> str:
    return f"{shlex.quote(sys.executable)}{MARKER}{event}"


def merged_settings(existing: dict, stop_timeout: int) -> dict:
    """`existing` with notyet's hooks replacing any earlier notyet hooks."""
    settings = json.loads(json.dumps(existing))  # deep copy
    hooks = settings.setdefault("hooks", {})
    for event, host_event in EVENTS.items():
        groups = [g for g in hooks.get(host_event, [])
                  if not any(MARKER in h.get("command", "") for h in g.get("hooks", []))]
        groups.append({"hooks": [{
            "type": "command",
            "command": hook_command(event),
            "timeout": stop_timeout if event == "stop" else 60,
        }]})
        hooks[host_event] = groups
    return settings


def install(root: str, scope: str, stop_timeout: int, confirm) -> tuple[Path, bool]:
    """Show the settings change and write it if `confirm(diff_text)` says yes."""
    path = settings_path(root, scope)
    before = json.loads(path.read_text()) if path.exists() else {}
    after = merged_settings(before, stop_timeout)
    old_text = json.dumps(before, indent=2) + "\n" if before else ""
    new_text = json.dumps(after, indent=2) + "\n"
    if old_text == new_text:
        return path, False
    diff = "".join(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                                        f"{path} (before)", f"{path} (after)"))
    if not confirm(diff):
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text)
    return path, True
