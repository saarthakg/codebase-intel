"""notyet command line.

  notyet init                 write .notyet.toml (detects the test command; asks first)
  notyet install claude       add notyet's hooks to Claude Code's settings (shows the diff; asks first)
  notyet status               the current session and its last check
  notyet receipt              print the latest receipt
  notyet check [--base REF]   run the gate now, outside a hook
  notyet export [--out F]     every finding from every session as CSV, for hand-labeling
  notyet ack ID[,ID] "reason" acknowledge findings (what agents run when blocked)
  notyet hook claude EVENT    hook entry point (used by the installed hooks)
"""
import argparse
import sys
import time
from pathlib import Path

from notyet import config as config_mod
from notyet import gate, snapshot, store
from notyet.findings import Finding


def _root(path: str = ".") -> str:
    try:
        return snapshot.repo_root(path)
    except snapshot.GitError:
        print(f"{Path(path).resolve()} is not inside a git repository.", file=sys.stderr)
        raise SystemExit(2)


def _confirm(prompt: str, yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        print("Not changing anything without confirmation; re-run with --yes.", file=sys.stderr)
        return False
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def cmd_init(args) -> int:
    root = _root(args.path)
    path = Path(root) / config_mod.CONFIG_FILE
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite).")
        return 1
    command = args.test_command or config_mod.detect_test_command(root)
    if not command:
        print("Couldn't find a pytest setup. Pass --test-command, e.g. --test-command 'python -m pytest'.")
        return 1
    text = config_mod.render_template(root, command)
    print(f"Will write {path}:\n\n{text}")
    if not _confirm("Write it?", args.yes):
        return 1
    path.write_text(text)
    print(f"Wrote {path}. Review the test command, then run `notyet install claude`.")
    return 0


def cmd_install(args) -> int:
    from notyet.hooks import claude
    root = _root(args.path)
    cfg = config_mod.load(root)
    if not cfg.test_command:
        print("Note: no test command configured yet; the gate will report 'not checked' until you run `notyet init`.")

    def confirm(diff: str) -> bool:
        print(diff)
        return _confirm("Apply this change to Claude Code's settings?", args.yes)

    path, changed = claude.install(root, args.scope, stop_timeout=cfg.hook_timeout, confirm=confirm)
    print(f"Installed notyet's hooks in {path}." if changed else f"No change made to {path}.")
    if changed:
        print(f"Mode: {cfg.mode}. The gate takes effect in Claude Code sessions started from now on.")
    return 0


def cmd_status(args) -> int:
    root = _root(args.path)
    session = store.latest_session(root)
    if session is None:
        print("No sessions yet.")
        return 0
    cfg = config_mod.load(root, session.baseline_tree)
    print(f"Session {session.session_id} · started {time.strftime('%Y-%m-%d %H:%M', time.localtime(session.started))} "
          f"· baseline: {session.baseline_source} · mode: {cfg.mode}")
    if session.runs:
        last = session.runs[-1]
        print(f"Last check: {last.verdict} · {len(last.finding_ids)} open finding(s) · receipt: {last.receipt}")
    else:
        print("No checks run yet.")
    if session.stops:
        skipped = sum(1 for st in session.stops if st.get("busy"))
        print(f"Stop hook fired {len(session.stops)} time(s); {skipped} skipped because background tasks were running")
    return 0


EXPORT_COLUMNS = ["repo", "session", "check", "at", "verdict", "finding_id", "rule", "severity", "location",
                  "title", "ack_category", "ack_reason", "label", "label_note"]


def cmd_export(args) -> int:
    """Every finding from every check, one row each, for hand-labeling (dogfooding).
    `label` is left empty: fill in true-positive / false-positive / unclear."""
    import csv
    root = _root(args.path)
    out = open(args.out, "w", newline="") if args.out else sys.stdout
    writer = csv.DictWriter(out, fieldnames=EXPORT_COLUMNS)
    writer.writeheader()
    rows = 0
    for session in store.all_sessions(root):
        for n, run in enumerate(session.runs, 1):
            for d in run.result.get("findings", []):
                f = Finding(**d)
                ack = session.acks.get(f.id, {})
                writer.writerow({
                    "repo": Path(root).name, "session": session.session_id, "check": n,
                    "at": time.strftime("%Y-%m-%d %H:%M", time.localtime(run.at)), "verdict": run.verdict,
                    "finding_id": f.id, "rule": f.rule, "severity": f.severity, "location": f.location,
                    "title": f.title, "ack_category": ack.get("category", ""), "ack_reason": ack.get("reason", ""),
                    "label": "", "label_note": ""})
                rows += 1
    if args.out:
        out.close()
        print(f"Wrote {rows} finding(s) to {args.out}.", file=sys.stderr)
    return 0


def cmd_receipt(args) -> int:
    root = _root(args.path)
    latest = store.receipts_dir(root) / "latest.md"
    if not latest.exists():
        print("No receipts yet.")
        return 1
    print(latest.read_text())
    return 0


def cmd_check(args) -> int:
    root = _root(args.path)
    session = store.start_session(root, f"manual-{int(time.time())}", "head")
    try:
        base = snapshot.git(root, "rev-parse", f"{args.base}^{{tree}}").strip()
    except snapshot.GitError as e:
        print(str(e), file=sys.stderr)
        return 2
    session.baseline_tree, session.baseline_head = base, base
    store.save_session(root, session)
    decision = gate.check(root, session)
    if decision.verdict == "no-change":
        print(f"No changes compared with {args.base}.")
        return 0
    print(Path(decision.receipt_path).read_text())
    return 1 if decision.verdict in ("blocked", "unresolved") or (
        decision.verdict == "reported" and args.strict) else 0


def cmd_ack(args) -> int:
    root = _root(args.path)
    session = store.load_session(root, args.session) if args.session else store.latest_session(root)
    if session is None or not session.runs:
        print("No gate run to acknowledge findings from.", file=sys.stderr)
        return 1
    findings = {f.id: f for f in (Finding(**d) for d in session.runs[-1].result.get("findings", []))}
    ids = [i for i in args.id.split(",") if i]
    missing = [i for i in ids if i not in findings]
    if missing:
        print(f"No finding {', '.join(missing)} in the latest check. Open findings: {', '.join(findings) or 'none'}.",
              file=sys.stderr)
        return 1
    reason = " ".join(args.reason).strip()
    if len(reason) < 15:
        print("Give a specific reason (at least a sentence); it's shown to the user on the receipt.", file=sys.stderr)
        return 1
    blocking = [i for i in ids if findings[i].severity == "block"]
    if blocking and not args.needs_human:
        print(f"{', '.join(blocking)}: [block] finding(s): fix them, or hand them to the user with --needs-human.",
              file=sys.stderr)
        return 1
    category = "needs-human" if args.needs_human else args.category
    for i in ids:
        session.acks[i] = {"category": category, "reason": reason, "at": time.time(), "title": findings[i].title}
    store.save_session(root, session)
    print(f"Recorded: {', '.join(ids)} ({category}). {'They' if len(ids) > 1 else 'It'} will appear on the receipt.")
    return 0


def cmd_hook(args) -> int:
    if args.agent != "claude":
        print(f"Unknown agent {args.agent!r}; supported: claude", file=sys.stderr)
        return 0  # never fail a host's hook
    from notyet.hooks import claude
    return claude.main(args.event)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="notyet", description="A completion gate for AI coding agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Write .notyet.toml")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--test-command", help="How to run this repo's tests (a pytest invocation)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--yes", action="store_true", help="Don't ask for confirmation")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("install", help="Install notyet's hooks for an agent")
    p.add_argument("agent", choices=["claude"])
    p.add_argument("--scope", choices=["local", "project", "user"], default="local",
                   help="local: just you, this repo (default); project: shared via .claude/settings.json; user: every repo")
    p.add_argument("--path", default=".")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_install)

    for name, func, help_ in (("status", cmd_status, "The current session and its last check"),
                              ("receipt", cmd_receipt, "Print the latest receipt")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--path", default=".")
        p.set_defaults(func=func)

    p = sub.add_parser("export", help="All findings from all sessions as CSV, for hand-labeling")
    p.add_argument("--out", help="CSV file to write (default: stdout)")
    p.add_argument("--path", default=".")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("check", help="Run the gate now, against a git ref")
    p.add_argument("--base", default="HEAD")
    p.add_argument("--strict", action="store_true", help="Exit 1 on findings even in report mode")
    p.add_argument("--path", default=".")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("ack", help="Acknowledge a finding with a reason")
    p.add_argument("id", help="finding id, or several separated by commas")
    p.add_argument("reason", nargs="+")
    p.add_argument("--needs-human", action="store_true", help="Hand it to the user (required for [block] findings)")
    p.add_argument("--category", default="intended",
                   choices=["intended", "false-positive", "out-of-scope", "follow-up"])
    p.add_argument("--session")
    p.add_argument("--path", default=".")
    p.set_defaults(func=cmd_ack)

    p = sub.add_parser("hook", help=argparse.SUPPRESS)
    p.add_argument("agent")
    p.add_argument("event", choices=["session-start", "prompt", "stop"])
    p.set_defaults(func=cmd_hook)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
