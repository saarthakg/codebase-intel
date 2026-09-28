import json
import os
import subprocess

import pytest

from notyet import config, gate, snapshot, store
from notyet.findings import EngineResult, Finding
from notyet.hooks import claude

_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t"}


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=_ENV).stdout


def _write(repo, rel, text):
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(text)


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, "app.py", "def f():\n    return 1\n")
    _write(repo, ".gitignore", "*.log\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def hook(repo, event, **payload):
    return claude.handle(event, {"cwd": str(repo), "session_id": "s1", **payload})


def fake_engine(findings):
    """An engine that returns the given findings (a callable, re-evaluated each run)."""
    def engine(ctx):
        return EngineResult(findings=list(findings()), checks=["fake check"])
    return engine


# ── Snapshots ─────────────────────────────────────────────────────────────────

def test_snapshot_includes_untracked_excludes_ignored_and_leaves_index_alone(repo):
    _write(repo, "new.py", "x = 1\n")
    _write(repo, "debug.log", "noise\n")
    _write(repo, "app.py", "def f():\n    return 2\n")
    _git(repo, "add", "app.py")                      # the user has something staged
    status_before = _git(repo, "status", "--porcelain")
    tree = snapshot.snapshot(str(repo))
    assert _git(repo, "status", "--porcelain") == status_before   # index and files untouched
    files = _git(repo, "ls-tree", "-r", "--name-only", tree).split()
    assert "new.py" in files and "debug.log" not in files
    assert snapshot.show(str(repo), tree, "app.py") == "def f():\n    return 2\n"


def test_same_size_edits_in_the_same_second_are_seen(repo):
    """Racy git: an edit that keeps the file's size, made within the same
    second as the index was written, must still change the snapshot."""
    for i in range(20):
        _write(repo, "app.py", f"def f():\n    return {i % 10}\n")
        _git(repo, "add", "app.py")                  # index written now
        before = snapshot.snapshot(str(repo))
        _write(repo, "app.py", f"def f():\n    return {(i + 1) % 10}\n")   # same size, same second
        assert snapshot.snapshot(str(repo)) != before


def test_change_is_measured_from_session_start(repo):
    _write(repo, "app.py", "def f():\n    return 'user edit before the session'\n")
    hook(repo, "session-start", source="startup")
    _write(repo, "agent.py", "y = 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "agent commits mid-session")       # HEAD moves; the change must not vanish
    session = store.load_session(str(repo), "s1")
    deltas = snapshot.diff_trees(str(repo), session.baseline_tree, snapshot.snapshot(str(repo)))
    assert [(d.status, d.path) for d in deltas] == [("A", "agent.py")]   # the user's earlier edit isn't in it


# ── Hook flow and verdicts ────────────────────────────────────────────────────

def test_report_mode_summarizes_and_never_blocks(repo, monkeypatch):
    finding = Finding(rule="test-regression", severity="block", title="tests/test_app.py::test_f fails now",
                      location="tests/test_app.py::test_f")
    monkeypatch.setattr(gate, "default_engines", lambda: [fake_engine(lambda: [finding])])
    hook(repo, "session-start", source="startup")
    _write(repo, "app.py", "def f():\n    return 3\n")
    out = hook(repo, "stop", stop_hook_active=False)
    assert "decision" not in out
    assert out["systemMessage"].startswith("notyet: REPORTED")
    assert "test_f fails now" in out["systemMessage"]
    receipt = store.receipts_dir(str(repo)) / "latest.md"
    assert "M app.py" in receipt.read_text()
    assert hook(repo, "stop", stop_hook_active=False) is None   # same tree: nothing re-run, nothing said


def test_conversation_only_turns_are_not_checked(repo, monkeypatch):
    calls = []
    monkeypatch.setattr(gate, "default_engines", lambda: [lambda ctx: calls.append(1) or EngineResult()])
    hook(repo, "session-start", source="startup")
    assert hook(repo, "stop") is None and calls == []


def _enforce(repo):
    _write(repo, ".notyet.toml", '[test]\ncommand = "python -m pytest"\n[gate]\nmode = "enforce"\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "config")


def test_enforce_blocks_then_releases_an_unchanged_set_as_unresolved(repo, monkeypatch):
    _enforce(repo)
    finding = Finding(rule="test-regression", severity="block", title="test_f fails now", location="t::test_f",
                      evidence=["AssertionError: 3 != 1"], action="Fix the code, not the test.")
    monkeypatch.setattr(gate, "default_engines", lambda: [fake_engine(lambda: [finding])])
    hook(repo, "session-start", source="startup")

    _write(repo, "app.py", "def f():\n    return 3\n")
    first = hook(repo, "stop")
    assert first["decision"] == "block"
    assert "test_f fails now" in first["reason"] and f"id={finding.id}" in first["reason"]
    assert "last automatic check" not in first["reason"]

    _write(repo, "app.py", "def f():\n    return 4\n")        # tries again, same failure
    second = hook(repo, "stop", stop_hook_active=True)
    assert second["decision"] == "block" and "last automatic check" in second["reason"]

    _write(repo, "app.py", "def f():\n    return 5\n")        # and again: released, but flagged
    third = hook(repo, "stop", stop_hook_active=True)
    assert "decision" not in third and third["systemMessage"].startswith("notyet: UNRESOLVED")
    receipt = (store.receipts_dir(str(repo)) / "latest.md").read_text()
    assert receipt.index("## Needs your attention") < receipt.index("## Change since")


def test_fixing_the_finding_passes(repo, monkeypatch):
    _enforce(repo)
    broken = {"yes": True}
    finding = Finding(rule="test-regression", severity="block", title="test_f fails now", location="t::test_f")
    monkeypatch.setattr(gate, "default_engines",
                        lambda: [fake_engine(lambda: [finding] if broken["yes"] else [])])
    hook(repo, "session-start", source="startup")
    _write(repo, "app.py", "def f():\n    return 3\n")
    assert hook(repo, "stop")["decision"] == "block"
    broken["yes"] = False
    _write(repo, "app.py", "def f():\n    return 1  # fixed\n")
    out = hook(repo, "stop", stop_hook_active=True)
    assert out["systemMessage"].startswith("notyet: PASSED")


def test_acknowledging(repo, monkeypatch, capsys):
    from notyet import cli
    _enforce(repo)
    blocker = Finding(rule="test-regression", severity="block", title="test_f fails now", location="t::test_f")
    advisory = Finding(rule="test-weakened", severity="resolve", title="assert removed in test_g", location="t::test_g")
    monkeypatch.setattr(gate, "default_engines", lambda: [fake_engine(lambda: [blocker, advisory])])
    hook(repo, "session-start", source="startup")
    _write(repo, "app.py", "def f():\n    return 3\n")
    assert hook(repo, "stop")["decision"] == "block"

    path = ["--path", str(repo)]
    assert cli.main(["ack", advisory.id, "the", "old", "assertion", "tested", "removed", "behavior", *path]) == 0
    assert cli.main(["ack", blocker.id, "intended", "change", "of", "return", "value", *path]) == 1   # block: fix or needs-human
    assert cli.main(["ack", blocker.id, "too", "short", *path]) == 1
    assert cli.main(["ack", f"{blocker.id},nope", "--needs-human", "return", "value", "change", "needs", "a", "product", "call",
                     *path]) == 1                                                     # unknown id: nothing recorded
    assert cli.main(["ack", f"{blocker.id},{advisory.id}", "--needs-human", "return", "value", "change", "needs", "a",
                     "product", "call", *path]) == 0

    out = hook(repo, "stop", stop_hook_active=True)          # same tree: re-decided from saved findings + acks
    assert "decision" not in out and out["systemMessage"].startswith("notyet: NEEDS YOUR REVIEW")
    assert "PASSED" not in out["systemMessage"]
    receipt = (store.receipts_dir(str(repo)) / "latest.md").read_text()
    assert "**needs human**" in receipt and "product call" in receipt
    assert receipt.count("**needs human**") == 2          # the batch ack re-filed the advisory item too


def test_config_edits_during_a_session_do_not_take_effect(repo, monkeypatch):
    _enforce(repo)
    finding = Finding(rule="test-regression", severity="block", title="test_f fails now", location="t::test_f")
    monkeypatch.setattr(gate, "default_engines", lambda: [fake_engine(lambda: [finding])])
    hook(repo, "session-start", source="startup")
    _write(repo, ".notyet.toml", '[gate]\nmode = "report"\n')    # the agent tries to switch the gate off
    out = hook(repo, "stop")
    assert out["decision"] == "block"
    receipt = (store.receipts_dir(str(repo)) / "latest.md").read_text()
    assert ".notyet.toml changed during this session" in receipt


def test_background_work_defers_the_check(repo, monkeypatch):
    calls = []
    monkeypatch.setattr(gate, "default_engines", lambda: [lambda ctx: calls.append(1) or EngineResult()])
    hook(repo, "session-start", source="startup")
    _write(repo, "app.py", "def f():\n    return 3\n")
    assert hook(repo, "stop", background_tasks=[{"id": "t1", "type": "shell", "status": "running"}]) is None
    assert calls == []


def test_stop_without_a_recorded_start_measures_from_head(repo, monkeypatch):
    monkeypatch.setattr(gate, "default_engines", lambda: [fake_engine(lambda: [])])
    _write(repo, "app.py", "def f():\n    return 3\n")
    out = hook(repo, "stop")                                   # hooks installed mid-session
    assert out["systemMessage"].startswith("notyet: PASSED")
    receipt = (store.receipts_dir(str(repo)) / "latest.md").read_text()
    assert "installed mid-session" in receipt and "M app.py" in receipt


def test_hooks_never_break_the_session(repo, tmp_path, monkeypatch):
    assert claude.handle("stop", {"cwd": str(tmp_path), "session_id": "x"}) is None   # not a git repo

    def boom(ctx):
        raise RuntimeError("engine exploded")
    monkeypatch.setattr(gate, "default_engines", lambda: [boom])
    hook(repo, "session-start", source="startup")
    _write(repo, "app.py", "def f():\n    return 3\n")
    out = hook(repo, "stop")                                   # an engine failure is reported, not raised
    assert "engine exploded" in (store.receipts_dir(str(repo)) / "latest.md").read_text()
    assert "decision" not in out

    monkeypatch.setattr(gate, "check", lambda *a, **k: 1 / 0)
    out = hook(repo, "stop")
    assert "internal error" in out["systemMessage"]
    assert "ZeroDivisionError" in (store.state_dir(str(repo)) / "errors.log").read_text()


def test_resume_keeps_the_session_baseline(repo):
    hook(repo, "session-start", source="startup")
    baseline = store.load_session(str(repo), "s1").baseline_tree
    _write(repo, "app.py", "def f():\n    return 3\n")
    hook(repo, "session-start", source="resume")
    hook(repo, "session-start", source="compact")
    assert store.load_session(str(repo), "s1").baseline_tree == baseline
    hook(repo, "session-start", source="clear")               # /clear starts a new task
    assert store.load_session(str(repo), "s1").baseline_tree != baseline


# ── Installing ────────────────────────────────────────────────────────────────

def test_install_merges_asks_and_is_idempotent(repo):
    settings = repo / ".claude" / "settings.local.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "run-lint"}]}]},
                                    "model": "opus"}))
    seen = []
    path, changed = claude.install(str(repo), "local", 240, confirm=lambda d: seen.append(d) or False)
    assert not changed and "+" in seen[0] and json.loads(settings.read_text())["hooks"]["Stop"][0]["hooks"][0]["command"] == "run-lint"

    path, changed = claude.install(str(repo), "local", 240, confirm=lambda d: True)
    data = json.loads(settings.read_text())
    assert changed and data["model"] == "opus"
    stop_cmds = [h["command"] for g in data["hooks"]["Stop"] for h in g["hooks"]]
    assert stop_cmds[0] == "run-lint" and stop_cmds[1].endswith("-m notyet hook claude stop")
    assert {"SessionStart", "UserPromptSubmit", "Stop"} <= set(data["hooks"])

    _, changed_again = claude.install(str(repo), "local", 240, confirm=lambda d: True)
    assert not changed_again                                   # nothing duplicated


def test_ack_command_in_the_agent_message_survives_paths_with_spaces(monkeypatch):
    import re
    import shlex
    import sys

    from notyet import receipt
    monkeypatch.setattr(sys, "executable", "/Users/x/My Project/.venv/bin/python")
    f = Finding(rule="test-regression", severity="block", title="tests/t.py::test_a fails now", location="tests/t.py::test_a")
    text = receipt.agent_message([f], last_chance=False)
    cmd = re.search(r"`([^`]*ack <id> --needs-human[^`]*)`", text).group(1)
    assert shlex.split(cmd)[:4] == ["/Users/x/My Project/.venv/bin/python", "-m", "notyet", "ack"]
    assert text.count("tests/t.py::test_a") == 1          # the location isn't repeated after the title


def test_engines_past_the_check_deadline_are_reported_not_run(repo):
    import time as _time

    from notyet.findings import Context
    ran = []
    ctx = Context(root=str(repo), session=None, config=config.Config(), baseline_tree="", current_tree="", deltas=[],
                  deadline=_time.monotonic() - 1)
    result = gate.run_engines(ctx, [lambda c: ran.append(1)])
    assert ran == [] and "skipped; the check used its" in result.not_checked[0]


def test_export_lists_every_finding_for_labeling(repo, monkeypatch, tmp_path):
    import csv

    from notyet import cli
    _enforce(repo)
    finding = Finding(rule="test-regression", severity="block", title="test_f fails now", location="t::test_f")
    monkeypatch.setattr(gate, "default_engines", lambda: [fake_engine(lambda: [finding])])
    hook(repo, "session-start", source="startup")
    _write(repo, "app.py", "def f():\n    return 3\n")
    hook(repo, "stop")
    out = tmp_path / "labels.csv"
    assert cli.main(["export", "--out", str(out), "--path", str(repo)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert [(r["rule"], r["verdict"], r["label"]) for r in rows] == [("test-regression", "blocked", "")]
