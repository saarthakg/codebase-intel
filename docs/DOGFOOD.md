# Dogfooding notyet (MVP week 4)

The goal is 40–50 real Claude Code sessions across 3 or more Python repos, with every finding
hand-labeled. It uses the Claude subscription, not the API, so it costs no extra money. It does
use session quota.

## Setup, per repo (asks before changing anything)

```sh
pip install git+https://github.com/saarthakg/notyet   # or: pip install -e <this checkout>
cd <repo>
notyet init              # proposes .notyet.toml (test command, budget); review it
notyet install claude    # shows the diff to .claude/settings.local.json; asks first
```

- **Start in `report` mode for the first few sessions.** notyet never blocks in report mode; it
  leaves a receipt after each change (`notyet receipt`). Switch to `mode = "enforce"` once the
  receipts look sane.
- **To remove it:** delete the three `notyet hook claude` entries from
  `.claude/settings.local.json`, and `.notyet.toml`. Its state lives only in `.git/notyet/`.

## During sessions

Work as usual. Things worth noting when they happen:
- the agent said it was done and notyet disagreed (the case this exists for);
- notyet blocked and you think it was wrong;
- the agent acknowledged something with a reason you wouldn't accept;
- a check felt slow.

## Labeling

```sh
notyet export --out labels-<repo>.csv
```

- **The export:** one row per finding per check.
- **`label` column:** fill in with `tp` (a real problem, or something you'd want to know),
  `fp` (wrong, or noise), or `unclear`. Use `label_note` for why.

The write-up (`docs/EVAL.md`) will report:
- findings per session;
- precision per rule;
- false blocks per session;
- how often a block led to a fix vs. an acknowledgment vs. a hand-off to you;
- check latency.
