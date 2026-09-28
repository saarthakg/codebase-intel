#!/usr/bin/env python3
"""Build eval/flask_bench.yaml: a second, out-of-sample benchmark on pallets/flask.

Everything in codebase-intel was tuned on psf/requests. Flask is a different
kind of codebase (a web framework: app/request contexts, blueprints, a
sans-IO core under src/flask/sansio, CLI, templating), and nothing was tuned
on it, so this measures whether the gains generalize. The labels were written
from the source alone and committed before scoring.

Usage:
  python eval/build_flask_bench.py --source <pallets/flask checkout>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchlib import build_bench, write_bench

S = "src/flask/"

DEFINITIONS = [
    ("Flask", "app.py"), ("Config", "config.py"), ("AppContext", "ctx.py"), ("FlaskClient", "testing.py"),
    ("MethodView", "views.py"), ("TaggedJSONSerializer", "json/tag.py"), ("render_template", "templating.py"),
    ("send_from_directory", "helpers.py"), ("find_best_app", "cli.py"), ("jsonify", "json/__init__.py"),
    ("has_request_context", "ctx.py"), ("stream_with_context", "helpers.py"), ("create_logger", "logging.py"),
    ("Scaffold", "sansio/scaffold.py"), ("from_pyfile", "config.py"), ("full_dispatch_request", "app.py"),
    # ambiguous bare names: a top-level function vs. a Flask method, and two Blueprint classes
    ("url_for", "helpers.py"), ("make_response", "helpers.py"), ("Blueprint", "blueprints.py"),
    # qualified method names
    ("Flask.make_response", "app.py"), ("Flask.url_for", "app.py"), ("AppContext.push", "ctx.py"),
    ("SecureCookieSessionInterface.open_session", "sessions.py"), ("View.dispatch_request", "views.py"),
]

REF_SYMBOLS = [
    "has_request_context", "stream_with_context", "_split_blueprint_path", "TaggedJSONSerializer",
    "SecureCookieSession", "get_debug_flag", "ScriptInfo", "setupmethod", "find_package",
    "DispatchingJinjaLoader", "NullSession", "BlueprintSetupState",
]

# Directories Flask makes importable beyond the repo root and src/, each checked
# by hand: the example apps are separate projects (own pyproject.toml) whose
# tests import them by their own names, and tests/conftest.py prepends
# tests/test_apps to sys.path for the test_apps fixture.
EXTRA_ROOTS = ("examples/celery/src", "examples/javascript", "examples/tutorial", "tests/test_apps")

IMPACT_TARGETS = [
    "config.py", "ctx.py", "helpers.py", "sessions.py", "templating.py", "json/tag.py",
    "views.py", "testing.py", "sansio/scaffold.py", "cli.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to the pallets/flask checkout")
    parser.add_argument("--out", default=str(Path(__file__).parent / "flask_bench.yaml"))
    args = parser.parse_args()
    bench = build_bench(
        Path(args.source),
        repo="pallets/flask",
        builder="eval/build_flask_bench.py",
        prefix=S,
        definitions=DEFINITIONS,
        ref_symbols=REF_SYMBOLS,
        impact_targets=IMPACT_TARGETS,
        extra_roots=EXTRA_ROOTS,
    )
    write_bench(bench, args.out)


if __name__ == "__main__":
    main()
