#!/usr/bin/env python3
"""Build eval/flask_bench.yaml: a second, out-of-sample benchmark on pallets/flask.

Everything in codebase-intel was tuned on psf/requests. Flask is a different
kind of codebase (a web framework: app/request contexts, blueprints, a
sans-IO core under src/flask/sansio, CLI, templating), and nothing was tuned
on it, so this measures whether the gains generalize. The labels were written
from the source alone, before any search was run on Flask, and committed
before scoring.

Usage:
  python eval/build_flask_bench.py --source <pallets/flask checkout>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchlib import build_bench, write_bench

S = "src/flask/"
SA = "src/flask/sansio/"

# query → [(file, qualified symbol that answers it), ...]  (any one is a hit)
SEARCH = [
    ("WSGI application entry point that handles each request",
     [(S + "app.py", "Flask.wsgi_app"), (S + "app.py", "Flask.__call__")]),
    ("dispatch the request to the matching view function",
     [(S + "app.py", "Flask.dispatch_request"), (S + "app.py", "Flask.full_dispatch_request")]),
    ("convert a view's return value into a response object", [(S + "app.py", "Flask.make_response")]),
    ("run the before_request functions", [(S + "app.py", "Flask.preprocess_request")]),
    ("apply after_request functions to the response", [(S + "app.py", "Flask.process_response")]),
    ("handle an unhandled exception raised during a request",
     [(S + "app.py", "Flask.handle_exception"), (S + "app.py", "Flask.handle_user_exception")]),
    ("handle HTTP errors like 404 with registered error handlers", [(S + "app.py", "Flask.handle_http_exception")]),
    ("build a URL for an endpoint", [(S + "app.py", "Flask.url_for"), (S + "helpers.py", "url_for")]),
    ("register a blueprint on the application",
     [(SA + "app.py", "App.register_blueprint"), (SA + "blueprints.py", "Blueprint.register")]),
    ("add a URL rule that maps a path to a view function",
     [(SA + "app.py", "App.add_url_rule"), (SA + "scaffold.py", "Scaffold.add_url_rule")]),
    ("route decorator for registering views", [(SA + "scaffold.py", "Scaffold.route")]),
    ("register an error handler for an exception type or status code",
     [(SA + "scaffold.py", "Scaffold.errorhandler"), (SA + "scaffold.py", "Scaffold.register_error_handler")]),
    ("load configuration from a Python file", [(S + "config.py", "Config.from_pyfile")]),
    ("load configuration from environment variables with a prefix", [(S + "config.py", "Config.from_prefixed_env")]),
    ("load configuration from an object or import string", [(S + "config.py", "Config.from_object")]),
    ("store the session in a signed cookie",
     [(S + "sessions.py", "SecureCookieSessionInterface"),
      (S + "sessions.py", "SecureCookieSessionInterface.save_session")]),
    ("decide whether the session cookie should be set on the response",
     [(S + "sessions.py", "SessionInterface.should_set_cookie")]),
    ("SameSite and Secure attributes of the session cookie",
     [(S + "sessions.py", "SessionInterface.get_cookie_samesite"),
      (S + "sessions.py", "SessionInterface.get_cookie_secure")]),
    ("push and pop the application context",
     [(S + "ctx.py", "AppContext.push"), (S + "ctx.py", "AppContext.pop")]),
    ("check whether a request context is active", [(S + "ctx.py", "has_request_context")]),
    ("run a function after the current request finishes", [(S + "ctx.py", "after_this_request")]),
    ("keep the request context available inside a streaming generator",
     [(S + "helpers.py", "stream_with_context")]),
    ("render a Jinja template with context variables", [(S + "templating.py", "render_template")]),
    ("look up templates across the app and blueprints",
     [(S + "templating.py", "DispatchingJinjaLoader.get_source"),
      (S + "templating.py", "DispatchingJinjaLoader._iter_loaders")]),
    ("flash a message to show on the next request",
     [(S + "helpers.py", "flash"), (S + "helpers.py", "get_flashed_messages")]),
    ("send a file from a directory without path traversal", [(S + "helpers.py", "send_from_directory")]),
    ("abort the request with an HTTP error code", [(S + "helpers.py", "abort")]),
    ("return a JSON response", [(S + "json/__init__.py", "jsonify"),
                                (S + "json/provider.py", "DefaultJSONProvider.response")]),
    ("serialize dates, UUIDs and dataclasses to JSON", [(S + "json/provider.py", "_default")]),
    ("tagged JSON serializer used for session data", [(S + "json/tag.py", "TaggedJSONSerializer")]),
    ("find the application object in a module for the CLI",
     [(S + "cli.py", "find_best_app"), (S + "cli.py", "locate_app")]),
    ("load environment variables from .env and .flaskenv files", [(S + "cli.py", "load_dotenv")]),
    ("the command that runs the development server", [(S + "cli.py", "run_command")]),
    ("command that lists all registered routes", [(S + "cli.py", "routes_command")]),
    ("test client for sending requests in tests",
     [(S + "testing.py", "FlaskClient"), (S + "app.py", "Flask.test_client")]),
    ("modify the session inside a test", [(S + "testing.py", "FlaskClient.session_transaction")]),
    ("class-based view that dispatches on the HTTP method", [(S + "views.py", "MethodView")]),
    ("turn a view class into a view function", [(S + "views.py", "View.as_view")]),
    ("set up the application logger", [(S + "logging.py", "create_logger")]),
    ("maximum allowed size of the request body", [(S + "wrappers.py", "Request.max_content_length")]),
    ("serve a static file with a cache timeout",
     [(S + "app.py", "Flask.send_static_file"), (S + "app.py", "Flask.get_send_file_max_age")]),
    ("find the root path of a package or module", [(S + "helpers.py", "get_root_path")]),
    ("run async view functions in a sync app",
     [(S + "app.py", "Flask.ensure_sync"), (S + "app.py", "Flask.async_to_sync")]),
]

# Queries phrased the way developers type them.
SEARCH_IDENTIFIER = [
    ("full_dispatch_request", [(S + "app.py", "Flask.full_dispatch_request")]),
    ("SecureCookieSessionInterface.save_session", [(S + "sessions.py", "SecureCookieSessionInterface.save_session")]),
    ("from_prefixed_env", [(S + "config.py", "Config.from_prefixed_env")]),
    ("register_blueprint options", [(SA + "app.py", "App.register_blueprint")]),
    ("stream_with_context", [(S + "helpers.py", "stream_with_context")]),
    ("find_best_app", [(S + "cli.py", "find_best_app")]),
    ("AppContext.push", [(S + "ctx.py", "AppContext.push")]),
    ("session_transaction", [(S + "testing.py", "FlaskClient.session_transaction")]),
    ("get_flashed_messages with_categories", [(S + "helpers.py", "get_flashed_messages")]),
    ("MethodView dispatch_request", [(S + "views.py", "MethodView.dispatch_request")]),
    ("TaggedJSONSerializer register", [(S + "json/tag.py", "TaggedJSONSerializer.register")]),
    ("send_from_directory", [(S + "helpers.py", "send_from_directory")]),
]

# symbol (bare or Class.method) → defining file (relative to src/flask/).
# Several names are defined twice in Flask (a public layer and the sans-IO
# core); the expected answer is the public one a user would mean.
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
        search_sets={"search": SEARCH, "search_identifier": SEARCH_IDENTIFIER},
        definitions=DEFINITIONS,
        ref_symbols=REF_SYMBOLS,
        impact_targets=IMPACT_TARGETS,
    )
    write_bench(bench, args.out)


if __name__ == "__main__":
    main()
