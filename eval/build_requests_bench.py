#!/usr/bin/env python3
"""Build eval/requests_bench.yaml from hand-written labels + the requests source.

Labels below are hand-written (which symbols and files to probe);
everything mechanical is derived from the source by eval/benchlib.py.

Usage:
  python eval/build_requests_bench.py --source ../requests-demo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchlib import build_bench, write_bench

S = "src/requests/"

DEFINITIONS = [
    ("HTTPAdapter", "adapters.py"), ("Session", "sessions.py"), ("PreparedRequest", "models.py"),
    ("Response", "models.py"), ("CaseInsensitiveDict", "structures.py"), ("RequestsCookieJar", "cookies.py"),
    ("HTTPDigestAuth", "auth.py"), ("dispatch_hook", "hooks.py"), ("get_netrc_auth", "utils.py"),
    ("super_len", "utils.py"), ("to_native_string", "_internal_utils.py"), ("check_compatibility", "__init__.py"),
    ("cert_verify", "adapters.py"), ("resolve_redirects", "sessions.py"), ("prepare_url", "models.py"),
    ("raise_for_status", "models.py"), ("build_digest_header", "auth.py"), ("merge_environment_settings", "sessions.py"),
    # Qualified method names — the bare name is ambiguous (e.g. `send` exists on several classes).
    ("HTTPAdapter.send", "adapters.py"), ("Session.send", "sessions.py"), ("Session.request", "sessions.py"),
    ("PreparedRequest.prepare_body", "models.py"), ("Response.json", "models.py"), ("HTTPBasicAuth.__call__", "auth.py"),
]

# Distinctive names whose usages we can find unambiguously.
REF_SYMBOLS = [
    "HTTPAdapter", "PreparedRequest", "CaseInsensitiveDict", "dispatch_hook", "to_native_string",
    "extract_cookies_to_jar", "super_len", "get_netrc_auth", "cookiejar_from_dict", "TooManyRedirects",
    "merge_cookies", "rewind_body",
]

IMPACT_TARGETS = [
    "adapters.py", "certs.py", "hooks.py", "structures.py", "help.py", "cookies.py",
    "_internal_utils.py", "status_codes.py", "api.py", "exceptions.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to the requests checkout")
    parser.add_argument("--out", default=str(Path(__file__).parent / "requests_bench.yaml"))
    args = parser.parse_args()
    bench = build_bench(
        Path(args.source),
        repo="psf/requests",
        builder="eval/build_requests_bench.py",
        prefix=S,
        definitions=DEFINITIONS,
        ref_symbols=REF_SYMBOLS,
        impact_targets=IMPACT_TARGETS,
    )
    write_bench(bench, args.out)


if __name__ == "__main__":
    main()
