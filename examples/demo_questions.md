# Demo Questions — psf/requests

Indexed with:
```
python scripts/ingest_repo.py --repo ../requests-demo --repo-id requests
# Indexed 47 files, 405 chunks, 807 symbols, 107 graph edges.
```

All output below is real, regenerated against the current `requests-demo` source (post the
2026-09 upstream pull that added the `_types.py` typing pass).

---

## Q1: Where is SSL certificate verification handled?

```
python scripts/demo_query.py --repo-id requests "where is SSL certificate verification handled?"
```

```
Search: "where is SSL certificate verification handled?"
Top 10 results:

[1] src/requests/adapters.py  lines 296–337  score=0.457
    manager = self.proxy_manager[proxy] = proxy_from_url(...)
    (proxy connection pool setup, used by the TLS-verifying send() path)

[2] tests/certs/README.md  lines 1–11  score=0.442
    # Testing Certificates
    This is a collection of certificates useful for testing aspects of
    Requests' behaviour.

[3] src/requests/adapters.py  lines 431–464  score=0.423
    must both set "ssl_context" and based on what else they require,
    alter the other keys to ensure the desired behaviour.

[4] tests/certs/mtls/README.md  lines 1–5  score=0.420
    # Certificate Examples for mTLS

[5] src/requests/certs.py  lines 1–19  score=0.415
    #!/usr/bin/env python
    """
    requests.certs
    ~~~~~~~~~~~~~~
    This module returns the preferred default CA certificate bundle.

[6] src/requests/sessions.py  lines 470–503  score=0.385
    #: If verify is set to `False`, requests will accept any TLS certificate
    #: presented by the server, and will ignore hostname mismatches...

[7] src/requests/sessions.py  lines 602–639  score=0.373
    hostname to the URL of the proxy...

[8] tests/certs/README.md  lines 9–11  score=0.370
    * [mtls](./mtls) provides a valid client certificate with a 2 year validity

[9] tests/testserver/server.py  lines 153–177  score=0.370
    (test server TLS context setup for mTLS tests)

[10] src/requests/adapters.py  lines 640–679  score=0.355
    Sends PreparedRequest object. Returns Response object.
```

**Answer:** SSL certificate verification lives primarily in `src/requests/adapters.py` (TLS
context construction and the `verify`/`cert` parameter handling in `send()`/`cert_verify()`)
and `src/requests/certs.py` (CA bundle resolution via `certifi`).

---

## Q2: What files would be affected if I changed adapters.py?

```
python scripts/demo_query.py --repo-id requests --mode impact --target "src/requests/adapters.py"
```

```
Impact analysis: src/requests/adapters.py

HIGH CONFIDENCE (direct/transitive imports):
  [0.95] tests/test_requests.py  — direct import
  [0.95] src/requests/models.py  — direct import
  [0.95] src/requests/sessions.py  — direct import
  [0.95] tests/test_adapters.py  — direct import
  [0.75] src/requests/cookies.py  — transitive import (2 hops)
  [0.75] src/requests/utils.py  — transitive import (2 hops)
  [0.75] src/requests/__init__.py  — transitive import (2 hops)
  [0.75] src/requests/hooks.py  — transitive import (2 hops)
  [0.75] src/requests/auth.py  — transitive import (2 hops)
  [0.75] src/requests/exceptions.py  — transitive import (2 hops)
  [0.75] src/requests/_types.py  — transitive import (2 hops)
  [0.75] src/requests/api.py  — transitive import (2 hops)

MEDIUM CONFIDENCE:
  [0.50] tests/test_utils.py  — transitive import (3 hops)
  [0.50] tests/test_packages.py  — transitive import (3 hops)
  [0.50] tests/test_testserver.py  — transitive import (3 hops)
  [0.50] tests/test_lowlevel.py  — transitive import (3 hops)
  [0.50] docs/conf.py  — transitive import (3 hops)
  [0.50] tests/test_hooks.py  — transitive import (3 hops)

RELATED (semantic similarity):
  [0.35] pyproject.toml  — semantically related
  [0.35] README.md  — semantically related
```

The test files that exercise `adapters.py` (`tests/test_adapters.py`, `tests/test_requests.py`)
are now direct dependents. Before the import-resolution fix they were missing entirely:
`requests` uses a `src/` layout, so `import requests.adapters` from `tests/` never resolved
and every test file had zero graph edges. `test_adapters.py` only showed up as "semantically
related".

Note `adapters.py` does **not** appear in its own results, even though `adapters.py` and
`models.py` actually import each other (a real circular import in `requests`). That's a
deliberate fix — see the "Why the dependency-graph traversal excludes its own starting file"
design decision in the main README.

---

## Q3: What happens after Session.send() is called?

```
python scripts/demo_query.py --repo-id requests "what happens after Session.send() is called?" --top-k 5
```

```
Search: "what happens after Session.send() is called?"
Top 5 results:

[1] src/requests/sessions.py  lines 916–921  score=0.414
    to create a session. This may be removed at a future date.
    :rtype: Session
    """
    return Session()

[2] src/requests/sessions.py  lines 377–436  score=0.402
    # https://tools.ietf.org/html/rfc7231#section-6.4.4
    if response.status_code == codes.see_other and method != "HEAD":
        method = "GET"
    # Do what the browsers do, despite standards... (redirect handling)

[3] src/requests/models.py  lines 793–835  score=0.402
    #: Textual reason of responded HTTP Status, e.g. "Not Found" or "OK".
    self.reason = None
    #: A CookieJar of Cookies the server sent back.

[4] src/requests/sessions.py  lines 901–921  score=0.396
    return state
    def __setstate__(self, state): ...
    def session() -> Session: ...

[5] src/requests/sessions.py  lines 1–66  score=0.386
    """
    requests.sessions
    ~~~~~~~~~~~~~~~~~
    This module provides a Session object to manage and persist settings.
```

---

## Q4: Where is HTTPAdapter defined, and who uses it?

```
python scripts/demo_query.py --repo-id requests --mode definition --symbol HTTPAdapter
```

```
Definition: HTTPAdapter
  HTTPAdapter (class)  src/requests/adapters.py  lines 158–748
  Used in 4 files (26 places):
    - src/requests/models.py: 90, 750
    - src/requests/sessions.py: 21, 502, 503
    - tests/test_adapters.py: 6
    - tests/test_requests.py: 20, 1651, 1652, 1653, 1654, 1664, 1665, 1666 …
```

References come from an index of identifier usages built from the tree-sitter parse, so
mentions in comments, docstrings and `HISTORY.md` aren't counted. This used to print
`Referenced in (1 files): src/requests/adapters.py`, the defining file itself, because the
lookup queried the definitions table instead of usages.

Qualified names work too: `--symbol HTTPAdapter.send` resolves to the adapter's `send`
method (lines 634–748) rather than `BaseAdapter.send` or `Session.send`.

---

## Q5: What would break if I changed HTTPAdapter.send()?

```
python scripts/demo_query.py --repo-id requests --mode impact --target "HTTPAdapter.send"
```

```
Impact analysis: HTTPAdapter.send

HIGH CONFIDENCE (direct/transitive imports):
  [0.95] tests/test_requests.py  — direct import
  [0.95] src/requests/models.py  — direct import
  [0.95] src/requests/sessions.py  — direct import
  [0.95] tests/test_adapters.py  — direct import
  [0.75] src/requests/cookies.py  — transitive import (2 hops)
  ...
  [0.70] tests/test_lowlevel.py  — references symbol
  [0.70] tests/testserver/server.py  — references symbol

MEDIUM CONFIDENCE:
  [0.50] tests/test_utils.py  — transitive import (3 hops)
  ...
```

A qualified name pins the target to the right definition. A bare `send` is ambiguous (it's
defined on `BaseAdapter`, `HTTPAdapter`, `Session` and more); `/definition` lists the other
matches under `other_definitions`, and symbol lookup prefers source files over tests.

"references symbol" hits come from identifier usages, which are matched by name. Any
`.send(...)` call counts, not only calls on an `HTTPAdapter`. Resolving call targets by type
would need a call graph, which is on the roadmap.

---

## Q6: Where is authentication handled?

```
python scripts/demo_query.py --repo-id requests "where is authentication handled?" --top-k 5
```

```
Search: "where is authentication handled?"
Top 5 results:

[1] src/requests/auth.py  lines 330–355  score=0.358
    r.headers["Authorization"] = _digest_auth
    if (tell := getattr(r.body, "tell", None)) is not None: ...

[2] src/requests/auth.py  lines 1–52  score=0.355
    """
    requests.auth
    ~~~~~~~~~~~~~
    This module contains the authentication handlers for Requests.
    """

[3] src/requests/auth.py  lines 93–143  score=0.339
    @overload
    def __init__(self, username: bytes, password: bytes) -> None: ...

[4] HISTORY.md  lines 2022–2093  score=0.325
    - Internal Refactor
    - Bytes data upload Bugfix ...

[5] src/requests/sessions.py  lines 306–343  score=0.317
    url = self.get_redirect_target(resp)
    yield resp
    def rebuild_auth(self, prepared_request, response): ...
```

**Answer:** Authentication is handled in `src/requests/auth.py` (`HTTPBasicAuth`,
`HTTPDigestAuth`, `HTTPProxyAuth`) and applied/re-applied across redirects in
`src/requests/sessions.py`.

---

## Q7: Where is SSL certificate verification handled? (grounded Q&A)

```
python scripts/demo_query.py --repo-id requests --mode ask \
  "Where is SSL certificate verification handled?"
```

```
Q: Where is SSL certificate verification handled?

A: Based on the provided excerpts, SSL/TLS certificate verification is handled in
the following places:
* HTTPAdapter.cert_verify method in src/requests/adapters.py: Specifically
  dedicated to verifying an SSL certificate (exposed for subclassing
  HTTPAdapter) [1]. Adapter methods also accept and process verify (a boolean
  to control TLS certificate verification or a path string to a CA bundle)
  and client cert parameters [3, 10].
* Session in src/requests/sessions.py: Manages the self.verify configuration
  (defaulting to True) and the verify parameter on session requests to
  determine whether to verify the server's TLS certificate or specify a CA
  bundle path [6, 7].
* src/requests/certs.py: Provides the preferred default CA certificate bundle
  used for verification via certifi.where() [5].
* Test Server in tests/testserver/server.py: Configures server-side
  verification using ssl.SSLContext (verify_mode and load_verify_locations)
  for mutual TLS tests [9].

Citations:
  src/requests/adapters.py    lines 296–337  (cited as [1])
  src/requests/certs.py       lines 1–19     (cited as [5])
  tests/testserver/server.py  lines 153–177  (cited as [9])
```

---

## Q8: What does a PR touching adapters.py and certs.py affect? (batch impact)

```
python scripts/demo_query.py --repo-id requests --mode impact-batch \
  --targets "src/requests/adapters.py,src/requests/certs.py"
```

```
Batch impact analysis: src/requests/adapters.py, src/requests/certs.py

HIGH CONFIDENCE (direct/transitive imports):
  [0.95] tests/test_requests.py    — direct import  (via src/requests/adapters.py, src/requests/certs.py)
  [0.95] src/requests/models.py    — direct import  (via src/requests/adapters.py, src/requests/certs.py)
  [0.95] src/requests/sessions.py  — direct import  (via src/requests/adapters.py, src/requests/certs.py)
  [0.95] tests/test_adapters.py    — direct import  (via src/requests/adapters.py, src/requests/certs.py)
  [0.95] src/requests/utils.py     — direct import  (via src/requests/adapters.py, src/requests/certs.py)
  [0.75] src/requests/cookies.py   — transitive import (2 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  ...
  [0.75] tests/test_utils.py       — transitive import (2 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.75] src/requests/adapters.py  — transitive import (2 hops)  (via src/requests/certs.py)

MEDIUM CONFIDENCE:
  [0.50] tests/test_packages.py    — transitive import (3 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  ...

RELATED (semantic similarity):
  [0.35] pyproject.toml  — semantically related  (via src/requests/adapters.py)
  [0.35] README.md       — semantically related  (via src/requests/adapters.py, src/requests/certs.py)
```

`triggered_by` shows which changed files cause each hit, in one call instead of running
`/impact` twice and diffing by hand. `certs.py` now has its real importer: `utils.py` does
`from . import certs`, which used to resolve to the package `__init__.py` and left `certs.py`
with no dependents at all. The old output of this example credited `sessions.py` as a direct
importer of `certs.py`, which it isn't. `adapters.py` itself shows up via `certs.py`
(`certs.py` → `utils.py` → `adapters.py`): the two changes are linked.

---

## Notes on /ask mode

The `/ask` endpoint requires a `GEMINI_API_KEY` (free at [aistudio.google.com](https://aistudio.google.com))
or `ANTHROPIC_API_KEY` set in `.env`, along with `LLM_BACKEND=gemini` or `LLM_BACKEND=anthropic`.
The LLM reads retrieved code chunks and answers with `[N]` citations — it cannot invent
behavior not present in the excerpts.
