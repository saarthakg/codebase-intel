# Demo Questions — psf/requests

Indexed with:
```
python scripts/ingest_repo.py --repo ../requests-demo --repo-id requests
# Indexed 47 files, 399 chunks, 807 symbols, 2591 references, 107 graph edges.
```

All output below is real, regenerated against the current `requests-demo` source (post the
2026-09 upstream pull that added the `_types.py` typing pass). The one exception is the
`/ask` answer in Q7, which is marked as such.

---

## Q1: Where is SSL certificate verification handled?

```
python scripts/demo_query.py --repo-id requests "where is SSL certificate verification handled?" --top-k 5
```

```
Search: "where is SSL certificate verification handled?"
Top 5 results:
[1] HISTORY.md  lines 1799–1875  score=0.031
    -   danger\_mode for automatic Response.raise\_for\_status()
    -   Response.iter\_lines refactor

[2] src/requests/adapters.py  lines 307–348  score=0.031
    def cert_verify(
    self, conn: Any, url: str, verify: _t.VerifyType, cert: _t.CertType

[3] src/requests/adapters.py  lines 428–453  score=0.031
    To override these settings, one may subclass this class, call this
    method and use the above logic to change parameters as desired. For

[4] HISTORY.md  lines 163–196  score=0.030
    - Fixed an issue where setting `verify=False` on the first request from a
    Session will cause subsequent requests to the _same origin_ to also ignore

[5] tests/testserver/server.py  lines 138–176  score=0.029
    class TLSServer(Server):
    def __init__(
```

Snippets are trimmed to their first two lines. `HTTPAdapter.cert_verify` (lines 307–348) is
the method that does the verification. The top hit is a changelog entry ("verify ssl is
default"), which keyword search matches strongly; hybrid search caps how many chunks one file
can contribute to the keyword list, so `HISTORY.md` doesn't take over the results, but it can
still rank first.

**Answer:** SSL certificate verification lives primarily in `src/requests/adapters.py`
(`HTTPAdapter.cert_verify`, and TLS context setup around it) and `src/requests/certs.py` (CA
bundle resolution via `certifi`).

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
[1] src/requests/sessions.py  lines 752–793  score=0.046
    def send(self, request: PreparedRequest, **kwargs: Any) -> Response:
    """Send a given PreparedRequest.

[2] src/requests/api.py  lines 67–99  score=0.029
    # By using the 'with' statement we are sure the session is closed, thus we
    # cases, and look like a memory leak in

[3] tests/test_requests.py  lines 2608–2646  score=0.028
    class RedirectSession(SessionRedirectMixin):
    def __init__(self, order_of_redirects):

[4] src/requests/sessions.py  lines 108–132  score=0.028
    def merge_hooks(
    request_hooks: _t.HooksType,

[5] HISTORY.md  lines 1653–1704  score=0.028
    -   Session cookies not saved when Session.request is called with
    return\_response=False
```

`Session.send()` itself is the top hit. The query contains the identifier `Session.send`, so
hybrid search also looks up the chunk that *defines* it and ranks it first.

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
[1] src/requests/auth.py  lines 273–310  score=0.032
    def handle_401(self, r: Response, **kwargs: Any) -> Response:
    """

[2] HISTORY.md  lines 2049–2102  score=0.031
    -   Smarter Query URL Parameterization
    -   Allow file uploads and POST data together

[3] src/requests/auth.py  lines 252–271  score=0.031
    # XXX should the partial digests be encoded too?
    base = (

[4] HISTORY.md  lines 1935–1996  score=0.028
    ------------------
    -   Automatic decoding of unicode, based on HTTP Headers.

[5] src/requests/sessions.py  lines 309–332  score=0.028
    def rebuild_auth(
    self, prepared_request: PreparedRequest, response: Response
```

**Answer:** Authentication is handled in `src/requests/auth.py` (`HTTPBasicAuth`,
`HTTPDigestAuth`, `HTTPProxyAuth`) and applied/re-applied across redirects in
`src/requests/sessions.py` (`rebuild_auth`, lines 309–332).

---

## Q7: Where is SSL certificate verification handled? (grounded Q&A)

> This answer was generated before hybrid search and structure-aware chunking; it hasn't been
> re-run since, because `/ask` calls a paid/rate-limited LLM API. Retrieval for the same
> question now returns the chunks shown in Q1, and `/ask` now sends full chunks rather than
> the first 800 characters of each, so a fresh run will differ.

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
