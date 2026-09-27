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

HIGH CONFIDENCE:
  [0.97] tests/test_adapters.py  — test named for this file
  [0.95] tests/test_requests.py  — direct import
  [0.95] src/requests/models.py  — direct import
  [0.95] src/requests/sessions.py  — direct import
  [0.75] src/requests/cookies.py  — transitive import (2 hops)
  [0.75] src/requests/utils.py  — transitive import (2 hops)
  [0.75] src/requests/__init__.py  — transitive import (2 hops)
  [0.75] src/requests/auth.py  — transitive import (2 hops)
  [0.75] src/requests/hooks.py  — transitive import (2 hops)
  [0.75] src/requests/exceptions.py  — transitive import (2 hops)
  [0.75] src/requests/_types.py  — transitive import (2 hops)
  [0.75] src/requests/api.py  — transitive import (2 hops)

MEDIUM CONFIDENCE:
  [0.65] pyproject.toml  — changed together in 2 of 4 commits
  [0.65] src/requests/compat.py  — changed together in 2 of 4 commits
  [0.65] src/requests/help.py  — changed together in 2 of 4 commits
  [0.50] tests/test_utils.py  — transitive import (3 hops)
  [0.50] tests/test_packages.py  — transitive import (3 hops)
  [0.50] tests/test_testserver.py  — transitive import (3 hops)
  [0.50] tests/test_lowlevel.py  — transitive import (3 hops)
  [0.50] docs/conf.py  — transitive import (3 hops)
  [0.50] tests/test_hooks.py  — transitive import (3 hops)

TESTS TO RUN:
  tests/test_adapters.py
  tests/test_requests.py
  tests/test_utils.py
  tests/test_packages.py
  tests/test_testserver.py
  tests/test_lowlevel.py
  tests/test_hooks.py
```

Five signals are at work here:
- `tests/test_adapters.py` is named for the file, so it ranks first.
- Direct and transitive importers come from the import graph. The test files that exercise
  `adapters.py` used to be missing entirely: `requests` uses a `src/` layout, so
  `import requests.adapters` from `tests/` never resolved.
- "changed together in 2 of 4 commits" comes from git history.
- References to the file's symbols (not shown, since the target here is a file).
- Semantic neighbours, which add nothing new for this file.

`requests-demo` is a shallow clone with only 64 commits, so history contributes little here:
4 commits touching `adapters.py`. With full history (see `eval/run_history_eval.py`),
co-change is the signal that moved the real-commit eval most.

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

HIGH CONFIDENCE:
  [0.97] tests/test_adapters.py  — test named for this file
  [0.95] tests/test_requests.py  — direct import
  [0.95] src/requests/models.py  — direct import
  [0.95] src/requests/sessions.py  — direct import
  [0.75] src/requests/cookies.py  — transitive import (2 hops)
  ...
  [0.70] tests/test_lowlevel.py  — references symbol
  [0.70] tests/testserver/server.py  — references symbol

MEDIUM CONFIDENCE:
  [0.65] pyproject.toml  — changed together in 2 of 4 commits
  [0.65] src/requests/compat.py  — changed together in 2 of 4 commits
  ...
```

A qualified name pins the target to the right definition. A bare `send` is ambiguous (it's
defined on `BaseAdapter`, `HTTPAdapter`, `Session` and more); `/definition` lists the other
matches under `other_definitions`, and symbol lookup prefers source files over tests.

"references symbol" hits come from identifier usages, which are matched by name. Any
`.send(...)` call counts, not only calls on an `HTTPAdapter`. For a change you've actually
made, `/impact/diff` (Q9) works from the changed functions instead.

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

HIGH CONFIDENCE:
  [0.97] tests/test_adapters.py  — test named for this file  (via src/requests/adapters.py, src/requests/certs.py)
  [0.95] tests/test_requests.py  — direct import  (via src/requests/adapters.py, src/requests/certs.py)
  [0.95] src/requests/models.py  — direct import  (via src/requests/adapters.py, src/requests/certs.py)
  [0.95] src/requests/sessions.py  — direct import  (via src/requests/adapters.py, src/requests/certs.py)
  [0.95] src/requests/utils.py  — direct import  (via src/requests/adapters.py, src/requests/certs.py)
  [0.75] src/requests/cookies.py  — transitive import (2 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.75] src/requests/__init__.py  — transitive import (2 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.75] src/requests/auth.py  — transitive import (2 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.75] src/requests/hooks.py  — transitive import (2 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.75] src/requests/exceptions.py  — transitive import (2 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.75] src/requests/_types.py  — transitive import (2 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.75] src/requests/api.py  — transitive import (2 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.75] tests/test_utils.py  — transitive import (2 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.75] src/requests/adapters.py  — transitive import (2 hops)  (via src/requests/certs.py)

MEDIUM CONFIDENCE:
  [0.65] pyproject.toml  — changed together in 2 of 4 commits  (via src/requests/adapters.py)
  [0.65] src/requests/compat.py  — changed together in 2 of 4 commits  (via src/requests/adapters.py)
  [0.65] src/requests/help.py  — changed together in 2 of 4 commits  (via src/requests/adapters.py, src/requests/certs.py)
  [0.50] tests/test_packages.py  — transitive import (3 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.50] tests/test_testserver.py  — transitive import (3 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.50] tests/test_lowlevel.py  — transitive import (3 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.50] docs/conf.py  — transitive import (3 hops)  (via src/requests/adapters.py, src/requests/certs.py)
  [0.50] tests/test_hooks.py  — transitive import (3 hops)  (via src/requests/adapters.py)

RELATED:
  [0.35] HISTORY.md  — semantically related  (via src/requests/certs.py)
  ...
```

`triggered_by` (the "via" list) shows which changed files surface each hit, in one call
instead of running `/impact` twice. `certs.py` has its real importer: `utils.py` does
`from . import certs`, which used to resolve to the package `__init__.py`. That left
`certs.py` with no dependents at all, and the old output of this example credited
`sessions.py` as a direct importer instead.

---

## Q9: What does this diff affect, function by function? (diff impact)

```
git diff 6f66281a^ HEAD -- src/requests/_types.py src/requests/models.py > change.diff
python scripts/demo_query.py --repo-id requests --mode impact-diff --diff change.diff
# or: git diff HEAD | python scripts/demo_query.py --repo-id requests --mode impact-diff --diff -
```

```
Diff impact: src/requests/_types.py, src/requests/models.py

CHANGED SYMBOLS:
  src/requests/_types.py::SupportsRead.read  → used in src/requests/adapters.py, src/requests/models.py, src/requests/sessions.py, src/requests/utils.py, tests/test_requests.py, tests/test_utils.py
  src/requests/_types.py::has_read  → used in src/requests/models.py
  src/requests/_types.py::SupportsItems.items  → used in src/requests/adapters.py, src/requests/models.py, src/requests/sessions.py, src/requests/utils.py, tests/test_requests.py
  src/requests/models.py::RequestEncodingMixin._encode_params
  src/requests/models.py::RequestEncodingMixin._encode_files
  src/requests/models.py::PreparedRequest.prepare_body

HIGH CONFIDENCE:
  [0.96] src/requests/sessions.py  — uses changed SupportsRead.read  (via src/requests/_types.py, src/requests/models.py)
  [0.96] src/requests/utils.py  — uses changed SupportsRead.read  (via src/requests/_types.py, src/requests/models.py)
  [0.96] src/requests/adapters.py  — uses changed SupportsRead.read  (via src/requests/_types.py, src/requests/models.py)
  [0.96] tests/test_requests.py  — uses changed SupportsRead.read  (via src/requests/_types.py, src/requests/models.py)
  [0.96] tests/test_utils.py  — uses changed SupportsRead.read  (via src/requests/_types.py, src/requests/models.py)
  [0.95] src/requests/models.py  — direct import  (via src/requests/_types.py)
  [0.95] src/requests/cookies.py  — direct import  (via src/requests/_types.py, src/requests/models.py)
  [0.95] src/requests/hooks.py  — direct import  (via src/requests/_types.py, src/requests/models.py)
  [0.95] src/requests/api.py  — direct import  (via src/requests/_types.py, src/requests/models.py)
  [0.95] src/requests/auth.py  — direct import  (via src/requests/_types.py, src/requests/models.py)
  [0.95] src/requests/__init__.py  — direct import  (via src/requests/_types.py, src/requests/models.py)
  [0.95] src/requests/exceptions.py  — direct import  (via src/requests/_types.py, src/requests/models.py)
  [0.95] src/requests/_types.py  — direct import  (via src/requests/models.py)
  ...
```

The diff's changed lines are mapped to the innermost functions they touch, and files that use
those functions rank above other importers. Usages are matched by name within files that
import the changed module, which is why a generic method name like `read` (from the
`SupportsRead` protocol) pulls in every dependent that calls `.read()`. Requiring the class
name as well was tested and removed: it threw away every gain on real commits, because
methods are usually called on instances obtained elsewhere.

---

## Notes on /ask mode

The `/ask` endpoint requires a `GEMINI_API_KEY` (free at [aistudio.google.com](https://aistudio.google.com))
or `ANTHROPIC_API_KEY` set in `.env`, along with `LLM_BACKEND=gemini` or `LLM_BACKEND=anthropic`.
The LLM reads retrieved code chunks and answers with `[N]` citations — it cannot invent
behavior not present in the excerpts.
