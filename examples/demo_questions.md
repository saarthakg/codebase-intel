# Demo Questions — psf/requests

Indexed with:
```
python scripts/ingest_repo.py --repo ../requests-demo --repo-id requests
# Indexed 46 files, 378 chunks, 757 symbols, 62 graph edges.
```

---

## Q1: Where is SSL certificate verification handled?

```
python scripts/demo_query.py --repo-id requests "where is SSL certificate verification handled?"
```

```
Search: "where is SSL certificate verification handled?"
Top 10 results:

[1] tests/certs/README.md  lines 1–11  score=0.442
    # Testing Certificates
    This is a collection of certificates useful for testing aspects of Requests' behaviour.

[2] tests/certs/mtls/README.md  lines 1–5  score=0.420
    # Certificate Examples for mTLS

[3] src/requests/certs.py  lines 1–19  score=0.415
    #!/usr/bin/env python
    """
    requests.certs
    ~~~~~~~~~~~~~~
    This module returns the preferred default CA certificate bundle. There is
    only one — the one from the certifi package.

[4] src/requests/adapters.py  lines 395–425  score=0.410
    (TLS context construction, verify/cert parameter handling)

[5] src/requests/adapters.py  lines 289–334  score=0.381
    the server's TLS certificate, or a string, in which case it must be a path
        to a CA bundle to use
    :param cert: The SSL certificate to verify.
```

**Answer:** SSL certificate verification lives primarily in `src/requests/certs.py` (CA bundle
resolution via certifi) and `src/requests/adapters.py` (the `send()` method's `verify`/`cert`
parameter handling and TLS context construction).

---

## Q2: What files would be affected if I changed adapters.py?

```
python scripts/demo_query.py --repo-id requests --mode impact --target "src/requests/adapters.py"
```

```
Impact analysis: src/requests/adapters.py

HIGH CONFIDENCE (direct/transitive imports):
  [0.95] src/requests/sessions.py  — direct import
  [0.75] src/requests/__init__.py  — transitive import (2 hops)

MEDIUM CONFIDENCE:
  [0.50] src/requests/utils.py  — transitive import (3 hops)
  [0.50] src/requests/api.py  — transitive import (3 hops)
  [0.50] src/requests/help.py  — transitive import (3 hops)

RELATED (semantic similarity):
  [0.35] README.md  — semantically related
  [0.35] tests/test_adapters.py  — semantically related
```

---

## Q3: What happens after Session.send() is called?

```
python scripts/demo_query.py --repo-id requests "what happens after Session.send() is called?"
```

```
Search: "what happens after Session.send() is called?"
Top 5 results:

[1] src/requests/api.py  lines 56–106  score=0.418
    with sessions.Session() as session:
        return session.request(...)

[2] src/requests/sessions.py  lines 340–398  score=0.416
    (redirect handling, response reconstruction)

[3] src/requests/sessions.py  lines 829–834  score=0.414
    return Session()

[4] src/requests/api.py  lines 1–31  score=0.411
    (top-level API wiring)

[5] src/requests/models.py  lines 327–379  score=0.411
    >>> s = requests.Session()
    >>> s.send(r)
    <Response [200]>
```

---

## Q4: Where is PreparedRequest defined?

```
python scripts/demo_query.py --repo-id requests --mode definition --symbol PreparedRequest
```

```
Definition: PreparedRequest
  Defined in: src/requests/models.py  line 315  (class)
  Referenced in (1 files):
    - src/requests/models.py
```

---

## Q5: What would break if I changed the send() method?

```
python scripts/demo_query.py --repo-id requests --mode impact --target "send"
```

```
Impact analysis: send

HIGH CONFIDENCE (direct/transitive imports):
  [0.95] src/requests/sessions.py  — direct import
  [0.75] src/requests/__init__.py  — transitive import (2 hops)

MEDIUM CONFIDENCE:
  [0.50] src/requests/utils.py  — transitive import (3 hops)
  [0.50] src/requests/api.py  — transitive import (3 hops)
  [0.50] src/requests/help.py  — transitive import (3 hops)

RELATED (semantic similarity):
  [0.35] tests/test_lowlevel.py  — semantically related
  [0.35] tests/test_requests.py  — semantically related
```

---

## Q6: Where is authentication handled?

```
python scripts/demo_query.py --repo-id requests "where is authentication handled?"
```

```
Search: "where is authentication handled?"
Top 5 results:

[1] src/requests/auth.py  lines 96–143  score=0.342
    class HTTPProxyAuth(HTTPBasicAuth):
        """Attaches HTTP Proxy Authentication to a given Request object."""

[2] src/requests/auth.py  lines 1–49  score=0.321
    """
    requests.auth
    ~~~~~~~~~~~~~
    This module contains the authentication handlers for Requests.
    """

[3] src/requests/sessions.py  lines 461–509  score=0.306
    (session.request() — where auth is merged and applied)

[4] src/requests/auth.py  lines 46–102  score=0.303
    (HTTPDigestAuth implementation)

[5] src/requests/sessions.py  lines 272–309  score=0.289
    (redirect handling with auth re-application)
```

**Answer:** Authentication is handled in `src/requests/auth.py` (HTTPBasicAuth, HTTPDigestAuth,
HTTPProxyAuth classes) and applied in `src/requests/sessions.py` where session.request() merges
and calls the auth callable.

---

## Notes on /ask mode

The `/ask` endpoint (and `--mode ask`) requires `ANTHROPIC_API_KEY` set in `.env`. It calls
`claude-sonnet-4-6` with retrieved chunks as context and returns a grounded answer with
`[N]` citations. Example:

```
python scripts/demo_query.py --repo-id requests --mode ask \
  "How does requests handle redirects?"
```
