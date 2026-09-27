"""Tokenization helpers for keyword search over code."""
import re

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_WORD_RE = re.compile(r"\w+")

# Question filler that would otherwise match nearly every chunk.
STOPWORDS = frozenset("""
a an and are as at be by can do does for from how i if in into is it its of on or should
that the their them then there these this to use used uses using was what when where which
who why will with you your code file files function functions method methods class
""".split())


def camel_parts(identifier: str) -> list[str]:
    """'HTTPAdapter' → ['HTTP', 'Adapter']; 'getNetrcAuth' → ['get', 'Netrc', 'Auth'].

    snake_case needs no help: FTS5's unicode61 tokenizer already splits on '_'.
    """
    return [p for seg in identifier.split("_") for p in _CAMEL_PART_RE.findall(seg)]


def expand_identifiers(text: str) -> str:
    """Append the camelCase parts of every identifier so word-level queries
    ("adapter", "cookie jar") match `HTTPAdapter` and `RequestsCookieJar`."""
    extra = []
    for ident in set(_IDENT_RE.findall(text)):
        parts = camel_parts(ident)
        if len(parts) > 1:
            extra.append(" ".join(parts))
    return text + ("\n" + " ".join(sorted(extra)) if extra else "")


def keyword_query_terms(query: str) -> list[str]:
    """Lowercased search terms: query words plus camelCase parts, minus stopwords."""
    terms = []
    for word in _WORD_RE.findall(expand_identifiers(query).lower()):
        if word not in STOPWORDS and word not in terms and (len(word) > 1 or word.isdigit()):
            terms.append(word)
    return terms


def looks_like_identifier(token: str) -> bool:
    """True for tokens a user typed as code: snake_case, camelCase, Dotted.names."""
    if not _IDENT_RE.fullmatch(token.replace(".", "_")):
        return False
    return "_" in token or "." in token or any(
        a.islower() and b.isupper() for a, b in zip(token, token[1:])
    )


def query_identifiers(query: str) -> list[str]:
    """Code-looking tokens in a query ("HTTPAdapter.send", "get_netrc_auth")."""
    tokens = re.findall(r"[A-Za-z_][\w.]*[\w]|[A-Za-z_]", query)
    return list(dict.fromkeys(t for t in tokens if looks_like_identifier(t)))
