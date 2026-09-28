"""Turn the evidence for a candidate file (features.py) into the probability
that it belongs in the change, and into reasons a person can check.

The model is a logistic regression trained by eval/train_model.py on real
changes replayed from several open-source repos (eval/build_replay.py). Its
probabilities are calibrated: on repos it wasn't trained on, suggestions
scored 0.2-0.3 were right 23% of the time, 0.3-0.5 32%, above 0.5 49%.
"""
import math
from typing import Any, Mapping

from codebase_intel.core.model_weights import BIAS, WEIGHTS

FEATURES = [
    "cc_p", "cc_rev_p", "log_cc_n", "cc_queries", "no_cc", "rc_p", "log_rc_n", "import1", "import23", "imported",
    "log_uses", "named_test", "tested_by_change", "same_dir", "path_sim", "log_commits",
    "is_test", "is_docs", "is_other",
]


def _get(c: Any, key: str, default=0):
    return c.get(key, default) if isinstance(c, Mapping) else getattr(c, key, default)


def featurize(c: Any) -> list[float]:
    """Model inputs for a candidate (a features.Candidate, or its dict form)."""
    return [
        _get(c, "cc_p"), _get(c, "cc_rev_p"), math.log1p(_get(c, "cc_n")), min(_get(c, "cc_queries"), 5),
        float(_get(c, "cc_n") == 0), _get(c, "rc_p", 0.0), math.log1p(_get(c, "rc_n", 0)),
        float(_get(c, "import_hops") == 1), float(_get(c, "import_hops") in (2, 3)), float(_get(c, "imported_hops") > 0),
        math.log1p(_get(c, "symbol_uses")), float(bool(_get(c, "named_test"))), float(bool(_get(c, "tested_by_change"))),
        float(_get(c, "same_dir")), _get(c, "path_sim"), math.log1p(_get(c, "commits")),
        float(_get(c, "kind") == "test"), float(_get(c, "kind") == "docs"), float(_get(c, "kind") == "other"),
    ]


def probability(c: Any) -> float:
    z = BIAS + sum(WEIGHTS[name] * x for name, x in zip(FEATURES, featurize(c)))
    return 1 / (1 + math.exp(-z))


def reasons(c: Any) -> list[str]:
    """Human-readable evidence, strongest contribution first."""
    x = dict(zip(FEATURES, featurize(c)))
    out: list[tuple[float, str]] = []

    def add(features: tuple[str, ...], text: str) -> None:
        out.append((sum(WEIGHTS[f] * x[f] for f in features), text))

    if _get(c, "cc_n"):
        add(("cc_p", "cc_rev_p", "log_cc_n", "cc_queries", "no_cc"),
            f"changed together with {_get(c, 'cc_with')} in {_get(c, 'cc_n')} of its {_get(c, 'cc_total')} changes")
    if _get(c, "rc_n", 0):
        add(("rc_p", "log_rc_n"), f"changed together {_get(c, 'rc_n')} times in the last {_get(c, 'rc_window', 0)} changes")
    uses = [u[1] if isinstance(u, (tuple, list)) else u for u in (_get(c, "uses", []) or [])]
    if uses:
        shown = ", ".join(dict.fromkeys(uses[:3]))
        add(("log_uses",), f"uses changed {shown}" + (f" (+{len(set(uses)) - 3} more)" if len(set(uses)) > 3 else ""))
    if _get(c, "named_test"):
        add(("named_test",), f"test named after {_get(c, 'named_test')}")
    if _get(c, "tested_by_change"):
        add(("tested_by_change",), f"module under test in {_get(c, 'tested_by_change')}")
    if _get(c, "import_hops") == 1:
        add(("import1",), f"imports {_get(c, 'imports')}")
    elif _get(c, "import_hops"):
        add(("import23",), f"imports {_get(c, 'imports')} indirectly ({_get(c, 'import_hops')} hops)")
    if _get(c, "imported_hops"):
        add(("imported",), f"imported by {_get(c, 'imported_by')}")
    out.sort(key=lambda t: -t[0])
    return [text for _, text in out]
