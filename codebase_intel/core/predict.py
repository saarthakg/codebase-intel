"""Which files a change is likely to need, with calibrated probabilities."""
from dataclasses import dataclass, field

from codebase_intel.core.features import gather
from codebase_intel.core.scoring import probability, reasons
from codebase_intel.state import RepoState


@dataclass
class Prediction:
    file: str
    probability: float
    reasons: list[str]
    because_of: list[str]           # changed files that led here
    kind: str                       # source / test / docs / other
    uses: list[tuple[str, str]] = field(default_factory=list)  # (changed file, symbol) it uses


def predict(changed: dict[str, list[str]], repo_id: str, state: RepoState, depth: int = 3) -> list[Prediction]:
    """Every candidate file for a change, most likely first. `changed` maps
    each changed, indexed file to the qualified names of symbols it touched
    (empty: unknown or module-level)."""
    cands = gather(changed, repo_id, state.graph, state.metadata_store, state.cochange,
                   depth=depth, recent=state.recent)
    out = [
        Prediction(file=c.file, probability=probability(c), reasons=reasons(c), because_of=sorted(c.because_of),
                   kind=c.kind, uses=list(dict.fromkeys(c.uses)))
        for c in cands.values()
    ]
    out.sort(key=lambda p: (-p.probability, p.file))
    return out


def target_change(target: str, repo_id: str, state: RepoState) -> dict[str, list[str]]:
    """A single file or symbol as a change: a file path → that file; a symbol
    ("HTTPAdapter.send") → its defining file, with that symbol changed."""
    if target in state.graph.G.nodes:
        return {target: []}
    from codebase_intel.core.definitions import best_definition
    found = best_definition(target, state.metadata_store, repo_id)
    if found is None:
        return {}
    return {found["file_path"]: [found.get("qualified_name") or target]}
