from __future__ import annotations
from typing import Dict, Any, List

def _parse_indices(s: str) -> List[int]:
    toks = [t for t in s.strip().replace(",", " ").split() if t]
    try:
        return [int(t) for t in toks]
    except ValueError:
        return []

def verify(instance: Dict[str, Any], candidate: Any):
    if instance.get("type") != "solving":
        return False, "Verifier only for solving."
    U = int(instance["U"]); M = int(instance["M"]); K = int(instance["K"])
    sets = [[int(x) for x in S] for S in instance["sets"]]
    if isinstance(candidate, list):
        idx = [int(x) for x in candidate]
    else:
        idx = _parse_indices(str(candidate))
    if len(idx) == 0:
        return False, "Need at least one index (unless U empty)."
    if len(set(idx)) != len(idx):
        return False, "Duplicate indices."
    if any(not (0 <= i < M) for i in idx):
        return False, "Index out of range."
    if len(idx) > K:
        return False, f"Must use at most K={K} sets."
    covered = set()
    for i in idx:
        covered.update(sets[i])
    if U > 0 and len(covered) < U:
        return False, "Not all elements covered."
    return True, "OK"
