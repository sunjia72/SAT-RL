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
    if len(idx) != K:
        return False, f"Expect exactly {K} indices."
    if any(not (0 <= i < M) for i in idx):
        return False, "Index out of range."
    if len(set(idx)) != K:
        return False, "Duplicate indices."
    used = set()
    for i in idx:
        for e in sets[i]:
            if e in used:
                return False, "Sets are not pairwise disjoint."
            used.add(e)
    return True, "OK"
