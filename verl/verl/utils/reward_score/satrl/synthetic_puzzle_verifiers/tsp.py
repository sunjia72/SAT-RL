from __future__ import annotations
from typing import Dict, Any, List

def _parse_perm(s: str) -> List[int]:
    toks = [t for t in s.strip().replace(',', ' ').split() if t]
    try:
        return [int(t) for t in toks]
    except ValueError:
        return []

def verify(instance: Dict[str, Any], candidate: Any):
    if instance.get("type") != "solving":
        return False, "Verifier only for solving."
    N = int(instance["N"]); T = int(instance["T"])
    W = [[int(v) for v in row] for row in instance["W"]]
    if isinstance(candidate, list):
        tour = [int(x) for x in candidate]
    else:
        tour = _parse_perm(str(candidate))
    if len(tour) != N: return False, "Need N vertices."
    if set(tour) != set(range(N)): return False, "Not a permutation."
    # compute cost
    cost = 0
    for k in range(N):
        i = tour[k]
        j = tour[(k+1) % N]
        if i == j: return False, "Self-loop."
        cost += int(W[i][j])
    if cost > T:
        return False, f"Cost {cost} > T {T}."
    return True, "OK"
