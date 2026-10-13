from __future__ import annotations
from typing import Dict, Any, List, Tuple

def _parse_indices(s: str) -> List[int]:
    toks = [t for t in s.strip().replace(",", " ").split() if t]
    try:
        return [int(t) for t in toks]
    except ValueError:
        return []

def verify(instance: Dict[str, Any], candidate: Any):
    if instance.get("type") != "solving":
        return False, "Verifier only for solving."
    N = int(instance["N"]); K = int(instance["K"])
    edges = {(int(u), int(v)) for (u, v) in instance["edges"]}
    # undirected edge set with u<v
    if isinstance(candidate, list):
        S = [int(x) for x in candidate]
    else:
        S = _parse_indices(str(candidate))
    if len(S) != K:
        return False, f"Need exactly {K} vertex indices."
    if any(not (0 <= v < N) for v in S):
        return False, "Vertex index out of range."
    if len(set(S)) != K:
        return False, "Duplicate vertices."
    SS = sorted(S)
    # every pair must be an edge
    for i in range(K):
        for j in range(i+1, K):
            u, v = SS[i], SS[j]
            e = (u, v) if u < v else (v, u)
            if e not in edges:
                return False, f"Missing edge between {u} and {v}."
    return True, "OK"
