from __future__ import annotations
from typing import Dict, Any, List, Tuple
from collections import deque

def _parse_idx(s: str) -> List[int]:
    toks = [t for t in s.strip().replace(',', ' ').split() if t]
    try:
        return [int(t) for t in toks]
    except ValueError:
        return []

def _acyclic_after_removal(N: int, edges: List[Tuple[int,int]], remove: List[int]) -> bool:
    rem = set(remove)
    kept = [v for v in range(N) if v not in rem]
    idmap = {v:i for i,v in enumerate(kept)}
    M = len(kept)
    indeg = [0]*M
    adj = [[] for _ in range(M)]
    for (u, v) in edges:
        if u in rem or v in rem: continue
        uu, vv = idmap[u], idmap[v]
        adj[uu].append(vv)
        indeg[vv] += 1
    # Kahn
    q = deque([i for i in range(M) if indeg[i]==0])
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for w in adj[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return seen == M

def verify(instance: Dict[str, Any], candidate: Any):
    if instance.get("type") != "solving":
        return False, "Verifier only for solving."
    N = int(instance["N"]); K = int(instance["K"])
    edges = [(int(u), int(v)) for (u,v) in instance["edges"]]
    if isinstance(candidate, list):
        rem = [int(x) for x in candidate]
    else:
        rem = _parse_idx(str(candidate))
    if any(not (0 <= i < N) for i in rem):
        return False, "Index out of range."
    if len(set(rem)) != len(rem):
        return False, "Duplicate indices."
    if len(rem) > K:
        return False, f"Removal size {len(rem)} > K {K}."
    if not _acyclic_after_removal(N, edges, rem):
        return False, "Remaining graph is cyclic."
    return True, "OK"
