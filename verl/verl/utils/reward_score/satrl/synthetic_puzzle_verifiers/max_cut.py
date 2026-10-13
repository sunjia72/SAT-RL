from __future__ import annotations
from typing import Dict, Any, List, Tuple

def _parse_bits(s: str, n: int) -> List[int]:
    toks = [t for t in s.strip().replace(",", " ").split() if t]
    if len(toks) != n:
        return []
    try:
        bs = [int(t) for t in toks]
        return bs
    except ValueError:
        return []

def verify(instance: Dict[str, Any], candidate: Any):
    if instance.get("type") != "solving":
        return False, "Verifier only for solving."
    N = int(instance["N"]); T = int(instance["T"])
    edges = [(int(u), int(v)) for (u, v) in instance["edges"]]
    if isinstance(candidate, list):
        lab = [int(x) for x in candidate]
    else:
        lab = _parse_bits(str(candidate), N)
    if len(lab) != N or any(b not in (0,1) for b in lab):
        return False, "Expect N bits 0/1."
    cut = 0
    for (u, v) in edges:
        if lab[u] != lab[v]:
            cut += 1
    if cut < T:
        return False, f"Cut size {cut} < T {T}."
    return True, "OK"
