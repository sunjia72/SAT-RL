# logic_generators/puzzles/graph_coloring/verifier.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple

def _parse_colors(s: str, N: int) -> List[int]:
    toks = [t for t in s.strip().replace(",", " ").split() if t]
    if len(toks) != N:
        return []
    try:
        return [int(t) for t in toks]
    except ValueError:
        return []

def verify(instance: Dict[str, Any], candidate: Any):
    if instance.get("type") != "solving":
        return False, "Verifier only for solving."
    N = int(instance["N"]); K = int(instance["K"])
    edges = [(int(u), int(v)) for (u, v) in instance["edges"]]
    if isinstance(candidate, list):
        colors = [int(x) for x in candidate]
    else:
        colors = _parse_colors(str(candidate), N)
    if len(colors) != N:
        return False, "Wrong number of colors."
    if any(c < 1 or c > K for c in colors):
        return False, "Colors out of range."
    for (u, v) in edges:
        if colors[u] == colors[v]:
            return False, f"Edge ({u},{v}) is monochromatic."
    return True, "OK"
