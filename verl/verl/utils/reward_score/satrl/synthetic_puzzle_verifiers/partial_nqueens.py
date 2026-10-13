from __future__ import annotations
from typing import Dict, Any, List, Tuple

def _parse_cols(s: str, N: int) -> List[int]:
    toks = [t for t in s.strip().replace(',', ' ').split() if t]
    if len(toks) != N: return []
    try:
        return [int(t) for t in toks]
    except ValueError:
        return []

def verify(instance: Dict[str, Any], candidate: Any):
    if instance.get("type") != "solving":
        return False, "Verifier only for solving."
    N = int(instance["N"])
    pre = [(int(i), int(j)) for (i, j) in instance["pre"]]
    if isinstance(candidate, list):
        cols = [int(x) for x in candidate]
    else:
        cols = _parse_cols(str(candidate), N)
    if len(cols) != N: return False, "Need N columns."
    if any(not (0 <= c < N) for c in cols): return False, "Column out of range."
    # respect pre
    for (i, j) in pre:
        if cols[i] != j: return False, f"Pre-placement mismatch at row {i}."
    # columns distinct
    if len(set(cols)) != N: return False, "Duplicate column."
    # diagonals
    d1 = set(); d2 = set()
    for r, c in enumerate(cols):
        a = c - r; b = c + r
        if a in d1 or b in d2: return False, "Diagonal attack."
        d1.add(a); d2.add(b)
    return True, "OK"
