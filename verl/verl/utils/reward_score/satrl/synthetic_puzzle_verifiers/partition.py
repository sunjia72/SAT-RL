from __future__ import annotations
from typing import Dict, Any, List

def _parse_idx(s: str) -> List[int]:
    toks = [t for t in s.strip().replace(',', ' ').split() if t]
    try:
        return [int(t) for t in toks]
    except ValueError:
        return []

def verify(instance: Dict[str, Any], candidate: Any):
    if instance.get("type") != "solving":
        return False, "Verifier only for solving."
    N = int(instance["N"])
    arr = [int(x) for x in instance["a"]]
    total = sum(arr)
    if total % 2 == 1:
        return False, "Total is odd; no solution."
    target = total // 2
    if isinstance(candidate, list):
        idx = [int(x) for x in candidate]
    else:
        idx = _parse_idx(str(candidate))
    if any(not (0 <= i < N) for i in idx):
        return False, "Index out of range."
    if len(set(idx)) != len(idx):
        return False, "Duplicate indices."
    s = sum(arr[i] for i in idx)
    if s != target:
        return False, f"Sum {s} != target {target}."
    return True, "OK"
