# logic_generators/puzzles/nfa/verifier.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple

def _parse_path(s: str, L: int) -> List[int]:
    toks = [t for t in s.strip().replace(",", " ").split() if t]
    if len(toks) != L:
        return []
    try:
        return [int(t) for t in toks]
    except ValueError:
        return []

def verify(instance: Dict[str, Any], candidate: Any):
    if instance.get("type") != "solving":
        return False, "Verifier only for solving."
    S = int(instance["S"]); L = int(instance["L"])
    start = int(instance["start"])
    accepts = [int(x) for x in instance["accepts"]]
    trans = [(int(s), int(a), int(t)) for (s,a,t) in instance["transitions"]]
    E = len(trans)
    if isinstance(candidate, list):
        path = [int(x) for x in candidate]
    else:
        path = _parse_path(str(candidate), L)
    if len(path) != L:
        return False, "Wrong path length."
    # simulate
    s = start
    for k, idx in enumerate(path):
        if not (0 <= idx < E):
            return False, f"Transition index out of range at step {k}."
        (src, _a, dst) = trans[idx]
        if src != s:
            return False, f"Transition {idx} not chaining at step {k}."
        s = dst
    if s not in set(accepts):
        return False, "Final state is not accepting."
    return True, "OK"
