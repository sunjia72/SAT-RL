# logic_generators/puzzles/bnn/verifier.py
from __future__ import annotations
from typing import Dict, Any, List

def _parse_bits(s: str, n: int) -> List[int]:
    toks = [t for t in s.strip().replace(",", " ").split() if t]
    if len(toks) != n:
        return []
    try:
        bits = [int(t) for t in toks]
        return bits
    except ValueError:
        return []

def _b(p):  # "True"/"False" to bool
    s = str(p).strip().lower()
    return s in ("true","t","1","+","yes","y")

def _eval_bnn(n_in: int, layers: list, x_bits: List[int]) -> int:
    val = {f"x{i}": int(x_bits[i]) for i in range(n_in)}
    for layer in layers:
        for neu in layer:
            s = 0
            for (v, pol) in neu["inputs"]:
                s += val[str(v)] if _b(pol) else (1 - val[str(v)])
            val[str(neu["out"])] = 1 if s >= int(neu["k"]) else 0
    return int(val["y"])

def verify(instance: Dict[str, Any], candidate: Any):
    if instance.get("type") != "solving":
        return False, "Verifier only for solving."
    n = int(instance["n_in"])
    layers = instance["layers"]
    prop = instance["property"]
    seed = [int(b) for b in prop["seed"]]
    eps = int(prop["eps"])
    y_seed = int(prop["y_seed"])

    if isinstance(candidate, list):
        x = [int(v) for v in candidate]
    else:
        x = _parse_bits(str(candidate), n)
    if len(x) != n or any(v not in (0,1) for v in x):
        return False, "Bad bit vector."

    # Hamming distance
    dist = sum(1 for i in range(n) if int(x[i]) != int(seed[i]))
    if dist > eps:
        return False, f"Hamming distance {dist} > eps {eps}."
    y = _eval_bnn(n, layers, x)
    if int(y) == int(y_seed):
        return False, "Output did not flip."
    return True, "OK"
