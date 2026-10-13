# logic_generators/puzzles/mini_sudoku/verifier.py
from __future__ import annotations
from typing import List, Tuple, Any, Dict
import math, json

def _parse_grid(s: str) -> List[List[int]]:
    rows = []
    for line in s.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        row = []
        for tok in line.split():
            if tok == ".":
                row.append(0)
            else:
                try:
                    row.append(int(tok))
                except ValueError:
                    row.append(0)
        rows.append(row)
    return rows

def _is_valid_solution(grid: List[List[int]]) -> bool:
    N = len(grid)
    if any(len(row) != N for row in grid):
        return False
    # must be 1..N everywhere
    for row in grid:
        if any(v < 1 or v > N for v in row):
            return False
    n = int(math.isqrt(N))
    if n * n != N:
        return False
    target = set(range(1, N + 1))

    # rows / cols
    for i in range(N):
        if set(grid[i]) != target:
            return False
    for j in range(N):
        col = [grid[i][j] for i in range(N)]
        if set(col) != target:
            return False
    # blocks
    for bi in range(n):
        for bj in range(n):
            block = [
                grid[i][j]
                for i in range(bi * n, (bi + 1) * n)
                for j in range(bj * n, (bj + 1) * n)
            ]
            if set(block) != target:
                return False
    return True

def verify(instance: Dict[str, Any], candidate: Any) -> Tuple[bool, str]:
    """
    Returns (ok, msg). Only intended for `type == "solving"`.
    """
    if not isinstance(instance, dict) or instance.get("type") != "solving":
        return False, "Verifier only applies to solving tasks."

    # candidate is expected to be a full grid string
    if isinstance(candidate, (list, dict)):
        # In case a program returns a matrix; allow it:
        grid = candidate if isinstance(candidate, list) else candidate.get("grid", [])
    else:
        grid = _parse_grid(str(candidate))

    N = int(instance["N"])
    given = _parse_grid(instance["grid"])

    # Check shape
    if len(grid) != N or any(len(row) != N for row in grid):
        return False, "Wrong shape."

    # Check given clues are respected
    for i in range(N):
        for j in range(N):
            if given[i][j] not in (0, None, "."):
                if int(grid[i][j]) != int(given[i][j]):
                    return False, f"Clue mismatch at ({i},{j})."

    # Check Sudoku validity
    if not _is_valid_solution([[int(x) for x in row] for row in grid]):
        return False, "Not a valid Sudoku solution."
    return True, "OK"
