"""Exact logical equivalence scoring for SATBench Boolean formulas.

The reward is binary: a proposed formula receives ``1.0`` only when Z3 proves
that the ground-truth formula and proposed formula are logically equivalent.
Equivalence is checked by asking whether ``F_gt != F_pred`` is satisfiable.
If the symmetric difference is unsatisfiable, the formulas are equivalent.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import site
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

try:  # pragma: no cover - import path depends on how the script is launched.
    from .formula_extractor import (
        BoolFormula,
        ExtractedFormula,
        extract_formula,
        extract_raw_formula,
    )
except ImportError:  # pragma: no cover
    from formula_extractor import (
        BoolFormula,
        ExtractedFormula,
        extract_formula,
        extract_raw_formula,
    )


Backend = Literal["z3", "ortools", "sympy", "pysmt", "pysat"]
VarStyle = Literal["multi", "one"]

_VAR_ID_RE = re.compile(r"^v_(\d+)$")
_X_UNDERSCORE_RE = re.compile(r"^x_(-?\d+(?:_-?\d+)*)$")
_X_COMPACT_RE = re.compile(r"^x(-?\d+)$")
_X_CALL_RE = re.compile(r"^x\((.*)\)$")


@dataclass(frozen=True)
class FormulaEquivalenceResult:
    score: float
    equivalent: bool
    gt_clause_count: int
    pred_clause_count: int
    gt_satisfiable: bool | None
    pred_satisfiable: bool | None
    unknown_checks: int
    timed_out: bool
    backend: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _Deadline:
    def __init__(self, timeout_s: float | None) -> None:
        self.timeout_s = timeout_s
        self.started_at = time.monotonic()

    def expired(self) -> bool:
        if self.timeout_s is None:
            return False
        return time.monotonic() - self.started_at >= self.timeout_s

    def remaining_ms(self, cap_ms: int | None = None) -> int:
        if self.timeout_s is None:
            return int(cap_ms if cap_ms is not None else 0)
        remaining = max(0.0, self.timeout_s - (time.monotonic() - self.started_at))
        ms = max(1, int(remaining * 1000))
        if cap_ms is not None:
            ms = min(ms, int(cap_ms))
        return ms


def score_program_against_row(
    row: dict[str, object],
    program: str,
    *,
    backend: Backend,
    timeout_s: float | None = 20.0,
    solver_timeout_ms: int = 20000,
) -> FormulaEquivalenceResult:
    extracted = extract_formula(program, backend=backend, run_check=False, quiet=True)
    return score_extracted_against_row(
        row,
        extracted,
        backend=backend,
        timeout_s=timeout_s,
        solver_timeout_ms=solver_timeout_ms,
    )


def score_extracted_against_row(
    row: dict[str, object],
    extracted: ExtractedFormula,
    *,
    backend: Backend,
    timeout_s: float | None = 20.0,
    solver_timeout_ms: int = 20000,
) -> FormulaEquivalenceResult:
    gt_field = "one_dimension_formula" if backend == "pysat" else "multi_dimension_formula"
    gt_style: VarStyle = "one" if backend == "pysat" else "multi"
    gt = extract_raw_formula(str(row[gt_field]))
    return score_extracted_formulas(
        gt,
        extracted,
        dims=[int(dim) for dim in row["dims"]],  # type: ignore[index]
        gt_var_style=gt_style,
        pred_var_style=("one" if backend == "pysat" else "multi"),
        backend=backend,
        timeout_s=timeout_s,
        solver_timeout_ms=solver_timeout_ms,
    )


def score_formula_strings(
    gt_formula: str,
    pred_formula: str,
    *,
    dims: Sequence[int],
    gt_var_style: VarStyle = "multi",
    pred_var_style: VarStyle = "multi",
    backend: str | None = None,
    timeout_s: float | None = 20.0,
    solver_timeout_ms: int = 20000,
) -> FormulaEquivalenceResult:
    return score_extracted_formulas(
        extract_raw_formula(gt_formula),
        extract_raw_formula(pred_formula),
        dims=dims,
        gt_var_style=gt_var_style,
        pred_var_style=pred_var_style,
        backend=backend,
        timeout_s=timeout_s,
        solver_timeout_ms=solver_timeout_ms,
    )


def _import_z3_solver() -> object:
    """Import the real ``z3-solver`` module, bypassing broken user-site shadows."""
    try:
        z3 = importlib.import_module("z3")
        if callable(getattr(z3, "Solver", None)):
            return z3
    except Exception:
        pass

    user_site_paths: set[str] = set()
    try:
        user_site_paths.add(site.getusersitepackages())
    except Exception:
        pass
    try:
        user_site = site.getuserbase()
        user_site_paths.update(path for path in sys.path if path.startswith(user_site))
    except Exception:
        pass

    old_path = list(sys.path)
    old_modules = {name: module for name, module in sys.modules.items() if name == "z3" or name.startswith("z3.")}
    try:
        sys.path = [path for path in sys.path if path not in user_site_paths]
        for name in list(sys.modules):
            if name == "z3" or name.startswith("z3."):
                del sys.modules[name]

        z3 = importlib.import_module("z3")
        if callable(getattr(z3, "Solver", None)):
            return z3
    finally:
        sys.path = old_path
        if not callable(getattr(sys.modules.get("z3"), "Solver", None)):
            for name in list(sys.modules):
                if name == "z3" or name.startswith("z3."):
                    del sys.modules[name]
            sys.modules.update(old_modules)

    raise ImportError("z3-solver is unavailable or shadowed by a non-solver z3 package")


def score_extracted_formulas(
    gt: ExtractedFormula,
    pred: ExtractedFormula,
    *,
    dims: Sequence[int],
    gt_var_style: VarStyle,
    pred_var_style: VarStyle,
    backend: str | None = None,
    timeout_s: float | None = 20.0,
    solver_timeout_ms: int = 20000,
) -> FormulaEquivalenceResult:
    z3 = _import_z3_solver()

    deadline = _Deadline(timeout_s)
    gt_clauses = tuple(canonicalize_formula(clause, dims, gt_var_style) for clause in gt.clauses())
    pred_clauses = tuple(canonicalize_formula(clause, dims, pred_var_style) for clause in pred.clauses())

    result = _check_equivalence(
        gt_clauses,
        pred_clauses,
        z3_module=z3,
        deadline=deadline,
        solver_timeout_ms=solver_timeout_ms,
    )
    equivalent, unknown, timed_out = result
    score = 1.0 if equivalent else 0.0

    gt_satisfiable = _formula_set_satisfiable(gt_clauses, z3, deadline, solver_timeout_ms)
    pred_satisfiable = _formula_set_satisfiable(pred_clauses, z3, deadline, solver_timeout_ms)

    return FormulaEquivalenceResult(
        score=score,
        equivalent=equivalent,
        gt_clause_count=len(gt_clauses),
        pred_clause_count=len(pred_clauses),
        gt_satisfiable=gt_satisfiable,
        pred_satisfiable=pred_satisfiable,
        unknown_checks=int(unknown),
        timed_out=timed_out or deadline.expired(),
        backend=backend,
    )


def _check_equivalence(
    gt_clauses: Sequence[BoolFormula],
    pred_clauses: Sequence[BoolFormula],
    *,
    z3_module: object,
    deadline: _Deadline,
    solver_timeout_ms: int,
) -> tuple[bool, bool, bool]:
    """Return ``(equivalent, unknown, timed_out)`` for two conjunctions."""

    if deadline.expired():
        return False, True, True

    variables = _z3_vars(_collect_vars(tuple(gt_clauses) + tuple(pred_clauses)), z3_module)
    gt_expr = _z3_conjunction(gt_clauses, variables, z3_module)
    pred_expr = _z3_conjunction(pred_clauses, variables, z3_module)

    solver = z3_module.Solver()
    solver.set(timeout=deadline.remaining_ms(solver_timeout_ms))
    solver.add(gt_expr != pred_expr)
    result = solver.check()
    if result == z3_module.unsat:
        return True, False, deadline.expired()
    if result == z3_module.sat:
        return False, False, deadline.expired()

    timed_out = deadline.expired()
    try:
        timed_out = timed_out or "timeout" in str(solver.reason_unknown()).lower()
    except Exception:
        pass
    return False, True, timed_out


def _formula_set_satisfiable(
    formulas: Sequence[BoolFormula],
    z3_module: object,
    deadline: _Deadline,
    solver_timeout_ms: int,
) -> bool | None:
    if deadline.expired():
        return None

    variables = _z3_vars(_collect_vars(formulas), z3_module)
    solver = z3_module.Solver()
    solver.set(timeout=deadline.remaining_ms(solver_timeout_ms))
    solver.add(_z3_conjunction(formulas, variables, z3_module))
    result = solver.check()
    if result == z3_module.sat:
        return True
    if result == z3_module.unsat:
        return False
    return None


def _z3_conjunction(
    formulas: Sequence[BoolFormula],
    variables: dict[str, object],
    z3_module: object,
) -> object:
    if not formulas:
        return z3_module.BoolVal(True)
    return z3_module.And(*[_to_z3(formula, variables, z3_module) for formula in formulas])


def canonicalize_formula(formula: BoolFormula, dims: Sequence[int], var_style: VarStyle) -> BoolFormula:
    if formula.op == "var":
        assert formula.name is not None
        return BoolFormula.var(canonical_var_name(formula.name, dims, var_style))
    if formula.op == "const":
        return formula
    return BoolFormula(formula.op, tuple(canonicalize_formula(arg, dims, var_style) for arg in formula.args))


def canonical_var_name(name: str, dims: Sequence[int], var_style: VarStyle) -> str:
    dims = tuple(int(dim) for dim in dims)
    stripped = name.strip()

    match = _VAR_ID_RE.match(stripped)
    if match:
        return _check_var_id(int(match.group(1)), dims)

    indices = _parse_index_tuple(stripped)
    if indices is not None:
        if var_style == "one" and len(indices) == 1:
            return _check_var_id(indices[0], dims)
        if len(indices) == len(dims):
            return _check_var_id(_index_tuple_to_var_id(indices, dims), dims)

    # Leave non-canonical names intact.  This makes extra or malformed variables
    # visible to the SAT checks instead of silently conflating them.
    return stripped


def _parse_index_tuple(name: str) -> tuple[int, ...] | None:
    match = _X_UNDERSCORE_RE.match(name)
    if match:
        return tuple(int(part) for part in match.group(1).split("_"))

    match = _X_COMPACT_RE.match(name)
    if match:
        return (int(match.group(1)),)

    match = _X_CALL_RE.match(name)
    if match:
        raw = match.group(1).strip()
        if raw.endswith(","):
            raw = raw[:-1]
        if not raw:
            return ()
        try:
            return tuple(int(part.strip()) for part in raw.split(","))
        except ValueError:
            return None

    return None


def _index_tuple_to_var_id(indices: Sequence[int], dims: Sequence[int]) -> int:
    if len(indices) != len(dims):
        raise ValueError(f"index tuple length {len(indices)} does not match dims length {len(dims)}")
    offset = 0
    for axis, (index, dim) in enumerate(zip(indices, dims)):
        if index < 0 or index >= dim:
            raise ValueError(f"index {tuple(indices)} is out of bounds for dims {tuple(dims)}")
        stride = math.prod(dims[axis + 1 :])
        offset += int(index) * int(stride)
    return offset + 1


def _check_var_id(var_id: int, dims: Sequence[int]) -> str:
    max_var = math.prod(dims)
    if var_id < 1 or var_id > max_var:
        raise ValueError(f"variable id {var_id} is out of bounds for dims {tuple(dims)}")
    return f"v_{var_id}"


def _collect_vars(formulas: Iterable[BoolFormula]) -> set[str]:
    names: set[str] = set()

    def visit(formula: BoolFormula) -> None:
        if formula.op == "var":
            assert formula.name is not None
            names.add(formula.name)
            return
        for arg in formula.args:
            visit(arg)

    for formula in formulas:
        visit(formula)
    return names


def _z3_vars(names: Iterable[str], z3_module: object, *, suffix: str = "") -> dict[str, object]:
    return {name: z3_module.Bool(f"{name}{suffix}") for name in names}


def _to_z3(formula: BoolFormula, variables: dict[str, object], z3_module: object) -> object:
    if formula.op == "var":
        assert formula.name is not None
        return variables[formula.name]
    if formula.op == "const":
        return z3_module.BoolVal(bool(formula.value))
    if formula.op == "not":
        return z3_module.Not(_to_z3(formula.args[0], variables, z3_module))
    if formula.op == "and":
        return z3_module.And(*[_to_z3(arg, variables, z3_module) for arg in formula.args])
    if formula.op == "or":
        return z3_module.Or(*[_to_z3(arg, variables, z3_module) for arg in formula.args])
    if formula.op == "implies":
        return z3_module.Implies(
            _to_z3(formula.args[0], variables, z3_module),
            _to_z3(formula.args[1], variables, z3_module),
        )
    if formula.op == "iff":
        return _to_z3(formula.args[0], variables, z3_module) == _to_z3(
            formula.args[1],
            variables,
            z3_module,
        )
    raise ValueError(f"unsupported formula op: {formula.op}")


def _load_jsonl_row(path: Path, row_index: int) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == row_index:
                return json.loads(line)
    raise IndexError(f"row index {row_index} not found in {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, help="SATBench JSONL path")
    parser.add_argument("--row-index", type=int, help="SATBench row index")
    parser.add_argument("--program-file", type=Path, help="Proposed solver program")
    parser.add_argument("--backend", choices=("z3", "ortools", "sympy", "pysmt", "pysat"), default="z3")
    parser.add_argument("--gt-formula", help="Raw ground-truth formula")
    parser.add_argument("--pred-formula", help="Raw proposed formula")
    parser.add_argument("--dims", nargs="*", type=int, help="Variable dimensions for raw formulas")
    parser.add_argument("--timeout", type=float, default=1.0, help="Total scoring timeout in seconds")
    args = parser.parse_args(argv)

    if args.jsonl and args.row_index is not None and args.program_file:
        row = _load_jsonl_row(args.jsonl, args.row_index)
        result = score_program_against_row(
            row,
            args.program_file.read_text(encoding="utf-8"),
            backend=args.backend,
            timeout_s=args.timeout,
        )
    elif args.gt_formula and args.pred_formula and args.dims:
        result = score_formula_strings(
            args.gt_formula,
            args.pred_formula,
            dims=args.dims,
            gt_var_style=("one" if args.backend == "pysat" else "multi"),
            pred_var_style=("one" if args.backend == "pysat" else "multi"),
            backend=args.backend,
            timeout_s=args.timeout,
        )
    else:
        parser.error("provide either --jsonl/--row-index/--program-file or --gt-formula/--pred-formula/--dims")

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
