"""Extract Boolean formulas from Python programs.

The extractor executes trusted/generated snippets and returns the Boolean
constraints they build for one of five supported backends: Z3, OR-Tools CP-SAT,
SymPy, PySMT, or PySAT. This is not a Python security sandbox.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import inspect
import importlib
import re
import site
import sys
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from types import MethodType
from typing import Iterable, Literal


Backend = Literal["z3", "ortools", "sympy", "pysmt", "pysat"]
SUPPORTED_BACKENDS: tuple[str, ...] = ("z3", "ortools", "sympy", "pysmt", "pysat")


@dataclass(frozen=True)
class BoolFormula:
    """Backend-neutral propositional formula.

    Normalized formulas use only variables, constants, and the connectives
    ``∧``, ``∨``, ``¬``, ``→``, and ``↔``.
    """

    op: str
    args: tuple["BoolFormula", ...] = ()
    name: str | None = None
    value: bool | None = None

    @classmethod
    def var(cls, name: str) -> "BoolFormula":
        return cls("var", name=name)

    @classmethod
    def const(cls, value: bool) -> "BoolFormula":
        return cls("const", value=value)

    def __str__(self) -> str:
        if self.op == "var":
            return self.name or "<unnamed>"
        if self.op == "const":
            return "true" if self.value else "false"
        if self.op == "not":
            return f"¬{self.args[0]}"

        names = {
            "and": "∧",
            "or": "∨",
            "implies": "→",
            "iff": "↔",
        }
        if self.op in names:
            return f"({' '.join(_interleave(names[self.op], (str(arg) for arg in self.args)))})"
        raise ValueError(f"unknown formula op: {self.op}")


@dataclass(frozen=True)
class ExtractedFormula:
    """A backend program's extracted Boolean assertions."""

    backend: str
    source: object | None
    assertions: tuple[BoolFormula, ...]

    def conjunction(self) -> BoolFormula:
        """Return the assertions as one Boolean expression."""

        return _bool_and(*self.assertions)

    def clauses(self) -> tuple[BoolFormula, ...]:
        """Return top-level clauses, flattening only outer ``And`` nodes."""

        return _bool_clauses(self.assertions)


def extract_formula(
    program: str,
    *,
    backend: Backend = "z3",
    run_check: bool = False,
    quiet: bool = True,
) -> ExtractedFormula:
    """Execute ``program`` and extract its Boolean formula.

    Z3 and PySMT snippets are expected to create exactly one solver. OR-Tools
    snippets are expected to create exactly one ``CpModel``. PySAT snippets
    should create exactly one CNF-like formula or solver. SymPy snippets should
    either call ``satisfiable(formula)`` or assign a Boolean expression to one
    of ``formula``, ``boolean_formula``, ``phi``, ``constraint``, ``constraints``,
    or ``assertions``.
    """

    if backend == "z3":
        return _extract_z3_formula(program, run_check=run_check, quiet=quiet)
    if backend == "ortools":
        return _extract_ortools_formula(program, run_check=run_check, quiet=quiet)
    if backend == "sympy":
        return _extract_sympy_formula(program, run_check=run_check, quiet=quiet)
    if backend == "pysmt":
        return _extract_pysmt_formula(program, run_check=run_check, quiet=quiet)
    if backend == "pysat":
        return _extract_pysat_formula(program, run_check=run_check, quiet=quiet)
    raise ValueError(f"unknown backend: {backend}")


def extract_z3_formula(program: str, *, run_check: bool = False, quiet: bool = True) -> ExtractedFormula:
    """Backward-compatible wrapper for the old Z3-only entry point."""

    return extract_formula(program, backend="z3", run_check=run_check, quiet=quiet)


def parse_bool_formula(formula: str) -> BoolFormula:
    """Parse a raw propositional formula string into a ``BoolFormula``.

    Supported connectives are ``¬``, ``∧``, ``∨``, ``→``, and ``↔``. Common
    ASCII aliases are accepted as well: ``!``/``~``/``not``, ``&``/``and``,
    ``|``/``or``, ``->``/``=>``/``implies``, and ``<->``/``<=>``/``iff``.
    """

    return _FormulaParser(formula).parse()


def extract_raw_formula(formula: str) -> ExtractedFormula:
    """Return an ``ExtractedFormula`` for a raw propositional formula string."""

    return ExtractedFormula(backend="raw", source=formula, assertions=(parse_bool_formula(formula),))


def _extract_z3_formula(program: str, *, run_check: bool, quiet: bool) -> ExtractedFormula:
    import z3

    created_solvers: list[object] = []
    original_solver = z3.Solver
    z3_module = getattr(z3, "z3", None)
    original_z3_module_solver = getattr(z3_module, "Solver", None)

    def tracking_solver(*args: object, **kwargs: object) -> object:
        solver = original_solver(*args, **kwargs)
        created_solvers.append(solver)
        if not run_check:
            solver.check = MethodType(_z3_unknown_check, solver)
        return solver

    z3.Solver = tracking_solver
    if original_z3_module_solver is not None:
        z3_module.Solver = tracking_solver
    try:
        _execute_program(program, "z3", quiet, ignore_errors=True)
    finally:
        z3.Solver = original_solver
        if original_z3_module_solver is not None:
            z3_module.Solver = original_z3_module_solver

    solvers = _require_exactly_one(created_solvers, "Solver() instance")
    solver = solvers[0]
    assertions = tuple(_z3_to_formula(assertion) for assertion in solver.assertions())
    return ExtractedFormula(backend="z3", source=solver, assertions=assertions)


def _extract_pysmt_formula(program: str, *, run_check: bool, quiet: bool) -> ExtractedFormula:
    import pysmt.shortcuts as pysmt_shortcuts

    created_solvers: list[object] = []
    original_solver = pysmt_shortcuts.Solver

    def tracking_solver(*args: object, **kwargs: object) -> object:
        solver = original_solver(*args, **kwargs)
        created_solvers.append(solver)
        if not run_check:
            solver.solve = MethodType(_pysmt_false_solve, solver)
        return solver

    pysmt_shortcuts.Solver = tracking_solver
    try:
        _execute_program(program, "pysmt", quiet, ignore_errors=True)
    finally:
        pysmt_shortcuts.Solver = original_solver

    solvers = _require_exactly_one(created_solvers, "Solver() instance")
    solver = solvers[0]
    assertions_result = solver.get_assertions() if hasattr(solver, "get_assertions") else None
    if assertions_result is not None:
        assertions = tuple(assertions_result)
    elif hasattr(solver, "assertions"):
        assertions = tuple(solver.assertions)
    else:
        raise RuntimeError("PySMT solver does not expose assertions")
    normalized_assertions = tuple(_pysmt_to_formula(assertion) for assertion in assertions)
    return ExtractedFormula(backend="pysmt", source=solver, assertions=normalized_assertions)


def _extract_sympy_formula(program: str, *, run_check: bool, quiet: bool) -> ExtractedFormula:
    import sympy
    import sympy.logic.inference as sympy_inference

    captured_formulas: list[object] = []
    original_sympy_satisfiable = getattr(sympy, "satisfiable", None)
    original_inference_satisfiable = sympy_inference.satisfiable

    def tracking_satisfiable(formula: object, *args: object, **kwargs: object) -> object:
        captured_formulas.append(formula)
        if run_check:
            return original_inference_satisfiable(formula, *args, **kwargs)
        return {}

    sympy.satisfiable = tracking_satisfiable
    sympy_inference.satisfiable = tracking_satisfiable
    try:
        namespace = _execute_program(program, "sympy", quiet, ignore_errors=True)
    finally:
        if original_sympy_satisfiable is not None:
            sympy.satisfiable = original_sympy_satisfiable
        sympy_inference.satisfiable = original_inference_satisfiable

    assertions = tuple(captured_formulas) if captured_formulas else _find_sympy_assertions(namespace)
    normalized_assertions = tuple(_sympy_to_formula(assertion) for assertion in assertions)
    return ExtractedFormula(backend="sympy", source=namespace, assertions=normalized_assertions)


def _extract_ortools_formula(program: str, *, run_check: bool, quiet: bool) -> ExtractedFormula:
    from ortools.sat.python import cp_model

    created_models: list[object] = []
    original_model = cp_model.CpModel
    original_solver_solve = cp_model.CpSolver.solve
    original_solver_camel_solve = cp_model.CpSolver.Solve

    def tracking_model(*args: object, **kwargs: object) -> object:
        model = original_model(*args, **kwargs)
        created_models.append(model)
        return model

    def unknown_solve(self: object, *args: object, **kwargs: object) -> object:
        return cp_model.UNKNOWN

    cp_model.CpModel = tracking_model
    if not run_check:
        cp_model.CpSolver.solve = unknown_solve
        cp_model.CpSolver.Solve = unknown_solve
    try:
        _execute_program(program, "ortools", quiet, ignore_errors=True)
    finally:
        cp_model.CpModel = original_model
        cp_model.CpSolver.solve = original_solver_solve
        cp_model.CpSolver.Solve = original_solver_camel_solve

    models = _require_exactly_one(created_models, "CpModel() instance")
    model = models[0]
    proto = model.proto
    var_names = tuple(variable.name or f"var_{index}" for index, variable in enumerate(proto.variables))
    assertions = tuple(_ortools_constraint_to_formula(constraint, var_names) for constraint in proto.constraints)
    return ExtractedFormula(backend="ortools", source=model, assertions=assertions)


def _extract_pysat_formula(program: str, *, run_check: bool, quiet: bool) -> ExtractedFormula:
    pysat_formula = _import_solver_module("pysat.formula", required_attr="CNF")
    pysat_solvers = _import_solver_module("pysat.solvers")

    created_formulas: list[object] = []
    created_solvers: list[_PySatSolverProxy] = []
    original_cnf = pysat_formula.CNF
    original_wcnf = getattr(pysat_formula, "WCNF", None)
    solver_classes = _pysat_solver_classes(pysat_solvers)
    original_solvers = {name: cls for name, cls in solver_classes}
    solver_tracking_depth = 0

    def tracking_cnf(*args: object, **kwargs: object) -> object:
        formula = original_cnf(*args, **kwargs)
        created_formulas.append(formula)
        return formula

    def tracking_wcnf(*args: object, **kwargs: object) -> object:
        if original_wcnf is None:
            raise RuntimeError("PySAT WCNF is not available")
        formula = original_wcnf(*args, **kwargs)
        created_formulas.append(formula)
        return formula

    def make_tracking_solver(name: str, solver_cls: type) -> object:
        signature = inspect.signature(solver_cls)

        def tracking_solver(*args: object, **kwargs: object) -> "_PySatSolverProxy":
            nonlocal solver_tracking_depth

            if solver_tracking_depth:
                return solver_cls(*args, **kwargs)

            bootstrap_with = _pysat_bootstrap_with(signature, args, kwargs)
            solver_tracking_depth += 1
            try:
                solver = solver_cls(*args, **kwargs)
            finally:
                solver_tracking_depth -= 1
            proxy = _PySatSolverProxy(solver, bootstrap_with=bootstrap_with, run_check=run_check)
            created_solvers.append(proxy)
            return proxy

        tracking_solver.__name__ = name
        return tracking_solver

    pysat_formula.CNF = tracking_cnf
    if original_wcnf is not None:
        pysat_formula.WCNF = tracking_wcnf
    for name, solver_cls in solver_classes:
        setattr(pysat_solvers, name, make_tracking_solver(name, solver_cls))

    try:
        namespace = _execute_program(program, "pysat", quiet, ignore_errors=True)
    finally:
        pysat_formula.CNF = original_cnf
        if original_wcnf is not None:
            pysat_formula.WCNF = original_wcnf
        for name, solver_cls in original_solvers.items():
            setattr(pysat_solvers, name, solver_cls)

    if created_solvers:
        solvers = _require_exactly_one(created_solvers, "PySAT solver")
        source = solvers[0]
        clauses = source.clauses
    elif created_formulas:
        formulas = _require_exactly_one(created_formulas, "PySAT CNF-like formula")
        source = formulas[0]
        clauses = _pysat_clauses_from_value(source)
    else:
        source = namespace
        clauses = _find_pysat_clauses(namespace)

    variable_names = _pysat_variable_names(namespace, source, clauses)
    assertions = tuple(_pysat_clause_to_formula(clause, variable_names) for clause in clauses)
    return ExtractedFormula(backend="pysat", source=source, assertions=assertions)


def _execute_program(program: str, backend: str, quiet: bool, *, ignore_errors: bool = False) -> dict[str, object]:
    namespace: dict[str, object] = {"__name__": f"__{backend}_formula_extractor__"}
    stdout = io.StringIO() if quiet else sys.stdout
    stderr = io.StringIO() if quiet else sys.stderr
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            exec(compile(program, f"<{backend}-program>", "exec"), namespace, namespace)
        except Exception as exc:
            if not ignore_errors:
                raise
            namespace["__formula_extractor_execution_error__"] = exc
    return namespace


def _import_solver_module(module_name: str, *, required_attr: str | None = None) -> object:
    try:
        module = importlib.import_module(module_name)
        if required_attr is None or hasattr(module, required_attr):
            return module
    except Exception:
        pass

    user_site_paths: set[str] = set()
    try:
        user_site_paths.add(site.getusersitepackages())
    except Exception:
        pass
    try:
        user_base = site.getuserbase()
        user_site_paths.update(path for path in sys.path if path.startswith(user_base))
    except Exception:
        pass

    root_name = module_name.split(".", 1)[0]
    old_path = list(sys.path)
    old_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == root_name or name.startswith(root_name + ".")
    }
    try:
        sys.path = [path for path in sys.path if path not in user_site_paths]
        for name in list(sys.modules):
            if name == root_name or name.startswith(root_name + "."):
                del sys.modules[name]
        module = importlib.import_module(module_name)
        if required_attr is None or hasattr(module, required_attr):
            return module
    finally:
        sys.path = old_path
        loaded = sys.modules.get(module_name)
        if loaded is None or (required_attr is not None and not hasattr(loaded, required_attr)):
            for name in list(sys.modules):
                if name == root_name or name.startswith(root_name + "."):
                    del sys.modules[name]
            sys.modules.update(old_modules)

    raise ImportError(f"{module_name} is unavailable or shadowed by an incompatible user-site package")


def _z3_unknown_check(self: object, *args: object, **kwargs: object) -> object:
    import z3

    return z3.unknown


def _pysmt_false_solve(self: object, *args: object, **kwargs: object) -> bool:
    return False


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


class _FormulaParser:
    _WORD_OPERATORS = {
        "not": "NOT",
        "and": "AND",
        "or": "OR",
        "implies": "IMPLIES",
        "iff": "IFF",
        "true": "CONST",
        "false": "CONST",
    }

    _SYMBOL_OPERATORS = {
        "¬": "NOT",
        "~": "NOT",
        "!": "NOT",
        "∧": "AND",
        "&": "AND",
        "&&": "AND",
        "/\\": "AND",
        "∨": "OR",
        "|": "OR",
        "||": "OR",
        "\\/": "OR",
        "→": "IMPLIES",
        "->": "IMPLIES",
        "=>": "IMPLIES",
        "↔": "IFF",
        "<->": "IFF",
        "<=>": "IFF",
    }

    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = self._tokenize(text)
        self.index = 0

    def parse(self) -> BoolFormula:
        formula = self._parse_iff()
        self._expect("EOF")
        return formula

    def _parse_iff(self) -> BoolFormula:
        terms = [self._parse_implies()]
        while self._match("IFF"):
            terms.append(self._parse_implies())
        return _chain_iff(tuple(terms))

    def _parse_implies(self) -> BoolFormula:
        lhs = self._parse_or()
        if self._match("IMPLIES"):
            return _bool_implies(lhs, self._parse_implies())
        return lhs

    def _parse_or(self) -> BoolFormula:
        terms = [self._parse_and()]
        while self._match("OR"):
            terms.append(self._parse_and())
        return _bool_or(*terms)

    def _parse_and(self) -> BoolFormula:
        terms = [self._parse_not()]
        while self._match("AND"):
            terms.append(self._parse_not())
        return _bool_and(*terms)

    def _parse_not(self) -> BoolFormula:
        if self._match("NOT"):
            return _bool_not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> BoolFormula:
        if self._match("LPAREN"):
            formula = self._parse_iff()
            self._expect("RPAREN")
            return formula

        token = self._peek()
        if token.kind == "CONST":
            self.index += 1
            return BoolFormula.const(token.value.lower() in ("true", "1"))
        if token.kind == "IDENT":
            self.index += 1
            return BoolFormula.var(token.value)
        raise ValueError(f"expected formula atom at position {token.position}: {self.text!r}")

    def _match(self, kind: str) -> bool:
        if self._peek().kind == kind:
            self.index += 1
            return True
        return False

    def _expect(self, kind: str) -> _Token:
        token = self._peek()
        if token.kind != kind:
            raise ValueError(f"expected {kind} at position {token.position}, found {token.value!r}")
        self.index += 1
        return token

    def _peek(self) -> _Token:
        return self.tokens[self.index]

    def _tokenize(self, text: str) -> list[_Token]:
        tokens: list[_Token] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char.isspace():
                index += 1
                continue

            matched = False
            for symbol in sorted(self._SYMBOL_OPERATORS, key=len, reverse=True):
                if text.startswith(symbol, index):
                    tokens.append(_Token(self._SYMBOL_OPERATORS[symbol], symbol, index))
                    index += len(symbol)
                    matched = True
                    break
            if matched:
                continue

            if char == "(":
                tokens.append(_Token("LPAREN", char, index))
                index += 1
                continue
            if char == ")":
                tokens.append(_Token("RPAREN", char, index))
                index += 1
                continue

            if char.isalpha() or char == "_":
                start = index
                index += 1
                while index < len(text) and (text[index].isalnum() or text[index] == "_"):
                    index += 1

                value = text[start:index]
                lower_value = value.lower()
                if lower_value in self._WORD_OPERATORS:
                    tokens.append(_Token(self._WORD_OPERATORS[lower_value], value, start))
                    continue

                if index < len(text) and text[index] == "(":
                    value, index = self._read_function_like_variable(text, start, index)
                tokens.append(_Token("IDENT", value, start))
                continue

            if char in ("0", "1"):
                tokens.append(_Token("CONST", char, index))
                index += 1
                continue

            raise ValueError(f"unexpected character at position {index}: {char!r}")

        tokens.append(_Token("EOF", "", len(text)))
        return tokens

    def _read_function_like_variable(self, text: str, start: int, open_paren: int) -> tuple[str, int]:
        depth = 0
        index = open_paren
        while index < len(text):
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1], index + 1
            elif char in "¬∧∨→↔|&!~":
                break
            index += 1
        return text[start:open_paren], open_paren


class _PySatSolverProxy:
    def __init__(self, solver: object, *, bootstrap_with: object | None, run_check: bool) -> None:
        self._solver = solver
        self._run_check = run_check
        self.clauses: list[tuple[int, ...]] = []
        if bootstrap_with is not None:
            self.clauses.extend(_pysat_clauses_from_value(bootstrap_with))

    def __getattr__(self, name: str) -> object:
        return getattr(self._solver, name)

    def __enter__(self) -> "_PySatSolverProxy":
        if hasattr(self._solver, "__enter__"):
            self._solver.__enter__()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> object:
        if hasattr(self._solver, "__exit__"):
            return self._solver.__exit__(exc_type, exc_value, traceback)
        if hasattr(self._solver, "delete"):
            self._solver.delete()
        return None

    def add_clause(self, clause: Iterable[int], *args: object, **kwargs: object) -> object:
        literals = _pysat_normalize_clause(clause)
        self.clauses.append(literals)
        return self._solver.add_clause(list(literals), *args, **kwargs)

    def append_formula(self, formula: object, *args: object, **kwargs: object) -> object:
        clauses = _pysat_clauses_from_value(formula)
        self.clauses.extend(clauses)
        return self._solver.append_formula([list(clause) for clause in clauses], *args, **kwargs)

    def solve(self, *args: object, **kwargs: object) -> bool:
        if self._run_check:
            return self._solver.solve(*args, **kwargs)
        return False

    def solve_limited(self, *args: object, **kwargs: object) -> bool:
        if self._run_check:
            return self._solver.solve_limited(*args, **kwargs)
        return False

    def get_core(self, *args: object, **kwargs: object) -> list[int]:
        if self._run_check:
            return self._solver.get_core(*args, **kwargs)
        return []


def _pysat_solver_classes(pysat_solvers: object) -> tuple[tuple[str, type], ...]:
    classes: list[tuple[str, type]] = []
    for name, value in vars(pysat_solvers).items():
        if isinstance(value, type) and all(hasattr(value, attr) for attr in ("add_clause", "append_formula", "solve")):
            classes.append((name, value))
    return tuple(classes)


def _pysat_bootstrap_with(signature: inspect.Signature, args: tuple[object, ...], kwargs: dict[str, object]) -> object | None:
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        return kwargs.get("bootstrap_with")
    return bound.arguments.get("bootstrap_with")


def _find_pysat_clauses(namespace: dict[str, object]) -> tuple[tuple[int, ...], ...]:
    for name in ("cnf", "formula", "clauses", "assertions"):
        if name in namespace:
            return _pysat_clauses_from_value(namespace[name])
    raise RuntimeError("expected a PySAT CNF-like formula, solver, or clauses list")


def _pysat_clauses_from_value(value: object) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, _PySatSolverProxy):
        return tuple(value.clauses)
    if hasattr(value, "clauses"):
        return tuple(_pysat_normalize_clause(clause) for clause in getattr(value, "clauses"))
    if hasattr(value, "hard") and hasattr(value, "soft"):
        clauses = list(getattr(value, "hard")) + list(getattr(value, "soft"))
        return tuple(_pysat_normalize_clause(clause) for clause in clauses)
    if isinstance(value, (str, bytes)):
        raise RuntimeError(f"expected PySAT clauses, found {value!r}")

    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RuntimeError(f"expected PySAT clauses, found {value!r}") from exc

    if not items:
        return ()
    if all(isinstance(item, int) for item in items):
        return (_pysat_normalize_clause(items),)
    return tuple(_pysat_normalize_clause(clause) for clause in items)


def _pysat_normalize_clause(clause: Iterable[int]) -> tuple[int, ...]:
    literals = tuple(int(literal) for literal in clause)
    if any(literal == 0 for literal in literals):
        raise RuntimeError(f"PySAT literal 0 is not a valid variable literal: {literals}")
    return literals


def _pysat_variable_names(
    namespace: dict[str, object],
    source: object,
    clauses: tuple[tuple[int, ...], ...],
) -> dict[int, str]:
    max_var = max((abs(literal) for clause in clauses for literal in clause), default=0)
    names: dict[int, str] = {}
    for pool in _pysat_id_pool_candidates(namespace, source):
        for var in range(1, max_var + 1):
            if var in names:
                continue
            try:
                obj = pool.obj(var)
            except Exception:
                continue
            if obj is not None:
                names[var] = str(obj)
    return names


def _pysat_id_pool_candidates(namespace: dict[str, object], source: object) -> tuple[object, ...]:
    candidates = list(namespace.values()) + [source]
    pools: list[object] = []
    seen: set[int] = set()

    for candidate in candidates:
        for pool in (candidate, getattr(candidate, "_vpool", None), getattr(candidate, "vpool", None)):
            if pool is None or id(pool) in seen or not callable(getattr(pool, "obj", None)):
                continue
            seen.add(id(pool))
            pools.append(pool)
    return tuple(pools)


def _pysat_clause_to_formula(clause: tuple[int, ...], variable_names: dict[int, str]) -> BoolFormula:
    return _bool_or(*(_pysat_literal_to_formula(literal, variable_names) for literal in clause))


def _pysat_literal_to_formula(literal: int, variable_names: dict[int, str]) -> BoolFormula:
    variable = BoolFormula.var(variable_names.get(abs(literal), f"x_{abs(literal)}"))
    if literal < 0:
        return _bool_not(variable)
    return variable


def _z3_to_formula(expr: object) -> BoolFormula:
    import z3

    if z3.is_true(expr):
        return BoolFormula.const(True)
    if z3.is_false(expr):
        return BoolFormula.const(False)

    kind = expr.decl().kind()
    if z3.is_const(expr) and kind == z3.Z3_OP_UNINTERPRETED:
        return BoolFormula.var(expr.decl().name())

    args = tuple(_z3_to_formula(child) for child in expr.children())
    if kind == z3.Z3_OP_NOT:
        _require_arity("Z3 Not", args, 1)
        return _bool_not(args[0])
    if kind == z3.Z3_OP_AND:
        return _bool_and(*args)
    if kind == z3.Z3_OP_OR:
        return _bool_or(*args)
    if kind == z3.Z3_OP_IMPLIES:
        _require_arity("Z3 Implies", args, 2)
        return _bool_implies(args[0], args[1])
    if kind in (z3.Z3_OP_IFF, z3.Z3_OP_EQ):
        _require_arity("Z3 equivalence", args, 2)
        return _bool_iff(args[0], args[1])
    if kind == z3.Z3_OP_XOR:
        return _bool_xor(*args)
    if kind == z3.Z3_OP_DISTINCT:
        return _bool_and(*(_bool_neq(lhs, rhs) for lhs, rhs in combinations(args, 2)))

    raise RuntimeError(f"unsupported Z3 Boolean expression: {expr}")


def _sympy_to_formula(expr: object) -> BoolFormula:
    import sympy
    from sympy.logic.boolalg import And, Equivalent, Implies, Not, Or, Xor

    if expr is True or expr == sympy.true:
        return BoolFormula.const(True)
    if expr is False or expr == sympy.false:
        return BoolFormula.const(False)
    if isinstance(expr, sympy.Symbol):
        return BoolFormula.var(str(expr))

    args = tuple(_sympy_to_formula(arg) for arg in getattr(expr, "args", ()))
    func = getattr(expr, "func", None)
    if func is Not:
        _require_arity("SymPy Not", args, 1)
        return _bool_not(args[0])
    if func is And:
        return _bool_and(*args)
    if func is Or:
        return _bool_or(*args)
    if func is Implies:
        _require_arity("SymPy Implies", args, 2)
        return _bool_implies(args[0], args[1])
    if func is Equivalent:
        return _chain_iff(args)
    if func is Xor:
        return _bool_xor(*args)

    raise RuntimeError(f"unsupported SymPy Boolean expression: {expr}")


def _pysmt_to_formula(expr: object) -> BoolFormula:
    if expr.is_bool_constant():
        return BoolFormula.const(bool(expr.constant_value()))
    if expr.is_symbol():
        return BoolFormula.var(expr.symbol_name())

    args = tuple(_pysmt_to_formula(arg) for arg in expr.args())
    if expr.is_not():
        _require_arity("PySMT Not", args, 1)
        return _bool_not(args[0])
    if expr.is_and():
        return _bool_and(*args)
    if expr.is_or():
        return _bool_or(*args)
    if expr.is_implies():
        _require_arity("PySMT Implies", args, 2)
        return _bool_implies(args[0], args[1])
    if expr.is_iff() or expr.is_equals():
        _require_arity("PySMT equivalence", args, 2)
        return _bool_iff(args[0], args[1])

    raise RuntimeError(f"unsupported PySMT Boolean expression: {expr}")


def _bool_and(*args: BoolFormula) -> BoolFormula:
    flattened: list[BoolFormula] = []
    for arg in args:
        if arg.op == "const":
            if arg.value is False:
                return BoolFormula.const(False)
            continue
        if arg.op == "and":
            flattened.extend(arg.args)
        else:
            flattened.append(arg)
    if not flattened:
        return BoolFormula.const(True)
    if len(flattened) == 1:
        return flattened[0]
    return BoolFormula("and", tuple(flattened))


def _bool_or(*args: BoolFormula) -> BoolFormula:
    flattened: list[BoolFormula] = []
    for arg in args:
        if arg.op == "const":
            if arg.value is True:
                return BoolFormula.const(True)
            continue
        if arg.op == "or":
            flattened.extend(arg.args)
        else:
            flattened.append(arg)
    if not flattened:
        return BoolFormula.const(False)
    if len(flattened) == 1:
        return flattened[0]
    return BoolFormula("or", tuple(flattened))


def _bool_not(arg: BoolFormula) -> BoolFormula:
    if arg.op == "const":
        return BoolFormula.const(not bool(arg.value))
    if arg.op == "not":
        return arg.args[0]
    return BoolFormula("not", (arg,))


def _bool_xor(*args: BoolFormula) -> BoolFormula:
    if not args:
        return BoolFormula.const(False)
    if len(args) == 1:
        return args[0]
    result = _bool_neq(args[0], args[1])
    for arg in args[2:]:
        result = _bool_neq(result, arg)
    return result


def _bool_implies(lhs: BoolFormula, rhs: BoolFormula) -> BoolFormula:
    return BoolFormula("implies", (lhs, rhs))


def _bool_iff(lhs: BoolFormula, rhs: BoolFormula) -> BoolFormula:
    return BoolFormula("iff", (lhs, rhs))


def _bool_eq(lhs: BoolFormula, rhs: BoolFormula) -> BoolFormula:
    return _bool_iff(lhs, rhs)


def _bool_neq(lhs: BoolFormula, rhs: BoolFormula) -> BoolFormula:
    return _bool_not(_bool_iff(lhs, rhs))


def _bool_at_most_one(*args: BoolFormula) -> BoolFormula:
    return _bool_and(*(_bool_or(_bool_not(lhs), _bool_not(rhs)) for lhs, rhs in combinations(args, 2)))


def _chain_iff(args: tuple[BoolFormula, ...]) -> BoolFormula:
    if not args:
        return BoolFormula.const(True)
    if len(args) == 1:
        return args[0]
    return _bool_and(*(_bool_iff(lhs, rhs) for lhs, rhs in zip(args, args[1:])))


def _bool_clauses(assertions: tuple[BoolFormula, ...]) -> tuple[BoolFormula, ...]:
    clauses: list[BoolFormula] = []
    for assertion in assertions:
        if assertion.op == "and":
            clauses.extend(assertion.args)
        else:
            clauses.append(assertion)
    return tuple(clauses)


def _interleave(separator: str, values: Iterable[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        if items:
            items.append(separator)
        items.append(value)
    return items


def _require_arity(name: str, args: tuple[BoolFormula, ...], expected: int) -> None:
    if len(args) != expected:
        raise RuntimeError(f"{name} expected {expected} arguments, found {len(args)}")


def _find_sympy_assertions(namespace: dict[str, object]) -> tuple[object, ...]:
    for name in ("formula", "boolean_formula", "phi", "constraint", "constraints", "assertions"):
        if name in namespace:
            return _coerce_sympy_assertions(name, namespace[name])

    named_assertions = _find_named_sympy_clause_assertions(namespace)
    if named_assertions:
        return named_assertions

    raise RuntimeError(
        "expected a SymPy formula assigned to one of: "
        "formula, boolean_formula, phi, constraint, constraints, assertions, "
        "or clause-like names such as cond1/c1"
    )


_SYMPY_CLAUSE_NAME_RE = re.compile(r"^(?:cond|condition|clause|assertion|c)(\d+)$", re.IGNORECASE)


def _find_named_sympy_clause_assertions(namespace: dict[str, object]) -> tuple[object, ...]:
    candidates: list[tuple[int, int, str, object]] = []
    prefix_rank = {
        "cond": 0,
        "condition": 0,
        "clause": 1,
        "assertion": 2,
        "c": 3,
    }

    for name, value in namespace.items():
        match = _SYMPY_CLAUSE_NAME_RE.match(name)
        if not match or not _is_sympy_boolean(value):
            continue

        prefix = re.sub(r"\d+$", "", name).lower()
        candidates.append((prefix_rank.get(prefix, 99), int(match.group(1)), name, value))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(value for _, _, _, value in candidates)


def _coerce_sympy_assertions(name: str, value: object) -> tuple[object, ...]:
    if isinstance(value, (list, tuple, set, frozenset)):
        assertions = tuple(value)
    else:
        assertions = (value,)

    invalid = [assertion for assertion in assertions if not _is_sympy_boolean(assertion)]
    if invalid:
        raise RuntimeError(f"{name} contains non-Boolean SymPy values: {invalid!r}")
    return assertions


def _is_sympy_boolean(value: object) -> bool:
    import sympy
    from sympy.logic.boolalg import Boolean

    return isinstance(value, Boolean) or value in (True, False, sympy.true, sympy.false)


def _ortools_constraint_to_formula(constraint: object, var_names: tuple[str, ...]) -> object:
    enforcement_literals = tuple(getattr(constraint, "enforcement_literal", ()))

    if constraint.has_bool_or():
        formula = _bool_or(*(_ortools_literal_to_formula(literal, var_names) for literal in constraint.bool_or.literals))
    elif constraint.has_bool_and():
        formula = _bool_and(*(_ortools_literal_to_formula(literal, var_names) for literal in constraint.bool_and.literals))
    elif constraint.has_bool_xor():
        formula = _bool_xor(*(_ortools_literal_to_formula(literal, var_names) for literal in constraint.bool_xor.literals))
    elif constraint.has_at_most_one():
        formula = _bool_at_most_one(
            *(_ortools_literal_to_formula(literal, var_names) for literal in constraint.at_most_one.literals)
        )
    elif constraint.has_exactly_one():
        literals = tuple(_ortools_literal_to_formula(literal, var_names) for literal in constraint.exactly_one.literals)
        formula = _bool_and(_bool_or(*literals), _bool_at_most_one(*literals))
    elif constraint.has_linear():
        formula = _ortools_linear_to_formula(constraint.linear, var_names)
    else:
        raise RuntimeError(f"unsupported OR-Tools propositional constraint: {constraint}")

    if enforcement_literals:
        enforcement = _bool_and(*(_ortools_literal_to_formula(literal, var_names) for literal in enforcement_literals))
        return _bool_implies(enforcement, formula)
    return formula


def _ortools_literal_to_formula(literal: int, var_names: tuple[str, ...]) -> object:
    if literal >= 0:
        return BoolFormula.var(var_names[literal])
    return _bool_not(BoolFormula.var(var_names[-literal - 1]))


def _ortools_linear_to_formula(linear: object, var_names: tuple[str, ...]) -> object:
    variables = list(linear.vars)
    coeffs = list(linear.coeffs)
    domain = list(linear.domain)

    if len(variables) == 1 and coeffs == [1]:
        variable = BoolFormula.var(var_names[variables[0]])
        if domain == [1, 1]:
            return variable
        if domain == [0, 0]:
            return _bool_not(variable)

    if len(variables) == 2 and abs(coeffs[0]) == 1 and coeffs[0] == -coeffs[1]:
        lhs = BoolFormula.var(var_names[variables[0]])
        rhs = BoolFormula.var(var_names[variables[1]])
        if domain == [0, 0]:
            return _bool_eq(lhs, rhs)
        if (
            not _domain_contains(domain, 0)
            and _domain_contains(domain, -1)
            and _domain_contains(domain, 1)
        ):
            return _bool_neq(lhs, rhs)

    if variables:
        satisfying_terms: list[BoolFormula] = []
        variable_formulas = [BoolFormula.var(var_names[variable]) for variable in variables]
        for assignment in product((False, True), repeat=len(variables)):
            value = sum(coeff for bit, coeff in zip(assignment, coeffs) if bit)
            if _domain_contains(domain, value):
                satisfying_terms.append(
                    _bool_and(
                        *(
                            variable if bit else _bool_not(variable)
                            for variable, bit in zip(variable_formulas, assignment)
                        )
                    )
                )
        return _bool_or(*satisfying_terms)

    return BoolFormula.const(_domain_contains(domain, 0))


def _domain_contains(domain: list[int], value: int) -> bool:
    return any(lower <= value <= upper for lower, upper in zip(domain[0::2], domain[1::2]))


def _require_exactly_one(values: Iterable[object], label: str) -> list[object]:
    unique_values = _unique_by_id(values)
    if len(unique_values) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {len(unique_values)}")
    return unique_values


def _unique_by_id(values: Iterable[object]) -> list[object]:
    seen: set[int] = set()
    unique_values: list[object] = []
    for value in values:
        key = id(value)
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(value)
    return unique_values


def _format_extracted(extracted: ExtractedFormula, output_format: str) -> str:
    if output_format == "formula":
        return str(extracted.conjunction())
    if output_format == "assertions":
        return "\n".join(str(assertion) for assertion in extracted.assertions)
    if output_format == "clauses":
        return "\n".join(str(clause) for clause in extracted.clauses())
    raise ValueError(f"unknown output format: {output_format}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "program",
        nargs="?",
        type=Path,
        help="Path to a Python file containing a backend program. Reads stdin if omitted.",
    )
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default="z3",
        help="Formula backend used by the input program.",
    )
    parser.add_argument(
        "--format",
        choices=("formula", "assertions", "clauses"),
        default="formula",
        help="Output format for the normalized extracted formula.",
    )
    parser.add_argument(
        "--run-check",
        action="store_true",
        help="Allow solver/check calls in the input program to run.",
    )
    parser.add_argument(
        "--no-quiet",
        action="store_true",
        help="Do not suppress stdout/stderr from the input program.",
    )
    args = parser.parse_args(argv)

    if args.program is None:
        program = sys.stdin.read()
    else:
        program = args.program.read_text()

    extracted = extract_formula(program, backend=args.backend, run_check=args.run_check, quiet=not args.no_quiet)
    print(_format_extracted(extracted, args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
