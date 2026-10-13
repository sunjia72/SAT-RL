import subprocess
import tempfile
import os
import re
from pysat.formula import CNF
from pysat.solvers import Solver as PySolver
import z3
from z3.z3util import get_vars



def _count_z3_models(self):
    # collect all Int vars from assertions
    vars_set = set()
    for a in self.assertions():
        vars_set.update(z3.z3util.get_vars(a))
    vars_list = list(vars_set)

    # fresh solver with same assertions
    s = z3.Solver()
    s.append(self.assertions())

    count = 0
    while s.check() == z3.sat:
        m = s.model()
        count += 1
        # block this exact model
        s.add(z3.Or(*[v != m[v] for v in vars_list]))
    return count

# monkey-patch z3.Solver
z3.Solver.count_models = _count_z3_models




# ─── PySAT CNF + Ganak ──────────────────────────────────────────────────────────

def _count_models_from_cnf(self, method: str = 'naive', solver_name: str = None) -> int:
    """
    Count models of this CNF, either by:
      - 'ganak':  writing it to a temp file and calling the Ganak counter
      - 'naive':  brute‐force enumeration via a PySAT Solver
    """
    if method == 'ganak':
        # write to disk
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.cnf', delete=False) as tmp:
            self.to_file(tmp.name)
            path = tmp.name
        try:
            res = subprocess.run(
                ['ganak', path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # parse "c s exact arb int <num>"
            for line in res.stdout.splitlines():
                m = re.match(r'^c s exact arb int (\d+)', line.strip())
                if m:
                    return int(m.group(1))
            raise RuntimeError("Ganak did not report a solution count.")
        finally:
            os.remove(path)

    elif method == 'naive':
        # build a solver and enumerate
        solver = PySolver(name=solver_name) if solver_name else PySolver()
        for clause in self.clauses:
            solver.add_clause(clause)

        count = 0
        while solver.solve():
            model = solver.get_model()
            count += 1
            # block this exact model
            solver.add_clause([-lit for lit in model])

        solver.delete()
        return count

    else:
        raise ValueError(f"Unknown method '{method}', use 'ganak' or 'naive'.")

CNF.count_models = _count_models_from_cnf




def count_dimacs_cnf_models(cnf_content: str) -> int:
    """
    Runs the Ganak model counter on a CNF formula provided as a string
    in DIMACS format and returns the exact model count.

    Parameters:
    ----------
    cnf_content : str
        A string containing the CNF formula in DIMACS format.

    Returns:
    -------
    int
        The number of satisfying assignments (models) of the formula.

    Raises:
    ------
    RuntimeError
        If Ganak does not return a valid model count.

    Example:
    -------
    >>> cnf_str = \"\"\"
    ... p cnf 2 2
    ... 1 -2 0
    ... -1 2 0
    ... \"\"\"
    >>> run_ganak_on_cnf(cnf_str)
    3
    """
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.cnf', delete=False) as tmp_file:
        tmp_file.write(cnf_content)
        tmp_file_path = tmp_file.name

    try:
        result = subprocess.run(
            ['ganak', tmp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )

        for line in result.stdout.splitlines():
            match = re.match(r'^c s exact arb int (\d+)', line.strip())
            if match:
                return int(match.group(1))

        raise RuntimeError("Number of solutions not found in Ganak output.")
    finally:
        os.remove(tmp_file_path)