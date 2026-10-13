import os
import tempfile

from run_smt_binaries.counter import *
from pysat.formula import CNF
from pysat.solvers import Solver
import z3


def test_count_z3_models():
    # Z3: Boolean model counting
    x, y = z3.Bools('x y')
    s = z3.Solver()
    s.add(z3.Or(x, y), z3.Or(z3.Not(x), y))

    count = s.count_models()
    print(f"Z3 model count: {count}")
    assert count == 2




def test_count_pysat_cnf_models():
    # Programmatic CNF
    cnf = CNF()
    cnf.append([1, -2])
    cnf.append([-1, 2])

    # CNF-level interface
    default_count = cnf.count_models()
    print(f"PySAT CNF default (ganak) count: {default_count}")
    assert default_count == 2

    naive_count = cnf.count_models(method='naive')
    print(f"PySAT CNF naive count: {naive_count}")
    assert naive_count == 2






if __name__ == '__main__':
    test_count_z3_models()
    test_count_pysat_cnf_models()
    print('All tests passed.')
