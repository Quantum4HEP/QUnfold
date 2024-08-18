import sympy
import numpy as np
from hypothesis import settings, given
from hypothesis import strategies as st
from qunfold.utils import approx_hessian


deg = st.integers(min_value=0, max_value=4)
xi = st.integers(min_value=0, max_value=1e6)


@settings(deadline=None)
@given(degrees=st.tuples(deg, deg, deg), point=st.tuples(xi, xi, xi))
def test_approx_hessian(degrees, point):
    varlist = [sympy.Symbol(f"x{i}") for i in range(len(degrees))]
    poly_sympy = 0.0
    for deg, var in zip(degrees, varlist):
        for exp in range(deg + 1):
            coeff = np.random.rand() * 5 - 2
            poly_sympy += coeff * var**exp
    n = len(varlist)
    hess_sympy = sympy.Matrix(np.zeros(shape=(n, n)))
    for i in range(n):
        for j in range(n):
            hess_sympy[i, j] = sympy.diff(poly_sympy, varlist[i], varlist[j])
    func = sympy.lambdify(args=varlist, expr=poly_sympy, modules="numpy")
    hess = sympy.lambdify(args=varlist, expr=hess_sympy, modules="numpy")
    hessian1 = approx_hessian(f=lambda x: func(*x), x=point)
    hessian2 = hess(*point)
    assert np.allclose(hessian1, hessian2)
