#!/usr/bin/env python
#this module will computer various MLEs for distributions

from __future__ import print_function, division
import sys
import numpy as np
from sympy import *
from sympy.core.compatibility import is_sequence
import operator
import scipy.optimize
from sympy.utilities.autowrap import ufuncify # yo

#infinity = float("inf") # this gives warnings with numpy
infinity = sys.float_info.max


def curve_fit(xvar, expr, xdata, ydata, pars, sigma=1.0, options=None, cache=None):
    """ 'curve_fit()' uses sympy to break the fitting into appropriate
    sub-problems to speed up the fitting process. For example, if there are
    linear parameters in 'expr', then those will be fit with an efficient
    linear regression algorithm, while the non-linear parameters will be fit
    more elaborately.

    The syntax is the similar as for scipy.optimize.curve_fit(), except that
    'pars' is a dictionary of symbol:value pairs, and expr is a sympy
    expression, which needs 'xvar' so that we know what the independent
    variable in 'expr' is.

    For examples, check out the testing section below.

    Currently, this exposes a bug in sympy's ufuncify(), see #2971 at
    https://github.com/sympy/sympy/issues/2971. There is a proposed solution at
    https://github.com/sympy/sympy/pull/3000, which is in the
    'ufuncify_argument_order' branch at https://github.com/hsgg/sympy.git. """

    # get derivatives:
    if cache == None:
        cache = make_cache(xvar, pars.keys(), expr)
    assert xvar == cache['xvar']
    assert equal_list_elements(pars.keys(), cache['variables'])
    assert expr == cache['expr']

    variables = cache['variables']
    jacobian = cache['jacobian']
    hessian = cache['hessian']
    linpars = cache['linpars']
    nonlinpars = cache['nonlinpars']
    funcset = cache['funcset']
    nonlinfunc = cache['nonlinfunc']
    fisherfuncs = cache['fisherfuncs']

    # set nonlinpar values, order matters
    nonlinvals = [pars[p] for p in nonlinpars]

    # fit!
    success = True
    if len(nonlinpars) > 0:
        extraargs = (xdata, ydata, sigma, funcset, nonlinfunc)
        nonlinvals, mlogl, success = fit_nonlinearly(mloglikelihood, nonlinvals,
                extraargs, options)

        # We expect an array later on:
        if not is_sequence(nonlinvals):
            nonlinvals = [nonlinvals]

    assert len(nonlinvals) == len(nonlinpars)

    # regain linear parameters
    try:
        ylin = ydata - nonlinfunc(xdata, *nonlinvals)
        linvals, mlogl, fisher = fit_linear_parameters(nonlinvals, xdata,
                ylin, sigma, funcset)
    except np.linalg.linalg.LinAlgError:
        success = False
        linvals = [0.0] * len(linpars)
        mlogl = infinity

    # reconstruct full parameter list
    parameters = {}
    for p, v in zip(nonlinpars, nonlinvals):
        parameters[p] = v
    for p, v in zip(linpars, linvals):
        parameters[p] = v

    # ensure correct order:
    parametervalues = [parameters[var] for var in variables]
    fisher = calc_fisher(fisherfuncs, parametervalues, xdata, ydata, sigma)

    return parameters, fisher, mlogl, success

def fit_linear_parameters(nonlinvals, x, y, sigma, funcset):
    num_linpar = len(funcset)

    gi = np.empty(shape=(num_linpar, len(y)))
    for i in range(0, num_linpar):
        gi[i,:] = funcset[i](x, *nonlinvals)

    Bi = np.empty(shape=(num_linpar))
    for i in range(0, num_linpar):
        Bi[i] = gi[i].T.dot(sigma).dot(y)

    Aij = np.empty(shape=(num_linpar, num_linpar))
    for i in range(0, num_linpar):
        for j in range(0, num_linpar):
            Aij[i,j] = gi[i].T.dot(sigma).dot(gi[j])

    linvals = np.linalg.solve(Aij, Bi)

    # calculate mlogl here, because we can, and it should make fitting faster
    mlogl = + 0.5 * linvals.T.dot(Aij).dot(linvals) - Bi.T.dot(linvals) + 0.5 * y.T.dot(sigma).dot(y)

    return linvals, mlogl, Aij

def mloglikelihood(nonlinvals, x, y, sigma, funcset, nonlinfunc=None):
    if nonlinfunc == None:
        ylin = y # copy should not be needed
    else:
        ylin = y - nonlinfunc(x, *nonlinvals)

    try:
        linvals, mlogl, fish = fit_linear_parameters(nonlinvals, x, ylin, sigma, funcset)
        return mlogl
    except np.linalg.linalg.LinAlgError:
        return infinity

def fit_nonlinearly(func, nonlinvals, extraargs, options=None):
    method = [
            "Nelder-Mead" # works well
            #"Powell" # fastest, also works well
            #"CG" # sucks
            #"BFGS" # close, but has precision problems
            #"Newton-CG" # Need Jacobian
            #"Anneal" # sucks
            #"L-BFGS-B" # close, but aborted abnormally
            #"TNC" # close, but not sure how to optimize
            #"COBYLA" # sucks
            #"SLSQP" # sucks
            ]

    r = scipy.optimize.minimize(func, nonlinvals, args=extraargs,
            method=method[0], options=options)
    return r.x, r.fun, r.success


def create_fishy_function(fitfunc, jacfunc1, jacfunc2, hessfunc):
    return lambda xdata, ydata, sigma, *parameters: (0.5) * (
            hessfunc(xdata, *parameters).T.dot(sigma).dot(fitfunc(xdata, *parameters) - ydata)
            + jacfunc2(xdata, *parameters).T.dot(sigma).dot(jacfunc1(xdata, *parameters))
            + jacfunc1(xdata, *parameters).T.dot(sigma).dot(jacfunc2(xdata, *parameters))
            + (fitfunc(xdata, *parameters) - ydata).T.dot(sigma).dot(hessfunc(xdata, *parameters)))

def get_fisherfuncs(variables, expr, jacobian, hessian):
    fitfunc = numerical_func(variables, expr)
    jacfunc = numerical_funcdict(variables, jacobian)
    hessfunc = numerical_funcdict(variables, hessian)
    pars = variables[1:]
    fisherfuncs = {}
    for var1 in pars:
        for var2 in pars:
            fisherfuncs[var1, var2] = create_fishy_function(fitfunc, jacfunc[var1], jacfunc[var2], hessfunc[var1, var2])
    return fisherfuncs

def calc_fisher(fisherfuncs, parametervalues, xdata, ydata, sigma):
    fisher = {}
    for key in fisherfuncs.keys():
        fisher[key] = fisherfuncs[key](xdata, ydata, sigma, *parametervalues)
    return fisher

######### helper functions #########
def get_jacobian(expr, variables):
    jacobian = {}
    for var in variables:
        print("Jacobian[", var, "] = ", sep='', end='', file=sys.stderr)
        jacobian[var] = diff(expr, var)
        print(jacobian[var], sep='', file=sys.stderr)
    return jacobian

def get_hessian(expr, variables):
    hessian = {}
    for var1 in variables:
        for var2 in variables:
            print("Hessian[", var1, ",", var2, "] = ", sep='', end='', file=sys.stderr)
            hessian[var1, var2] = diff(expr, var1, var2)
            print(hessian[var1, var2], sep='', file=sys.stderr)
    return hessian

def get_linpars_nonlinpars(hessian, variables):
    """ Decides which variables are linear parameters, and which are non-linear
    parameters. """
    linpars = list(variables)
    nonlinpars = []

    # Mark non-linear parameters one-by-one
    while True:
        # Count how many potential non-linear "cross-talks" there might be:
        crosses = {}
        for var in linpars:
            crosses[var] = 0
            for var2 in linpars:
                if hessian[var, var2] != 0:
                    crosses[var] += 1

        maxcross = max(crosses.iteritems(), key=operator.itemgetter(1))
        if maxcross[1] == 0:
            break # Only linear parameters left.

        # Remove the nonlinear parameter most influencing others:
        linpars.remove(maxcross[0])
        nonlinpars += [maxcross[0]]

        if len(linpars) == 0:
            break

    return linpars, nonlinpars

def make_cache(xvar, variables, expr):
    cache = {'xvar': xvar, 'variables': variables, 'expr': expr}

    jacobian = get_jacobian(expr, variables)
    hessian = get_hessian(expr, variables)
    linpars, nonlinpars = get_linpars_nonlinpars(hessian, variables)
    print("Linear:", linpars, file=sys.stderr)
    print("Nonlinear:", nonlinpars, file=sys.stderr)

    # yeah, ufuncify. yo. order matters.
    funcset = []
    nonlinexpr = expr
    for p in linpars:
        funcset += [numerical_func([xvar] + nonlinpars, jacobian[p])]
        nonlinexpr -= p * jacobian[p]
    nonlinfunc = numerical_func([xvar] + nonlinpars, nonlinexpr)
    if len(nonlinpars) == 0:
        assert nonlinexpr == 0

    cache['jacobian'] = jacobian
    cache['hessian'] = hessian
    cache['linpars'] = linpars
    cache['nonlinpars'] = nonlinpars
    cache['funcset'] = funcset
    cache['nonlinfunc'] = nonlinfunc

    cache['fisherfuncs'] = get_fisherfuncs([xvar] + variables, expr, jacobian, hessian)

    return cache


def numerical_func(arg_list, expr):
    """ Turn an expression into a function. Theano might be an interesting
    option, too. Check this:
    http://docs.sympy.org/latest/modules/numeric-computation.html """
    if expr == 0:
        def zero_func(*args):
            return np.zeros_like(args[0])
        print(arg_list, expr, "...zero-like!", file=sys.stderr)
        return zero_func
    print(arg_list, expr, "...ufuncify!", file=sys.stderr)
    return ufuncify(arg_list, expr)

def numerical_funcdict(arg_list, expr_dict):
    func_dict = {}
    for key in expr_dict.keys():
        func_dict[key] = numerical_func(arg_list, expr_dict[key])
    return func_dict

def convert_matrixdict_to_matrix(matrix, variables):
    mat = np.empty(shape=(len(variables), len(variables)))
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            mat[i, j] = matrix[var1, var2]
    return mat

def invert_matrixdict(matrix, variables):
    mat = convert_matrixdict_to_matrix(matrix, variables)

    mat = np.linalg.inv(mat)

    matrix_result = {}
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            matrix_result[var1, var2] = mat[i, j]

    return matrix_result

############## TESTS #####################
def equal_list_elements(l1, l2):
    if len(l1) != len(l2):
        return False
    for i in l1:
        try: l2.index(i)
        except ValueError: return False
    for i in l2:
        try: l1.index(i)
        except ValueError: return False
    return True

def equal_listorder(l1, l2):
    if len(l1) != len(l2):
        return False
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return False
    return True

def reldiff(a, b):
    return np.abs(2 * (a - b) / (a + b))

def equal_dicts_tol(m1, m2, tol=1e-10):
    for key in m1.keys():
        if reldiff(m1[key], m2[key]) > tol:
            return False
    for key in m2.keys():
        if reldiff(m1[key], m2[key]) > tol:
            return False
    return True

def make_data():
    xdata = np.array([0.0, 1.0, 2.0, 3.0])
    ydata = np.array([3.1, 1.9, 1.03, -0.2])
    return xdata, ydata

def make_gaussdata(normalization=1.0, yerr=0.00001):
    xdata = np.linspace(-2.0, 2.0, 100)
    ydata = np.exp(-0.5 * (xdata-0.3)**2) * normalization / sqrt(2 * np.pi)
    rand = yerr * np.random.randn(len(xdata))
    return xdata, ydata + rand, yerr

def test_linear():
    print("Running test_linear()...")
    m = Symbol('m')
    c = Symbol('c')
    x = Symbol('x')
    expr = m * x + c
    linpars, nonlinpars = get_linpars_nonlinpars(get_hessian(expr, [m, c]), [m, c])
    assert equal_list_elements(linpars, [m, c])
    assert nonlinpars == []
    xdata, ydata = make_data()
    r = curve_fit(x, expr, xdata, ydata, {m: 0.0, c:0.0})
    print(r)
    assert np.abs(r[0][c] - 3.073) < 1e-13
    assert np.abs(r[0][m] + 1.077) < 1e-13
    assert r[-1] == True
    yerr = 1.0
    expected_fisher = {}
    expected_fisher[m, m] = xdata.T.dot(xdata) / yerr**2
    expected_fisher[m, c] = sum(xdata) / yerr**2
    expected_fisher[c, m] = sum(xdata) / yerr**2
    expected_fisher[c, c] = len(xdata) / yerr**2
    print("expected_fisher =", expected_fisher)
    assert equal_dicts_tol(r[1], expected_fisher)

def test_linear_nonlinear():
    print("Running test_linear_nonlinear()...")
    m = Symbol('m')
    c = Symbol('c')
    x = Symbol('x')
    expr = m * x + c * c
    linpars, nonlinpars = get_linpars_nonlinpars(get_hessian(expr, [m, c]), [m, c])
    assert linpars == [m]
    assert nonlinpars == [c]
    xdata, ydata = make_data()
    r = curve_fit(x, expr, xdata, ydata, {m: 0.0, c:0.0})
    print(r)
    assert np.abs(r[0][c]**2 - 3.073) < 1e-4
    assert np.abs(r[0][m] + 1.077) < 1e-4
    assert r[-1] == True

def test_cross_linear():
    print("Running test_cross_linear()...")
    m = Symbol('m')
    c = Symbol('c')
    x = Symbol('x')
    expr = m * c * x + c
    linpars, nonlinpars = get_linpars_nonlinpars(get_hessian(expr, [c, m]), [c, m])
    assert linpars == [m] 
    assert nonlinpars == [c]
    xdata, ydata = make_data()
    r = curve_fit(x, expr, xdata, ydata, {m: 0.0, c:0.1})
    print(r)
    assert np.abs(r[0][c] - 3.073) < 1e-4
    assert np.abs(r[0][m]*r[0][c] + 1.077) < 1e-4
    assert r[-1] == True

def test_nonlinear():
    print("Running test_nonlinear()...")
    m = Symbol('m')
    c = Symbol('c')
    x = Symbol('x')
    expr = log(m) * x + c * c
    linpars, nonlinpars = get_linpars_nonlinpars(get_hessian(expr, [m, c]), [m, c])
    assert equal_list_elements(linpars, [])
    assert equal_list_elements(nonlinpars, [m, c])
    xdata, ydata = make_data()
    r = curve_fit(x, expr, xdata, ydata, {m: 0.0, c:0.0})
    print(r)
    assert np.abs(r[0][c]**2 - 3.073) < 1e-4
    assert np.abs(np.log(r[0][m]) + 1.077) < 1e-4
    assert r[-1] == True

def test_gaussian():
    print("Running test_gaussian()...")
    x = Symbol('x')
    b = Symbol('b')
    mu = Symbol('mu')
    sigma = Symbol('sigma')
    expr = (b / sqrt(2 * pi * sigma**2)) * exp(-0.5 * (x-mu)**2 / sigma**2)
    linpars, nonlinpars = get_linpars_nonlinpars(get_hessian(expr, [b, mu, sigma]), [b, mu, sigma])
    assert equal_list_elements(linpars, [b])
    assert equal_list_elements(nonlinpars, [mu, sigma])
    xdata, ydata, yerr = make_gaussdata()
    r = curve_fit(x, expr, xdata, ydata, {b: 0.0, mu:0.0, sigma:1.0}, sigma=1/yerr**2)
    print(r)
    assert np.abs(r[0][sigma] - 1.0) < 1e-4
    assert np.abs(r[0][b] - 1.0) < 1e-4
    assert np.abs(r[0][mu] - 0.3) < 1e-4
    assert r[-1] == True

def test_linear_fisher():
    print("Running test_linear_fisher()...")
    x = Symbol('x')
    m = Symbol('m')
    c = Symbol('c')
    expr = m * x + c
    yerr = 1.0
    xdata = np.linspace(0.0, 100.0, 1000000)
    ydata = -0.345 * xdata + 33.0 + yerr * np.random.randn(len(xdata))
    r = curve_fit(x, expr, xdata, ydata, {m:0.0, c:0.0}, sigma=1/yerr**2)
    print(r)
    assert reldiff(r[0][m], -0.345) < 1e-3
    assert reldiff(r[0][c], 33.0) < 1e-3
    assert r[-1] == True
    expected_fisher = {}
    expected_fisher[m, m] = xdata.T.dot(xdata) / yerr**2
    expected_fisher[m, c] = sum(xdata) / yerr**2
    expected_fisher[c, m] = sum(xdata) / yerr**2
    expected_fisher[c, c] = len(xdata) / yerr**2
    print("expected_fisher =", expected_fisher)
    assert equal_dicts_tol(r[1], expected_fisher)


def test_cache_variables_order():
    print("Running test_cache_variables_order()...")
    x, m, c = symbols('x m c')
    expr = m * x + c
    xdata, ydata = make_data()
    cache = make_cache(x, [m, c], expr)
    r1 = curve_fit(x, expr, xdata, ydata, {m:0.0, c:0.0}, cache=cache)
    r2 = curve_fit(x, expr, xdata, ydata, {c:0.0, m:0.0}, cache=cache)
    print("r1 =", r1)
    print("r2 =", r2)
    assert equal_dicts_tol(r1[0], r2[0])
    assert equal_dicts_tol(r1[1], r2[1])
    assert np.abs(r1[2] - r2[2]) < 1e-10
    assert r1[3]
    assert r2[3]

def test_singular_matrix():
    print("Running test_singular_matrix()...")
    x, b, mu, sigma = symbols('x b mu sigma')
    expr = (b / sqrt(2 * pi * sigma**2)) * exp(-0.5 * (x-mu)**2 / sigma**2)
    xdata, ydata, yerr = make_gaussdata()
    # Make sure the initial guess is so far off that the matrix will be
    # singular:
    initpars = {b: 0.0, mu:-1e12, sigma:1.0}
    r = curve_fit(x, expr, xdata, ydata, initpars, sigma=1/yerr**2)
    print(r)
    if r[3] != False:
        raise Exception("This test may be ineffective. curve_fit() should fail,"
                + "but claims it didn't.")


def test_ufuncify_argument_order():
    print("Running test_ufuncify_argument_order()...")
    # if this test fails, your sympy version is too old
    a, b, c = symbols('a, b, c')
    expr = a + b - c
    fabc = ufuncify([a, b, c], expr)
    facb = ufuncify([a, c, b], expr)
    x = fabc(0.5, 1.1, 2.2) # = -0.6
    y = facb(0.5, 1.1, 2.2) # =  1.6
    print("difference:", x + 0.6, y - 1.6)
    print("reldiff:   ", reldiff(x, -0.6), reldiff(y, 1.6))
    assert reldiff(x, -0.6) <= 2.0**(-52)
    assert reldiff(y,  1.6) <= 2.0**(-52)

def test_invert_matrixdict():
    print("Running test_invert_matrixdict()...")
    variables = symbols('x y z v w s t')
    mat = {}
    for var1 in variables:
        for var2 in variables:
            mat[var1, var2] = np.random.randn()
    matinv = invert_matrixdict(mat, variables)
    matinvinv = invert_matrixdict(matinv, variables)
    assert equal_dicts_tol(mat, matinvinv)
    assert not equal_dicts_tol(mat, matinv)


if __name__ == "__main__":
    test_invert_matrixdict()
    print()
    test_ufuncify_argument_order()
    print()
    test_linear()
    print()
    test_linear_nonlinear()
    print()
    test_cross_linear()
    print()
    test_nonlinear()
    print()
    test_gaussian()
    print()
    test_linear_fisher()
    print()
    test_cache_variables_order()
    print()
    test_singular_matrix()
    print()
    print("All tests passed.")
