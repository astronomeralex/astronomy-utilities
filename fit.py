#!/usr/bin/env python
#this module will computer various MLEs for distributions

from __future__ import print_function, division
import sys
import numpy as np
from sympy import *
import operator
import scipy.optimize
from sympy.utilities.autowrap import ufuncify # yo

def curve_fit(xvar, expr, xdata, ydata, pars, sigma=1.0, options=None):
    """ 'curve_fit()' uses sympy to break the fitting into appropriate
    sub-problems to speed up the fitting process. For example, if there are
    linear parameters in 'expr', then those will be fit with an efficient
    linear regression algorithm, while the non-linear parameters will be fit
    more elaborately.

    The syntax is the similar as for scipy.optimize.curve_fit(), except that
    'pars' is a dictionary of symbol:value pairs, and expr is a sympy
    expression, which needs 'xvar' so that we know what the independent
    variable in 'expr' is.

    For examples, check out the testing section below. """

    # get derivatives:
    variables = pars.keys()
    jacobian = get_jacobian(expr, variables)
    hessian = get_hessian(expr, variables)
    linpars, nonlinpars = get_linpars_nonlinpars(hessian, pars.keys())

    # yeah, ufuncify. yo. order matters.
    funcset = []
    nonlinexpr = expr
    for p in linpars:
        funcset += [numerical_func([xvar] + nonlinpars, jacobian[p])]
        nonlinexpr -= p * jacobian[p]
    nonlinfunc = numerical_func([xvar] + nonlinpars, nonlinexpr)

    if len(nonlinpars) == 0:
        assert nonlinexpr == 0

    # set nonlinpar values, order matters
    nonlinvals = []
    for p in nonlinpars:
        nonlinvals += [pars[p]]

    # fit!
    success = True
    if len(nonlinpars) > 0:
        extraargs = (xdata, ydata, sigma, funcset, nonlinfunc)
        nonlinvals, mlogl, success = fit_nonlinearly(mloglikelihood, nonlinvals,
                extraargs, options)
        # We except an array later on:
        try:
            length = len(nonlinvals)
        except TypeError:
            nonlinvals = [nonlinvals]
            length = 1

    # regain linear parameters
    try:
        linvals, mlogl, fisher = fit_linear_parameters(nonlinvals, xdata,
                ydata - nonlinfunc(xdata, *nonlinvals), sigma, funcset)
    except np.linalg.linalg.LinAlgError:
        success = False

    # reconstruct full parameter list
    parameters = {}
    for p, v in zip(nonlinpars, nonlinvals):
        parameters[p] = v
    for p, v, in zip(linpars, linvals):
        parameters[p] = v

    # provide function
    fitexpr = simplify(expr.subs(parameters))

    return parameters, fitexpr, mlogl, success

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

def mloglikelihood(nonlinvals, x, y, sigma, funcset, nonlinfunc):
    ylin = y - nonlinfunc(x, *nonlinvals)
    linvals, mlogl, fish = fit_linear_parameters(nonlinvals, x, ylin, sigma, funcset)
    #print(mlogl, "linvals =", linvals, " nonlinpar =", nonlinvals)
    return mlogl

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


######### helper functions #########
def get_jacobian(expr, variables):
    jacobian = {}
    for var in variables:
        print("Jacobian[", var, "] = ", sep='', end='', file=sys.stderr)
        jacobian[var] = simplify(diff(expr, var))
        print(jacobian[var], sep='', file=sys.stderr)
    return jacobian

def get_hessian(expr, variables):
    hessian = {}
    for var1 in variables:
        for var2 in variables:
            print("Hessian[", var1, ",", var2, "] = ", sep='', end='', file=sys.stderr)
            hessian[var1, var2] = simplify(diff(expr, var1, var2))
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

def numerical_func(arg_list, expr):
    """ Turn an expression into a function. Theano might be an interesting
    option, too. Check this:
    http://docs.sympy.org/latest/modules/numeric-computation.html """
    print(arg_list, expr, "...ufuncify!")
    return ufuncify(arg_list, expr)

############## TESTS #####################
def equal_lists(l1, l2):
    if len(l1) != len(l2):
        return False
    for i in l1:
        try: l2.index(i)
        except ValueError: return False
    for i in l2:
        try: l1.index(i)
        except ValueError: return False
    return True

def make_data():
    xdata = np.array([0.0, 1.0, 2.0, 3.0])
    ydata = np.array([3.1, 1.9, 1.03, -0.2])
    return xdata, ydata

def test_linear():
    print("Running test_linear()...")
    m = Symbol('m')
    c = Symbol('c')
    x = Symbol('x')
    expr = m * x + c
    linpars, nonlinpars = get_linpars_nonlinpars(get_hessian(expr, [m, c]), [m, c])
    assert equal_lists(linpars, [m, c])
    assert nonlinpars == []
    xdata, ydata = make_data()
    r = curve_fit(x, expr, xdata, ydata, {m: 0.0, c:0.0})
    print(r)

def test_linear_nonlinear():
    print()
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

def test_cross_linear():
    print()
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

def test_nonlinear():
    print()
    print("Running test_nonlinear()...")
    m = Symbol('m')
    c = Symbol('c')
    x = Symbol('x')
    expr = log(m) * x + c * c
    linpars, nonlinpars = get_linpars_nonlinpars(get_hessian(expr, [m, c]), [m, c])
    assert equal_lists(linpars, [])
    assert equal_lists(nonlinpars, [m, c])
    xdata, ydata = make_data()
    r = curve_fit(x, expr, xdata, ydata, {m: 0.0, c:0.0})
    print(r)

if __name__ == "__main__":
    test_linear()
    test_linear_nonlinear()
    test_cross_linear()
    test_nonlinear()
