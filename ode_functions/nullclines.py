from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton

from ode_functions.diff_eq import h_inf, f, m_inf, default_parameters


def nullcline_h(v):
    """
    h nullcline

    Simply a call to h_inf as they're the same. This function provides a similar naming as to nullcline_v

    :param v: Membrane potential
    :return: h nullcline
    """

    return h_inf(v)


def nullcline_v(v, i_app, hs=1):
    """
    Compute the v nullcline

    Computes the v nullcline for all values of v specified via newton's method

    :param v: Membrane potential
    :param i_app: Applied current
    :param hs: hs variable
    :return: v nullcline
    """

    nullcline = np.zeros((len(v),))
    parameters = default_parameters(i_app=i_app)

    """Find self-consistent h value for every v on the nullcline"""
    for ix, _ in enumerate(v):
        solvable_nullcline = partial(__nullcline_v_implicit__, v[ix], parameters, hs)  # make a function that takes h
        nullcline[ix] = newton(solvable_nullcline, x0=0)

    return nullcline


def intersect(fa, fb):
    """
    Compute intersection between 2 curves lazily (where the two functions are closest).

    This is a helper function for plotting intersections of nullclines. This is not a true intersection

    :param fa: Numeric values for function a
    :param fb: Numeric values for function b
    :return: Intersection index
    """

    return np.argmin(np.abs(fa - fb))


def __nullcline_v_implicit__(v, parameters, hs, h):
    """
    Implicit form of the v nullcline evaluated at h, v, and hs

    :param v: Membrane potential
    :param h: h gating variable
    :param hs: hs gating variable
    :param parameters: Parameters
    :return: v nullcline in implicit form
    """

    i_app = parameters['i_app']
    g_na = parameters['g_na']
    g_k = parameters['g_k']
    g_l = parameters['g_l']
    e_na = parameters['e_na']
    e_k = parameters['e_k']
    e_l = parameters['e_l']

    return i_app - g_l * (v - e_l) - g_k * (f(h) ** 3) * (v - e_k) - g_na * h * hs * (m_inf(v) ** 3) * (v - e_na)


def nullcline_figure(v, i_app, stability, hs=1, h_color='black', v_color='grey'):
    """
    Helper function for creating nullcline figure

    :param v: Set of voltages to create nullcline for
    :param i_app: Injected current
    :param stability: Whether or not the intersection is stable
    :param hs: Optional parameter for value of hs on nullcline: defaults to 1
    :param h_color: Optional color for the h_nullcline color: defaults to black
    :param v_color: Optional color for the v_nullcline color: defaults to grey
    :return: None
    """

    """Extract and plot the h nullcline"""
    nh = nullcline_h(v)
    plt.plot(v, nh, h_color, zorder=-1000)  # large negative zorder forces plotting on bottom

    """Extract and plot the v nullcline"""
    nv = nullcline_v(v, i_app, hs=hs)
    plt.plot(v, nv, v_color)

    """Lazily compute intersection and plot the stability (where closest only: not true intersection) """
    cross_index = intersect(nh, nv)
    style = 'k' if stability else 'none'
    plt.scatter(v[cross_index], nv[cross_index], edgecolors='k', facecolor=style, zorder=1000)
