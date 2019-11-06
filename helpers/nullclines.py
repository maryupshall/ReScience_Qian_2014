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
    v nullcline

    Computes the v nullcline for all values of v specified via newton's method

    :param v: Membrane potential
    :param i_app: Applied current
    :param hs: hs variable
    :return: v nullcline
    """

    nullcline = np.zeros((len(v),))
    parameters = default_parameters(i_app=i_app)
    for i in range(len(v)):
        h_solve = newton(lambda h: __nullcline_v_implicit__(h, v[i], parameters, hs=hs), 0)
        nullcline[i] = h_solve

    return nullcline


def __nullcline_v_implicit__(h, v, parameters, hs=1):
    """
    Implicit form of the v nullcline evaluated at h, v, and hs

    :param h: h gating variable
    :param v: Membrane potential
    :param parameters: Parameters
    :param hs: hs gating variable
    :return: Implicit nullcline
    """

    i_app = parameters['i_app']
    g_na = parameters['g_na']
    g_k = parameters['g_k']
    g_l = parameters['g_l']
    e_na = parameters['e_na']
    e_k = parameters['e_k']
    e_l = parameters['e_l']

    return i_app - g_l * (v - e_l) - g_k * (f(h) ** 3) * (v - e_k) - g_na * h * hs * (m_inf(v) ** 3) * (v - e_na)
