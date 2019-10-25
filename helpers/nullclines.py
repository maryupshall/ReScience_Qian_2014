import numpy as np
from scipy.optimize import newton

from ode_functions.diff_eq import h_inf, f, m_inf, default_parameters


def nullcline_h(v):
    return h_inf(v)


def nullcline_v(v, i_app, hs=1):
    nv = np.zeros((len(v),))
    parameters = default_parameters(i_app=i_app)
    for i in range(len(v)):
        h_solve = newton(lambda h: __nullcline_v_implicit__(h, v[i], parameters, hs=hs), 0)
        nv[i] = h_solve

    return nv


def __nullcline_v_implicit__(h, v, parameters, hs=1):
    p = parameters
    i_app, g_na, g_k, g_l, e_na, e_k, e_l, _ = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7:]
    return i_app - g_l * (v - e_l) - g_k * (f(h) ** 3) * (v - e_k) - g_na * h * hs * (m_inf(v) ** 3) * (v - e_na)
