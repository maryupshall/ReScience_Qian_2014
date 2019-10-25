import numpy as np

from ode_functions.gating import f, m_inf


def sodium_current(v, m, parameters, h=1, hs=1):
    g_na = parameters[1]
    e_na = parameters[4]
    return g_na * (v - e_na) * (m ** 3) * h * hs


def total_current(v, h, parameters, hs=1):
    p = parameters
    i_app, g_na, g_k, g_l, e_na, e_k, e_l, _ = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7:]

    return - (g_l * (v - e_l)) - (g_na * (m_inf(v) ** 3) * h * hs * (v - e_na)) - (
            g_k * (f(h[-1]) ** 3) * (v - e_k))


def nmda_current(v, p, mg=1.4):
    g_syn, e_syn = p
    return (g_syn * (v - e_syn)) / (1 + (mg / 3.57) * np.exp(0.062 * v))


def ampa_current(v, p):
    g_syn, e_syn = p
    return g_syn * (v - e_syn)
