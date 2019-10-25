from ode_functions.gating import *
import sympy


def ode_2d(state, t, parameters, exp=np.exp):
    i_app, g_na, g_k, g_l, e_na, e_k, e_l = parameters
    v, h = state
    n = f(h)

    dv = i_app - g_l * (v - e_l) - g_na * (m_inf(v, exp=exp) ** 3) * h * (v - e_na) - g_k * ((n ** 3) * (v - e_k))
    dh = - (h - (h_inf(v, exp=exp))) / (tau_h(v, exp=exp))

    return [dv, dh]


def ode_3d(state, t, parameters, synapse=None, scale=1, exp=np.exp):
    p = parameters
    i_app, g_na, g_k, g_l, e_na, e_k, e_l, p_synapse = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7:]
    v, h, hs = state
    n = f(h)

    i_syn = 0 if synapse is None else synapse(v, p_synapse)

    dv = i_app - (g_l * (v - e_l)) - (g_na * (m_inf(v, exp=exp) ** 3) * h * hs * (v - e_na)) - (
            g_k * (n ** 3) * (v - e_k)) - i_syn
    dh = - (h - (h_inf(v, exp=exp))) / (tau_h(v, exp=exp))
    dhs = - (hs - (hs_inf(v, exp=exp))) / (tah_hs(v, exp=exp)) * scale

    return [dv, dh, dhs]


def synaptic_3d(state, t, parameters):
    p, func = parameters[:-1], parameters[-1]

    return ode_3d(state, t, p, synapse=func)


def ode_5d(state, t, parameters):
    i_app, g_na, g_k, g_l, e_na, e_k, e_l = parameters
    v, h, hs, m, n = state

    dv = i_app - g_l * (v - e_l) - g_na * (m ** 3) * h * hs * (v - e_na) - g_k * (n ** 3) * (v - e_k)  # 3D model
    dh = - (h - (h_inf(v))) / (tau_h(v))
    dhs = - (hs - (hs_inf(v))) / (tah_hs(v))
    dm = - (m - (m_inf(v))) / (tau_m(v))
    dn = - (n - (n_inf(v))) / (tau_n(v))

    return [dv, dh, dhs, dm, dn]


def voltage_clamp(state, t, parameters):
    p, func = parameters[:-1], parameters[-1]
    return __clamp__(state, t, p, func, 0)


def hs_clamp(state, t, parameters):
    return __clamp__(state, t, parameters, ode_3d, 2, exp=sympy.exp)


def __clamp__(state, t, p, func, ix, exp=np.exp):
    ddt = func(state, t, p, exp=exp)
    ddt[ix] = 0
    return ddt


def default_parameters(i_app=0, g_na=8):
    g_k = 0.6  # mS/cm^2
    e_na = 60  # mV
    e_k = -85  # mV
    e_l = -60  # mV
    g_l = 0.013  # mS/cm^2

    return [i_app, g_na, g_k, g_l, e_na, e_k, e_l]
