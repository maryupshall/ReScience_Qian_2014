import sympy

from ode_functions.gating import *


def ode_2d(state, t, parameters, exp=np.exp):
    """
    2 dimensional ODE

    :param state: State of the ODE (r.h.s of the ODE vector)
    :param t: Current time
    :param parameters: Parameters
    :param exp: Which exponential function to use, defaults to numpy.exp
    :return: The r.h.s of the ODE
    """

    i_app = parameters['i_app']
    g_na = parameters['g_na']
    g_k = parameters['g_k']
    g_l = parameters['g_l']
    e_na = parameters['e_na']
    e_k = parameters['e_k']
    e_l = parameters['e_l']

    v, h = state
    n = f(h)

    dv = i_app - g_l * (v - e_l) - g_na * (m_inf(v, exp=exp) ** 3) * h * (v - e_na) - g_k * ((n ** 3) * (v - e_k))
    dh = - (h - (h_inf(v, exp=exp))) / (tau_h(v, exp=exp))

    return [dv, dh]


def ode_3d(state, t, parameters, synapse=None, scale=1, exp=np.exp):
    """
    3 dimensional ODE

    :param state: State of the ODE (r.h.s of the ODE vector)
    :param t: Current time
    :param parameters: Parameters
    :param synapse: Optional synapse function
    :param scale: Scaling factor for dhs todo: justify
    :param exp: Which exponential function to use, defaults to numpy.exp
    :return: The r.h.s of the ODE
    """

    i_app = parameters['i_app']
    g_na = parameters['g_na']
    g_k = parameters['g_k']
    g_l = parameters['g_l']
    e_na = parameters['e_na']
    e_k = parameters['e_k']
    e_l = parameters['e_l']

    v, h, hs = state
    n = f(h)

    i_syn = 0
    if synapse is not None:
        g_syn = parameters['g_syn']
        e_syn = parameters['e_syn']

        i_syn = synapse(v, g_syn, e_syn)

    dv = i_app - (g_l * (v - e_l)) - (g_na * (m_inf(v, exp=exp) ** 3) * h * hs * (v - e_na)) - (
            g_k * (n ** 3) * (v - e_k)) - i_syn
    dh = - (h - (h_inf(v, exp=exp))) / (tau_h(v, exp=exp))
    dhs = - (hs - (hs_inf(v, exp=exp))) / (tau_hs(v, exp=exp)) * scale

    return [dv, dh, dhs]


def synaptic_3d(state, t, parameters, func):
    return ode_3d(state, t, parameters, synapse=func)


def ode_5d(state, t, parameters):
    """
    5 dimensional ODE

    :param state: State of the ODE (r.h.s of the ODE vector)
    :param t: Current time
    :param parameters: Parameters
    :return: The r.h.s of the ODE
    """

    i_app = parameters['i_app']
    g_na = parameters['g_na']
    g_k = parameters['g_k']
    g_l = parameters['g_l']
    e_na = parameters['e_na']
    e_k = parameters['e_k']
    e_l = parameters['e_l']

    v, h, hs, m, n = state

    dv = i_app - g_l * (v - e_l) - g_na * (m ** 3) * h * hs * (v - e_na) - g_k * (n ** 3) * (v - e_k)  # 3D model
    dh = - (h - (h_inf(v))) / (tau_h(v))
    dhs = - (hs - (hs_inf(v))) / (tau_hs(v))
    dm = - (m - (m_inf(v))) / (tau_m(v))
    dn = - (n - (n_inf(v))) / (tau_n(v))

    return [dv, dh, dhs, dm, dn]


def voltage_clamp(state, t, parameters, func):
    """
    Perform voltage clamp on any ode

    Voltage is always the first variable in all ODEs

    :param state: State of the ODE (r.h.s of the ODE vector)
    :param t: Current time
    :param parameters: Parameters
    :param func: ODE function to clamp
    :return: The ODE with voltage clamped
    """

    # todo: smart way to handle function included with parameters
    return __clamp__(state, t, parameters, func, 0)


def hs_clamp(state, t, parameters):
    """
    Clamp hs in ode_3d (hs is the 3rd variable)

    :param state: State of the ODE (r.h.s of the ODE vector)
    :param t: Current time
    :param parameters: Parameters
    :return: The 3d ODE with hs clamped
    """

    return __clamp__(state, t, parameters, ode_3d, 2, exp=sympy.exp)  # only used for the bifn so use sympy.exp


def __clamp__(state, t, p, func, ix, exp=np.exp):
    """
    Internal function for clamping an ode variable

    :param state: State of the ODE (r.h.s of the ODE vector)
    :param t: Current time
    :param p: Parameters
    :param func: Which ode function to call
    :param ix: Index of ODE variable to clamp
    :param exp: Which exponential function to use
    :return: The r.h.s. of the ODE with a particular variables d/dt set to 0
    """

    ddt = func(state, t, p, exp=exp)
    ddt[ix] = 0
    return ddt


def default_parameters(i_app=0, g_na=8):
    """
    Generate a dictionary for the default parameters

    :param i_app: Applied current - defaults to 0 if not specified
    :param g_na: Default sodium conductance - defaults to 8 if not specified
    :return: Default parameters
    """

    g_k = 0.6  # mS/cm^2
    e_na = 60  # mV
    e_k = -85  # mV
    e_l = -60  # mV
    g_l = 0.013  # mS/cm^2

    params = {'g_k': g_k,
              'e_k': e_k,
              'e_na': e_na,
              'g_na': g_na,
              'e_l': e_l,
              'g_l': g_l,
              'i_app': i_app}

    return params
