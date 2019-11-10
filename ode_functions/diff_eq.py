import sympy
from scipy.integrate import odeint

from ode_functions.current import sodium_current, total_current
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
    :param scale: Scaling factor for dhs. Used to explore convergence rate in paper
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
        e_syn = 0  # parameters['e_syn'] I'm not sure about this?

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


def pattern_to_window_and_value(pattern: dict, end_time):
    times = list(pattern.keys())
    values = list(pattern.values())

    # make a set of time windows (times[0], times[1]), (times[1], times[2])... (times[-1], end_time)
    window = list(zip(times[:-1], times[1:])) + [(times[-1], end_time)]

    return list(zip(window, values))


def pulse(function, parameter, pattern: dict, end_time, ic, **kwargs):
    """
    Apply a time dependent "pulse" to an ODE system.

    A pulse is a sequence of discrete changes to a constant parameters. The ODE is solved for each value seperately and
    stitched together with the IC of the next step being the end point of the current step

    For example injecting a dc-current at t=1000
    :param function: ODE function to apply pulse to
    :param parameter: Parameter to
    :param pattern: A dictionary of time:value pairs {0: 0, 1000:1} will set the parameter to 0 at t=0 and 1 and t=1000
    :param end_time: Time to end the simulation at
    :param ic: Simulation initial condition
    :param kwargs: Additional parameters to set for default parameters (?)
    :return: The solved continuous ode, the time points and the waveform of the property
    """

    solution = np.array([0] * len(ic))  # needs dummy data to keep shape for vstack
    t_solved = np.array([])
    stimulus = np.array([])

    sequence = pattern_to_window_and_value(pattern, end_time)
    for (t0, t1), value in sequence:  # iterate over time windows and the value
        parameters = default_parameters()
        parameters[parameter] = value  # set the target parameter to a value

        t = np.arange(t0, t1, 0.1)
        t_solved = np.concatenate((t_solved, t))

        state = odeint(function, ic, t, args=(parameters,))
        ic = state[-1, :]  # maintain the initial condition for when this re-initializes at the next step

        solution = np.vstack((solution, state))  # keep track of solution
        stimulus = np.concatenate((stimulus, np.ones(t.shape) * value))

    solution = solution[1:, :]  # first row is [0,0] dummy data so omit

    return solution, t_solved, stimulus


def current_voltage_curve(ode_function, clamp_voltage, time, ic, use_system_hs=True, current_function="PeakNa",
                          follow=False, **kwargs):
    parameters = default_parameters(**kwargs)
    current = np.zeros(clamp_voltage.shape)  # initialize empty IV curve
    for iy, v in enumerate(clamp_voltage):  # for every voltage
        state = odeint(voltage_clamp, ic, time, args=(parameters, ode_function))

        if use_system_hs:
            hs = state[:, 2]
        else:
            hs = 1

        h = state[:, 1]
        if current_function == "PeakNa":
            ss_current = sodium_current(v, m_inf(v), parameters, h=h, hs=hs)  # last time is steady state
        elif current_function == "Balance":
            ss_current = -total_current(v, h, parameters, hs=hs)
        else:
            raise ValueError("Unknown Current")
        if follow:  ## s.s. analysis.
            # this makes the search more effecient for IV curves but breaks depolarization experiments
            ic = [v] + state[-1, 1:].tolist()  # update the ic so it's near f.p
            ss_current = ss_current[-1]
        else:
            pk = np.argmax(np.abs(ss_current))
            ss_current = ss_current[pk]  # take peak (amplitude) current
        current[iy] = ss_current

    return current
