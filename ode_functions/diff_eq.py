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


def ode_5d(state, t, parameters, shift=60):
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
    dn = - (n - (n_inf(v))) / (tau_n(v, shift=shift))

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


def pattern_to_window_and_value(pattern, end_time):
    """
    Convert a pattern dictionary to an iterable tuple of time windows (start, stop) and parameter values

    :param pattern: Pattern dict formatted like {t_event:value}
    :param end_time: End of the simulation
    :return: Iterable pattern of windows and values
    """

    times = list(pattern.keys())
    values = list(pattern.values())

    """make a set of time windows (times[0], times[1]), (times[1], times[2])... (times[-1], end_time)"""
    window = list(zip(times[:-1], times[1:])) + [(times[-1], end_time)]

    """zip window and parameter value for convenient unpacking"""
    return list(zip(window, values))


def pulse(function, parameter, pattern: dict, end_time, ic, clamp_function=None, **kwargs):  # todo refactor for clamp
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
    :param clamp_function: Optional function to clamp voltage for
    :param kwargs: Additional parameters to set for default parameters (?)
    :return: The solved continuous ode, the time points and the waveform of the property
    """

    solution = np.array([0] * len(ic))  # needs dummy data to keep shape for vstack
    t_solved = np.array([])
    stimulus = np.array([])

    sequence = pattern_to_window_and_value(pattern, end_time)
    for (t0, t1), value in sequence:  # iterate over time windows and the value
        parameters = default_parameters(**kwargs)
        if parameter is not None:  # voltage clamp does not set a parameter
            parameters[parameter] = value  # set the target parameter to a value

        t = np.arange(t0, t1, 0.1)
        t_solved = np.concatenate((t_solved, t))
        if clamp_function is None:
            state = odeint(function, ic, t, args=(parameters,))
        else:
            ic[0] = value
            state = odeint(function, ic, t, args=(parameters, clamp_function))

        ic = state[-1, :]  # maintain the initial condition for when this re-initializes at the next step

        solution = np.vstack((solution, state))  # keep track of solution
        stimulus = np.concatenate((stimulus, np.ones(t.shape) * value))

    solution = solution[1:, :]  # first row is [0,0] dummy data so omit

    return solution, t_solved, stimulus


def compute_iv_current(solution, parameters, follow):
    """
    Compute the current of the IV curve at a given clamp voltage given the follow rule

    If follow mode: I(V) is the steady state current, otherwise we use the peak amplitude current of the transient

    :param solution: odeint solution
    :param parameters: Model parameters
    :param follow: Follow mode or peak mode (True/False)
    :return: I(V)
    """

    if follow:
        return -total_current(solution, parameters)[-1]  # [-1] give steady state
    else:
        i_na = sodium_current(solution, parameters)
        pk = np.argmax(np.abs(i_na))  # intermediate pk allows for sign preservation
        return i_na[pk]


def current_voltage_curve(ode_function, clamp_voltage, time, ic, follow=False, **kwargs):
    """
    Compute IV curve in either follow mode or peak mode

    In follow mode a traditional IV curve is computed where the I(V) is the steady state current at a clamped voltage
    for efficiency the initial condition of the next voltage level is the steady state of the present clamp.

    In peak mode (follow = False) the voltage is held at some reset voltage. The voltage is the clamped at the voltage
    for the IV curve and the peak (transient) current is used

    :param ode_function: The function to call for voltage_clamp (the model used)
    :param clamp_voltage: Voltages to clamp during the IV curve
    :param time: Time steps to solve the ode
    :param ic: Initial condition or reset condition
    :param follow: Optional flag is follow model or peak mode is used: defaults to peak mode, follow=False
    :param kwargs: Optional settings for the parameters such as g_na or i_leak
    :return: I(V)
    """

    """Initialize IV curve"""
    parameters = default_parameters(**kwargs)
    current = np.zeros(clamp_voltage.shape)
    state = np.array([ic])  # inital state is ic; array([ic]) gives a 2d array

    """Update model inital state according to IV curve type, run voltage clamp and save I(V)"""
    for ix, v in enumerate(clamp_voltage):
        ic = update_ic(v, ic, state, follow)
        state = odeint(voltage_clamp, ic, time, args=(parameters, ode_function))
        current[ix] = compute_iv_current(state, parameters, follow)

    return current


def update_ic(v, ic, state, follow):
    """
    Set the initial condition for the IV curve at a particular clamp voltage depending on the mode

    If follow mode then the clamp voltage is combined with the state. Otherwise the clamp voltage is combined with the
    reset ic

    :param v: Clamp voltage
    :param ic: Reset conditions
    :param state: Steady state
    :param follow: Follow mode or peak mode (true/false)
    :return: Updated ic according to follow rule
    """

    if follow:
        return np.concatenate([[v], state[-1, 1:]])  # ic = [set voltage, ss of old simulation]
    else:
        return np.concatenate([[v], ic[1:]])  # ic = [set voltage, ss of holding]


def clamp_steady_state(v_clamp):
    """
    Determine the steady-state of ode_3d when clamped to a voltage

    :param v_clamp: Voltage to clamp to
    :return: Steady state of the neuron at v_clamp
    """
    state = odeint(voltage_clamp, [v_clamp, 1, 1], [0, 500], args=(default_parameters(), ode_3d))
    return state[-1, :]
