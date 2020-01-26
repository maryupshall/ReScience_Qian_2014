"""Monolithic module for solving all ODE problems.

Provides functionality to create 2,3,5D ode systems. Clamping variables, and applying discrete current pulses.
"""
import sympy
from scipy.integrate import odeint

from ode_functions.current import sodium_current, total_current
from ode_functions.gating import *


def ode_2d(state, _, parameters, exp=np.exp):
    """Compute the two dimensional ODE.

    :param state: State of the ODE (r.h.s of the ODE vector)
    :param _: Current time
    :param parameters: Parameters
    :param exp: Which exponential function to use, defaults to numpy.exp
    :return: The r.h.s of the ODE
    """
    i_app = parameters["i_app"]
    v, h = state

    dv = i_app + total_current(state, parameters, exp=exp)
    dh = -(h - (h_inf(v, exp=exp))) / (tau_h(v, exp=exp))

    return [dv, dh]


def ode_3d(state, _, parameters, synapse=None, scale=1, exp=np.exp):
    """Compute the three dimensional ODE.

    :param state: State of the ODE (r.h.s of the ODE vector)
    :param _: Current time
    :param parameters: Parameters
    :param synapse: Optional synapse function
    :param scale: Scaling factor for dhs. Used to explore convergence rate in paper
    :param exp: Which exponential function to use, defaults to numpy.exp
    :return: The r.h.s of the ODE
    """
    i_app = parameters["i_app"]
    v, h, hs = state

    i_syn = 0
    if synapse is not None:
        g_syn = parameters["g_syn"]
        e_syn = parameters["e_syn"]

        i_syn = synapse(v, g_syn, e_syn)

    dv = i_app + total_current(state, parameters, exp=exp) - i_syn
    dh = -(h - (h_inf(v, exp=exp))) / (tau_h(v, exp=exp))
    dhs = -(hs - (hs_inf(v, exp=exp))) / (tau_hs(v, exp=exp)) * scale

    return [dv, dh, dhs]


def ode_5d(state, _, parameters, use_modified_tau_n=True):
    """Compute the five dimensional ODE.

    :param state: State of the ODE (r.h.s of the ODE vector)
    :param _: Current time
    :param parameters: Parameters
    :param use_modified_tau_n: Optional parameter to use the original tau_n which does not work. Defaults to our tau_n
    :return: The r.h.s of the ODE
    """
    i_app = parameters["i_app"]
    v, h, hs, m, n = state

    dv = i_app + total_current(state, parameters)
    dh = -(h - (h_inf(v))) / (tau_h(v))
    dhs = -(hs - (hs_inf(v))) / (tau_hs(v))
    dm = -(m - (m_inf(v))) / (tau_m(v))
    dn = -(n - (n_inf(v))) / (tau_n(v, use_modified_tau_n=use_modified_tau_n))

    return [dv, dh, dhs, dm, dn]


def voltage_clamp(state, _, parameters, func):
    """Perform voltage clamp on any ode.

    Voltage is always the first variable in all ODEs

    :param state: State of the ODE (r.h.s of the ODE vector)
    :param _: Ignored
    :param parameters: Parameters
    :param func: ODE function to clamp
    :return: The ODE with voltage clamped
    """
    return clamp_variable(state, _, parameters, func, 0)


def hs_clamp(state, _, parameters):
    """Clamp hs in ode_3d (hs is the 3rd variable).

    :param state: State of the ODE (r.h.s of the ODE vector)
    :param _: Ignored
    :param parameters: Parameters
    :return: The 3d ODE with hs clamped
    """
    return clamp_variable(
        state, _, parameters, ode_3d, 2, exp=sympy.exp
    )  # only used for the bifn so use sympy.exp


def clamp_variable(state, _, p, func, ix, exp=np.exp):
    """Clamp a specifed ode variable.

    :param state: State of the ODE (r.h.s of the ODE vector)
    :param _: Ignored
    :param p: Parameters
    :param func: Which ode function to call
    :param ix: Index of ODE variable to clamp (i.e. ix=0 clamps V)
    :param exp: Which exponential function to use
    :return: The r.h.s. of the ODE with a particular variables d/dt set to 0
    """
    ddt = func(state, _, p, exp=exp)
    ddt[ix] = 0
    return ddt


def default_parameters(i_app=0, g_na=8, g_syn=0):
    """Generate a dictionary for the default parameters.

    :param i_app: Applied current - defaults to 0 if not specified
    :param g_na: Default sodium conductance - defaults to 8 if not specified
    :param g_syn: Default synaptic conductance - defaults to 0 if not specified
    :return: Default parameters
    """
    g_k = 0.6  # mS/cm^2
    e_na = 60  # mV
    e_k = -85  # mV
    e_l = -60  # mV
    g_l = 0.013  # mS/cm^2

    params = {
        "g_k": g_k,
        "e_k": e_k,
        "e_na": e_na,
        "g_na": g_na,
        "e_l": e_l,
        "g_l": g_l,
        "i_app": i_app,
        "g_syn": g_syn,
        "e_syn": 0,
    }

    return params


def pattern_to_window_and_value(pattern, end_time):
    """Convert a pattern dictionary to an iterable tuple of time windows (start, stop) and parameter values.

    :param pattern: Pattern dict formatted like {t_event:value}
    :param end_time: End of the simulation
    :return: Iterable pattern of windows and values
    """
    times = list(pattern.keys())
    values = list(pattern.values())

    # make a set of time windows (times[0], times[1]), (times[1], times[2])... (times[-1], end_time)
    window = list(zip(times[:-1], times[1:])) + [(times[-1], end_time)]

    # zip window and parameter value for convenient unpacking
    return list(zip(window, values))


def pulse_ode_call(model_function, ic, t_max, parameter_name, value):
    """Handle the decision making for how to call odeint when parameter is v_clamp or not.

    Swaps model_function for voltage_clap when clamping and passes the model function in arguments

    :param model_function: Either function to pulse or function to clamp and clamp is pulsed
    :param ic: Initial conditions
    :param t_max: Simulation end_time
    :param parameter_name: Name of parameter or flag to v_clamp
    :param value: Value of parameter or clamp potential
    :return: ODE solution
    """
    # v_clamp the ic, set voltage_clamp to be solved and pass the model function in args
    if parameter_name is "v_clamp":
        ic[0] = value
        ode_function = voltage_clamp
        add_param = (model_function,)
        kwargs = {}
    else:
        # Pass parameters as args and solve the specified function
        ode_function = model_function
        add_param = ()
        kwargs = {parameter_name: value}

    t, sol = solve_ode(
        model=ode_function, ic=ic, t_max=t_max, additional_params=add_param, **kwargs,
    )
    return sol


def pulse(model, parameter_name, temporal_pattern, t_max, ic, **kwargs):
    """Apply a time dependent "pulse" to an ODE system.

    A pulse is a sequence of discrete changes to a constant parameters. The ODE is solved for each value seperately and
    stitched together with the IC of the next step being the end point of the current step

    For example injecting a dc-current at t=1000

    :param model: ODE function to apply pulse to
    :param parameter_name: Parameter to
    :param temporal_pattern: A dictionary of time:value pairs {0: 0, 1000:1} will set the parameter to 0 at t=0 and 1 and t=1000
    :param t_max: Time to end the simulation at
    :param ic: Simulation initial condition
    :param kwargs: Additional parameters to set for default parameters
    :return: The solved continuous ode, the time points and the waveform of the property
    """
    # Initialize data holders
    solution = np.array([0] * len(ic))  # needs dummy data to keep shape for vstack
    t_solved = np.array([])
    stimulus = np.array([])

    sequence = pattern_to_window_and_value(temporal_pattern, t_max)
    # iterate over the time windows and set the parameter to value during the window
    for (t0, t1), value in sequence:
        parameters = default_parameters(**kwargs)
        # update parameter name
        parameters[parameter_name] = value

        #  generate new time steps and save
        t = np.arange(t0, t1, 0.1)
        t_solved = np.concatenate((t_solved, t))

        # Call ode and get model solution update
        block_solution = pulse_ode_call(
            model_function=model,
            ic=ic,
            t_max=t1 - t0,
            parameter_name=parameter_name,
            value=value,
        )

        # Save solution and stimulus wave form and update ic for next window
        solution = np.vstack((solution, block_solution))  # keep track of solution
        stimulus = np.concatenate((stimulus, np.ones(t.shape) * value))
        ic = block_solution[-1, :]

    #  first row of solution is dummy data so omit
    return solution[1:, :], t_solved, stimulus


def compute_iv_current(solution, parameters, follow):
    """Compute the current of the IV curve at a given clamp voltage given the follow rule.

    If follow mode: I(V) is the steady state current, otherwise we use the peak amplitude current of the transient

    :param solution: odeint solution
    :param parameters: Model parameters
    :param follow: Follow mode or peak mode (True/False)
    :return: I(V)
    """
    if follow:
        return -total_current(solution.T, parameters)[-1]  # [-1] give steady state
    else:
        i_na = sodium_current(solution.T, parameters)
        pk = np.argmax(np.abs(i_na))  # intermediate pk allows for sign preservation
        return i_na[pk]


def solve_ode(model, ic, t_max, additional_params=(), dt=0.1, rtol=None, **kwargs):
    """Solve an ode model with settings.

    :param model: ODE Model to solve
    :param ic: Initial conditions
    :param t_max: Simulation end time
    :param additional_params: Additional parameters (function to clamp) to pass to odeint
    :param dt: Time step to save at (defaults to 0.1ms)
    :param rtol: Relative tolerance (defaults to none)
    :param kwargs: Optional arguments to set parameters
    :return: t, and solution
    """
    # Initialize time and paraters
    parameters = default_parameters(**kwargs)
    t = np.arange(0, t_max, dt)

    # Solve and return solution
    solution = odeint(model, ic, t, args=(parameters,) + additional_params, rtol=rtol)
    return t, solution


def current_voltage_curve(model, clamp_range, t_max, ic, follow=False, **kwargs):
    """Compute IV curve in either follow mode or peak mode.

    In follow mode a traditional IV curve is computed where the I(V) is the steady state current at a clamped voltage
    for efficiency the initial condition of the next voltage level is the steady state of the present clamp.

    In peak mode (follow = False) the voltage is held at some reset voltage. The voltage is the clamped at the voltage
    for the IV curve and the peak (transient) current is used

    :param model: The function to call for voltage_clamp (the model used)
    :param clamp_range: Upper and lower range of voltage to clamp during the IV curve
    :param t_max: Time to run clamp for to equilibrate
    :param ic: Initial condition or reset condition
    :param follow: Optional flag is follow model or peak mode is used: defaults to peak mode, follow=False
    :param kwargs: Optional settings for the parameters such as g_na or i_leak
    :return: I(V)
    """
    # Initialize IV curve
    parameters = default_parameters(**kwargs)
    voltage = np.arange(*clamp_range)
    current = np.zeros(voltage.shape)
    state = np.array([ic])  # inital state is ic; array([ic]) gives a 2d array

    # Update model inital state according to IV curve type, run voltage clamp and save I(V)
    for ix, v in enumerate(voltage):
        ic = update_ic(v, ic, state, follow)
        _, state = solve_ode(
            model=voltage_clamp, ic=ic, t_max=t_max, additional_params=(model,)
        )
        current[ix] = compute_iv_current(state, parameters, follow)

    return current, voltage


def update_ic(v, ic, state, follow):
    """Set the initial condition for the IV curve at a particular clamp voltage depending on the mode.

    If follow mode then the clamp voltage is combined with the state. Otherwise the clamp voltage is combined with the
    reset ic

    :param v: Clamp voltage
    :param ic: Reset conditions
    :param state: Steady state
    :param follow: Follow mode or peak mode (true/false)
    :return: Updated ic according to follow rule
    """
    if follow:
        return np.concatenate(
            [[v], state[-1, 1:]]
        )  # ic = [set voltage, ss of old simulation]
    else:
        return np.concatenate([[v], ic[1:]])  # ic = [set voltage, ss of holding]


def steady_state_when_clamped(v_clamp):
    """Determine the steady-state of ode_3d when clamped to a voltage.

    :param v_clamp: Voltage to clamp to
    :return: Steady state of the neuron at v_clamp
    """
    _, state = solve_ode(
        model=voltage_clamp, ic=[v_clamp, 1, 1], t_max=500, additional_params=(ode_3d,)
    )
    return state[-1, :]


def resize_initial_condition(ic, model, fill=1):
    """Correct the dimension of an initial condition.

    Given a potentially valid initial condition (correct number of dimensions) either cut off the excess dimensions or
    append `fill` n times to fill the inital conditions to the correct size

    :param ic: Initial condition list
    :param model: Model to be solved
    :param fill: How to fill ic if ic is to small
    :return: Appropriately sized ic
    """
    # Determine correct ic size
    if model == ode_3d:
        target_ic_length = 3
    elif model == ode_2d:
        target_ic_length = 2
    else:
        target_ic_length = 5

    actual_ic_length = len(ic)
    # trim ic if too large
    if actual_ic_length >= target_ic_length:
        return ic[:target_ic_length]
    else:
        # fill ic if too large
        return ic + (target_ic_length - actual_ic_length) * [fill]
