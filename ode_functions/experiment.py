"""Module with helpers for performing experimental manipulations.

Manipulations include clamping, sequence of discrete pulses (of current or 'variable')"""
import numpy as np
import sympy

from ode_functions.current import total_current, sodium_current
from ode_functions.diff_eq import ode_3d, solve_ode, default_parameters, update_ic


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

    A pulse is a sequence of discrete changes to a constant parameters. The ODE is solved for each value separately and
    stitched together with the IC of the next step being the end point of the current step

    For example injecting a dc-current at t=1000

    :param model: ODE function to apply pulse to
    :param parameter_name: Name of parameter to update during pulse
    :param temporal_pattern: A dictionary of time:value pairs {0: 0, 1000:1} will set param to 0 at t=0 and 1 and t=1000
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


def steady_state_when_clamped(v_clamp):
    """Determine the steady-state of ode_3d when clamped to a voltage.

    :param v_clamp: Voltage to clamp to
    :return: Steady state of the neuron at v_clamp
    """
    _, state = solve_ode(
        model=voltage_clamp, ic=[v_clamp, 1, 1], t_max=500, additional_params=(ode_3d,)
    )
    return state[-1, :]


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
