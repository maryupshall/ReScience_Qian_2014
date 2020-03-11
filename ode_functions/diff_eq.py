"""Monolithic module for specifying and solving the ODE problems.

Provides functionality to create 2,3,5D ode systems as well as solving them.
"""
from scipy.integrate import odeint

from ode_functions.current import total_current
from ode_functions.gating import *
from units import uA_PER_CM2, mS_PER_CM2, ureg, strip_dimension


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


def default_parameters(i_app=0 * uA_PER_CM2, g_na=8 * mS_PER_CM2, g_syn=0 * mS_PER_CM2):
    """Generate a dictionary for the default parameters.

    :param i_app: Applied current - defaults to 0 if not specified
    :param g_na: Default sodium conductance - defaults to 8 if not specified
    :param g_syn: Default synaptic conductance - defaults to 0 if not specified
    :return: Default parameters
    """
    g_k = 0.6 * mS_PER_CM2
    e_na = 60 * ureg.mV
    e_k = -85 * ureg.mV
    e_l = -60 * ureg.mV
    g_l = 0.013 * mS_PER_CM2

    params = {
        "g_k": g_k,
        "e_k": e_k,
        "e_na": e_na,
        "g_na": g_na,
        "e_l": e_l,
        "g_l": g_l,
        "i_app": i_app,
        "g_syn": g_syn,
        "e_syn": 0 * ureg.mV,
    }

    return params


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
    t_max = strip_dimension(t_max)
    t = np.arange(0, t_max, dt)

    ic = list(map(strip_dimension, ic))
    striped_parameters = {k: strip_dimension(v) for k, v in parameters.items()}

    # Solve and return solution
    solution = odeint(
        model, ic, t, args=(striped_parameters,) + additional_params, rtol=rtol
    )
    return t, solution


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
