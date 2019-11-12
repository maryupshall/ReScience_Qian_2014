import numpy as np

from ode_functions.gating import f, m_inf


def sodium_current(solution, parameters):
    """Compute sodium current from 2d,3d and 5d model.

    Variables are extracted from the model and missing variables (low dimension) are assumed to be 1 such as hs

    :param solution: odeint solution
    :param parameters: Model parameters # todo make optional?
    :return: Sodium current
    """
    # Extract sodium current parameters
    g_na = parameters["g_na"]
    e_na = parameters["e_na"]

    # Extract or assign variables v,h, and hs
    hs = 1
    if solution.shape[1] >= 3:  # hs exists in both the 3d and 5d model
        hs = solution[:, 2]
    h = solution[:, 1]
    v = solution[:, 0]

    return g_na * (v - e_na) * (m_inf(v) ** 3) * h * hs


def total_current(solution, parameters):
    """Compute sodium current from 2d,3d and 5d model.

    Variables are extracted from the model and missing variables (low dimension) are assumed to be 1 such as hs

    :param solution: odeint solution
    :param parameters: Model parameters # todo make optional?
    :return: Sodium current
    """
    # Extract model parameters
    g_k = parameters["g_k"]
    g_l = parameters["g_l"]
    e_k = parameters["e_k"]
    e_l = parameters["e_l"]

    # Extract or assign variables v,h, and hs
    h = solution[:, 1]
    v = solution[:, 0]

    return (
        -(g_l * (v - e_l))
        - sodium_current(solution, parameters)
        - (g_k * (f(h) ** 3) * (v - e_k))
    )


def nmda_current(v, g_syn, e_syn, mg=1.4):
    """Current from an nmda synapse

    :param v: Membrane potential
    :param g_syn: Synaptic conductance
    :param e_syn: Reversal potential of synapse
    :param mg: Model mg parameter: todo: check paper for what this is
    :return: nmda current
    """
    return (g_syn * (v - e_syn)) / (1 + (mg / 3.57) * np.exp(0.062 * v))


def ampa_current(v, g_syn, e_syn):
    """Current from an ampa synapse

    :param v: Membrane potential
    :param g_syn: Synaptic conductance
    :param e_syn: Reversal potential of synapse
    :return: ampa current
    """
    return g_syn * (v - e_syn)
