"""Collection of functions for computing different types of current.

Membrane current
Individual ionic current
AMPA/NMDA current
"""
import numpy as np

from ode_functions.gating import f_approx, m_inf
from units import strip_dimension


def leak_current(solution, parameters):
    """Compute leak current from 2d,3d and 5d model.

    Solution must either be 1d (1 time point) or variables on rows for slicing

    :param solution: odeint solution
    :param parameters: Model parameters
    :return: Sodium current
    """
    # Extract leak current parameters
    g_l = parameters["g_l"]
    e_l = parameters["e_l"]

    v = solution[0]
    return g_l * (v - e_l)


def sodium_current(solution, parameters, exp=np.exp):
    """Compute sodium current from 2d,3d and 5d model.

    Variables are extracted from the model and missing variables (low dimension) are assumed to be 1 such as hs

    Solution must either be 1d (1 time point) or variables on rows for slicing

    :param solution: odeint solution
    :param parameters: Model parameters
    :param exp: Exponential function to use
    :return: Sodium current
    """
    # Extract sodium current parameters
    g_na = strip_dimension(parameters["g_na"])
    e_na = strip_dimension(parameters["e_na"])

    # Extract or assign variables v,h, and hs
    hs = 1 if len(solution) == 2 else solution[2]
    v = solution[0]
    m = m_inf(v, exp=exp) if len(solution) < 5 else solution[3]
    h = solution[1]

    return g_na * (v - e_na) * (m ** 3) * h * hs


def potassium_current(solution, parameters):
    """Compute potassium current from 2d,3d and 5d model.

    Solution must either be 1d (1 time point) or variables on rows for slicing

    :param solution: odeint solution
    :param parameters: Model parameters
    :return: Sodium current
    """
    # Extract potassium current parameters
    g_k = parameters["g_k"]
    e_k = parameters["e_k"]

    v = solution[0]
    h = solution[1]

    n = f_approx(h) if len(solution) < 5 else solution[4]
    return g_k * (n ** 3) * (v - e_k)


def total_current(solution, parameters, exp=np.exp):
    """Compute membrane current from 2d,3d and 5d model.

    Solution must either be 1d (1 time point) or variables on rows for slicing

    :param solution: odeint solution
    :param parameters: Model parameters
    :param exp: Exponential function to use
    :return: Sodium current
    """
    return (
            -leak_current(solution, parameters)
            - sodium_current(solution, parameters, exp=exp)
            - potassium_current(solution, parameters)
    )


def nmda_current(v, g_syn, e_syn, mg=1.4):
    """Compute the current from an nmda synapse.

    :param v: Membrane potential
    :param g_syn: Synaptic conductance
    :param e_syn: Reversal potential of synapse
    :param mg: Model mg concentration: defaults to 1.4mM
    :return: nmda current
    """
    return (g_syn * (v - e_syn)) / (1 + (mg / 3.57) * np.exp(0.062 * v))


def ampa_current(v, g_syn, e_syn):
    """Compute the current from an ampa synapse.

    :param v: Membrane potential
    :param g_syn: Synaptic conductance
    :param e_syn: Reversal potential of synapse
    :return: ampa current
    """
    return g_syn * (v - e_syn)
