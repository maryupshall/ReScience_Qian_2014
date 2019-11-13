from functools import partial

import matplotlib.pyplot as plt
from scipy.optimize import newton

from ode_functions.current import total_current
from ode_functions.diff_eq import h_inf, default_parameters


def nullcline_h(v):
    """h nullcline

    Simply a call to h_inf as they're the same. This function provides a similar naming as to nullcline_v

    :param v: Membrane potential
    :return: h nullcline
    """
    return h_inf(v)


def nullcline_v(voltages, i_app, hs=1):
    """Compute the v nullcline

    Computes the v nullcline for all values of v specified via newton's method

    :param voltages: Membrane potential
    :param i_app: Applied current
    :param hs: hs variable
    :return: v nullcline
    """
    nullcline = np.zeros((len(voltages),))
    parameters = default_parameters(i_app=i_app)

    # Find self-consistent h value for every v on the nullcline
    for ix, v in enumerate(voltages):
        f_solve = partial(__nullcline_v_implicit__, v, parameters, hs)
        nullcline[ix] = newton(f_solve, x0=0)

    return nullcline


import numpy as np


def x_inf(v, x_half, x_slope, exp=np.exp):
    """Steady-state activation for variable

    :param v: Membrane potential
    :param x_half: Half max voltage
    :param x_slope: Slope at half max
    :param exp: Function to call for exponential: should be overridden for symbolic exponential (sympy.exp)
    :return: x_inf(v)
    """
    return 1 / (1 + exp(-(v - x_half) / x_slope))


def m_inf(v, exp=np.exp):
    """Steady-state activation variable for m

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: m_inf(v)
    """
    m_half = -30.0907
    m_slope = 9.7264
    return x_inf(v, m_half, m_slope, exp=exp)


def h_inf(v, exp=np.exp):
    """Steady-state activation variable for h

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: h_inf(v)
    """
    h_half = -54.0289
    h_slope = -10.7665
    return x_inf(v, h_half, h_slope, exp=exp)


def hs_inf(v, exp=np.exp):
    """Steady-state activation variable for hs

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: hs_inf(v)
    """
    hs_half = -54.8
    hs_slope = -1.57
    return x_inf(v, hs_half, hs_slope, exp=exp)


def n_inf(v, exp=np.exp):
    """Steady-state activation variable for n

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: hs_inf(v)
    """
    n_half = -25
    n_slope = 12
    return x_inf(v, n_half, n_slope, exp=exp)


def tau_h(v, exp=np.exp):
    """Time constant (\tau) for variable h

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: tau_h(v)
    """
    a = 0.00050754 * exp(-0.063213 * v)
    b = 9.7529 * exp(0.13442 * v)

    return 0.4 + 1 / (a + b)


def tau_hs(v, exp=np.exp):
    """Time constant (\tau) for variable hs

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: tau_hs(v)
    """
    return 20 + 160 / (1 + exp((v + 47.2) / 1))


def tau_m(v, exp=np.exp):
    """Time constant (\tau) for variable m

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: tau_m(v)
    """
    a = -(15.6504 + (0.4043 * v)) / (exp(-19.565 - (0.50542 * v)) - 1)
    b = 3.0212 * exp(-0.0074630 * v)

    return 0.01 + 1 / (a + b)


def tau_n(v, exp=np.exp, use_modified_tau_n=True):
    """Time constant (\tau) for variable n

    :param v: Membrane potential
    :param exp: Exponential function to use
    :param use_modified_tau_n: Optional parameter to use the original tau_n which does not work. Defaults to our tau_n
    :return: tau_n(v)
    """
    shift = 60 if use_modified_tau_n else 40
    return 1 + 19 * exp((-((np.log(1 + 0.05 * (v + shift)) / 0.05) ** 2)) / 300)


def __nullcline_v_implicit__(v, parameters, hs, h):
    """Implicit form of the v nullcline evaluated at h, v, and hs

    :param v: Membrane potential
    :param h: h gating variable
    :param hs: hs gating variable
    :param parameters: Parameters
    :return: v nullcline in implicit form
    """
    i_app = parameters["i_app"]

    effective_state = [v, h, hs]
    return i_app + total_current(effective_state, parameters)


def nullcline_figure(v_range, i_app, stability, hs=1, color_h="black", color_v="grey"):
    """Helper function for creating nullcline figure

    :param v_range: Min and max voltage to use
    :param i_app: Injected current
    :param stability: Whether or not the intersection is stable
    :param hs: Optional parameter for value of hs on nullcline: defaults to 1
    :param color_h: Optional color for the h_nullcline color: defaults to black
    :param color_v: Optional color for the v_nullcline color: defaults to grey
    :return: None
    """
    voltages = np.arange(*v_range)

    # Compute nullclines
    nh = nullcline_h(voltages)
    nv = nullcline_v(voltages, i_app, hs=hs)

    # Plot nullclines
    plt.plot(voltages, nh, color_h, zorder=-1000)
    plt.plot(voltages, nv, color_v)

    # Lazily compute intersection and plot the stability (where closest only: not true intersection)
    style = "k" if stability else "none"
    x_i, y_i = nullcline_intersection(nh, nv, voltages)
    plt.scatter(x_i, y_i, edgecolors="k", facecolor=style, zorder=1000)


def nullcline_intersection(nh, nv, v):
    """Lazy intersection of nullclines

    This is a helper function for plotting intersections of nullclines. This is not a true intersection

    Intersection is defined where the difference between nh and nv is minimum

    :param nh: Numeric nullcline for h
    :param nv: Numeric nullcline for v
    :param v: Voltage for nx(v)
    :return: (v,nh) where the intersection occurs
    """
    intersection_index = np.argmin(np.abs(nh - nv))
    return v[intersection_index], nh[intersection_index]
