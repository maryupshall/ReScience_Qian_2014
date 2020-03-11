"""Collection of functions for computing nullclines and generating traces with them."""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton

from ode_functions.current import total_current
from ode_functions.diff_eq import h_inf, default_parameters
from units import strip_dimension


def nullcline_h(v):
    """Compute the h nullcline.

    Simply a call to h_inf as they're the same. This function provides a similar naming as to nullcline_v

    :param v: Membrane potential
    :return: h nullcline
    """
    return h_inf(v)


def nullcline_v(voltages, i_app, hs=1):
    """Compute the v nullcline.

    Computes the v nullcline for all values of v specified via newton's method

    :param voltages: Membrane potential
    :param i_app: Applied current
    :param hs: hs variable
    :return: v nullcline
    """
    nullcline = np.zeros((len(voltages),))
    parameters = default_parameters(i_app=i_app)
    striped_parameters = {k: strip_dimension(v) for k, v in parameters.items()}

    # Find self-consistent h value for every v on the nullcline
    for ix, v in enumerate(voltages):
        f_solve = partial(nullcline_v_implicit, v, striped_parameters, hs)
        nullcline[ix] = newton(f_solve, x0=0)

    return nullcline


def nullcline_v_implicit(v, parameters, hs, h):
    """Implicit form of the v nullcline evaluated at h, v, and hs.

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
    """Create tandard nullcline-curve figure.

    :param v_range: Min and max voltage to use
    :param i_app: Injected current
    :param stability: Whether or not the intersection is stable
    :param hs: Optional parameter for value of hs on nullcline: defaults to 1
    :param color_h: Optional color for the h_nullcline color: defaults to black
    :param color_v: Optional color for the v_nullcline color: defaults to grey
    :return: None
    """
    voltages = np.arange(*map(strip_dimension, v_range))

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
    """Lazy intersection of nullclines.

    This is a helper function for plotting intersections of nullclines. This is not a true intersection

    Intersection is defined where the difference between nh and nv is minimum

    :param nh: Numeric nullcline for h
    :param nv: Numeric nullcline for v
    :param v: Voltage for nx(v)
    :return: (v,nh) where the intersection occurs
    """
    intersection_index = np.argmin(np.abs(nh - nv))
    return v[intersection_index], nh[intersection_index]
