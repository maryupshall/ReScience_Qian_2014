import numpy as np


def x_inf(v, x_half, x_slope, exp=np.exp):
    """
    Steady-state activation for variable

    :param v: Membrane potential
    :param x_half: Half max voltage
    :param x_slope: Slope at half max
    :param exp: Function to call for exponential: should be overridden for symbolic exponential (sympy.exp)
    :return: x_inf(v)
    """

    return 1 / (1 + exp(-(v - x_half) / x_slope))


def m_inf(v, exp=np.exp):
    """
    Steady-state activation variable for m

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: m_inf(v)
    """

    m_half = -30.0907
    m_slope = 9.7264
    return x_inf(v, m_half, m_slope, exp=exp)


def h_inf(v, exp=np.exp):
    """
    Steady-state activation variable for h

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: h_inf(v)
    """

    h_half = -54.0289
    h_slope = -10.7665
    return x_inf(v, h_half, h_slope, exp=exp)


def hs_inf(v, exp=np.exp):
    """
    Steady-state activation variable for hs

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: hs_inf(v)
    """

    hs_half = -54.8
    hs_slope = -1.57
    return x_inf(v, hs_half, hs_slope, exp=exp)


def n_inf(v, exp=np.exp):
    """
    Steady-state activation variable for n

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: hs_inf(v)
    """

    n_half = -25
    n_slope = 12
    return x_inf(v, n_half, n_slope, exp=exp)


def tau_h(v, exp=np.exp):
    """
    Time constant (\tau) for variable h

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: tau_h(v)
    """

    a = 0.00050754 * exp(-0.063213 * v)
    b = 9.7529 * exp(0.13442 * v)

    return 0.4 + 1 / (a + b)


def tau_hs(v, exp=np.exp):
    """
    Time constant (\tau) for variable hs

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: tau_hs(v)
    """

    return 20 + 160 / (1 + exp((v + 47.2) / 1))


def tau_m(v, exp=np.exp):
    """
    Time constant (\tau) for variable m

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: tau_m(v)
    """

    a = -(15.6504 + (0.4043 * v)) / (exp(-19.565 - (0.50542 * v)) - 1)
    b = 3.0212 * exp(-0.0074630 * v)

    return 0.01 + 1 / (a + b)


def tau_n(v, exp=np.exp):
    """
    Time constant (\tau) for variable n

    :param v: Membrane potential
    :param exp: Exponential function to use
    :return: tau_n(v)
    """

    return 1 + 19 * exp((-(np.log(1 + 0.05 * (v + 60)) / 0.05) ** 2) / 300)


def f(h):
    """
    f(h) is a dimension reduction where n is represented as a function f(h)

    :param h: Gating variable h
    :return: f(h)
    """

    a0 = 0.8158
    a1 = -3.8768
    a2 = 6.8838
    a3 = -4.2079

    fh = np.asarray(a0 + a1 * h + a2 * (h ** 2) + a3 * (h ** 3))  # make array for logical indexing

    try:  # if its an array
        fh[fh < 0] = 0
        fh[fh > 1] = 1
    except TypeError:  # will not work for symbolic functions however not important
        pass

    return fh
