from functools import partial

import PyDSTool
from scipy.integrate import odeint
from scipy.signal import argrelmax
from sympy import *

from ode_functions.diff_eq import ode_3d, default_parameters, hs_clamp, pulse
from ode_functions.gating import *
from ode_functions.nullclines import nullcline_figure
from plotting import *


def run():
    """
    Top level runner for figure 3
    :return: None
    """

    print("Running: Figure 3")

    init_figure(size=(6, 8))
    for ix in np.arange(2):
        plt.subplot2grid((5, 4), (ix, 0), colspan=4, rowspan=1)
        __figure3a__("A" + str(ix + 1), ix=ix)

    for ix in np.arange(4):
        plt.subplot2grid((5, 4), (2, ix), colspan=1, rowspan=1)
        __figure3b__("B" + str(ix + 1), ix)

    plt.subplot2grid((5, 4), (3, 0), colspan=4, rowspan=1)
    __figure3c__("C")

    plt.subplot2grid((5, 4), (4, 0), colspan=4, rowspan=1)
    __figure3d__("D")

    save_fig("3")


def __plot_secondary_frequency__(times, frequency):
    """
    Function to plot instantaneous frequency

    :param times: List of times frequency is calculated
    :param frequency: List of frequencies
    :return: Nine
    """

    original_axis = plt.gca()  # keep reference to main axis

    """Plot frequency on a secondary axis and set limits to scale values to near original figure"""
    ax2 = original_axis.twinx()
    ax2.plot(times, frequency, "ko", markersize=1)
    ax2.set_ylim([1, 10])  # gives nice scaling
    ax2.set_yticks([7, 9])
    ax2.set_ylabel("Frequency (Hz)")

    plt.sca(original_axis)  # return plt.* command to the original axis


def __figure3a__(title, ix=0):
    """
    Compute 3d model response into depolarization block for step current input for figure 3A

    :param title: Plot title (panel label)
    :param ix: Which figure to run: voltage (fig_num=0) or h (fig_num=1)
    :return: None
    """

    """Compute a 6000ms simulation with i_app=0 at t=0 and then i_app=0.16 at t=2000"""
    pattern = {0: 0, 2000: 0.16}  # set i_app=0 at t=0. Change to i_app=0,16 at t=2000
    end_time = 6000
    ic = [-55, 0, 0]

    """Solve 3d model for above parameters and compute frequency"""
    solution, t_solved, stimulus = pulse(ode_3d, "i_app", pattern, end_time, ic)
    spike_time, instantaneous_frequency = __compute_instantaneous_frequency__(
        solution[:, 0], t_solved
    )

    """Plot voltage data and add frequency axis for first panel"""
    if ix == 0:
        v = solution[:, 0]
        plt.plot(t_solved, v, "k")
        plt.plot(t_solved, 10 * stimulus - 80, "grey")
        __plot_secondary_frequency__(spike_time, instantaneous_frequency)

        set_properties(
            title,
            y_label="v (mV)",
            y_tick=[-60, -40, -20, 0, 20],
            x_tick=[0, 3000, 6000],
            x_ticklabel=[],
            x_limits=[0, 6000],
        )
    else:
        h, hs = solution[:, 1], solution[:, 2]
        plt.plot(t_solved, h * hs, "k")
        plt.plot(t_solved, hs, "k--")

        set_properties(
            title,
            x_label="time (ms)",
            y_label="h$_{total}$, h$_s$",
            y_tick=[0, 0.2, 0.4, 0.6, 0.8],
            x_tick=[0, 3000, 6000],
            x_limits=[0, 6000],
        )
        make_legend(["h$_{total}$", "h$_s$"], loc="upper right")


def __compute_instantaneous_frequency__(voltage, time, threshold=-40):
    """
    Get per-spike frequency from peak times

    :param voltage: Voltage trace time series
    :param time: Time of voltage points
    :param threshold: Set threshold s.t. peaks below threshold are excluded
    :return: Spike times and frequency
    """

    """Find peaks and filter spikelets if any"""
    spike_index = argrelmax(voltage)[0]
    spike_index = spike_index[voltage[spike_index] > threshold]

    """Compute frequency as 1/isi"""
    instantaneous_frequency = 1 / np.diff(time[spike_index])

    """Drop first spike times (isi has 1 less point) and return frequency in Hz (period is in ms)"""
    return time[spike_index[1:]], 1000 * instantaneous_frequency


def __figure3b__(title, ix=0):
    """
    Plot nullclines for different model regimes in different panels for 3B

    :param title: Plot title (panel label)
    :param ix: Which plot to make ix referes to the index if the below i_app_list and hs_list
    :return: None
    """

    """Compute isi for voltage between -90 and 50"""
    voltage = np.arange(-90, 50)

    """different panels (ix) use a different parameters: set the appropriate one"""
    i_app = [0, 0.16, 0.16, 0.16][ix]
    hs = [0.6, 0.6, 0.2, 0.05][ix]
    nullcline_figure(voltage, i_app, stability=ix == 3, hs=hs)

    y_label = "h" if ix == 0 else ""
    y_ticklabel = None if ix == 0 else []

    set_properties(
        title,
        y_label=y_label,
        x_tick=[-40, 40],
        y_tick=[0, 0.2, 0.4, 0.6, 0.8],
        x_limits=(-80, 50),
        y_limits=(0, 0.6),
        y_ticklabel=y_ticklabel,
        x_label="V (mV)",
    )


def __figure3c__(title):
    """
    Perform bifurcation analysis of 3D system for 3C

    :param title: Plot title (panel label)
    :return: None
    """

    """Compute contunuation and plot bifurcation diagram"""
    __figure3c_continuation__()

    """Set parameters for crashing trajectory"""
    parameters = default_parameters(i_app=0.16)
    t = np.arange(0, 10000, 0.1)
    ic = [-60, 0, 1]

    """solve system and overlay - zorder plotting behind bifuraction diagram"""
    trajectory = odeint(ode_3d, ic, t, args=(parameters,))
    plt.plot(
        trajectory[:, 2], trajectory[:, 0], c="grey", zorder=-1e5, linewidth=0.5
    )  # draw trajectory below bifn

    set_properties(
        title,
        y_label="v (mV)",
        x_limits=[0, 1],
        x_tick=[0, 0.5, 1],
        y_tick=[-80, -40, 0, 40],
        x_label="hs",
    )


def __figure3c_continuation__():
    """
    Actual continuation analysis for 3C. Contains commands to pyDSTool. Performs some formatting and continuation

    Plotting commands are contained with continuation commands to keep pycont objects together

    :return: None
    """

    """Set parameters and convert to symbolic representation"""
    parameters = default_parameters(i_app=0.16)
    v, h, h_s = symbols("v h h_s")
    dydt = hs_clamp(
        [v, h, h_s], 0, parameters
    )  # returns a symbolic expression since variables are symbolic

    DSargs_3 = PyDSTool.args(name="bifn_3")
    DSargs_3.pars = {"h_s": 0}
    DSargs_3.varspecs = {
        "v": PyDSTool.convertPowers(str(dydt[0])),
        "h": PyDSTool.convertPowers(str(dydt[1])),
    }  # convert **2 to ^2
    DSargs_3.ics = {"v": 0, "h": 0}

    ode_3 = PyDSTool.Generator.Vode_ODEsystem(DSargs_3)
    ode_3.set(pars={"h_s": 0})
    ode_3.set(ics={"v": -49, "h": 0.4})
    PyCont_3 = PyDSTool.ContClass(ode_3)

    PCargs_3 = PyDSTool.args(name="EQ1_3", type="EP-C")
    PCargs_3.freepars = ["h_s"]
    PCargs_3.MaxNumPoints = 350
    PCargs_3.MaxStepSize = 0.1
    PCargs_3.MinStepSize = 1e-5
    PCargs_3.StepSize = 1e-2
    PCargs_3.LocBifPoints = "all"
    PCargs_3.SaveEigen = True
    PyCont_3.newCurve(PCargs_3)
    PyCont_3["EQ1_3"].backward()

    PyCont_3["EQ1_3"].display(["h_s", "v"], stability=True, figure=1)

    PCargs_3.name = "LC1_3"
    PCargs_3.type = "LC-C"
    PCargs_3.initpoint = "EQ1_3:H2"
    PCargs_3.freepars = ["h_s"]
    PCargs_3.MaxNumPoints = 500
    PCargs_3.MaxStepSize = 0.1
    PCargs_3.LocBifPoints = "all"
    PCargs_3.SaveEigen = True
    PyCont_3.newCurve(PCargs_3)
    PyCont_3["LC1_3"].backward()
    PyCont_3["LC1_3"].display(("h_s", "v_min"), stability=True, figure=1)
    PyCont_3["LC1_3"].display(("h_s", "v_max"), stability=True, figure=1)

    PyCont_3.plot.toggleLabels(
        visible="off", bytype=["P", "RG", "LPC"]
    )  # remove unused labels
    PyCont_3.plot.togglePoints(
        visible="off", bytype=["P", "RG", "LPC"]
    )  # remove unused points

    # hopefully remove rightmost hopf point - this may not always be H1? todo
    PyCont_3.plot.toggleLabels(visible="off", byname="H1")

    plt.gca().set_title("")


def __figure3d__(title):
    """
    Explore 3D model response to a faster hs rate for figure 3D

    :param title: Plot title (panel label)
    :return: None
    """

    """Compute a 10000ms simulation with i_app=0 at t=0 and then i_app=0.16 at t=2000"""
    pattern = {0: 0, 2000: 0.16}
    end_time = 10000
    ic = [-65, 1, 1]

    """Create curried function with partial to hide the scale kwargs"""
    solution, t_solved, stimulus = pulse(
        partial(ode_3d, scale=2), "i_app", pattern, end_time, ic
    )

    plt.plot(t_solved, solution[:, 0], "k")
    plt.plot(t_solved, 30 * stimulus - 80, "grey")
    set_properties(
        title,
        y_label="$v_m$ (mV)",
        y_tick=[-60, -40, -20, 0, 20],
        y_limits=(-80, 20),
        x_label="t (ms)",
        x_tick=[0, 5000, 10000],
        x_limits=(0, 10000),
    )
