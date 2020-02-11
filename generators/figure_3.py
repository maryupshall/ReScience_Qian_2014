"""Run figure 3.

run() will create all subplots and save them to ../figures
"""
from functools import partial

import PyDSTool
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmax
from sympy import *

from ode_functions.diff_eq import ode_3d, default_parameters, solve_ode
from ode_functions.experiment import hs_clamp, pulse
from ode_functions.nullclines import nullcline_figure
from plotting import init_figure, save_fig, set_properties


def run():
    """Top level runner for figure 3.

    :return: None
    """
    print("Running: Figure 3")

    init_figure(size=(6, 8))
    for ix in np.arange(2):
        plt.subplot2grid((5, 4), (ix, 0), colspan=4, rowspan=1)
        figure3a("A" + str(ix + 1), ix=ix)

    for ix in np.arange(4):
        plt.subplot2grid((5, 4), (2, ix), colspan=1, rowspan=1)
        figure3b("B" + str(ix + 1), ix)

    plt.subplot2grid((5, 4), (3, 0), colspan=4, rowspan=1)
    figure3c("C")

    plt.subplot2grid((5, 4), (4, 0), colspan=4, rowspan=1)
    figure3d("D")

    save_fig("3")


def plot_secondary_frequency(times, frequency):
    """Plot instantaneous frequency on a secondary axis.

    :param times: List of times frequency is calculated
    :param frequency: List of frequencies
    :return: None
    """
    original_axis = plt.gca()  # keep reference to main axis

    # Plot frequency on a secondary axis and set limits to scale values to near original figure
    ax2 = original_axis.twinx()
    ax2.plot(times, frequency, "ko", markersize=1)
    ax2.set_ylim([1, 10])  # gives nice scaling
    ax2.set_yticks([7, 9])
    ax2.set_ylabel("Frequency (Hz)")

    plt.sca(original_axis)  # return plt.* command to the original axis


def figure3a(title, ix=0):
    """Compute 3d model response into depolarization block for step current input for figure 3A.

    :param title: Plot title (panel label)
    :param ix: Which figure to run: voltage (fig_num=0) or h (fig_num=1)
    :return: None
    """
    # Compute a 6000ms simulation with i_app=0 at t=0 and then i_app=0.16 at t=2000
    pattern = {0: 0, 2000: 0.16}
    ic = [-55, 0, 0]

    # Solve 3d model for above parameters and compute frequency
    solution, t_solved, stimulus = pulse(
        model=ode_3d,
        parameter_name="i_app",
        temporal_pattern=pattern,
        t_max=6000,
        ic=ic,
    )
    t_spike, f_spike = compute_instantaneous_frequency(solution[:, 0], t_solved)

    # Plot voltage data and add frequency axis for first panel
    if ix == 0:
        v = solution[:, 0]
        plot_secondary_frequency(t_spike, f_spike)
        plt.plot(t_solved, v, "k")
        plt.plot(t_solved, 10 * stimulus - 80, "grey")

        y_tick = [-60, -40, -20, 0, 20]

    else:
        h, hs = solution[:, 1], solution[:, 2]
        plt.plot(t_solved, h * hs, "k")
        plt.plot(t_solved, hs, "k--")
        plt.legend(["h$_{total}$", "h$_s$"], loc="upper right")

        y_tick = [0, 0.2, 0.4, 0.6, 0.8]

    xlabel = "" if ix == 0 else "Time (ms)"
    ylabel = "V (mV)" if ix == 0 else "h$_{total}$, h$_s$"
    x_ticklabel = [] if ix == 0 else None

    set_properties(
        title,
        y_label=ylabel,
        y_tick=y_tick,
        x_tick=[0, 3000, 6000],
        x_ticklabel=x_ticklabel,
        x_limits=[0, 6000],
        x_label=xlabel,
    )


def compute_instantaneous_frequency(
        voltage, time, voltage_threshold=-40, time_threshold=2000
):
    """Get per-spike frequency from peak times.

    :param voltage: Voltage trace time series
    :param time: Time of voltage points
    :param voltage_threshold: Set threshold s.t. peaks below threshold are excluded
    :param time_threshold: Only return spikes after this
    :return: Spike times and frequency
    """
    # Find peaks and filter spikelets if any
    spike_index = argrelmax(voltage)[0]
    spike_index = spike_index[voltage[spike_index] > voltage_threshold]
    # Restrict spikes to after interesting time
    spike_index = spike_index[time[spike_index] > time_threshold]

    # Compute frequency as 1/isi
    instantaneous_frequency = 1 / np.diff(time[spike_index])

    # Drop first spike times (isi has 1 less point) and return frequency in Hz (period is in ms)
    return time[spike_index[1:]], 1000 * instantaneous_frequency


def figure3b(title, panel=0):
    """Plot nullclines for different model regimes in different panels for 3B.

    :param title: Plot title (panel label)
    :param panel: Which plot to make ix referes to the index if the below i_app_list and hs_list
    :return: None
    """
    # different panels (ix) use a different parameters: set the appropriate one
    i_app = [0, 0.16, 0.16, 0.16][panel]
    hs = [0.6, 0.6, 0.2, 0.05][panel]

    s = panel == 3  # 4th panel only is stable
    nullcline_figure(v_range=[-90, 50], i_app=i_app, stability=s, hs=hs)

    y_label = "h" if panel == 0 else ""
    y_ticklabel = None if panel == 0 else []

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


def figure3c(title):
    """Perform bifurcation analysis of 3D system for 3C.

    :param title: Plot title (panel label)
    :return: None
    """
    # Compute contunuation and plot bifurcation diagram
    figure3c_continuation()
    ic = [-60, 0, 1]

    # solve system and overlay hs,v trajectory - zorder plotting behind bifuraction diagram
    t, sol = solve_ode(model=ode_3d, ic=ic, t_max=10000, i_app=0.16)
    plt.plot(sol[:, 2], sol[:, 0], c="grey", zorder=-1e5, linewidth=0.5)

    set_properties(
        title,
        y_label="V (mV)",
        x_limits=[0, 1],
        x_tick=[0, 0.5, 1],
        y_tick=[-80, -40, 0, 40],
        x_label="hs",
    )


def figure3c_continuation():
    """Continuation analysis for 3C. Contains commands to pyDSTool.

    Performs some formatting and continuation
    Plotting commands are contained with continuation commands to keep pycont objects together

    :return: None
    """
    # Set parameters and convert to symbolic representation
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

    # hopefully remove rightmost hopf point - this may not always be H1?
    PyCont_3.plot.toggleLabels(visible="off", byname="H1")

    plt.gca().set_title("")


def figure3d(title):
    """Explore 3D model response to a faster hs rate for figure 3D.

    :param title: Plot title (panel label)
    :return: None
    """
    # Compute a 10000ms simulation with i_app=0 at t=0 and then i_app=0.16 at t=2000
    pattern = {0: 0, 2000: 0.16}
    ic = [-65, 1, 1]

    # Create curried function with partial to hide the scale kwargs and solve
    partial_ode = partial(ode_3d, scale=2)
    sol, t, stimulus = pulse(
        model=partial_ode,
        parameter_name="i_app",
        temporal_pattern=pattern,
        t_max=10000,
        ic=ic,
    )

    # Plot voltage trace
    plt.plot(t, sol[:, 0], "k")
    plt.plot(t, 30 * stimulus - 80, "grey")

    # Plot properties
    set_properties(
        title,
        y_label="V (mV)",
        y_tick=[-60, -40, -20, 0, 20],
        y_limits=(-80, 20),
        x_label="Time (ms)",
        x_tick=[0, 5000, 10000],
        x_limits=(0, 10000),
    )
