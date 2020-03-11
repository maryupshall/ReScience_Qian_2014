"""Run figure 4.

run() will create all subplots and save them to ../figures
"""
import PyDSTool
import matplotlib.pyplot as plt
import numpy as np
from sympy import *

from ode_functions.diff_eq import (
    ode_2d,
    ode_3d,
    default_parameters,
    resize_initial_condition,
)
from ode_functions.experiment import current_voltage_curve
from ode_functions.nullclines import nullcline_figure
from plotting import init_figure, save_fig, set_properties
from units import uA_PER_CM2, strip_dimension, ureg


def run():
    """Top level runner for figure 4.

    :return: None
    """
    print("Running: Figure 4")

    init_figure(size=(6, 7))
    for ix in [0, 1]:
        plt.subplot2grid((4, 2), (0, ix), colspan=1, rowspan=1)
        figure4a("A" + str(ix + 1), panel=ix)

    for ix in [0, 1]:
        plt.subplot2grid((4, 2), (ix + 1, 0), colspan=2, rowspan=1)
        figure4b("B" + str(ix + 1), panel=ix)

    for ix in [0, 1]:
        plt.subplot2grid((4, 2), (3, ix), colspan=1, rowspan=1)
        figure4c("C" + str(ix + 1), panel=ix)

    save_fig("4")


def figure4a(title, panel=0):
    """Plot nullclines for different model currents (ix).

    :param title: Plot title (panel label)
    :param panel: Which plot to make ix refers to the index if the below i_app_list and hs_list
    :return: None
    """
    # Select appropriate i_app and hs for the panel used
    i_app_list = ([[0, 3.5], [0.16, 0.16, 0.16]] * uA_PER_CM2)[panel]
    hs_list = [[1, 1], [0.6, 0.2, 0.05]][panel]

    # Stability for each curve on each panel
    stability = [[False, True], [False, False, True]]

    # Iterate over the different v nullclines from the different i_app and hs values
    for iy, (i_app, hs) in enumerate(zip(i_app_list, hs_list)):
        nullcline_figure(
            v_range=[-90 * ureg.mV, 50 * ureg.mV],
            i_app=i_app,
            stability=stability[panel][iy],
            hs=hs,
            color_h="g",
            color_v="r",
        )

    if panel == 0:
        set_properties(
            title,
            x_label="V (mV)",
            y_label="h",
            x_tick=[-40, 0],
            y_tick=[0, 0.05, 0.1, 0.15],
            x_limits=(-40, 5),
            y_limits=(0, 0.15),
        )
        plt.annotate(
            "", xy=(-15, 0.05), xytext=(-20, 0.07), arrowprops=dict(arrowstyle="->")
        )
    else:
        set_properties(
            title,
            x_label="V (mV)",
            x_tick=[-60, 20],
            y_tick=[0, 0.2, 0.4],
            x_limits=(-80, 20),
            y_limits=(0, 0.4),
        )
        plt.annotate(
            "", xy=(-25, 0.3), xytext=(-20, 0.2), arrowprops=dict(arrowstyle="->")
        )


def figure4b(title, panel=0):
    """Perform bifurcation analysis of 2D and 3D system for 4B1/2.

    :param title: Plot title (panel label)
    :param panel: Which plot to make, 2D (panel=0) or 3d (panel=1)
    :return: None
    """
    # Compute contunuation and plot bifurcation diagram depending on the panel
    if panel == 0:
        figure4b1_continuation()
        x_label = ""
        x_tick = [-6, 0, 6]
    else:
        figure4b2_continuation()
        x_label = r"I$_{app}$($\mu$A/cm$^2$)"
        x_tick = [-0.1, 0, 0.2, 0.1]

    set_properties(
        title,
        y_label="V(mV)",
        y_tick=[-80, 0, 30],
        x_label=x_label,
        x_tick=x_tick,
        x_limits=(min(x_tick), max(x_tick)),
    )


def figure4b1_continuation():
    """Actual continuation analysis for 4B1. Contains commands to pyDSTool.

    Performs some formatting and continuation.
    Plotting commands are contained with continuation commands to keep pycont objects together

    :return: None
    """
    # Set parameters and convert to symbolic representation
    parameters = default_parameters(i_app=0 * uA_PER_CM2)
    striped_parameters = {k: strip_dimension(v) for k, v in parameters.items()}

    v, h, i_app = symbols("v h i_app")
    striped_parameters["i_app"] = i_app
    dydt = ode_2d([v, h], 0, striped_parameters, exp=exp)

    DSargs_1 = PyDSTool.args(name="bifn_1")
    DSargs_1.pars = {"i_app": 0}
    DSargs_1.varspecs = {
        "v": PyDSTool.convertPowers(str(dydt[0])),
        "h": PyDSTool.convertPowers(str(dydt[1])),
    }
    DSargs_1.ics = {"v": 0, "h": 0}

    ode_1 = PyDSTool.Generator.Vode_ODEsystem(DSargs_1)
    ode_1.set(pars={"i_app": 0})
    ode_1.set(ics={"v": -49, "h": 0.4})
    PyCont_1 = PyDSTool.ContClass(ode_1)

    PCargs_1 = PyDSTool.args(name="EQ1_1", type="EP-C")
    PCargs_1.freepars = ["i_app"]
    PCargs_1.MaxNumPoints = 500
    PCargs_1.MaxStepSize = 0.05
    PCargs_1.MinStepSize = 1e-5
    PCargs_1.StepSize = 1e-2
    PCargs_1.LocBifPoints = "all"
    PCargs_1.SaveEigen = True
    PyCont_1.newCurve(PCargs_1)
    PyCont_1["EQ1_1"].backward()
    PyCont_1["EQ1_1"].forward()
    PyCont_1["EQ1_1"].backward()

    PyCont_1["EQ1_1"].display(["i_app", "v"], stability=True, figure=1)

    PCargs_1.name = "LC1_1"
    PCargs_1.type = "LC-C"
    PCargs_1.initpoint = "EQ1_1:H1"
    PCargs_1.freepars = ["i_app"]
    PCargs_1.MaxNumPoints = 500
    PCargs_1.MaxStepSize = 0.1
    PCargs_1.LocBifPoints = "all"
    PCargs_1.SaveEigen = True
    PyCont_1.newCurve(PCargs_1)
    PyCont_1["LC1_1"].backward()
    PyCont_1["LC1_1"].display(("i_app", "v_min"), stability=True, figure=1)
    PyCont_1["LC1_1"].display(("i_app", "v_max"), stability=True, figure=1)

    PyCont_1.plot.toggleLabels(visible="off", bytype=["P", "RG", "LP"])
    PyCont_1.plot.togglePoints(visible="off", bytype=["P", "RG", "LP"])
    plt.gca().set_title("")


def figure4b2_continuation():
    """Actual continuation analysis for 4B2. Contains commands to pyDSTool.

    Performs some formatting and continuation.
    Plotting commands are contained with continuation commands to keep pycont objects together

    :return: None
    """
    # Set parameters and convert to symbolic representation
    parameters = default_parameters(i_app=-0.1 * uA_PER_CM2)
    striped_parameters = {k: strip_dimension(v) for k, v in parameters.items()}

    v, h, h_s, i_app = symbols("v h h_s i_app")
    striped_parameters["i_app"] = i_app
    dydt = ode_3d([v, h, h_s], 0, striped_parameters, exp=exp)

    DSargs_2 = PyDSTool.args(name="bifn_2")
    DSargs_2.pars = {"i_app": 0}
    DSargs_2.varspecs = {
        "v": PyDSTool.convertPowers(str(dydt[0])),
        "h": PyDSTool.convertPowers(str(dydt[1])),
        "h_s": PyDSTool.convertPowers(str(dydt[2])),
    }
    DSargs_2.ics = {"v": 0, "h": 0, "h_s": 0}

    ode_2 = PyDSTool.Generator.Vode_ODEsystem(DSargs_2)
    ode_2.set(pars={"i_app": -0.1})
    ode_2.set(ics={"v": -67, "h": 0.77, "h_s": 1})
    PyCont_2 = PyDSTool.ContClass(ode_2)

    PCargs_2 = PyDSTool.args(name="EQ1_2", type="EP-C")
    PCargs_2.freepars = ["i_app"]
    PCargs_2.MaxNumPoints = 300
    PCargs_2.MaxStepSize = 0.1
    PCargs_2.MinStepSize = 1e-5
    PCargs_2.StepSize = 1e-2
    PCargs_2.LocBifPoints = "all"
    PCargs_2.SaveEigen = True
    PyCont_2.newCurve(PCargs_2)
    PyCont_2["EQ1_2"].backward()

    PyCont_2["EQ1_2"].display(["i_app", "v"], stability=True, figure=1)

    PCargs_2.name = "LC1_2"
    PCargs_2.type = "LC-C"
    PCargs_2.initpoint = "EQ1_2:H2"
    PCargs_2.freepars = ["i_app"]
    PCargs_2.MaxNumPoints = 400
    PCargs_2.MaxStepSize = 0.1
    PCargs_2.StepSize = 1e-2
    PCargs_2.LocBifPoints = "all"
    PCargs_2.SaveEigen = True
    PyCont_2.newCurve(PCargs_2)
    PyCont_2["LC1_2"].forward()
    PyCont_2["LC1_2"].display(("i_app", "v_min"), stability=True, figure=1)
    PyCont_2["LC1_2"].display(("i_app", "v_max"), stability=True, figure=1)

    PyCont_2.plot.toggleLabels(visible="off", bytype=["P", "RG"])
    PyCont_2.plot.togglePoints(visible="off", bytype=["P", "RG"])

    PyCont_2.plot.toggleLabels(visible="off", byname=["LPC2", "LPC3"])
    PyCont_2.plot.togglePoints(visible="off", byname=["LPC2", "LPC3"])

    plt.gca().set_title("")


def figure4c(title, panel=0):
    """Compute true IV curves for 2d and 3d model for figure 4C1/2.

    :param title: Plot title (panel label)
    :param panel: Which plot to make, 2D (label=0) or 3d (label=1)
    :return: None
    """
    # Select appropriate model given
    model = [ode_2d, ode_3d][panel]

    # Set IC
    ic = [-100 * ureg.mV, 1]
    ic = resize_initial_condition(ic, model, fill=1)

    # Compute IV curve
    current, voltage = current_voltage_curve(
        model=model,
        clamp_range=[-100 * ureg.mV, 20 * ureg.mV],
        t_max=3000 * ureg.ms,
        ic=ic,
        follow=True,
    )

    # plot IV curve
    plt.plot(voltage, current, "k")
    plt.plot(voltage, np.zeros(np.shape(voltage)), "--", color="grey")

    if panel == 0:
        set_properties(
            title,
            x_label="V (mV)",
            y_label=r"I$_{stim}$($\mu$A/cm$^2$)",
            x_tick=[-80, -40],
            y_tick=[-5, 0, 5],
            x_limits=(-100, -20),
            y_limits=(-5, 5),
        )
    else:
        set_properties(
            title,
            x_label="V (mV)",
            x_tick=[-70, -60, -50],
            y_tick=[-0.1, 0, 0.1, 0.2],
            x_limits=(-70, -50),
            y_limits=(-0.1, 0.2),
        )


if __name__ == "__main__":
    run()
