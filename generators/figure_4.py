import PyDSTool
from sympy import *

from ode_functions.diff_eq import ode_2d, ode_3d, default_parameters, current_voltage_curve
from ode_functions.gating import *
from ode_functions.nullclines import nullcline_figure
from plotting import *


def run():
    """
    Top level runner for figure 4
    :return: None
    """

    print("Running: Figure 4")

    init_figure(size=(6, 7))
    for ix in [0, 1]:
        plt.subplot2grid((4, 2), (0, ix), colspan=1, rowspan=1)
        __figure4a__("A" + str(ix + 1), panel=ix)

    for ix in [0, 1]:
        plt.subplot2grid((4, 2), (ix + 1, 0), colspan=2, rowspan=1)
        __figure4b__("B" + str(ix + 1), panel=ix)

    for ix in [0, 1]:
        plt.subplot2grid((4, 2), (3, ix), colspan=1, rowspan=1)
        __figure4c__("C" + str(ix + 1), panel=ix)

    save_fig('4')


def __figure4a__(title, panel=0):
    """
    Plot nullclines for different model currents (ix)

    :param title: Plot title (panel label)
    :param panel: Which plot to make ix refers to the index if the below i_app_list and hs_list
    :return: None
    """

    v = np.arange(-90, 50)
    """Select appropriate i_app and hs for the panel used"""
    i_app_list = [[0, 3.5], [0.16, 0.16, 0.16]][panel]
    hs_list = [[1, 1], [0.6, 0.2, 0.05]][panel]

    """Stability for each curve on each panel"""
    stability = [[False, True], [False, False, True]]

    """Iterate over the different v nullclines from the different i_app and hs values"""
    for iy, (i_app, hs) in enumerate(zip(i_app_list, hs_list)):
        nullcline_figure(v, i_app, stability=stability[panel][iy], hs=hs, h_color='g', v_color='r')

    if panel == 0:
        set_properties(title, x_label="v (mV)", y_label="h", x_tick=[-40, 0], y_tick=[0, 0.05, 0.1, 0.15],
                       x_limits=(-40, 5), y_limits=(0, 0.15))
    else:
        set_properties(title, x_label="v (mV)", x_tick=[-60, 20], y_tick=[0, 0.2, 0.4], x_limits=(-80, 20),
                       y_limits=(0, 0.4))


def __figure4b__(title, panel=0):
    """
    Perform bifurcation analysis of 2D and 3D system for 4B1/2

    :param title: Plot title (panel label)
    :param panel: Which plot to make, 2D (panel=0) or 3d (panel=1)
    :return: None
    """

    """Compute contunuation and plot bifurcation diagram depending on the panel"""
    if panel == 0:
        __figure4b1_continuation__()
        x_label = ""
        x_tick = [-6, 0, 6]
    else:
        __figure4b2_continuation__()
        x_label = "I$_{app}$($\mu$A/cm$^2$)"
        x_tick = [-0.1, 0, 0.2, 0.1]

    set_properties(title, y_label='$V_m$ (mV)', y_tick=[-80, 0, 30], x_label=x_label, x_tick=x_tick,
                   x_limits=(min(x_tick), max(x_tick)))


def __figure4b1_continuation__():
    """
    Actual continuation analysis for 4B1. Contains commands to pyDSTool. Performs some formatting and continuation

    Plotting commands are contained with continuation commands to keep pycont objects together

    :return: None
    """

    """Set parameters and convert to symbolic representation"""
    parameters = default_parameters(i_app=0)
    v, h, i_app = symbols('v h i_app')
    parameters['i_app'] = i_app
    dydt = ode_2d([v, h], 0, parameters, exp=exp)

    DSargs_1 = PyDSTool.args(name='bifn_1')
    DSargs_1.pars = {'i_app': 0}
    DSargs_1.varspecs = {'v': PyDSTool.convertPowers(str(dydt[0])),
                         'h': PyDSTool.convertPowers(str(dydt[1]))}
    DSargs_1.ics = {'v': 0, 'h': 0}

    ode_1 = PyDSTool.Generator.Vode_ODEsystem(DSargs_1)
    ode_1.set(pars={'i_app': 0})
    ode_1.set(ics={'v': -49, "h": 0.4})
    PyCont_1 = PyDSTool.ContClass(ode_1)

    PCargs_1 = PyDSTool.args(name='EQ1_1', type='EP-C')
    PCargs_1.freepars = ['i_app']
    PCargs_1.MaxNumPoints = 500
    PCargs_1.MaxStepSize = 0.05
    PCargs_1.MinStepSize = 1e-5
    PCargs_1.StepSize = 1e-2
    PCargs_1.LocBifPoints = 'all'
    PCargs_1.SaveEigen = True
    PyCont_1.newCurve(PCargs_1)
    PyCont_1['EQ1_1'].backward()
    PyCont_1['EQ1_1'].forward()
    PyCont_1['EQ1_1'].backward()

    PyCont_1['EQ1_1'].display(['i_app', 'v'], stability=True, figure=1)

    PCargs_1.name = 'LC1_1'
    PCargs_1.type = 'LC-C'
    PCargs_1.initpoint = 'EQ1_1:H1'
    PCargs_1.freepars = ['i_app']
    PCargs_1.MaxNumPoints = 500
    PCargs_1.MaxStepSize = 0.1
    PCargs_1.LocBifPoints = 'all'
    PCargs_1.SaveEigen = True
    PyCont_1.newCurve(PCargs_1)
    PyCont_1['LC1_1'].backward()
    PyCont_1['LC1_1'].display(('i_app', 'v_min'), stability=True, figure=1)
    PyCont_1['LC1_1'].display(('i_app', 'v_max'), stability=True, figure=1)

    PyCont_1.plot.toggleLabels(visible='off', bytype=['P', 'RG', 'LP'])
    PyCont_1.plot.togglePoints(visible='off', bytype=['P', 'RG', 'LP'])
    plt.gca().set_title('')


def __figure4b2_continuation__():
    """
    Actual continuation analysis for 4B2. Contains commands to pyDSTool. Performs some formatting and continuation

    Plotting commands are contained with continuation commands to keep pycont objects together

    :return: None
    """

    """Set parameters and convert to symbolic representation"""

    parameters = default_parameters(i_app=-0.1)
    v, h, h_s, i_app = symbols('v h h_s i_app')
    parameters['i_app'] = i_app
    dydt = ode_3d([v, h, h_s], 0, parameters, exp=exp)

    DSargs_2 = PyDSTool.args(name='bifn_2')
    DSargs_2.pars = {'i_app': 0}
    DSargs_2.varspecs = {'v': PyDSTool.convertPowers(str(dydt[0])),
                         'h': PyDSTool.convertPowers(str(dydt[1])),
                         'h_s': PyDSTool.convertPowers(str(dydt[2]))}
    DSargs_2.ics = {'v': 0, 'h': 0, 'h_s': 0}

    ode_2 = PyDSTool.Generator.Vode_ODEsystem(DSargs_2)
    ode_2.set(pars={'i_app': -0.1})
    ode_2.set(ics={'v': -67, "h": 0.77, "h_s": 1})
    PyCont_2 = PyDSTool.ContClass(ode_2)

    PCargs_2 = PyDSTool.args(name='EQ1_2', type='EP-C')
    PCargs_2.freepars = ['i_app']
    PCargs_2.MaxNumPoints = 300
    PCargs_2.MaxStepSize = 0.1
    PCargs_2.MinStepSize = 1e-5
    PCargs_2.StepSize = 1e-2
    PCargs_2.LocBifPoints = 'all'
    PCargs_2.SaveEigen = True
    PyCont_2.newCurve(PCargs_2)
    PyCont_2['EQ1_2'].backward()

    PyCont_2['EQ1_2'].display(['i_app', 'v'], stability=True, figure=1)

    PCargs_2.name = 'LC1_2'
    PCargs_2.type = 'LC-C'
    PCargs_2.initpoint = 'EQ1_2:H2'
    PCargs_2.freepars = ['i_app']
    PCargs_2.MaxNumPoints = 400
    PCargs_2.MaxStepSize = 0.1
    PCargs_2.StepSize = 1e-2
    PCargs_2.LocBifPoints = 'all'
    PCargs_2.SaveEigen = True
    PyCont_2.newCurve(PCargs_2)
    PyCont_2['LC1_2'].forward()
    PyCont_2['LC1_2'].display(('i_app', 'v_min'), stability=True, figure=1)
    PyCont_2['LC1_2'].display(('i_app', 'v_max'), stability=True, figure=1)

    PyCont_2.plot.toggleLabels(visible='off', bytype=['P', 'RG'])
    PyCont_2.plot.togglePoints(visible='off', bytype=['P', 'RG'])

    PyCont_2.plot.toggleLabels(visible='off', byname=['LPC2', 'LPC3'])
    PyCont_2.plot.togglePoints(visible='off', byname=['LPC2', 'LPC3'])

    plt.gca().set_title('')


def __figure4c__(title, panel=0):  # todo slightly different
    """
    Compute true IV curves for 2d and 3d model for figure 4C1/2

    :param title: Plot title (panel label)
    :param panel: Which plot to make, 2D (label=0) or 3d (label=1)
    :return: None
    """

    """Select appropriate model given which panel"""
    model = [ode_2d, ode_3d][panel]

    """Solve IV curve for voltage between -100 and 20 using 3000ms to ensure equilibrium is reached"""
    voltage = np.arange(-100, 20)
    time = np.arange(0, 3000)

    ic = [-100, 1]  # at such hyperpolarized parameters h and hs are ~1
    """Add dimension to 3d model"""
    if model == ode_3d:
        ic += [1]
    ic = np.array(ic)  # needed before refactor: todo: here

    """Use system hs if the model is the 3d model"""
    use_system_hs = model == ode_3d

    current = current_voltage_curve(model, voltage, time, ic, use_system_hs=use_system_hs, current_function="Balance",
                                    follow=True)  # todo probably refactor this

    plt.plot(voltage, current, 'k')
    plt.plot(voltage, np.zeros(np.shape(voltage)), '--', color='grey')

    if panel == 0:
        set_properties(title, x_label="Voltage (mV)", y_label="I$_{stim}$($\mu$A/cm$^2$)", x_tick=[-80, -40],
                       y_tick=[-5, 0, 5], x_limits=(-100, -20), y_limits=(-5, 5))
    else:
        set_properties(title, x_label="Voltage (mV)", x_tick=[-70, -60, -50], y_tick=[-0.1, 0, 0.1, 0.2],
                       x_limits=(-70, -50), y_limits=(-0.1, 0.2))
