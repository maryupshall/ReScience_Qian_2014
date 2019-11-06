import PyDSTool
from scipy.integrate import odeint
from sympy import *

from helpers.nullclines import nullcline_h, nullcline_v
from helpers.plotting import *
from ode_functions.current import total_current
from ode_functions.diff_eq import ode_2d, ode_3d, voltage_clamp, default_parameters
from ode_functions.gating import *


def run():
    """
    Top level runner for figure 4
    :return: None
    """

    init_figure(size=(6, 7))
    plt.subplot2grid((4, 2), (0, 0), colspan=1, rowspan=1)
    __figure4a__("A1", ix=0)
    plt.subplot2grid((4, 2), (0, 1), colspan=1, rowspan=1)
    __figure4a__("A2", ix=1)

    plt.subplot2grid((4, 2), (1, 0), colspan=2, rowspan=1)
    __figure4b__("B1", ix=0)
    plt.subplot2grid((4, 2), (2, 0), colspan=2, rowspan=1)
    __figure4b__("B2", ix=1)

    plt.subplot2grid((4, 2), (3, 0), colspan=1, rowspan=1)
    __figure4c__("C1", ix=0)
    plt.subplot2grid((4, 2), (3, 1), colspan=1, rowspan=1)
    __figure4c__("C2", ix=1)

    save_fig('4')


def __figure4a__(title, ix=0):
    """
    Nullcline analysis of different shifts to nullclines

    :param title: Plot title (panel label)
    :param ix: Which plot to make, right shift (ix=0) or left shift (ix=1)
    :return: None
    """

    i_app_list_set = [[0, 3.5], [0.16, 0.16, 0.16]]
    hs_list_set = [[1, 1], [0.6, 0.2, 0.05]]
    v = np.arange(-90, 50)

    stability = [[False, True], [False, False, True]]

    nh = nullcline_h(v)

    i_app_list = i_app_list_set[ix]
    hs_list = hs_list_set[ix]

    plt.plot(v, nh, 'g')
    for iy, (I, hs) in enumerate(zip(i_app_list, hs_list)):
        nv = nullcline_v(v, I, hs=hs)
        cross_index = np.argmin(np.abs(nv - nh))  # where they are closest i.e. min(err)
        style = 'k' if stability[ix][iy] else 'none'
        plt.scatter(v[cross_index], nv[cross_index], edgecolors='k', zorder=1e6, facecolor=style)
        plt.plot(v, nv, 'r')

    if ix == 0:
        set_properties(title, x_label="v (mV)", y_label="h", x_tick=[-40, 0], y_tick=[0, 0.05, 0.1, 0.15],
                       x_limits=(-40, 5), y_limits=(0, 0.15))
    else:
        set_properties(title, x_label="v (mV)", x_tick=[-60, 20], y_tick=[0, 0.2, 0.4], x_limits=(-80, 20),
                       y_limits=(0, 0.4))


def __figure4b__(title, ix=0):
    """
    Bifurcation analysis for 2D and 3D model

    :param title: Plot title (panel label)
    :param ix: Which plot to make, 2D (ix=0) or 3d (ix=1)
    :return: None
    """

    if ix == 0:
        __figure4b1_continuation__()
        x_label = ""
        x_tick = [-6, 0, 6]
    else:
        __figure4b2_continuation__()
        x_label = "$I_{app}($\mu$A/cm$^2$)$"
        x_tick = [-0.1, 0, 0.2, 0.1]

    set_properties(title, y_label='$V_m$ (mV)', y_tick=[-80, 0, 30], x_label=x_label, x_tick=x_tick,
                   x_limits=(min(x_tick), max(x_tick)))


def __figure4b1_continuation__():
    """
    Actual continuation analysis for 4B1. Contains commands to pyDSTool. Performs some formatting and continuation

    :return: None
    """

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

    :return: None
    """

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


def __figure4c__(title, ix=0):
    """
    IV curves for 2D and 3D model

    :param title: Plot title (panel label)
    :param ix: Which plot to make, 2D (ix=0) or 3d (ix=1)
    :return: None
    """

    ode_functions = [ode_2d, ode_3d]
    v_list = np.arange(-100, 20, 0.5)
    t = np.arange(0, 1000, 0.1)

    membrane_current = np.zeros(len(v_list))
    parameters = default_parameters()

    for i in range(len(v_list)):
        initial_state = [v_list[i], 0] + ix * [0]

        state = odeint(voltage_clamp, initial_state, t, args=(parameters, ode_functions[ix]))  # run voltage clamp
        h = state[:, 1]
        v = state[:, 0]
        hs = 1 if np.shape(state)[1] == 2 else state[:, 2]  # if 2d then hs=1 otherwise hs is taken from simulation

        membrane_current[i] = -total_current(v, h, parameters, hs=hs)[-1]  # take last time point as steady state

    plt.plot(v_list, membrane_current, 'k')
    plt.plot(v_list, v_list * [0], '--', color='grey')

    if ix == 0:
        set_properties(title, x_label="Voltage (mV)", y_label="I$_{stim}($\mu$A/cm^{2}$)", x_tick=[-80, -40],
                       y_tick=[-5, 0, 5], x_limits=(-100, -20), y_limits=(-5, 5))
    else:
        set_properties(title, x_label="Voltage (mV)", x_tick=[-70, -60, -50], y_tick=[-0.1, 0, 0.1, 0.2],
                       x_limits=(-70, -50), y_limits=(-0.1, 0.2))
