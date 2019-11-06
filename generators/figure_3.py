import PyDSTool
from scipy.integrate import odeint
from scipy.signal import argrelmax
from sympy import *

from helpers.nullclines import nullcline_v, nullcline_h
from helpers.plotting import *
from ode_functions.diff_eq import ode_3d, default_parameters, hs_clamp
from ode_functions.gating import *


def run():
    """
    Top level runner for figure 3
    :return: None
    """

    init_figure(size=(6, 8))
    plt.subplot2grid((5, 4), (0, 0), colspan=4, rowspan=1)
    __figure3a__('A1', fig_num=0)

    plt.subplot2grid((5, 4), (1, 0), colspan=4, rowspan=1)
    __figure3a__('A2', fig_num=1)

    for ix in np.arange(4):
        plt.subplot2grid((5, 4), (2, ix), colspan=1, rowspan=1)
        __figure3b__('B' + str(ix + 1), ix)

    plt.subplot2grid((5, 4), (3, 0), colspan=4, rowspan=1)
    __figure3c__('C')

    plt.subplot2grid((5, 4), (4, 0), colspan=4, rowspan=1)
    __figure3d__('D')

    save_fig("3")


def __figure3a__(title, fig_num=0):
    """
    Model response: depolarization block

    :param title: Plot title (panel label)
    :param fig_num: Which figure to run: voltage (fig_num=0) or h (fig_num=1)
    :return: None
    """

    ic = [-55, 0, 0]
    t_solved = np.array([])
    solution = np.array([0, 0, 0])  # needs dummy data to keep shape for vstack
    currents = [0, 0.16]
    t0 = 0
    times = [2000, 6000]

    t = None
    state = None

    for ix, i_app in enumerate(currents):  # multiple applied currents and stitch together
        t = np.arange(t0, times[ix], 0.1)
        t_solved = np.concatenate((t_solved, t))
        t0 = times[ix]

        parameters = default_parameters(i_app=i_app)
        state = odeint(ode_3d, ic, t, args=(parameters,))
        ic = state[-1, :]  # maintain consistent initial condition

        solution = np.vstack((solution, state))

    # we want the spikes from the second half. Since the for loop runs only twice the last state
    # and t variables correspond to that
    spike_index = argrelmax(state[:, 0])[0]
    spike_index = spike_index[state[spike_index, 0] > -40]
    instantaneous_frequency = 1000 / np.diff(t[spike_index])

    solution = solution[1:, :]  # first row is [0,0] dummy data so omit

    stimulus = np.zeros(t_solved.shape)
    stimulus[t_solved > times[0]] = 1

    if fig_num == 0:  # plot voltage data
        original_axis = plt.gca()
        plt.plot(t_solved, solution[:, 0], 'k')
        plt.plot(t_solved, 10 * stimulus - 80, 'grey')

        ax2 = original_axis.twinx()  # second axis to plot the "inset frequency"
        ax2.plot(t[spike_index[1:]], instantaneous_frequency, 'ko', markersize=1)
        ax2.set_ylim([1, 10])  # gives nice scaling
        ax2.set_yticks([7, 9])
        ax2.set_ylabel('Frequency (Hz)')

        plt.sca(original_axis)  # return plt.* command to the original axis

        set_properties(title, y_label='v (mV)', y_tick=[-60, -40, -20, 0, 20], x_tick=[0, 3000, 6000], x_ticklabel=[],
                       x_limits=[0, 6000])

    else:  # plot h data
        plt.plot(t_solved, (solution[:, 1]) * (solution[:, 2]), 'k')
        plt.plot(t_solved, solution[:, 2], "k--")
        set_properties(title, x_label='time (ms)', y_label='h$_{total}$, h$_s$', y_tick=[0, 0.2, 0.4, 0.6, 0.8],
                       x_tick=[0, 3000, 6000], x_limits=[0, 6000])


def __figure3b__(title, ix=0):
    """
    Nullcline analysis of different regimes in depolarization block

    :param title: Plot title (panel label)
    :param ix: Which plot to make ix referes to the index if the below i_app_list and hs_list
    :return: None
    """

    i_app_list = [0, 0.16, 0.16, 0.16]
    hs_list = [0.6, 0.6, 0.2, 0.05]

    v = np.arange(-90, 50)
    nh = nullcline_h(v)

    I = i_app_list[ix]
    hs = hs_list[ix]
    plt.plot(v, nh, 'k')
    nv = nullcline_v(v, I, hs=hs)

    plt.plot(v, nv, '--', color='grey')
    style = 'k' if ix == 3 else 'none'
    cross_index = np.argmin(np.abs(nv - nh))
    plt.scatter(v[cross_index], nv[cross_index], edgecolors='k', facecolors=style)

    y_label = ""
    y_ticklabel = []
    if ix == 0:
        y_label = "h"
        y_ticklabel = None
    set_properties(title, y_label=y_label, x_tick=[-40, 40], y_tick=[0, 0.2, 0.4, 0.6, 0.8],
                   x_limits=(-80, 50), y_limits=(0, 0.6), y_ticklabel=y_ticklabel, x_label='V (mV)')


def __figure3c__(title):
    """
    Bifurcation analysis

    :param title: Plot title (panel label)
    :return: None
    """

    __figure3c_continuation__()
    parameters = default_parameters(i_app=0.16)
    t = np.arange(0, 10000, 0.1)
    ic = [-60, 0, 1]

    trajectory = odeint(ode_3d, ic, t, args=(parameters,))  # solve system and overlay - zorder ensures in background
    plt.plot(trajectory[:, 2], trajectory[:, 0], c='grey', zorder=-1e5, linewidth=0.5)

    set_properties(title, y_label="v (mV)", x_limits=[0, 1], x_tick=[0, 0.5, 1], y_tick=[-80, -40, 0, 40], x_label='hs')


def __figure3c_continuation__():
    """
    Actual continuation analysis for 3C. Contains commands to pyDSTool. Performs some formatting and continuation

    :return: None
    """

    parameters = default_parameters(i_app=0.16)
    v, h, h_s = symbols('v h h_s')
    dydt = hs_clamp([v, h, h_s], 0, parameters)  # create a symbolic version of the ode

    DSargs_3 = PyDSTool.args(name='bifn_3')
    DSargs_3.pars = {'h_s': 0}
    DSargs_3.varspecs = {'v': PyDSTool.convertPowers(str(dydt[0])),
                         'h': PyDSTool.convertPowers(str(dydt[1]))}
    DSargs_3.ics = {'v': 0, 'h': 0}

    ode_3 = PyDSTool.Generator.Vode_ODEsystem(DSargs_3)
    ode_3.set(pars={'h_s': 0})
    ode_3.set(ics={'v': -49, "h": 0.4})
    PyCont_3 = PyDSTool.ContClass(ode_3)

    PCargs_3 = PyDSTool.args(name='EQ1_3', type='EP-C')
    PCargs_3.freepars = ['h_s']
    PCargs_3.MaxNumPoints = 350
    PCargs_3.MaxStepSize = 0.1
    PCargs_3.MinStepSize = 1e-5
    PCargs_3.StepSize = 1e-2
    PCargs_3.LocBifPoints = 'all'
    PCargs_3.SaveEigen = True
    PyCont_3.newCurve(PCargs_3)
    PyCont_3['EQ1_3'].backward()

    PyCont_3['EQ1_3'].display(['h_s', 'v'], stability=True, figure=1)

    PCargs_3.name = 'LC1_3'
    PCargs_3.type = 'LC-C'
    PCargs_3.initpoint = 'EQ1_3:H2'
    PCargs_3.freepars = ['h_s']
    PCargs_3.MaxNumPoints = 500
    PCargs_3.MaxStepSize = 0.1
    PCargs_3.LocBifPoints = 'all'
    PCargs_3.SaveEigen = True
    PyCont_3.newCurve(PCargs_3)
    PyCont_3['LC1_3'].backward()
    PyCont_3['LC1_3'].display(('h_s', 'v_min'), stability=True, figure=1)
    PyCont_3['LC1_3'].display(('h_s', 'v_max'), stability=True, figure=1)

    PyCont_3.plot.toggleLabels(visible='off', bytype=['P', 'RG', 'LPC'])  # remove unused labels
    PyCont_3.plot.togglePoints(visible='off', bytype=['P', 'RG', 'LPC'])  # remove unused points

    # hopefully remove rightmost hopf point - this may not always be H1?
    PyCont_3.plot.toggleLabels(visible='off', byname='H1')

    plt.gca().set_title('')


def __figure3d__(title):
    """
    Model response with faster hs rate

    :param title: Plot title (panel label)
    :return: None
    """

    ic = [-65, 1, 1]
    t_solved = np.array([])
    solution = np.array([0, 0, 0])
    currents = [0, 0.16]
    times = [2000, 10000]
    t0 = 0

    for ix, I_app in enumerate(currents):
        t = np.arange(t0, times[ix], 0.1)
        t_solved = np.concatenate((t_solved, t))
        t0 = times[ix]

        parameters = default_parameters(i_app=I_app)
        state = odeint(lambda s, _, p: ode_3d(s, _, p, scale=2), ic, t, args=(parameters,))
        ic = state[-1, :]  # maintain initial condition

        solution = np.vstack((solution, state))

    solution = solution[1:, :]  # TODO: hack for starting shape

    stimulus = np.zeros(t_solved.shape)
    stimulus[t_solved > times[0]] = 1

    plt.plot(t_solved, solution[:, 0], 'k')
    plt.plot(t_solved, 10 * stimulus - 80, 'grey')
    set_properties(title, y_label='$v_m$ (mV)', y_tick=[-60, -40, -20, 0, 20], y_limits=(-80, 20), x_label='t (ms)',
                   x_tick=[0, 5000, 10000], x_limits=(0, 10000))
