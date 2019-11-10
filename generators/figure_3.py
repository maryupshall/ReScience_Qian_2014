import PyDSTool
from scipy.integrate import odeint
from scipy.signal import argrelmax
from sympy import *

from helpers.nullclines import nullcline_figure
from helpers.plotting import *
from ode_functions.diff_eq import ode_3d, default_parameters, hs_clamp, pulse
from ode_functions.gating import *


def run():
    """
    Top level runner for figure 3
    :return: None
    """

    print("Running: Figure 3")

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

    pattern = {0: 0, 2000: 0.16}  # set i_app=0 at t=0. Change to i_app=0,16 at t=2000
    end_time = 6000
    ic = [-55, 0, 0]
    solution, t_solved, stimulus = pulse(ode_3d, 'i_app', pattern, end_time, ic)

    spike_time, instantaneous_frequency = __compute_instantaneous_frequency__(solution[:, 0], t_solved)

    # we want the spikes from the second half. Since the for loop runs only twice the last state
    # and t variables correspond to that

    if fig_num == 0:  # plot voltage data
        original_axis = plt.gca()
        plt.plot(t_solved, solution[:, 0], 'k')
        plt.plot(t_solved, 10 * stimulus - 80, 'grey')

        ax2 = original_axis.twinx()  # second axis to plot the "inset frequency"
        ax2.plot(spike_time, instantaneous_frequency, 'ko', markersize=1)
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


def __compute_instantaneous_frequency__(voltage, time, threshold=-40):
    """
    Get per-spike frequency from peak times

    :param voltage: Voltage trace time series
    :param time: Time of voltage points
    :param threshold: Set threshold s.t. peaks below threshold are excluded
    :return: Spike times and frequency
    """

    spike_index = argrelmax(voltage)[0]  # returns ([peaks]) so extract array
    spike_index = spike_index[voltage[spike_index] > threshold]  # filter out spikelets if any?
    instantaneous_frequency = 1 / np.diff(time[spike_index])

    spike_times = time[spike_index[1:]]  # drop first spike time since its used to compute diff
    return spike_times, 1000 * instantaneous_frequency  # convert frequency to Hz


def __figure3b__(title, ix=0):
    """
    Nullcline analysis of different regimes in depolarization block

    :param title: Plot title (panel label)
    :param ix: Which plot to make ix referes to the index if the below i_app_list and hs_list
    :return: None
    """

    v = np.arange(-90, 50)
    i_app = [0, 0.16, 0.16, 0.16][ix]  # different panels (ix) use a different i_app: set the appropriate one
    hs = [0.6, 0.6, 0.2, 0.05][ix]  # different panels (ix) use a different hs: set the appropriate one

    nullcline_figure(v, i_app, hs=hs, plot_h_nullcline=True, stability=ix == 3)

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

    __figure3c_continuation__()  # also makes the plot
    parameters = default_parameters(i_app=0.16)
    t = np.arange(0, 10000, 0.1)
    ic = [-60, 0, 1]

    trajectory = odeint(ode_3d, ic, t, args=(parameters,))  # solve system and overlay - zorder ensures in background
    plt.plot(trajectory[:, 2], trajectory[:, 0], c='grey', zorder=-1e5, linewidth=0.5)  # draw trajectory below bifn

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

    # hopefully remove rightmost hopf point - this may not always be H1? todo
    PyCont_3.plot.toggleLabels(visible='off', byname='H1')

    plt.gca().set_title('')


def __figure3d__(title):
    """
    Model response with faster hs rate

    :param title: Plot title (panel label)
    :return: None
    """

    pattern = {0: 0, 2000: 0.16}  # set i_app=0 at t=0. Change to i_app=0.16 at t=2000
    end_time = 10000
    ic = [-65, 1, 1]

    # need lambda function to set additional scale flag
    solution, t_solved, stimulus = pulse(lambda s, t, p: ode_3d(s, t, p, scale=2), 'i_app', pattern, end_time, ic)

    plt.plot(t_solved, solution[:, 0], 'k')
    plt.plot(t_solved, 10 * stimulus - 80, 'grey')
    set_properties(title, y_label='$v_m$ (mV)', y_tick=[-60, -40, -20, 0, 20], y_limits=(-80, 20), x_label='t (ms)',
                   x_tick=[0, 5000, 10000], x_limits=(0, 10000))
