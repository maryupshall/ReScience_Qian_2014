from helpers.nullclines import nullcline_figure
from helpers.plotting import *
from ode_functions.diff_eq import ode_2d, pulse
from ode_functions.gating import *


def run():
    """
    Top level runner for figure 2
    :return: None
    """

    print("Running: Figure 2")

    init_figure(size=(5, 3))
    plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)
    __figure2a__('A')

    for ix, col in enumerate([0, 1]):
        plt.subplot2grid((2, 2), (1, col), colspan=1, rowspan=1)
        __figure2b__('B' + str(ix + 1), ix=ix)

    save_fig('2')


def __figure2a__(title):
    """
    Model response to step current input

    :param title: Plot title (panel label)
    :return: None
    """

    pattern = {0: 0, 2000: 3.5}  # set i_app=0 at t=0. Change to i_app=3.5 at t=2000
    end_time = 3000
    ic = [-35, 1]
    solution, t_solved, stimulus = pulse(ode_2d, 'i_app', pattern, end_time, ic)

    plt.plot(t_solved, solution[:, 0], "k")
    plt.plot(t_solved, stimulus - 70, "grey")
    block_potential = solution[-1, 0]
    plt.text(2500, block_potential + 10, '{0:.1f}'.format(block_potential), horizontalalignment='center')
    set_properties(title, y_label="v (mV)", y_tick=[-60, -30, 0, 30], x_tick=[0, 1500, 3000], x_limits=[0, 3000])


def __figure2b__(title, ix=0):
    """
    Nullcline analysis of different current regimes (no current or with current)

    :param title: Plot title (panel label)
    :param ix: Which plot to make, without current (ix=0) or without current (ix=1)
    :return: None
    """

    v = np.arange(-90, 50, 1)  # compute nullcline every 1mV between -90,V and 50mV
    i_app = [0, 3.5][ix]  # different panels use a different i_app: set the appropriate one

    nullcline_figure(v, i_app, hs=1, plot_h_nullcline=True, stability=ix == 1)

    y_label = "h"
    y_ticklabel = None

    if ix == 1:
        y_label = ""
        y_ticklabel = []
    set_properties(title, x_label="v (mV)", y_label=y_label, x_tick=[-50, 0, 50], y_tick=[0, 0.1, 0.2, 0.3, 0.4],
                   x_limits=[-75, 50], y_limits=[0, 0.4], y_ticklabel=y_ticklabel)
