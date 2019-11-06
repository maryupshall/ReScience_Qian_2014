from scipy.integrate import odeint

from helpers.nullclines import nullcline_h, nullcline_v
from helpers.plotting import *
from ode_functions.diff_eq import ode_2d, default_parameters
from ode_functions.gating import *


def run():
    """
    Top level runner for figure 2
    :return: None
    """

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

    ic = [-35, 1]
    t_solved = np.array([])
    solution = np.array([0, 0])  # needs dummy data to keep shape for vstack
    currents = [0, 3.5]
    times = [2000, 3000]
    t0 = 0

    for ix, i_app in enumerate(currents):  # run simulation for multiple i_app and stitch together
        t = np.arange(t0, times[ix], 0.1)
        t0 = times[ix]
        t_solved = np.concatenate((t_solved, t))

        parameters = default_parameters(i_app=i_app)
        state = odeint(ode_2d, ic, t, args=(parameters,))
        ic = state[-1, :]  # maintain the initial condition for when this re-initializes

        solution = np.vstack((solution, state))  # keep track of solution

    solution = solution[1:, :]  # first row is [0,0] dummy data so omit

    stimulus = np.zeros(t_solved.shape)
    stimulus[t_solved > times[0]] = currents[1]

    plt.plot(t_solved, solution[:, 0], "k")
    plt.plot(t_solved, stimulus - 70, "grey")
    set_properties(title, y_label="v (mV)", y_tick=[-60, -30, 0, 30], x_tick=[0, 1500, 3000], x_limits=[0, 3000])


def __figure2b__(title, ix=0):
    """
    Nullcline analysis of different current regimes (no current or with current)

    :param title: Plot title (panel label)
    :param ix: Which plot to make, without current (ix=0) or without current (ix=1)
    :return: None
    """

    i_app_list = [0, 3.5]
    v = np.arange(-90, 50)
    nh = nullcline_h(v)

    i_app = i_app_list[ix]

    plt.plot(v, nh, 'k')
    nv = nullcline_v(v, i_app)

    plt.plot(v, nv, '--', color='grey')
    style = 'k' if ix == 1 else 'none'

    cross_index = np.argmin(np.abs(nv - nh))  # where they are closest i.e. min(err)
    plt.scatter(v[cross_index], nv[cross_index], edgecolors='k', facecolors=style)

    y_label = "h"
    y_ticklabel = None

    if ix == 1:
        y_label = ""
        y_ticklabel = []
    set_properties(title, x_label="v (mV)", y_label=y_label, x_tick=[-50, 0, 50], y_tick=[0, 0.1, 0.2, 0.3, 0.4],
                   x_limits=[-75, 50], y_limits=[0, 0.4], y_ticklabel=y_ticklabel)
