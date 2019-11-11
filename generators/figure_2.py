from ode_functions.diff_eq import ode_2d, pulse
from ode_functions.gating import *
from ode_functions.nullclines import nullcline_figure
from plotting import *


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
        __figure2b__('B' + str(ix + 1), panel=ix)

    save_fig('2')


def __figure2a__(title):
    """
    Compute 2d model response to step current input for figure 2A

    :param title: Plot title (panel label)
    :return: None
    """

    """Compute a 3000ms simulation with i_app=0 at t=0 and then i_app=3.5 at t=2000"""
    pattern = {0: 0, 2000: 3.5}
    end_time = 3000
    initial_condition = [-35, 1]

    """Solve ode_2d for a current pulse with above parameters"""
    solution, t_solved, stimulus = pulse(ode_2d, 'i_app', pattern, end_time, initial_condition)
    v = solution[:, 0]

    """Annotate depolarization block potential"""
    block_potential = v[-1]  # since the model remains in depolarization block the last time step is sufficient
    plt.text(2500, block_potential + 10, '{0:.1f}'.format(block_potential), horizontalalignment='center')

    plt.plot(t_solved, v, "k")
    plt.plot(t_solved, stimulus - 70, "grey")
    set_properties(title, y_label="v (mV)", y_tick=[-60, -30, 0, 30], x_tick=[0, 1500, 3000], x_limits=[0, 3000])


def __figure2b__(title, panel=0):
    """
    Plot nullclines for different model regimes in different panels for 2B

    Model regimes are taken from before depolarization block and after

    :param title: Plot title (panel label)
    :param panel: Which plot to make, without current (panel=0) or without current (panel=1)
    :return: None
    """

    """Select appropriate current regime depending on panel"""
    i_app = [0, 3.5][panel]
    voltage = np.arange(-90, 50, 1)

    """Compute nullcline and set the stability"""
    nullcline_figure(voltage, i_app, stability=panel == 1)  # 2nd panel is stable

    y_label = "h" if panel == 0 else ""
    y_ticklabel = None if panel == 0 else []  # todo clean non for default

    set_properties(title, x_label="V (mV)", y_label=y_label, x_tick=[-50, 0, 50], y_tick=[0, 0.1, 0.2, 0.3, 0.4],
                   x_limits=[-75, 50], y_limits=[0, 0.4], y_ticklabel=y_ticklabel)
