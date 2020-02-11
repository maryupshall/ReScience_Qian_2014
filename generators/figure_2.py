"""Run figure 2.

run() will create all subplots and save them to ../figures
"""
import matplotlib.pyplot as plt

from ode_functions.diff_eq import ode_2d
from ode_functions.experiment import pulse
from ode_functions.nullclines import nullcline_figure
from plotting import init_figure, save_fig, set_properties


def run():
    """Top level runner for figure 2.

    :return: None
    """
    print("Running: Figure 2")

    init_figure(size=(5, 3))
    plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)
    figure2a("A")

    for ix, col in enumerate([0, 1]):
        plt.subplot2grid((2, 2), (1, col), colspan=1, rowspan=1)
        figure2b("B" + str(ix + 1), panel=ix)

    save_fig("2")


def figure2a(title):
    """Compute 2d model response to step current input for figure 2A.

    :param title: Plot title (panel label)
    :return: None
    """
    # Compute a 3000ms simulation with i_app=0 at t=0 and then i_app=3.5 at t=2000
    pattern = {0: 0, 2000: 3.5}
    end_time = 3000
    initial_condition = [-35, 1]

    # Solve ode_2d for a current pulse with above parameters
    solution, t, waveform = pulse(
        model=ode_2d,
        parameter_name="i_app",
        temporal_pattern=pattern,
        t_max=end_time,
        ic=initial_condition,
    )

    # since the model remains in depolarization block the last time step is sufficient
    v = solution[:, 0]
    block_potential = v[-1]

    plt.text(
        2500,
        block_potential + 10,
        "{0:.1f}".format(block_potential),
        horizontalalignment="center",
    )

    # Plot voltage trace
    plt.plot(t, v, "k")
    plt.plot(t, waveform - 70, "grey")

    # Plot properties
    set_properties(
        title,
        y_label="V (mV)",
        y_tick=[-60, -30, 0, 30],
        x_tick=[0, 1500, 3000],
        x_limits=[0, 3000],
    )


def figure2b(title, panel=0):
    """Plot nullclines for different model regimes in different panels for 2B.

    Model regimes are taken from before depolarization block and after

    :param title: Plot title (panel label)
    :param panel: Which plot to make, without current (panel=0) or without current (panel=1)
    :return: None
    """
    # Select appropriate current regime depending on panel
    i_app = [0, 3.5][panel]

    # Compute nullcline and set the stability
    s = panel == 1  # panel==1 means 2nd panel is stable
    nullcline_figure(v_range=[-90, 50], i_app=i_app, stability=s)

    # Plot Properties
    y_label = "h" if panel == 0 else ""
    y_ticklabel = None if panel == 0 else []
    set_properties(
        title,
        x_label="V (mV)",
        y_label=y_label,
        x_tick=[-50, 0, 50],
        y_tick=[0, 0.1, 0.2, 0.3, 0.4],
        x_limits=[-75, 50],
        y_limits=[0, 0.4],
        y_ticklabel=y_ticklabel,
    )
