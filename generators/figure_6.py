"""Run figure 6.

run() will create all subplots and save them to ../figures
"""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from ode_functions.current import nmda_current, ampa_current
from ode_functions.diff_eq import pulse, ode_3d
from plotting import init_figure, save_fig, set_properties


def run():
    """Top level runner for figure 6.

    :return: None
    """
    print("Running: Figure 6")

    init_figure(size=(6, 6))
    channels = ["nmda", "ampa", "i_app"]
    for ix, channel in enumerate(channels):
        for iy, title in enumerate(["A", "B"]):
            plt.subplot2grid((3, 2), (ix, iy), colspan=1, rowspan=1)
            figure6(title + str(ix + 1), channel, iy)

    save_fig("6")


def figure6(
        title, channel, version, ampa_scale=0.001, nmda_scale=3.7e-5, extract_time=7500,
):
    """Apply a synaptic pulse to the 3d model to determine potential at which depolarization block occurs.

    Original values in the paper do not seem to provide an accurate replication manual scaling was used
    (ampa and nmda_scale) to bring them into a working range

    :param title: Plot title (panel label)
    :param channel: Name of the channel nmda, ampa, i_app
    :param version: Version 0 or 1 of the parameters
    :param ampa_scale: Optional scale factor for ampa conductance
    :param nmda_scale: Optional scale factor for nmda conductance
    :param extract_time: Time where block voltage is computed (defaults to 7500ms)
    :return: None
    """
    # Containers for the function and parameter values for the different channels
    channel_types = {"nmda": nmda_current, "ampa": ampa_current, "i_app": None}
    all_parameters = {
        "nmda": [60 * nmda_scale, 60 * nmda_scale],
        "ampa": [2.3 * ampa_scale, 7 * ampa_scale],
        "i_app": [0.16, 0.32],
    }

    # Initialize and set properties
    channel_function = channel_types[channel]
    on_value = all_parameters[channel][version]
    parameter_name = "i_app" if channel == "i_app" else "g_syn"
    pattern = {
        0: 0,  # off at t=0
        2000: on_value,  # on at t=2000
        8000: 0,  # off at t=8000
    }
    ic = [-65, 1, 1]

    # Create a curried ode_3d to take the synapse and solve it
    synapse_model = partial(ode_3d, synapse=channel_function)
    solution, t_solved, stimulus = pulse(
        model=synapse_model,
        parameter_name=parameter_name,
        temporal_pattern=pattern,
        t_max=10000,
        ic=ic,
    )

    # Plot voltage trace and extract block potential
    plt.plot(t_solved, solution[:, 0], "k")

    extract_ix = np.where(t_solved > extract_time)[0][0]
    block_potential = solution[extract_ix, 0]
    plt.text(
        7500,
        block_potential + 10,
        "{0:.1f}".format(block_potential),
        horizontalalignment="center",
    )

    # Plot parameters
    y_ticklabel = None if version == 0 else []
    y_label = "V (mV)" if version == 0 else ""
    x_ticklabel = None if title[1] == "3" else []  # if row 3
    x_label = "Time (ms)" if title[1] == "3" else ""  # if row 3

    set_properties(
        title,
        y_label=y_label,
        y_tick=[-80, -40, 0],
        y_ticklabel=y_ticklabel,
        x_tick=[0, 5000, 10000],
        x_label=x_label,
        x_ticklabel=x_ticklabel,
    )
