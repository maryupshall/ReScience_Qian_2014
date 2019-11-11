from functools import partial

import numpy as np

from ode_functions.current import nmda_current, ampa_current
from ode_functions.diff_eq import pulse, ode_3d
from plotting import *


def run():
    """
    Top level runner for figure 6
    :return: None
    """

    print("Running: Figure 6")

    init_figure(size=(6, 6))
    __figure6__()

    save_fig('6')


def __figure6__(ampa_scale=1 / 1000, nmda_scale=1 / 27200):
    channel_types = {'nmda': nmda_current, 'ampa': ampa_current, 'i_app': None}

    # parameters for each channel. There are two sets of parameters for the 2 regimes

    parameters_left_figures = {'nmda': [0, 60 * nmda_scale, 0], 'ampa': [0, 2.3 * ampa_scale, 0], 'i_app': [0, 0.16, 0]}
    parameters_right_figures = {'nmda': [0, 60 * nmda_scale, 0], 'ampa': [0, 7 * ampa_scale, 0], 'i_app': [0, 0.32, 0]}

    """ iterate over fig a/b parameter sets"""
    for iz, parameter_sets in enumerate([parameters_left_figures, parameters_right_figures]):
        for ix, channel_type in enumerate(['nmda', 'ampa', 'i_app']):  # iterate over type of channel nmda, ampa, inj.

            extract_time = 7500

            plt.subplot(3, 2, 2 * ix + iz + 1)

            channel_parameters = parameter_sets[channel_type]
            channel_function = channel_types[channel_type]

            if channel_type == 'i_app':
                parameter = 'i_app'
            else:
                parameter = 'g_syn'

            pattern = {0: channel_parameters[0],
                       2000: channel_parameters[1],
                       8000: channel_parameters[2]}  # at t=0 set parameter to p[0], at t=2000 set parameter to p[1]...

            end_time = 10000
            ic = [-65, 1, 1]

            # lambda function is to set the channel_function
            synapse_model = partial(ode_3d, synapse=channel_function)
            solution, t_solved, stimulus = pulse(synapse_model, parameter, pattern, end_time, ic)

            extract_ix = np.where(t_solved > extract_time)[0][0]
            block_potential = solution[extract_ix, 0]
            plt.plot(t_solved, solution[:, 0], 'k')
            plt.text(7500, block_potential + 10, '{0:.1f}'.format(block_potential), horizontalalignment='center')

            # plot setting generation - not germaine to simulations
            title = "A"
            y_label = "V (mV)"
            y_ticklabel = None
            if iz == 1:
                y_label = ""
                y_ticklabel = []
                title = "B"
            title += str(ix + 1)
            x_label = ""
            x_ticklabel = []
            if (2 * ix + iz + 1) == 5 or (2 * ix + iz + 1) == 6:
                plt.plot(t_solved, 10 * stimulus - 80, 'grey')
                x_label = "t (ms)"
                x_ticklabel = None

            set_properties(title, y_label=y_label, y_tick=[-80, -40, 0], y_ticklabel=y_ticklabel,
                           x_tick=[0, 5000, 10000],
                           x_label=x_label, x_ticklabel=x_ticklabel)
